import argparse
import os
import sys
import pandas as pd
from datetime import datetime
import gc
import traceback

# --- 모델 및 전처리 관련 클래스/함수 임포트 ---
# 경로 문제 해결을 위해 src 디렉토리를 sys.path에 추가
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import torch
import math
from label.train import ContextRiskModel, ContextDataset, LABEL_ORDER # type: ignore
from summary.evaluation import load_model_and_tokenizer as load_summary_model # type: ignore
from transformers import AutoTokenizer

# --- 전처리 함수 임포트 ---
from label.preprocess import (
    parse_datetime_column,
    compute_time_features,
    apply_emotional_features
)
from summary.preprocess import clean_text as clean_text_for_summary


def validate_input_csv(df: pd.DataFrame, max_chars: int) -> None:
    """
    입력된 CSV 데이터의 유효성을 검사합니다.

    Args:
        df (pd.DataFrame): 로드된 데이터프레임.
        max_chars (int): 허용되는 최대 글자 수.

    Raises:
        ValueError: 유효성 검사 실패 시.
    """
    required_columns = ["doll_id", "text", "uttered_at"]
    if not all(col in df.columns for col in required_columns):
        raise ValueError(f"CSV 파일에 필수 컬럼({', '.join(required_columns)})이 없습니다.")

    if df.empty:
        raise ValueError("CSV 파일에 분석할 대화 내용이 없습니다.")

    if df['doll_id'].nunique() != 1:
        raise ValueError("모든 대화의 'doll_id'는 동일해야 합니다.")

    total_text_length = df['text'].astype(str).str.len().sum()
    if total_text_length > max_chars:
        raise ValueError(f"분석 가능한 최대 글자 수를 초과했습니다. ({total_text_length:,} > {max_chars:,})")


def _build_context_for_prediction(df: pd.DataFrame, k: int) -> pd.DataFrame:
    """
    예측을 위해 단일 세션으로 간주하고 문맥 시퀀스를 생성합니다.
    label/preprocess.py의 build_context_sequences와 유사하지만, session_id 없이 동작합니다.

    Args:
        df (pd.DataFrame): 전처리 중인 데이터프레임.
        k (int): 문맥에 포함할 이전 발화의 최대 개수.

    Returns:
        pd.DataFrame: 문맥 시퀀스(seq_*) 컬럼이 추가된 데이터프레임.
    """
    df = df.sort_values(["doll_id", "uttered_at"]).reset_index(drop=True)
    seq_data = []
    emo_cols = [c for c in df.columns if c.startswith("emo_")]

    # doll_id로만 그룹화하여 전체를 단일 세션으로 처리
    for _, group in df.groupby(["doll_id"], sort=False):
        texts, delta_ts, hours = group['text'].tolist(), group['delta_t'].tolist(), group['hour'].tolist()
        emo_vectors = group[emo_cols].values.tolist()
        # 예측 시점에는 실제 레이블이 없으므로 None으로 채웁니다.
        labels = [None] * len(group)
        for i in range(len(group)):
            start_index = max(0, i - k + 1)
            seq_data.append({
                'seq_labels': labels[start_index:i+1], 'seq_texts': texts[start_index:i+1], 'seq_delta_t': delta_ts[start_index:i+1],
                'seq_hours': hours[start_index:i+1], 'seq_emo_vectors': emo_vectors[start_index:i+1],
            })
    return pd.concat([df, pd.DataFrame(seq_data)], axis=1)


def run_label_prediction(df: pd.DataFrame, model: ContextRiskModel, tokenizer: AutoTokenizer, device: torch.device) -> pd.DataFrame:
    """
    위험도 분류 모델의 전처리 및 순차적 예측을 수행합니다.

    Args:
        df (pd.DataFrame): 입력 데이터프레임.
        model (ContextRiskModel): 로드된 위험도 분류 모델.
        tokenizer (AutoTokenizer): 로드된 토크나이저.
        device (torch.device): 연산을 수행할 장치.

    Returns:
        pd.DataFrame: 예측 결과가 추가된 데이터프레임.
    """
    # 1. 특성 공학 및 문맥 구성
    df = parse_datetime_column(df)
    df = df.sort_values("uttered_at").reset_index(drop=True)
    df = compute_time_features(df)
    df = apply_emotional_features(df)
    k = model.config.get('k_context', 20)
    df = _build_context_for_prediction(df, k)

    # 2. 토크나이징
    sep = tokenizer.sep_token
    df['joined_text'] = [f" {sep} ".join(str(t) for t in texts) for texts in df['seq_texts']]
    encodings = tokenizer(
        df['joined_text'].tolist(),
        truncation=True,
        padding=False,
        max_length=tokenizer.model_max_length
    )
    df['input_ids'] = encodings['input_ids']
    df['attention_mask'] = encodings['attention_mask']

    # 3. 순차적 예측
    label_map = {label: i for i, label in enumerate(LABEL_ORDER)}
    dataset = ContextDataset(df, label_map)
    
    # 예측 시 동적으로 문맥 위험도를 계산하기 위한 변수 초기화
    risk_scores = torch.tensor([i for i, _ in enumerate(LABEL_ORDER)], dtype=torch.float, device=device)
    decay_lambda = 0.00384  # 시간 경과에 따른 위험도 감쇠 계수
    session_context_risk = 0.0  # 단일 세션이므로 하나의 누적 위험도 변수만 사용
    all_logits = []

    model.eval()
    with torch.no_grad():
        for i in range(len(dataset)):
            item = dataset[i]
            inputs = {k: v.unsqueeze(0).to(device) for k, v in item.items() if k != 'label'}

            # 세션 시작 플래그 확인 (데이터의 첫 발화)
            if inputs["time_feats"][0, 3].item() == 1.0: # is_session_start
                session_context_risk = 0.0

            # 이전까지 누적된 문맥 위험도를 현재 예측의 입력으로 사용
            inputs["context_risk_feats"] = torch.tensor([[math.log1p(session_context_risk)]], dtype=torch.float, device=device)

            # 모델 예측
            with torch.amp.autocast(device_type=device.type, enabled=(device.type == 'cuda')):
                logits = model(**inputs)
            all_logits.append(logits)

            # 다음 예측을 위해 문맥 위험도 업데이트
            # 1. 시간 경과에 따라 이전 위험도를 감쇠시킴
            delta_t = torch.exp(inputs["time_feats"][0, 0]) - 1
            decay_factor = math.exp(-decay_lambda * delta_t.item())
            decayed_risk = session_context_risk * decay_factor

            # 2. 현재 발화의 예측된 위험도를 계산하여 감쇠된 위험도에 더함
            probs = torch.softmax(logits, dim=-1)
            predicted_risk_score = (probs * risk_scores).sum(dim=-1).item()

            # 예측된 위험도가 임계값(0.05)보다 클 때만 누적
            session_context_risk = decayed_risk + predicted_risk_score if predicted_risk_score > 0.05 else decayed_risk

    all_logits_tensor = torch.cat(all_logits, dim=0)
    probabilities = torch.softmax(all_logits_tensor, dim=-1)
    predicted_class_ids = torch.argmax(probabilities, dim=-1)

    # 4. 결과 포맷팅
    df['predicted_label_id'] = predicted_class_ids.cpu().numpy()
    # 정수 ID를 다시 'positive', 'danger' 등 문자열 라벨로 변환
    df['predicted_label'] = df['predicted_label_id'].map({v: k for k, v in label_map.items()})
    df['confidence_scores'] = [
        {label: f"{score:.4f}" for label, score in zip(LABEL_ORDER, prob_list)}
        for prob_list in probabilities.cpu().numpy().tolist()
    ]

    return df


def run_summary_prediction(text: str, model, tokenizer) -> str:
    """
    대화 요약 모델의 전처리 및 예측을 수행합니다.

    Args:
        text (str): 요약할 전체 대화 텍스트.
        model: 로드된 요약 모델.
        tokenizer: 로드된 요약 토크나이저.

    Returns:
        str: 생성된 요약문.
    """
    device = model.device
    cleaned_text = clean_text_for_summary(text)
    input_text = "summarize: " + cleaned_text
    input_ids = tokenizer.encode(input_text, return_tensors="pt", max_length=1024, truncation=True).to(device)

    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_length=256,
            num_beams=4,
            early_stopping=True
        )
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)


def format_and_save_results(df: pd.DataFrame, summary_text: str, output_dir: str, base_filename: str):
    """
    예측 결과를 JSON 구조에 맞춰 포맷팅하고 두 개의 CSV 파일로 저장합니다.

    Args:
        df (pd.DataFrame): 위험도 예측 결과가 포함된 데이터프레임.
        summary_text (str): 생성된 요약문.
        output_dir (str): 결과 파일을 저장할 디렉토리.
        base_filename (str): 출력 파일명의 기반이 될 원본 파일 이름.
    """
    # --- 1. 개별 발화 결과 (dialogue_result) 생성 및 저장 ---
    # confidence_scores 딕셔너리를 별도의 컬럼으로 확장
    confidence_df = pd.json_normalize(df['confidence_scores'])
    confidence_df.columns = [f"score_{col}" for col in confidence_df.columns]

    dialogue_result_df = df[['doll_id', 'text', 'uttered_at', 'predicted_label']].copy()
    dialogue_result_df.rename(columns={'predicted_label': 'label'}, inplace=True)
    dialogue_result_df.insert(0, 'seq', range(len(dialogue_result_df)))
    
    # 확장된 점수 컬럼을 병합
    dialogue_result_df = pd.concat([dialogue_result_df, confidence_df], axis=1)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    utterance_filepath = os.path.join(
        output_dir, f"{base_filename}_{timestamp}-prediction_utterance.csv"
    )
    dialogue_result_df.to_csv(utterance_filepath, index=False, encoding='utf-8')
    print(f"개별 발화 예측 결과 저장 완료: {utterance_filepath}")

    # --- 2. 전체 세션 결과 (overall_result) 생성 및 저장 ---
    # 세션 내에서 가장 높은 위험도 라벨을 전체 세션의 대표 라벨로 결정
    risk_map = {label: i for i, label in enumerate(LABEL_ORDER)}
    highest_risk_row = dialogue_result_df.loc[dialogue_result_df['label'].map(risk_map).idxmax()]
    
    full_text = " ".join(df['text'].astype(str).tolist()).strip()
    overall_label_name = highest_risk_row['label']

    treatment_plan_map = {
        "positive": "특별한 위험 징후는 없습니다. 지속적으로 모니터링해 주세요.",
        "danger": "주의가 필요한 발화가 감지되었습니다. 반복될 경우 주기적인 안부 확인 및 말벗 서비스 제공을 권장합니다.",
        "critical": "위험도가 높은 발화가 감지되었습니다. 상황에 따라 관리자가 직접 통화하여 심리적 안정을 유도하고, 방문 상담이 필요할 수 있습니다.",
        "emergency": "매우 위급한 발화가 감지되었습니다. 신속하게 상황을 파악한 후 관계 기관에 신고하거나 적극적인 대응이 요구됩니다."
    }

    # 증거(evidence) 추출: 대표 위험도에 대한 신뢰도가 가장 높은 발화 2개를 선택
    evidences = sorted(
        dialogue_result_df.to_dict('records'),
        key=lambda x: float(x[f"score_{overall_label_name}"]),
        reverse=True
    )
    evidence_list = [
        {"seq": v["seq"], "text": v["text"], "score": v[f"score_{overall_label_name}"]}
        for v in evidences
    ][:2] # 예시와 같이 2개 추출

    # confidence_scores를 개별 score_* 컬럼으로 확장
    scores_data = {label: score for label, score in highest_risk_row.items() if isinstance(label, str) and label.startswith('score_')}
    
    overall_result_data = {
        "doll_id": df['doll_id'].iloc[0],
        "dialogue_count": len(df),
        "char_length": len(full_text),
        "label": overall_label_name,
        "treatment_plan": treatment_plan_map.get(overall_label_name, "알 수 없는 위험도"),
        "full_text": full_text,
        "summary": summary_text,
        "evidence": evidence_list
    }
    overall_result_data.update(scores_data)

    # 원하는 컬럼 순서 지정
    column_order = [
        "doll_id", "dialogue_count", "char_length", "label",
        "score_positive", "score_danger", "score_critical", "score_emergency",
        "summary", "evidence", "treatment_plan", "full_text"
    ]

    # 2차원 형태로 변환하고 컬럼 순서를 지정하여 CSV 저장
    session_result_df = pd.DataFrame([overall_result_data])
    # DataFrame에 존재하는 컬럼만 순서를 맞춤
    session_result_df = session_result_df.reindex(columns=[col for col in column_order if col in session_result_df.columns])
    
    session_filepath = os.path.join(
        output_dir, f"{base_filename}_{timestamp}-prediction_session.csv"
    )
    session_result_df.to_csv(session_filepath, index=False, encoding='utf-8')
    print(f"대화 세션 요약 결과 저장 완료: {session_filepath}")


def main(args):
    """메인 실행 함수"""
    try:
        # --- 1. 데이터 로드 및 유효성 검사 ---
        print(f"CSV 파일 로드 중: {args.input_csv}")
        df = pd.read_csv(args.input_csv)
        validate_input_csv(df, args.max_chars)
        print("입력 데이터 유효성 검사 통과.")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"사용 장치: {device}")

        # --- 2. 모델 로드 ---
        print(f"위험도 분류 모델 로드 중: {'../model/label'}")
        label_model = ContextRiskModel.from_pretrained('../model/label').to(device)
        label_tokenizer = AutoTokenizer.from_pretrained('../model/label')

        print(f"대화 요약 모델 로드 중: {'../model/summary'}")
        summary_model, summary_tokenizer = load_summary_model('../model/summary')

        # --- 3. 위험도 분류 예측 ---
        print("위험도 분류 예측 진행 중...")
        prediction_df = run_label_prediction(df.copy(), label_model, label_tokenizer, device)

        # --- 4. 대화 요약 예측 ---
        print("대화 요약 예측 진행 중...")
        full_text = " ".join(df['text'].astype(str).tolist()).strip()
        summary_text = run_summary_prediction(full_text, summary_model, summary_tokenizer)

        # --- 5. 결과 포맷팅 및 저장 ---
        print("결과 포맷팅 및 파일 저장 중...")
        output_dir = os.path.dirname(args.input_csv)
        base_filename = os.path.splitext(os.path.basename(args.input_csv))[0]
        format_and_save_results(prediction_df, summary_text, output_dir, base_filename)

        print("\n모든 예측 작업이 성공적으로 완료되었습니다.")

    except FileNotFoundError:
        print(f"오류: 입력 파일 '{args.input_csv}'을(를) 찾을 수 없습니다.", file=sys.stderr)
        sys.exit(1)
    except ValueError as e:
        print(f"데이터 유효성 검사 오류: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"예상치 못한 오류가 발생했습니다: {e.__class__.__name__}: {e}", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)
    finally:
        # 메모리 정리
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="대화 내용의 위험도 분류 및 요약 예측 스크립트")
    parser.add_argument("--input_csv", type=str, default="../data/prediction/dialogue.csv", help="분석할 대화 내용이 담긴 CSV 파일 경로")
    parser.add_argument("--max_chars", type=int, default=10000, help="분석 가능한 최대 글자 수")

    args = parser.parse_args()
    main(args)