import argparse
import os
import sys
import pandas as pd
from datetime import datetime
import gc
import traceback
from tqdm import tqdm

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
    sessionize,
    compute_time_features,
    apply_emotional_features,
    build_context_sequences
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

    total_text_length = df['text'].astype(str).str.len().sum()
    if total_text_length > max_chars:
        raise ValueError(f"분석 가능한 최대 글자 수를 초과했습니다. ({total_text_length:,} > {max_chars:,})")


def run_label_prediction(df: pd.DataFrame, model: ContextRiskModel, tokenizer: AutoTokenizer, device: torch.device) -> pd.DataFrame:
    """
    단일 세션에 대한 위험도 분류 모델의 전처리 및 순차적 예측을 수행합니다.

    Args:
        df (pd.DataFrame): 단일 세션의 데이터프레임.
        model (ContextRiskModel): 로드된 위험도 분류 모델.
        tokenizer (AutoTokenizer): 로드된 토크나이저.
        device (torch.device): 연산을 수행할 장치.

    Returns:
        pd.DataFrame: 예측 결과가 추가된 데이터프레임.
    """
    # 1. 특성 공학 및 문맥 구성
    # 입력 df는 이미 시간 파싱 및 세션화가 완료된 상태로 가정
    df = df.sort_values("uttered_at").reset_index(drop=True)
    df = compute_time_features(df)
    df = apply_emotional_features(df)
    k = model.config.get('k_context', 20)
    # 예측 시점에는 실제 레이블이 없으므로, build_context_sequences는 seq_labels를 None으로 채웁니다.
    df = build_context_sequences(df, k)

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
    
    risk_scores = torch.tensor([i for i, _ in enumerate(LABEL_ORDER)], dtype=torch.float, device=device)
    decay_lambda = 0.00384
    session_context_risk = 0.0
    all_logits = []

    model.eval()
    with torch.no_grad():
        for i in range(len(dataset)):
            item = dataset[i]
            inputs = {k: v.unsqueeze(0).to(device) for k, v in item.items() if k != 'label'}

            if inputs["time_feats"][0, 3].item() == 1.0: # is_session_start
                session_context_risk = 0.0

            inputs["context_risk_feats"] = torch.tensor([[math.log1p(session_context_risk)]], dtype=torch.float, device=device)

            with torch.amp.autocast(device_type=device.type, enabled=(device.type == 'cuda')):
                logits = model(**inputs)
            all_logits.append(logits)

            delta_t = torch.exp(inputs["time_feats"][0, 0]) - 1
            decay_factor = math.exp(-decay_lambda * delta_t.item())
            decayed_risk = session_context_risk * decay_factor

            probs = torch.softmax(logits, dim=-1)
            predicted_risk_score = (probs * risk_scores).sum(dim=-1).item()

            session_context_risk = decayed_risk + predicted_risk_score if predicted_risk_score > 0.05 else decayed_risk

    all_logits_tensor = torch.cat(all_logits, dim=0)
    probabilities = torch.softmax(all_logits_tensor, dim=-1)
    predicted_class_ids = torch.argmax(probabilities, dim=-1)

    # 4. 결과 포맷팅
    df['predicted_label_id'] = predicted_class_ids.cpu().numpy()
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


def format_session_result(df: pd.DataFrame, summary_text: str) -> pd.DataFrame:
    """
    단일 세션의 예측 결과를 요약하여 DataFrame으로 포맷팅합니다.

    Args:
        df (pd.DataFrame): 위험도 예측 결과가 포함된 단일 세션의 데이터프레임.
        summary_text (str): 생성된 요약문.

    Returns:
        pd.DataFrame: 세션 요약 결과가 담긴 DataFrame.
    """
    confidence_df = pd.json_normalize(df['confidence_scores'])
    confidence_df.columns = [f"score_{col}" for col in confidence_df.columns]
    
    dialogue_result_df = df[['doll_id', 'text', 'uttered_at', 'predicted_label']].copy()
    dialogue_result_df.rename(columns={'predicted_label': 'label'}, inplace=True)
    dialogue_result_df.insert(0, 'seq', range(len(dialogue_result_df)))
    dialogue_result_df = pd.concat([dialogue_result_df.reset_index(drop=True), confidence_df.reset_index(drop=True)], axis=1)

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

    evidences = sorted(
        dialogue_result_df.to_dict('records'),
        key=lambda x: float(x[f"score_{overall_label_name}"]),
        reverse=True
    )
    evidence_list = [
        {"seq": v["seq"], "text": v["text"], "score": v[f"score_{overall_label_name}"]}
        for v in evidences
    ][:2]

    scores_data = {label: score for label, score in highest_risk_row.items() if isinstance(label, str) and label.startswith('score_')}
    
    overall_result_data = {
        "doll_id": df['doll_id'].iloc[0],
        "session_id": df['session_id'].iloc[0],
        "dialogue_count": len(df),
        "char_length": len(full_text),
        "label": overall_label_name,
        "treatment_plan": treatment_plan_map.get(overall_label_name, "알 수 없는 위험도"),
        "full_text": full_text,
        "summary": summary_text,
        "evidence": str(evidence_list) # CSV 저장을 위해 문자열로 변환
    }
    overall_result_data.update(scores_data)

    column_order = [
        "doll_id", "session_id", "dialogue_count", "char_length", "label",
        "score_positive", "score_danger", "score_critical", "score_emergency",
        "summary", "evidence", "treatment_plan", "full_text"
    ]

    session_result_df = pd.DataFrame([overall_result_data])
    return session_result_df.reindex(columns=[col for col in column_order if col in session_result_df.columns])


def main(args):
    """메인 실행 함수"""
    try:
        print(f"CSV 파일 로드 중: {args.input_csv}")
        df = pd.read_csv(args.input_csv)
        validate_input_csv(df, args.max_chars)
        print("입력 데이터 유효성 검사 통과.")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"사용 장치: {device}")

        print(f"위험도 분류 모델 로드 중: {'../model/label'}")
        label_model = ContextRiskModel.from_pretrained('../model/label').to(device)
        label_tokenizer = AutoTokenizer.from_pretrained('../model/label')

        print(f"대화 요약 모델 로드 중: {'../model/summary'}")
        summary_model, summary_tokenizer = load_summary_model('../model/summary')

        print("대화 세션 분할 중...")
        df = parse_datetime_column(df)
        df = sessionize(df, gap_seconds=600) # 10분 간격으로 세션 분할

        all_utterance_results = []
        all_session_results = []

        grouped = df.groupby(['doll_id', 'session_id'])
        for (doll_id, session_id), session_df in tqdm(grouped, desc="세션별 예측 처리 중"):
            print(f"\n처리 중: doll_id={doll_id}, session_id={session_id} ({len(session_df)} 발화)")
            
            # 위험도 분류 예측
            prediction_df = run_label_prediction(session_df.copy(), label_model, label_tokenizer, device)
            
            # 대화 요약 예측
            full_text = " ".join(session_df['text'].astype(str).tolist()).strip()
            summary_text = run_summary_prediction(full_text, summary_model, summary_tokenizer)

            # 세션 결과 포맷팅
            session_result_df = format_session_result(prediction_df, summary_text)
            all_session_results.append(session_result_df)

            # 개별 발화 결과 포맷팅
            confidence_df = pd.json_normalize(prediction_df['confidence_scores'])
            confidence_df.columns = [f"score_{col}" for col in confidence_df.columns]
            utterance_result = pd.concat([prediction_df[['doll_id', 'session_id', 'text', 'uttered_at', 'predicted_label']].reset_index(drop=True), confidence_df], axis=1)
            utterance_result.rename(columns={'predicted_label': 'label'}, inplace=True)
            all_utterance_results.append(utterance_result)

        # --- 결과 취합 및 저장 ---
        print("\n결과 취합 및 파일 저장 중...")
        final_session_df = pd.concat(all_session_results, ignore_index=True)
        final_utterance_df = pd.concat(all_utterance_results, ignore_index=True)
        final_utterance_df.insert(0, 'seq', final_utterance_df.groupby(['doll_id', 'session_id']).cumcount())

        output_dir = os.path.dirname(args.input_csv)
        base_filename = os.path.splitext(os.path.basename(args.input_csv))[0]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        session_filepath = os.path.join(output_dir, f"{base_filename}_{timestamp}-prediction_session_multi.csv")
        final_session_df.to_csv(session_filepath, index=False, encoding='utf-8')
        print(f"대화 세션 요약 결과 저장 완료: {session_filepath}")

        utterance_filepath = os.path.join(output_dir, f"{base_filename}_{timestamp}-prediction_utterance_multi.csv")
        final_utterance_df.to_csv(utterance_filepath, index=False, encoding='utf-8')
        print(f"개별 발화 예측 결과 저장 완료: {utterance_filepath}")

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
    parser = argparse.ArgumentParser(description="대화 내용의 위험도 분류 및 요약 예측 스크립트 (다중 세션 처리)")
    parser.add_argument("--input_csv", type=str, default="../data/prediction/val_data.csv", help="분석할 대화 내용이 담긴 CSV 파일 경로")
    parser.add_argument("--max_chars", type=int, default=10000, help="분석 가능한 최대 글자 수")

    args = parser.parse_args()
    main(args)