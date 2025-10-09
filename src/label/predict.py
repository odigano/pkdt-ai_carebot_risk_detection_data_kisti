import argparse
import os
import sys
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

import ast

# train.py에서 모델, 데이터셋 등 필요한 클래스와 함수를 임포트합니다.
# 이를 통해 학습과 예측 과정에서 동일한 데이터 처리 및 모델 구조를 보장합니다.
from train import ContextRiskModel, ContextDataset, collate_fn, LABEL_ORDER

# tqdm.pandas()를 호출하여 progress_apply를 활성화합니다.
tqdm.pandas(desc="Parsing list-like columns")

def load_and_parse_csv(path: str) -> pd.DataFrame:
    """
    CSV를 로드하고, CSV 저장으로 인해 문자열로 변환된 리스트 형태의 컬럼들을
    ast.literal_eval을 사용하여 다시 파이썬 객체(리스트)로 파싱합니다.

    Args:
        path (str): 로드할 CSV 파일 경로.

    Returns:
        pd.DataFrame: 리스트 컬럼이 파싱된 DataFrame.
    """
    df = pd.read_csv(path)
    list_columns = ['input_ids', 'attention_mask', 'seq_texts', 'seq_delta_t', 'seq_hours', 'seq_emo_vectors']
    for col in list_columns:
        if col in df.columns:
            df[col] = df[col].progress_apply(ast.literal_eval)
    return df

def setup_matplotlib_font():
    """
    Matplotlib에서 한글 폰트를 설정하여 그래프의 라벨이 깨지지 않도록 합니다.
    운영체제에 따라 적절한 폰트를 자동으로 선택합니다.
    """
    try:
        import platform
        if platform.system() == 'Windows': plt.rc('font', family='Malgun Gothic')
        elif platform.system() == 'Darwin': plt.rc('font', family='AppleGothic')
        else: plt.rc('font', family='NanumGothic')
        plt.rcParams['axes.unicode_minus'] = False # 마이너스 부호 깨짐 방지
    except Exception as e:
        print(f'Warning: Could not set Korean font for plots. Error: {e}', file=sys.stderr)

def run_predictions(model, dataloader, device):
    """
    모델 예측을 실행하고, 각 샘플에 대한 예측 라벨과 해당 라벨의 확률을 반환합니다.

    Args:
        model (nn.Module): 학습된 모델.
        dataloader (DataLoader): 예측할 데이터가 포함된 DataLoader.
        device (torch.device): 연산을 수행할 장치 (e.g., 'cuda' or 'cpu').

    Returns:
        Tuple[List[int], List[float]]: (예측된 라벨 ID 리스트, 예측된 라벨의 확률 리스트).
    """
    model.eval() # 모델을 평가 모드로 설정
    all_preds, all_top_probs = [], []
    with torch.no_grad(): # 그래디언트 계산 비활성화
        for batch in tqdm(dataloader, desc="Predicting"):
            # 평가/추론 시에는 라벨이 없을 수 있으므로, 배치에서 제거합니다.
            batch.pop("labels", None) 
            inputs = {k: v.to(device) for k, v in batch.items()}
            
            logits = model(**inputs)
            probs = torch.softmax(logits, dim=-1) # 로짓을 확률로 변환
            top_probs, preds = torch.max(probs, dim=-1) # 가장 높은 확률과 해당 인덱스(예측 라벨)를 추출
            
            all_preds.extend(preds.cpu().numpy().tolist())
            all_top_probs.extend(top_probs.cpu().numpy().tolist())
            
    return all_preds, all_top_probs

def generate_evaluation_report(df, label_order, output_dir):
    """
    실제 라벨이 있을 경우, Classification Report와 Confusion Matrix를 생성하고 저장합니다.
    모델의 성능을 다각도로 평가하기 위한 핵심 함수입니다.

    Args:
        df (pd.DataFrame): 실제 라벨('label')과 예측 라벨('predicted_label')이 포함된 DataFrame.
        label_order (List[str]): 평가에 사용할 라벨 순서.
        output_dir (str): 평가 결과물(리포트, 이미지)을 저장할 디렉토리.
    """
    if 'label' not in df.columns or df['label'].isnull().all():
        print("No 'label' column found or all labels are NaN. Skipping evaluation report.")
        return

    # 실제 라벨과 예측 라벨이 모두 있는 데이터만 필터링하여 평가
    eval_df = df.dropna(subset=['label', 'predicted_label'])
    if eval_df.empty: 
        print("No valid data for evaluation. Skipping report generation.")
        return

    y_true, y_pred = eval_df['label'], eval_df['predicted_label']
    
    # 1. Classification Report 생성 및 저장
    report_dict = classification_report(y_true, y_pred, labels=label_order, target_names=label_order, zero_division=0, output_dict=True)
    report_text = classification_report(y_true, y_pred, labels=label_order, target_names=label_order, zero_division=0)
    
    print("\n[Classification Report]")
    os.makedirs(output_dir, exist_ok=True)
    print(report_text)
    with open(os.path.join(output_dir, "evaluation-confusion_matrix_report.txt"), 'w', encoding='utf-8') as f:
        f.write(report_text)

    # 2. Confusion Matrix 생성 및 이미지 저장
    cm = confusion_matrix(y_true, y_pred, labels=label_order)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_order, yticklabels=label_order)
    plt.title('Confusion Matrix'); plt.xlabel('Predicted Label'); plt.ylabel('True Label')
    plt.savefig(os.path.join(output_dir, "evaluation-confusion_matrix_heatmap.png"))
    plt.close()

    # 3. Classification Report 막대 그래프 생성 및 저장
    report_df = pd.DataFrame(report_dict).iloc[:-1, :].T # support 행 제외
    report_df = report_df.drop(index=['accuracy', 'macro avg', 'weighted avg']) # 요약 행 제외
    report_df.plot(kind='bar', figsize=(12, 8), rot=0)
    plt.title('Classification Report')
    plt.xlabel('Labels')
    plt.ylabel('Scores')
    plt.ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "evaluation-confusion_matrix_report_plot.png"))
    plt.close()
    
    print(f"Evaluation reports saved to '{output_dir}'")

def generate_session_summary(df, label_order, output_dir):
    """
    세션 단위로 예측 결과를 요약하여, 각 세션의 최고 위험도를 파악하고 결과를 저장합니다.
    개별 발화가 아닌 대화의 전체적인 맥락에서 위험도를 판단하는 데 도움을 줍니다.

    Args:
        df (pd.DataFrame): 'doll_id', 'session_id', 'predicted_label' 등이 포함된 DataFrame.
        label_order (List[str]): 위험도 순서가 정의된 라벨 리스트.
        output_dir (str): 요약 결과를 저장할 디렉토리.
    """
    # 라벨 문자열을 위험도 수준(정수)으로 매핑 (e.g., 'positive':0, 'emergency':3)
    risk_map = {label: i for i, label in enumerate(label_order)}

    def get_highest_risk_label(series):
        """세션 내 발화들 중 가장 높은 위험도 라벨을 반환하는 함수."""
        valid_labels = [label for label in series if label in risk_map]
        # risk_map의 값을 기준으로 가장 큰 값을 가진 라벨을 찾음
        return max(valid_labels, key=risk_map.get) if valid_labels else None

    # 사용자(doll_id)와 세션(session_id)으로 그룹화하여 요약 정보 계산
    session_summary = df.groupby(['doll_id', 'session_id']).agg(
        session_start=pd.NamedAgg(column='uttered_at', aggfunc='min'),
        session_end=pd.NamedAgg(column='uttered_at', aggfunc='max'),
        utterance_count=pd.NamedAgg(column='text', aggfunc='size'),
        session_highest_risk=pd.NamedAgg(column='predicted_label', aggfunc=get_highest_risk_label),
        all_labels_in_session=pd.NamedAgg(column='predicted_label', aggfunc=lambda x: sorted(list(x.unique()), key=lambda l: risk_map.get(l, -1)))
    ).reset_index()

    print("\nSession summary generated.")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "evaluation-session_predictions.csv")
    session_summary.to_csv(output_path, index=False, encoding='utf-8')
    print(f"Session-level summary saved to: '{output_path}'")

def run_predict(args):
    """메인 평가/추론 파이프라인을 실행합니다."""
    setup_matplotlib_font()
    device = torch.device("cuda" if torch.cuda.is_available() and not args.force_cpu else "cpu")
    print(f"Using device: {device}")

    print(f"Loading model from: {args.model_dir}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, use_fast=True)
    model = ContextRiskModel.from_pretrained(args.model_dir).to(device)

    print(f"Loading and parsing data from: {args.preprocessed_path}...")
    df = load_and_parse_csv(args.preprocessed_path)

    label_map = {label: i for i, label in enumerate(LABEL_ORDER)}
    dataset = ContextDataset(df, label_map)
    
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        collate_fn=lambda b: collate_fn(b, pad_token_id),
        num_workers=args.num_workers, pin_memory=(device.type == 'cuda')
    )

    # 모델 예측 실행
    preds, top_probs = run_predictions(model, dataloader, device)
    
    # 예측 결과를 원본 DataFrame에 추가
    df['predicted_label_id'] = preds
    df['predicted_label'] = df['predicted_label_id'].map({v: k for k, v in label_map.items()})
    df['predicted_probability'] = top_probs

    # 발화 단위 예측 결과 저장
    os.makedirs(args.output_dir, exist_ok=True)
    utterance_output_path = os.path.join(args.output_dir, "evaluation-utterance_predictions.csv")
    df.to_csv(utterance_output_path, index=False, encoding='utf-8')
    print(f"\nUtterance-level predictions saved to: '{utterance_output_path}'")

    # 평가 모드일 경우, 추가적인 리포트 및 요약 생성
    if args.mode == 'evaluate':
        generate_evaluation_report(df, LABEL_ORDER, args.output_dir)
        generate_session_summary(df, LABEL_ORDER, args.output_dir)

# 스크립트 실행을 위한 ArgumentParser 설정
parser = argparse.ArgumentParser(description="학습된 ContextRiskModel을 평가하거나 추론하는 스크립트")
parser.add_argument("--preprocessed_path", type=str, default="../../data/label/preprocessed_real_data.csv", help="Preprocessed data file path (.csv)")
parser.add_argument("--model_dir", type=str, default="../../model/label", help="학습된 모델(pytorch_model.bin, config.json 등)이 저장된 디렉토리")
parser.add_argument("--output_dir", type=str, default="../../figures/label", help="평가 결과(CSV, 이미지 등)를 저장할 디렉토리")
parser.add_argument("--mode", type=str, choices=['inference', 'evaluate'], default='evaluate', help="'inference': 단순 추론, 'evaluate': 정답과 비교하여 성능 평가")
parser.add_argument("--batch_size", type=int, default=64, help="예측 시 사용할 배치 크기")
parser.add_argument("--num_workers", type=int, default=0, help="DataLoader를 위한 워커 수")
parser.add_argument("--force_cpu", action="store_true", help="CUDA 사용 가능 시에도 CPU를 강제로 사용")

if __name__ == "__main__":
    args = parser.parse_args()
    run_predict(args)
