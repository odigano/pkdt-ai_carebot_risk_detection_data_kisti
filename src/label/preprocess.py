import argparse
import os
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer
from typing import Dict

# 사전 학습된 언어 모델의 토크나이저 이름
TOKENIZER_NAME = "klue/roberta-base"

# 데이터 및 전처리 설정
K_CONTEXT = 20  # 모델이 참고할 이전 대화(문맥)의 최대 개수
SESSION_GAP_SECONDS = 10 * 60  # 새로운 대화 세션으로 구분하는 시간 간격 (초)
MAX_SEQ_LEN = 128  # 토크나이저가 처리할 시퀀스의 최대 길이
LABEL_ORDER = ["positive", "danger", "critical", "emergency"]  # 위험도 레이블 순서


def parse_datetime_column(df: pd.DataFrame, col: str = "uttered_at") -> pd.DataFrame:
    """
    DataFrame의 특정 컬럼(기본값: 'uttered_at')을 datetime 객체로 변환합니다.
    시간 기반의 계산을 위해 데이터 타입을 통일하는 단계입니다.

    Args:
        df (pd.DataFrame): 처리할 DataFrame.
        col (str): datetime으로 변환할 컬럼 이름.

    Returns:
        pd.DataFrame: 해당 컬럼이 datetime으로 변환된 DataFrame.
    """
    df[col] = pd.to_datetime(df[col])
    return df

def sessionize(df: pd.DataFrame, gap_seconds: int) -> pd.DataFrame:
    """
    사용자(doll_id)별 발화 데이터를 시간 간격을 기준으로 세션화합니다.
    일정 시간(gap_seconds) 이상 대화가 없으면 새로운 세션으로 간주하여 'session_id'를 부여합니다.

    Args:
        df (pd.DataFrame): 'doll_id'와 'uttered_at' 컬럼이 포함된 DataFrame.
        gap_seconds (int): 새로운 세션을 시작할 시간 간격(초).

    Returns:
        pd.DataFrame: 'session_id' 컬럼이 추가된 DataFrame.
    """
    # 사용자 ID와 발화 시간으로 정렬하여 순서를 보장합니다.
    df = df.sort_values(["doll_id", "uttered_at"]).reset_index(drop=True)
    
    # 사용자별로 이전 발화와의 시간 차이를 계산합니다.
    time_diffs = df.groupby("doll_id")["uttered_at"].diff().dt.total_seconds()
    
    # 새로운 세션 시작 지점을 식별합니다.
    # 1. 사용자가 바뀌는 경우 (doll_id.ne(doll_id.shift()))
    # 2. 시간 차이가 설정된 gap_seconds를 초과하는 경우
    new_session_flags = (df["doll_id"].ne(df["doll_id"].shift())) | (time_diffs > gap_seconds)
    
    # 각 사용자 그룹 내에서 새로운 세션 플래그의 누적 합계를 계산하여 session_id를 부여합니다.
    df['session_id'] = new_session_flags.groupby(df['doll_id']).cumsum()
    return df

def compute_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    시간 관련 특성(delta_t, hour)을 계산하여 DataFrame에 추가합니다.
    - delta_t: 이전 발화와의 시간 차이(초). 세션 시작 발화는 0으로 설정됩니다.
    - hour: 발화가 발생한 시간(0-23).

    Args:
        df (pd.DataFrame): 'doll_id', 'session_id', 'uttered_at' 컬럼이 포함된 DataFrame.

    Returns:
        pd.DataFrame: 'delta_t'와 'hour' 컬럼이 추가된 DataFrame.
    """
    if 'session_id' in df.columns:
        # 세션이 구분되어 있다면, 세션 내에서 시간 차이를 계산합니다.
        df['delta_t'] = df.groupby(['doll_id', 'session_id'])['uttered_at'].diff().dt.total_seconds().fillna(0)
    else:
        # 세션 구분이 없다면, 사용자별로 시간 차이를 계산합니다.
        df['delta_t'] = df.groupby('doll_id')['uttered_at'].diff().dt.total_seconds().fillna(0)
    df["hour"] = df["uttered_at"].dt.hour
    return df

def apply_emotional_features(df: pd.DataFrame, text_col: str = "text") -> pd.DataFrame:
    """
    텍스트 기반의 감정/위험 어휘 사전을 사용하여 감정적 특성을 추출하고 점수화합니다.
    미리 정의된 키워드의 등장 횟수와 가중치를 기반으로 점수를 계산하여 모델의 특성으로 활용합니다.

    Args:
        df (pd.DataFrame): 원본 DataFrame.
        text_col (str): 감정 특성을 추출할 텍스트 컬럼 이름.

    Returns:
        pd.DataFrame: 감정 특성 점수 컬럼(emo_*)이 추가된 DataFrame.
    """
    def extract_emotional_features(text: str) -> Dict[str, float]:
        """단일 텍스트에서 감정 관련 키워드를 찾아 점수를 계산합니다."""
        features = {}
        text = str(text)
        # 위험도별 키워드와 가중치 사전
        risk_lexicon = {
            'emergency': {'keywords': ['도와줘', '구해줘', '살려줘', '응급', '위험', '사고', '병원', '119', '112', '불이야', '죽고 싶어', '죽고 싶다', '자살'], 'weight': 3.0},
            'critical': {'keywords': ['아파', '아프다', '고통', '괴롭다', '괴로워', '우울', '외롭다', '외로워', '쓸쓸'], 'weight': 2.0},
            'danger': {'keywords': ['힘들어', '어려워', '스트레스', '불안', '걱정', '답답'], 'weight': 1.5},
            'positive': {'keywords': ['좋아', '행복', '기뻐', '만족', '감사', '고마워'], 'weight': 0.5}
        }
        for category, data in risk_lexicon.items():
            score = 0
            count = 0
            for keyword in data['keywords']:
                if keyword in text:
                    c = text.count(keyword)
                    score += c * data['weight']
                    count += c
            features[f'emo_{category}_score'] = score
            features[f'emo_{category}_count'] = count
        return features

    # DataFrame의 모든 텍스트에 대해 감정 특성 추출을 적용합니다.
    emo_feats = [extract_emotional_features(t) for t in tqdm(df[text_col].fillna(""), desc="Extracting emotional features")]
    emo_df = pd.DataFrame(emo_feats, index=df.index)
    
    # 생성된 특성 컬럼들의 데이터 타입을 숫자로 변환합니다.
    for col in emo_df.columns:
        if col.startswith('emo_'):
            emo_df[col] = pd.to_numeric(emo_df[col], errors='coerce').fillna(0).astype(float)
    
    # 원본 DataFrame에 감정 특성 DataFrame을 합칩니다.
    return pd.concat([df, emo_df], axis=1)

def build_context_sequences(df: pd.DataFrame, k: int) -> pd.DataFrame:
    """
    각 발화를 기준으로, 이전 k개의 발화를 포함하는 문맥 시퀀스를 생성합니다.
    모델이 현재 발화뿐만 아니라 이전 대화의 흐름을 이해하도록 돕습니다.

    Args:
        df (pd.DataFrame): 세션화 및 시간 특성 계산이 완료된 DataFrame.
        k (int): 문맥에 포함할 이전 발화의 최대 개수.

    Returns:
        pd.DataFrame: 문맥 시퀀스 컬럼(seq_*)이 추가된 DataFrame.
    """
    df = df.sort_values(["doll_id", "session_id", "uttered_at"]).reset_index(drop=True)
    seq_data = []
    emo_cols = [c for c in df.columns if c.startswith("emo_")]
    
    # 세션별로 그룹화하여 문맥이 다른 세션을 침범하지 않도록 합니다.
    for _, group in tqdm(df.groupby(["doll_id", "session_id"], sort=False), desc="Building context sequences"):
        texts = group['text'].tolist()
        delta_ts = group['delta_t'].tolist()
        hours = group['hour'].tolist()
        labels = group['label'].tolist() if 'label' in group.columns else [None] * len(group)
        emo_vectors = group[emo_cols].values.tolist()
        
        # 그룹 내 각 발화에 대해 문맥 시퀀스를 생성합니다.
        for i in range(len(group)):
            start_index = max(0, i - k + 1)
            seq_data.append({
                'seq_labels': labels[start_index:i+1],
                'seq_texts': texts[start_index:i+1],
                'seq_delta_t': delta_ts[start_index:i+1],
                'seq_hours': hours[start_index:i+1],
                'seq_emo_vectors': emo_vectors[start_index:i+1],
            })
            
    seq_df = pd.DataFrame(seq_data)
    return pd.concat([df.reset_index(drop=True), seq_df], axis=1)

def normalize_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    다양한 형태(숫자, 문자열)로 표현된 레이블을 'LABEL_ORDER'에 정의된 표준 형식으로 통일합니다.
    예: 0 -> "positive", "Danger" -> "danger"

    Args:
        df (pd.DataFrame): 'label' 컬럼이 포함된 DataFrame.

    Returns:
        pd.DataFrame: 'label' 컬럼이 표준화된 DataFrame.
    """
    if "label" not in df.columns:
        return df
        
    new_labels = df['label'].copy().astype(object)
    
    # 숫자 레이블을 문자열로 매핑 (e.g., 0 -> "positive")
    numeric_map = {i: label for i, label in enumerate(LABEL_ORDER)}
    # 다양한 문자열 형식을 표준 형식으로 매핑 (e.g., "danger" -> "danger", "Critical" -> "critical")
    valid_map = {lab.lower(): lab for lab in LABEL_ORDER}
    
    # 숫자 형식의 레이블을 먼저 변환합니다.
    numeric_series = pd.to_numeric(df['label'], errors='coerce')
    mask_numeric = numeric_series.notna()
    new_labels.loc[mask_numeric] = numeric_series[mask_numeric].astype(int).map(numeric_map)
    
    # 나머지(문자열 형식) 레이블을 변환합니다.
    new_labels.loc[~mask_numeric] = df.loc[~mask_numeric, 'label'].astype(str).str.strip().str.lower().map(valid_map)
    
    df['label'] = new_labels
    return df

def run_preprocess(args: argparse.Namespace):
    """
    전체 전처리 파이프라인을 실행하고 결과를 CSV 파일로 저장합니다.
    1. 데이터 로드
    2. 시간 컬럼 파싱 및 세션화
    3. 레이블 정규화
    4. 시간 및 감정 특성 추출
    5. 문맥 시퀀스 생성
    6. 텍스트 토크나이징
    7. 결과 저장

    Args:
        args (argparse.Namespace): 스크립트 실행 시 전달된 인자.
    """
    print("--- 1. Starting Data Preprocessing ---")
    df = pd.read_csv(args.input_csv)

    df = parse_datetime_column(df)
    df = sessionize(df, gap_seconds=args.session_gap_seconds)
    df = normalize_labels(df)
    df = compute_time_features(df)
    df = apply_emotional_features(df)
    df = build_context_sequences(df, k=args.k_context)
    
    # 레이블이 없는 데이터는 학습/평가에서 제외합니다.
    if "label" in df.columns:
        df = df.dropna(subset=["label"]).reset_index(drop=True)

    print("--- 2. Pre-tokenizing sequences for performance ---")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, use_fast=True)
    sep = tokenizer.sep_token
    
    # 문맥 시퀀스의 텍스트들을 토크나이저의 분리자(sep_token)로 연결합니다.
    # 이렇게 하면 여러 발화를 하나의 시퀀스로 처리할 수 있습니다.
    tqdm.pandas(desc="Joining context texts")
    df['joined_text'] = [f" {sep} ".join(str(t) for t in texts) for texts in tqdm(df['seq_texts'], desc="Joining context texts")]
    
    print("Batch encoding texts...")
    # 모든 텍스트를 한 번에 토크나이징하여 효율성을 높입니다.
    encodings = tokenizer(
        df['joined_text'].tolist(),
        truncation=True, 
        padding=False,  # 패딩은 추후 DataLoader에서 동적으로 처리
        max_length=args.max_seq_len
    )
    
    df['input_ids'] = encodings['input_ids']
    df['attention_mask'] = encodings['attention_mask']
    
    # 임시로 사용된 컬럼을 삭제합니다.
    df = df.drop(columns=['joined_text'])
        
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    print(f"--- 3. Saving preprocessed data to {args.output_path} ---")
    df.to_csv(args.output_path, index=False, encoding='utf-8')
        
    print("Preprocessing complete.")

# 스크립트 실행을 위한 ArgumentParser 설정
parser = argparse.ArgumentParser(description="데이터 전처리, 특징 추출 및 토크나이징 후 CSV로 저장하는 스크립트")
parser.add_argument("--input_csv", type=str, default="../../data/label/train_data.csv", help="원본 데이터 CSV 파일 경로")
parser.add_argument("--output_path", type=str, default="../../data/label/preprocessed_train_data.csv", help="전처리된 데이터(CSV)가 저장될 경로")
parser.add_argument("--tokenizer_name", type=str, default=TOKENIZER_NAME, help="토크나이저로 사용할 모델 이름")
parser.add_argument("--k_context", type=int, default=K_CONTEXT, help="문맥으로 사용할 이전 발화의 수")
parser.add_argument("--session_gap_seconds", type=int, default=SESSION_GAP_SECONDS, help="새로운 세션을 정의하기 위한 시간 간격(초)")
parser.add_argument("--max_seq_len", type=int, default=MAX_SEQ_LEN, help="입력 시퀀스의 최대 토큰 길이")

if __name__ == "__main__":
    args = parser.parse_args()
    run_preprocess(args)
