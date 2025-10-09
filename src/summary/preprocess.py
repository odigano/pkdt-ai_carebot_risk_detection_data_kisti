import argparse
import os
import re
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import T5TokenizerFast
from typing import Tuple
from tqdm.auto import tqdm

# pandas의 progress_apply 메서드에 tqdm을 적용하여 진행 상황을 시각적으로 표시합니다.
tqdm.pandas(desc="Applying clean_text")

def clean_text(text: str) -> str:
    """
    요약 모델 학습에 불필요한 특수문자를 제거하고 공백을 정규화하여 텍스트를 정리합니다.
    한글, 영어, 숫자, 그리고 기본적인 공백 문자만 남깁니다.

    Args:
        text (str): 전처리할 원본 문자열.

    Returns:
        str: 특수문자와 불필요한 공백이 제거된 문자열.
    """
    # 한글, 영어(대소문자), 숫자, 공백을 제외한 모든 문자를 공백으로 치환합니다.
    text = re.sub(r'[^가-힣ㄱ-ㅎㅏ-ㅣa-zA-Z0-9\s]', ' ', text)
    # 연속된 여러 개의 공백을 하나의 공백으로 압축합니다.
    text = re.sub(r'\s+', ' ', text)
    # 문자열 양 끝의 공백을 제거합니다.
    return text.strip()

def load_and_split_data(data_path: str, test_size: float = 0.1, random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    CSV 파일에서 데이터를 로드하고, 텍스트를 정제한 후 학습 및 검증 세트로 분할합니다.

    Args:
        data_path (str): 원본 데이터가 담긴 CSV 파일 경로.
        test_size (float): 검증 세트로 분할할 데이터의 비율.
        random_state (int): 재현 가능한 분할을 위한 시드 값.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: (학습용 DataFrame, 검증용 DataFrame).
    """
    df = pd.read_csv(data_path)
    # 요약에 필요한 'text'와 'summary' 컬럼만 선택합니다.
    df = df[['text', 'summary']]
    # 두 컬럼 중 하나라도 비어있는 행은 제거합니다.
    df.dropna(inplace=True)
    
    # 원본 텍스트와 요약 텍스트에 대해 전처리 함수를 적용합니다.
    df['text'] = df['text'].progress_apply(clean_text)
    df['summary'] = df['summary'].progress_apply(clean_text)
    
    # 데이터를 학습 세트와 검증 세트로 분할합니다.
    train_df, val_df = train_test_split(df, test_size=test_size, random_state=random_state)
    
    return train_df, val_df

def tokenize_dataframe(df: pd.DataFrame, tokenizer: T5TokenizerFast, max_input_length: int, max_target_length: int) -> pd.DataFrame:
    """
    DataFrame의 텍스트를 T5 모델에 맞게 토크나이징하여 새로운 컬럼으로 추가합니다.

    Args:
        df (pd.DataFrame): 토크나이징할 데이터가 포함된 DataFrame.
        tokenizer (T5TokenizerFast): T5 모델용 토크나이저.
        max_input_length (int): 입력 텍스트(원본)의 최대 토큰 길이.
        max_target_length (int): 타겟 텍스트(요약문)의 최대 토큰 길이.

    Returns:
        pd.DataFrame: 'input_ids', 'attention_mask', 'labels' 컬럼이 추가된 DataFrame.
    """
    # T5 모델이 요약 과업임을 인지하도록 입력 텍스트 앞에 "summarize: " 접두사를 추가합니다.
    inputs = ["summarize: " + text for text in df["text"]]
    
    # 원본 텍스트를 토크나이징합니다.
    # padding="max_length"는 모든 시퀀스를 max_input_length에 맞춰 패딩 처리합니다.
    model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True, padding="max_length")
    
    # 타겟(정답) 요약문을 토크나이징합니다. 모델의 디코더 입력으로 사용됩니다.
    labels = tokenizer(df["summary"].tolist(), max_length=max_target_length, truncation=True, padding="max_length")

    # 토크나이징 결과를 DataFrame의 새로운 컬럼으로 추가합니다.
    df['input_ids'] = model_inputs['input_ids']
    df['attention_mask'] = model_inputs['attention_mask']
    df['labels'] = labels['input_ids']
    
    return df

def run_preprocess(args: argparse.Namespace):
    """
    전체 전처리 파이프라인을 실행하고, 토크나이징된 데이터를 CSV 파일로 저장합니다.

    Args:
        args (argparse.Namespace): 스크립트 실행 시 전달된 인자.
    """
    print("1. Loading and splitting data...")
    train_df, val_df = load_and_split_data(args.data_path, random_state=args.seed)
    
    print(f"2. Loading tokenizer ('{args.model_name}')...")
    tokenizer = T5TokenizerFast.from_pretrained(args.model_name)
    
    print("3. Tokenizing train and validation datasets...")
    tokenized_train_df = tokenize_dataframe(train_df.copy(), tokenizer, args.max_input_length, args.max_target_length)
    tokenized_val_df = tokenize_dataframe(val_df.copy(), tokenizer, args.max_input_length, args.max_target_length)

    # 출력 디렉토리가 없으면 생성합니다.
    os.makedirs(args.output_dir, exist_ok=True)
    train_csv_path = os.path.join(args.output_dir, "train_data.csv")
    val_csv_path = os.path.join(args.output_dir, "val_data.csv")

    print("4. Saving tokenized datasets to CSV...")
    tokenized_train_df.to_csv(train_csv_path, index=False, encoding='utf-8')
    tokenized_val_df.to_csv(val_csv_path, index=False, encoding='utf-8')
    
    print("\nPreprocessing complete. Files saved to:")
    print(f"- Train data: {train_csv_path}")
    print(f"- Validation data: {val_csv_path}")

# 스크립트 실행을 위한 ArgumentParser 설정
parser = argparse.ArgumentParser(description="요약 모델 학습을 위한 데이터 전처리 스크립트")
parser.add_argument("--data_path", type=str, default="../../data/summary/session_data.csv", help="원본 데이터 CSV 파일 경로")
parser.add_argument("--model_name", type=str, default="paust/pko-t5-small", help="토크나이저로 사용할 사전 학습된 T5 모델 이름")
parser.add_argument("--output_dir", type=str, default="../../data/summary", help="전처리된 CSV 파일이 저장될 디렉토리")
parser.add_argument("--max_input_length", type=int, default=1024, help="입력 텍스트의 최대 토큰 길이")
parser.add_argument("--max_target_length", type=int, default=128, help="타겟 요약문의 최대 토큰 길이")
parser.add_argument("--seed", type=int, default=42, help="데이터 분할 시 사용할 랜덤 시드")

if __name__ == "__main__":
    args = parser.parse_args()
    run_preprocess(args)