import argparse
import os
import torch
import sys
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from transformers import T5ForConditionalGeneration, T5TokenizerFast, DataCollatorWithPadding
from torch.utils.data import DataLoader, Dataset
from sentence_transformers import SentenceTransformer, util
from typing import Tuple
from argparse import Namespace

import evaluate
import re
from collections import Counter
from konlpy.tag import Okt # Okt는 클래스 외부에서 초기화하여 정적으로 사용합니다.

class InferenceDataset(Dataset):
    """
    추론(inference) 시 DataLoader를 사용한 배치 처리를 위해
    텍스트 리스트를 토크나이징하여 PyTorch Dataset 객체로 변환하는 클래스입니다.
    """
    def __init__(self, texts, tokenizer, max_length=1024):
        """
        Args:
            texts (list[str]): 요약할 원본 텍스트 리스트.
            tokenizer (T5TokenizerFast): T5 모델용 토크나이저.
            max_length (int): 입력 시퀀스의 최대 토큰 길이.
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        # 데이터셋 생성 시점에 모든 텍스트를 미리 토크나이징합니다.
        self.encodings = self.tokenizer(texts, max_length=self.max_length, truncation=True)

    def __len__(self):
        """데이터셋의 전체 샘플 수를 반환합니다."""
        return len(self.encodings['input_ids'])

    def __getitem__(self, idx):
        """주어진 인덱스(idx)에 해당하는 샘플을 딕셔너리 형태로 반환합니다."""
        return {key: val[idx] for key, val in self.encodings.items()}

class KORougeEvaluator:
    """
    Hugging Face `evaluate` 라이브러리의 ROUGE가 한국어 형태소 분석을 제대로 지원하지 않아
    점수가 0으로 계산되는 문제를 해결하기 위해 직접 구현한 ROUGE 평가 클래스입니다.
    
    Okt 형태소 분석기를 사용하여 텍스트를 형태소 단위로 토큰화한 후,
    이를 기반으로 ROUGE-1, ROUGE-2, ROUGE-L 점수를 계산합니다.
    모든 메서드는 정적(static)으로 구현되어 객체 생성 없이 사용할 수 있습니다.
    """
    _okt_tokenizer = None # Okt 객체를 저장할 클래스 변수

    @classmethod
    def _get_okt_tokenizer(cls) -> Okt:
        """
        Okt 형태소 분석기 인스턴스를 싱글톤(Singleton)처럼 관리합니다.
        최초 호출 시에만 객체를 생성하고, 이후에는 계속 재사용하여 메모리 및 속도 효율을 높입니다.
        """
        if cls._okt_tokenizer is None:
            cls._okt_tokenizer = Okt()
        return cls._okt_tokenizer

    @staticmethod
    def _okt_tokenize(text: str) -> list[str]:
        """
        주어진 텍스트를 Okt 형태소 분석기로 토큰화합니다.
        정규식을 사용하여 알파벳, 숫자, 한글, 하이픈(-)만 유효한 토큰으로 간주하고,
        조사, 구두점 등 불필요한 요소는 제거합니다.
        """
        okt = KORougeEvaluator._get_okt_tokenizer() # 클래스 메서드를 통해 Okt 인스턴스 가져오기
        toks = okt.morphs(text)
        # ROUGE 점수 계산의 정확도를 위해 유의미한 형태소만 필터링합니다.
        cleaned = [t for t in toks if re.match(r'^[\w가-힣]+$', t)]
        return cleaned

    @staticmethod
    def _ngrams(tokens: list[str], n: int) -> list[tuple]:
        """
        ROUGE-N 점수 계산에 사용될 n-gram을 생성합니다.
        예: tokens=['a', 'b', 'c'], n=2 -> [('a', 'b'), ('b', 'c')]
        주어진 토큰 리스트에서 n-grams를 생성합니다.
        """
        if n <= 0 or len(tokens) < n:
            return []
        return [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]

    @staticmethod
    def _overlap_counts(ref_ngrams: list[tuple], pred_ngrams: list[tuple]) -> tuple[int, int, int]:
        """
        참조(reference)와 예측(prediction) n-gram 리스트 간의 겹치는 개수를 계산합니다.
        Counter를 사용하여 각 n-gram의 등장 횟수를 세고, 두 리스트에 공통으로 나타나는
        n-gram의 최소 등장 횟수를 합산하여 겹치는 수를 구합니다.

        Returns: (겹치는 n-gram 수, 참조 n-gram 총 개수, 예측 n-gram 총 개수)
        """
        ref_c = Counter(ref_ngrams)
        pred_c = Counter(pred_ngrams)
        overlap = 0
        for gram, cnt in pred_c.items():
            overlap += min(cnt, ref_c.get(gram, 0))
        return overlap, sum(ref_c.values()), sum(pred_c.values())

    @staticmethod
    def _lcs_length_list(a: list[str], b: list[str]) -> int:
        """
        두 토큰 리스트 간의 최장 공통 부분 수열(LCS) 길이를 계산합니다.
        메모리 효율성을 위해 2차원 DP 테이블 대신 1차원 배열 2개를 사용하는 동적 계획법을 사용합니다.
        """
        la, lb = len(a), len(b)
        if la == 0 or lb == 0:
            return 0
        
        # 1D rolling DP for space efficiency
        dp = [0] * (lb + 1)
        for i in range(1, la + 1):
            prev = 0 # dp[i-1][j-1]
            ai = a[i-1]
            for j in range(1, lb + 1):
                tmp = dp[j] # dp[i-1][j]
                if ai == b[j-1]:
                    dp[j] = prev + 1
                else:
                    # max(dp[j], dp[j-1])
                    if dp[j] < dp[j-1]: # dp[j] is dp[i-1][j], dp[j-1] is dp[i][j-1]
                        dp[j] = dp[j-1]
                prev = tmp
        return dp[lb]

    @staticmethod
    def _f1_from_pr(p: float, r: float) -> float:
        """
        정밀도(Precision)와 재현율(Recall)로부터 F1 점수를 계산합니다.
        """
        if p + r == 0:
            return 0.0
        return 2 * p * r / (p + r)

    @staticmethod
    def _rouge_scores_single_token(ref: str, pred: str) -> dict:
        """
        단일 참조-예측 텍스트 쌍에 대한 ROUGE-1, ROUGE-2, ROUGE-L 점수를 계산하는 내부 헬퍼 함수입니다.
        """
        r_tokens = KORougeEvaluator._okt_tokenize(ref)
        p_tokens = KORougeEvaluator._okt_tokenize(pred)

        # ROUGE-1 (unigram)
        r1_overlap, r1_ref_count, r1_pred_count = KORougeEvaluator._overlap_counts(
            KORougeEvaluator._ngrams(r_tokens, 1), KORougeEvaluator._ngrams(p_tokens, 1)
        )
        r1_recall = r1_overlap / r1_ref_count if r1_ref_count > 0 else 0.0
        r1_prec = r1_overlap / r1_pred_count if r1_pred_count > 0 else 0.0
        r1_f1 = KORougeEvaluator._f1_from_pr(r1_prec, r1_recall)

        # ROUGE-2 (bigram)
        r2_overlap, r2_ref_count, r2_pred_count = KORougeEvaluator._overlap_counts(
            KORougeEvaluator._ngrams(r_tokens, 2), KORougeEvaluator._ngrams(p_tokens, 2)
        )
        r2_recall = r2_overlap / r2_ref_count if r2_ref_count > 0 else 0.0
        r2_prec = r2_overlap / r2_pred_count if r2_pred_count > 0 else 0.0
        r2_f1 = KORougeEvaluator._f1_from_pr(r2_prec, r2_recall)

        # ROUGE-L (LCS on token sequences)
        lcs_len = KORougeEvaluator._lcs_length_list(r_tokens, p_tokens)
        rl_recall = lcs_len / len(r_tokens) if len(r_tokens) > 0 else 0.0
        rl_prec = lcs_len / len(p_tokens) if len(p_tokens) > 0 else 0.0
        rl_f1 = KORougeEvaluator._f1_from_pr(rl_prec, rl_recall)

        return {
            "ref_tokens": r_tokens,
            "pred_tokens": p_tokens,
            "rouge1": {"overlap": r1_overlap, "ref_count": r1_ref_count, "pred_count": r1_pred_count, "recall": r1_recall, "precision": r1_prec, "f1": r1_f1},
            "rouge2": {"overlap": r2_overlap, "ref_count": r2_ref_count, "pred_count": r2_pred_count, "recall": r2_recall, "precision": r2_prec, "f1": r2_f1},
            "rougeL": {"lcs": lcs_len, "ref_len": len(r_tokens), "pred_len": len(p_tokens), "recall": rl_recall, "precision": rl_prec, "f1": rl_f1},
        }

    @staticmethod
    def rouge_scores_batch(predictions: list[str], references: list[str]) -> tuple[list[dict], dict]:
        """
        예측 요약(predictions)과 참조 요약(references) 리스트 전체에 대한
        평균 ROUGE 점수를 계산하여 반환합니다.
        이 클래스에서 외부 호출을 위해 설계된 메인 메서드입니다.
        """
        assert len(references) == len(predictions), "refs와 preds 길이가 같아야 합니다."
        n = len(references)
        
        # 각 ROUGE 지표의 precision, recall, f1 합계를 저장할 변수 초기화
        sum_r1_p = sum_r1_r = sum_r1_f1 = 0.0
        sum_r2_p = sum_r2_r = sum_r2_f1 = 0.0
        sum_rl_p = sum_rl_r = sum_rl_f1 = 0.0

        for i, (r, p) in enumerate(zip(references, predictions)):
            res = KORougeEvaluator._rouge_scores_single_token(r, p)
            
            sum_r1_p += res["rouge1"]["precision"]
            sum_r1_r += res["rouge1"]["recall"]
            sum_r1_f1 += res["rouge1"]["f1"]
            sum_r2_p += res["rouge2"]["precision"]
            sum_r2_r += res["rouge2"]["recall"]
            sum_r2_f1 += res["rouge2"]["f1"]
            sum_rl_p += res["rougeL"]["precision"]
            sum_rl_r += res["rougeL"]["recall"]
            sum_rl_f1 += res["rougeL"]["f1"]

        # 평균 계산
        return {
            "rouge1_p_mean": sum_r1_p / n if n > 0 else 0.0,
            "rouge1_r_mean": sum_r1_r / n if n > 0 else 0.0,
            "rouge1_f1_mean": sum_r1_f1 / n if n > 0 else 0.0,
            "rouge2_p_mean": sum_r2_p / n if n > 0 else 0.0,
            "rouge2_r_mean": sum_r2_r / n if n > 0 else 0.0,
            "rouge2_f1_mean": sum_r2_f1 / n if n > 0 else 0.0,
            "rougeL_p_mean": sum_rl_p / n if n > 0 else 0.0,
            "rougeL_r_mean": sum_rl_r / n if n > 0 else 0.0,
            "rougeL_f1_mean": sum_rl_f1 / n if n > 0 else 0.0,
        }

def load_model_and_tokenizer(model_path: str) -> Tuple[T5ForConditionalGeneration, T5TokenizerFast]:
    """
    지정된 경로에서 사전 학습/파인튜닝된 T5 모델과 토크나이저를 로드합니다.
    사용 가능한 경우 모델을 CUDA 장치로 이동시킵니다.

    Args:
        model_path (str): 모델 파일과 토크나이저 파일이 저장된 디렉토리 경로.
    Returns:
        (T5ForConditionalGeneration, T5TokenizerFast): 로드된 모델과 토크나이저 객체.
    """
    print(f"Loading model and tokenizer from {model_path}...")
    model = T5ForConditionalGeneration.from_pretrained(model_path)
    tokenizer = T5TokenizerFast.from_pretrained(model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return model, tokenizer

def generate_summary(model: T5ForConditionalGeneration, tokenizer: T5TokenizerFast, text: str) -> str:
    """
    단일 텍스트에 대한 요약을 생성합니다.
    T5 모델의 과업 지시를 위해 입력 텍스트 앞에 "summarize: " 접두사를 추가합니다.
    빔 서치(beam search)를 사용하여 더 자연스러운 문장을 생성합니다.

    Args:
        model (T5ForConditionalGeneration): 요약 생성을 위한 T5 모델.
        tokenizer (T5TokenizerFast): 텍스트 인코딩/디코딩을 위한 토크나이저.
        text (str): 요약할 원본 텍스트.
    Returns:
        str: 생성된 요약문.
    """
    device = model.device
    input_text = "summarize: " + text
    input_ids = tokenizer.encode(input_text, return_tensors="pt", max_length=1024, truncation=True).to(device)

    with torch.no_grad():
        output_ids = model.generate( # 빔 서치(num_beams=4)를 사용하여 생성 품질 향상
            input_ids,
            max_length=128,
            num_beams=4,
            early_stopping=True
        )
    
    summary = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return summary

def batch_evaluate(model: T5ForConditionalGeneration, tokenizer: T5TokenizerFast, args: Namespace):
    """
    검증 데이터셋 전체에 대해 요약을 생성하고, 다양한 평가 지표(ROUGE, BLEU, BERTScore 등)를
    일괄적으로 계산하여 성능을 종합적으로 평가합니다.
    결과는 텍스트 리포트, 예측 결과 CSV, 성능 시각화 그래프로 저장됩니다.
    """
    # Matplotlib 한글 폰트 설정
    try:
        import platform
        if platform.system() == 'Windows': plt.rc('font', family='Malgun Gothic')
        elif platform.system() == 'Darwin': plt.rc('font', family='AppleGothic')
        else: plt.rc('font', family='NanumGothic')
        plt.rcParams['axes.unicode_minus'] = False
    except Exception as e:
        print(f'Warning: Could not set Korean font for plots. Error: {e}', file=sys.stderr)

    print(f"Loading dataset from {args.val_csv_path} for evaluation...")
    eval_df = pd.read_csv(args.val_csv_path)
    
    # 예측 결과를 담을 DataFrame 복사본 생성
    predictions_df = eval_df.copy()

    # --- 평가 지표 라이브러리 로드 ---
    print("Loading evaluation metrics...")
    bleu_metric = evaluate.load('bleu')
    meteor_metric = evaluate.load('meteor')
    bertscore_metric = evaluate.load('bertscore')
    sbert_model = SentenceTransformer(args.sbert_model) # 문장 임베딩 모델

    print("Generating summaries for evaluation using batch processing...")
    
    texts_to_summarize = ["summarize: " + text for text in eval_df['text']]
    references = eval_df['summary'].tolist()
    
    # DataLoader를 사용하여 배치 처리
    dataset = InferenceDataset(texts_to_summarize, tokenizer, max_length=args.max_input_length)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    dataloader = DataLoader(dataset, batch_size=args.eval_batch_size, collate_fn=data_collator)

    predictions = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Generating summaries"):
            input_ids = batch['input_ids'].to(model.device)
            attention_mask = batch['attention_mask'].to(model.device)
            
            output_ids = model.generate(input_ids, attention_mask=attention_mask, max_length=128, num_beams=4, early_stopping=True)
            batch_preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
            predictions.extend(batch_preds)

    # 예측 결과를 DataFrame에 추가
    predictions_df['generated_summary'] = predictions
    # 필요한 컬럼만 선택하여 저장 (예: 원본, 실제 요약, 생성된 요약)
    output_predictions_df = predictions_df[['text', 'summary', 'generated_summary']]

    # --- 각 평가 지표 계산 ---
    print("Calculating all metrics...")
    rouge_results = KORougeEvaluator.rouge_scores_batch(references=references, predictions=predictions) # 직접 구현한 ROUGE
    bleu_results = bleu_metric.compute(predictions=predictions, references=[[ref] for ref in references]) # BLEU 점수
    meteor_results = meteor_metric.compute(predictions=predictions, references=references) # METEOR 점수
    bertscore_results = bertscore_metric.compute(predictions=predictions, references=references, lang="ko") # BERTScore (F1 기준)

    # --- SBERT를 이용한 문장 임베딩 유사도 계산 ---
    print("Encoding sentences with SBERT...")
    pred_embeddings = sbert_model.encode(predictions, convert_to_tensor=True, show_progress_bar=True)
    ref_embeddings = sbert_model.encode(references, convert_to_tensor=True, show_progress_bar=True)
    # 코사인 유사도 계산 후, 각 쌍의 유사도(대각 행렬)의 평균을 구함
    cosine_scores = util.cos_sim(pred_embeddings, ref_embeddings)
    sbert_similarity = torch.mean(torch.diag(cosine_scores)).item()

    results_df = pd.DataFrame({
        'Metric': [
            'ROUGE-1',
            'ROUGE-2',
            'ROUGE-L',
            'BLEU',
            'METEOR',
            # 'BERT Score (P)',
            # 'BERT Score (R)',
            'BERT Score (F1)',
            'SBERT Similarity'
        ],
        'Score': [
            rouge_results['rouge1_f1_mean'] * 100,
            rouge_results['rouge2_f1_mean'] * 100,
            rouge_results['rougeL_f1_mean'] * 100,
            bleu_results['bleu'] * 100,
            meteor_results['meteor'] * 100,
            # pd.Series(bertscore_results['precision']).mean() * 100,
            # pd.Series(bertscore_results['recall']).mean() * 100,
            pd.Series(bertscore_results['f1']).mean() * 100,
            sbert_similarity * 100
        ]
    })
    results_df['Score'] = results_df['Score'].round(2)

    print("\n--- Comprehensive Evaluation Results ---")
    print(results_df.to_string(index=False))

    # --- 결과 저장 (텍스트 리포트, 예측 결과 CSV, 성능 그래프) ---
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"\nSaving results to {args.output_dir}...")

    results_txt_path = os.path.join(args.output_dir, "evaluation-metrics_report.txt")
    results_df.to_csv(results_txt_path, sep='\t', index=False)

    # 예측 결과 CSV 파일 저장
    predictions_csv_path = os.path.join(args.output_dir, "evaluation-summary_predictions.csv")
    output_predictions_df.to_csv(predictions_csv_path, index=False, encoding='utf-8')

    # --- 성능 시각화 그래프 생성 및 저장 ---
    plt.figure(figsize=(12, 8))
    sns.set_style("whitegrid")
    
    barplot = sns.barplot(x='Score', y='Metric', data=results_df, hue='Metric', palette='viridis', legend=False)
    plt.title('Summarization Model Evaluation Metrics', fontsize=16)
    plt.xlabel('Score', fontsize=12)
    plt.ylabel('Metric', fontsize=12)
    plt.xlim(0, 100)

    for p in barplot.patches:
        width = p.get_width()
        plt.text(width + 1, p.get_y() + p.get_height() / 2. + 0.1, f'{width:.2f}', ha="left")

    results_img_path = os.path.join(args.output_dir, "evaluation-metrics_plot.png")
    plt.savefig(results_img_path, bbox_inches='tight')
    plt.close()

    print(f"Results saved successfully to {results_txt_path}, {results_img_path}, and {predictions_csv_path}")

def run_evaluate(args: Namespace):
    """
    메인 평가 함수.
    데이터셋 전체 평가를 수행합니다.
    """
    model, tokenizer = load_model_and_tokenizer(args.model_path)
    # 검증 데이터셋에 대한 종합적인 성능 평가 수행
    batch_evaluate(model, tokenizer, args)

# --- 스크립트 실행을 위한 ArgumentParser 설정 ---
parser = argparse.ArgumentParser(description="학습된 T5 모델로 종합 평가를 수행합니다.")

parser.add_argument("--model_path", type=str, default="../../model/summary", help="파인튜닝된 모델이 저장된 경로")
parser.add_argument("--val_csv_path", type=str, default="../../data/summary/val_data.csv", help="평가에 사용할 검증 데이터셋 CSV 파일 경로")
parser.add_argument("--sbert_model", type=str, default="BM-K/KoSimCSE-roberta-multitask", help="SBERT 유사도 계산에 사용할 모델 이름")
parser.add_argument("--eval_batch_size", type=int, default=8, help="평가 시 사용할 배치 크기")
parser.add_argument("--max_input_length", type=int, default=1024, help="입력 텍스트의 최대 토큰 길이")
parser.add_argument("--output_dir", type=str, default="../../figures/summary", help="평가 결과(텍스트, 이미지)가 저장될 디렉토리")

if __name__ == "__main__":
    args = parser.parse_args()
    run_evaluate(args)