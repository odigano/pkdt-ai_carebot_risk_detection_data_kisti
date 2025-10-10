import argparse
import os
import json
import math
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from typing import List, Dict, Any

import ast

# --- 1. 설정 (Configurations) ---

LABEL_ORDER = ["positive", "danger", "critical", "emergency"]

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
    # CSV에 리스트 형태로 저장된 컬럼 목록
    list_columns = ['input_ids', 'attention_mask', 'seq_texts', 'seq_delta_t', 'seq_hours', 'seq_emo_vectors']
    for col in list_columns:
        if col in df.columns:
            # progress_apply를 사용하여 파싱 진행 상황을 시각적으로 보여줍니다.
            df[col] = df[col].progress_apply(ast.literal_eval)
    return df

# --- 2. 데이터셋 클래스 및 함수 ---

class ContextDataset(Dataset):
    """
    전처리된 데이터를 모델 학습에 사용할 수 있는 형태로 변환하는 PyTorch Dataset 클래스.
    텍스트 데이터 외에 시간, 감정, 문맥 기반의 추가 특성을 생성합니다.
    """
    def __init__(self, df: pd.DataFrame, label_map: Dict[str, int]):
        """
        Args:
            df (pd.DataFrame): 전처리 및 파싱이 완료된 DataFrame.
            label_map (Dict[str, int]): 레이블 문자열을 정수 인덱스로 매핑하는 딕셔너리.
        """
        self.df = df
        self.label_map = label_map
        # 감정 특성 관련 컬럼 이름을 미리 추출하여 사용합니다.
        self.emo_cols = [c for c in df.columns if c.startswith("emo_")]
        # 문맥 위험도 계산에 사용할 감정 점수 컬럼의 인덱스를 미리 찾아둡니다.
        self.emo_score_indices = {
            'emergency': self.emo_cols.index('emo_emergency_score') if 'emo_emergency_score' in self.emo_cols else None,
            'critical': self.emo_cols.index('emo_critical_score') if 'emo_critical_score' in self.emo_cols else None,
            'danger': self.emo_cols.index('emo_danger_score') if 'emo_danger_score' in self.emo_cols else None,
        }

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        하나의 데이터 샘플(발화)에 대한 모델 입력값을 생성합니다.
        
        Args:
            idx (int): 가져올 데이터의 인덱스.

        Returns:
            Dict[str, Any]: 모델 입력으로 사용될 텐서 딕셔너리.
        """
        row = self.df.iloc[idx]
        
        # 1. 토크나이징된 텍스트 데이터
        input_ids = row["input_ids"]
        attention_mask = row["attention_mask"]

        # 2. 시간 관련 특성
        last_delta = math.log1p(row["delta_t"])
        last_hour = row["hour"]
        # 시간(hour)을 순환적인 특성으로 변환하여 23시와 0시가 가깝다는 것을 표현
        hour_sin = math.sin(2 * math.pi * last_hour / 24)
        hour_cos = math.cos(2 * math.pi * last_hour / 24)
        
        # 3. 감정 어휘 기반 특성
        emo_vec = row[self.emo_cols].values.astype(np.float32)

        # 4. 문맥 기반 위험도 특성 (Contextual Risk Feature)
        # 이전 대화들의 위험도와 시간 경과를 함께 고려한 특성입니다.
        # 최근에 위험한 발화가 많았을수록 높은 값을 가집니다.
        seq_emo_vectors = row["seq_emo_vectors"]
        seq_delta_t = row["seq_delta_t"]
        
        weighted_context_risk = 0.0
        # 문맥에 2개 이상의 발화가 있을 때만 계산 (현재 발화 제외)
        if len(seq_emo_vectors) > 1:
            # 현재 발화를 제외한 이전 발화들에 대해 반복
            for i in range(len(seq_emo_vectors) - 1):
                emo_vec_context = seq_emo_vectors[i]
                delta_t = seq_delta_t[i+1]  # 해당 발화와 다음 발화 사이의 시간 간격
                
                # 각 위험도 레벨의 감정 점수에 가중치를 부여하여 합산
                utterance_risk_score = 0
                if self.emo_score_indices['emergency'] is not None: utterance_risk_score += emo_vec_context[self.emo_score_indices['emergency']] * 3.0
                if self.emo_score_indices['critical'] is not None: utterance_risk_score += emo_vec_context[self.emo_score_indices['critical']] * 2.0
                if self.emo_score_indices['danger'] is not None: utterance_risk_score += emo_vec_context[self.emo_score_indices['danger']] * 1.0
                
                # 위험 점수가 0보다 클 경우, 시간 경과(delta_t)로 나누어 점수를 감쇠시킴
                # (최근 발화일수록 더 큰 영향을 줌)
                if utterance_risk_score > 0:
                    weighted_context_risk += utterance_risk_score / (delta_t + 1.0) # 분모가 0이 되는 것을 방지
        
        # 최종 문맥 위험도 점수에 log1p를 적용하여 값의 범위를 안정화
        context_risk_feat = math.log1p(weighted_context_risk)

        # 모델에 입력될 최종 딕셔너리 구성
        item = {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "time_feats": torch.tensor([last_delta, hour_sin, hour_cos], dtype=torch.float),
            "emo_feats": torch.tensor(emo_vec, dtype=torch.float),
            "context_risk_feats": torch.tensor([context_risk_feat], dtype=torch.float),
        }
        # 레이블이 있는 경우 (학습/검증 데이터)
        if "label" in row.index and not pd.isna(row["label"]):
            item["label"] = torch.tensor(self.label_map.get(row["label"], -1), dtype=torch.long)

        return item

def collate_fn(batch: List[Dict[str, Any]], pad_token_id: int) -> Dict[str, Any]:
    """
    DataLoader에서 생성된 샘플 리스트를 미니배치(mini-batch)로 구성합니다.
    가변 길이의 시퀀스(input_ids)를 패딩하여 동일한 길이로 만듭니다.
    """
    input_ids = [b["input_ids"] for b in batch]
    attention_mask = [b["attention_mask"] for b in batch]
    
    # `pad_sequence`를 사용하여 배치 내 최대 길이에 맞춰 패딩을 동적으로 적용
    input_ids_padded = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=pad_token_id)
    attention_mask_padded = torch.nn.utils.rnn.pad_sequence(attention_mask, batch_first=True, padding_value=0)

    # 나머지 특성들은 텐서로 변환 후 쌓아줍니다 (stack).
    time_feats = torch.stack([b["time_feats"] for b in batch], dim=0)
    emo_feats = torch.stack([b["emo_feats"] for b in batch], dim=0)
    context_risk_feats = torch.stack([b["context_risk_feats"] for b in batch], dim=0)

    out = {
        "input_ids": input_ids_padded,
        "attention_mask": attention_mask_padded,
        "time_feats": time_feats,
        "emo_feats": emo_feats,
        "context_risk_feats": context_risk_feats,
    }
    if "label" in batch[0]:
        out["labels"] = torch.stack([b["label"] for b in batch], dim=0)
    return out

# --- 3. 모델 (Model) ---

class ContextRiskModel(nn.Module):
    """
    문맥을 고려한 위험도 분류 모델.
    사전 학습된 언어 모델(Encoder)과 LSTM, 추가 특성을 결합한 하이브리드 구조.
    """
    def __init__(self, encoder_name: str, emo_feat_dim: int, time_feat_dim: int = 3, num_labels: int = 4, lstm_hidden_size: int = 256, context_risk_feat_dim: int = 1, use_attention: bool = True):
        super().__init__()
        # 모델의 설정을 저장하여 나중에 모델을 불러올 때 동일한 구조를 재현할 수 있도록 함
        self.config = {
            "encoder_name": encoder_name, "emo_feat_dim": emo_feat_dim, "time_feat_dim": time_feat_dim,
            "num_labels": num_labels, "lstm_hidden_size": lstm_hidden_size, 
            "context_risk_feat_dim": context_risk_feat_dim, "use_attention": use_attention,
        }
        self.use_attention = use_attention
        self.encoder = AutoModel.from_pretrained(encoder_name)
        enc_dim = self.encoder.config.hidden_size
        
        # 양방향 LSTM: 텍스트 시퀀스의 순방향 및 역방향 문맥을 모두 학습
        self.lstm = nn.LSTM(input_size=enc_dim, hidden_size=lstm_hidden_size, num_layers=1, batch_first=True, bidirectional=True)
        
        if self.use_attention:
            # Multi-head Attention: LSTM 출력의 여러 부분에 가중치를 부여하여 중요한 정보를 강조
            self.attention = nn.MultiheadAttention(embed_dim=lstm_hidden_size * 2, num_heads=8, batch_first=True)
            self.attention_norm = nn.LayerNorm(lstm_hidden_size * 2) # 잔차 연결을 위한 Layer Normalization
            pooled_dim = lstm_hidden_size * 2
        else:
            # Attention을 사용하지 않을 경우, LSTM의 마지막 은닉 상태를 사용
            pooled_dim = lstm_hidden_size * 2
            
        # 최종 분류기(Classifier)의 입력 차원:
        # (언어 모델 출력 차원) + (시간 특성 차원) + (감정 특성 차원) + (문맥 위험도 특성 차원)
        input_dim = pooled_dim + time_feat_dim + emo_feat_dim + context_risk_feat_dim
        
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 512), nn.ReLU(), nn.Dropout(0.2), nn.Linear(512, num_labels)
        )

    def forward(self, input_ids, attention_mask, time_feats, emo_feats, context_risk_feats):
        # 1. 언어 모델(Encoder)을 통과시켜 토큰별 임베딩(hidden states)을 얻음
        sequence_output = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        
        # 2. LSTM에 입력하기 전, 패딩을 무시하도록 시퀀스를 압축 (성능 및 효율성 향상)
        lengths = attention_mask.sum(dim=1).long().cpu()
        packed_input = pack_padded_sequence(sequence_output, lengths, batch_first=True, enforce_sorted=False)
        packed_out, (h_n, c_n) = self.lstm(packed_input)
        lstm_output, _ = pad_packed_sequence(packed_out, batch_first=True) # 다시 패딩된 형태로 복원
        
        if self.use_attention:
            # 3a. Attention 적용 및 풀링
            attn_output, _ = self.attention(lstm_output, lstm_output, lstm_output, key_padding_mask=attention_mask == 0)
            # 잔차 연결(Residual Connection) 및 정규화
            pooled = self.attention_norm(lstm_output + attn_output)
            # 어텐션 마스크를 고려하여 평균 풀링 수행
            pooled = self._masked_mean_pooling(pooled, attention_mask)
        else:
            # 3b. Attention 미사용 시, LSTM의 마지막 은닉 상태를 결합하여 사용
            pooled = torch.cat((h_n[-2,:,:], h_n[-1,:,:]), dim=1)
        
        # 4. 언어 모델의 출력과 추가 특성들을 결합
        x = torch.cat([pooled, time_feats, emo_feats, context_risk_feats], dim=-1)
        
        # 5. 최종 분류기를 통과시켜 각 클래스에 대한 로짓(logits)을 반환
        return self.classifier(x)
    
    def _masked_mean_pooling(self, hidden_states, attention_mask):
        """어텐션 마스크를 고려하여 패딩 토큰을 제외하고 평균 풀링을 수행합니다."""
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
        sum_embeddings = torch.sum(hidden_states * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9) # 0으로 나누는 것을 방지
        return sum_embeddings / sum_mask

    def save_pretrained(self, save_directory):
        """모델의 가중치와 설정을 저장합니다."""
        os.makedirs(save_directory, exist_ok=True)
        json.dump(self.config, open(os.path.join(save_directory, "config.json"), 'w'), indent=4)
        torch.save(self.state_dict(), os.path.join(save_directory, "pytorch_model.bin"))

    @classmethod
    def from_pretrained(cls, load_directory):
        """저장된 가중치와 설정으로부터 모델을 불러옵니다."""
        config = json.load(open(os.path.join(load_directory, "config.json"), 'r'))
        model = cls(**config)
        model.load_state_dict(torch.load(os.path.join(load_directory, "pytorch_model.bin"), map_location=torch.device('cpu')))
        return model

class FocalLoss(nn.Module):
    """
    Focal Loss: 클래스 불균형 문제를 해결하기 위한 손실 함수.
    맞추기 쉬운 샘플(easy example)의 손실은 줄이고, 맞추기 어려운 샘플(hard example)의 손실에 더 집중합니다.
    """
    def __init__(self, alpha: List[float] = None, gamma: float = 2.0, reduction: str = 'mean'):
        """
        Args:
            alpha (List[float], optional): 각 클래스에 대한 가중치. 클래스 불균형이 심할 때 사용.
            gamma (float, optional): Focusing 파라미터. 높을수록 쉬운 샘플의 영향력을 줄임.
            reduction (str, optional): 손실 집계 방식 ('mean', 'sum', 'none').
        """
        super().__init__()
        self.alpha = torch.tensor(alpha) if alpha is not None else None
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # 표준 CrossEntropyLoss 계산
        BCE_loss = F.cross_entropy(inputs, targets, reduction='none')
        # pt는 모델이 정답을 맞출 확률
        pt = torch.exp(-BCE_loss)
        # Focal Loss 계산: (1-pt)^gamma * BCE_loss
        F_loss = (1-pt)**self.gamma * BCE_loss
        
        # alpha 가중치가 주어지면, 해당 클래스의 손실에 가중치를 적용
        if self.alpha is not None:
            self.alpha = self.alpha.to(inputs.device)
            F_loss = self.alpha[targets] * F_loss
        if self.reduction == 'mean': return torch.mean(F_loss)
        elif self.reduction == 'sum': return torch.sum(F_loss)
        else: return F_loss

# --- 4. 학습 (Training) ---

def run_train(args):
    """모델 학습 파이프라인 전체를 실행합니다."""
    device = torch.device("cuda" if torch.cuda.is_available() and not args.force_cpu else "cpu")
    print(f"Starting training on device: {device}")

    print(f"Loading and parsing data from {args.train_preprocessed_path}...")
    df = load_and_parse_csv(args.train_preprocessed_path)
    # 유효하지 않은 레이블을 가진 데이터를 필터링
    if "label" in df.columns:
        original_len = len(df)
        df = df[df['label'].isin(LABEL_ORDER)].copy()
        if len(df) < original_len: print(f"Filtered out {original_len - len(df)} rows with invalid labels from training data.")

    train_df = df

    print(f"Loading and parsing validation data from {args.val_preprocessed_path}...")
    val_df = load_and_parse_csv(args.val_preprocessed_path)
    if "label" in val_df.columns:
        original_len = len(val_df)
        val_df = val_df[val_df['label'].isin(LABEL_ORDER)].copy()
        if len(val_df) < original_len: print(f"Filtered out {original_len - len(val_df)} rows with invalid labels from validation data.")

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, use_fast=True)
    label_map = {label: i for i, label in enumerate(LABEL_ORDER)}
    
    # --- Focal Loss의 alpha 값 계산 로직 ---
    # 목표: 데이터가 적은 클래스(불균형)와 위험도가 높은 클래스에 더 높은 가중치를 부여
    # 1. 클래스별 데이터 수의 역빈도(Inverse Frequency)를 기반으로 가중치 계산
    class_counts = train_df['label'].value_counts().reindex(LABEL_ORDER).fillna(0)
    total_samples = len(train_df)
    num_classes = len(LABEL_ORDER)
    inverse_freq_weights = [total_samples / (num_classes * count) if count > 0 else 0.0 for count in class_counts]

    # 2. 위험도에 따른 수동 가중치 부여
    # 이 값들을 조정하여 특정 위험 클래스에 대한 민감도를 제어할 수 있습니다.
    risk_level_weights = [1.0, 4.0, 8.0, 12.0]

    # 3. 두 가중치를 곱하여 최종 alpha 값 생성
    final_alpha_weights = [inv_freq * risk_weight for inv_freq, risk_weight in zip(inverse_freq_weights, risk_level_weights)]
    print(f"Using FocalLoss with final alpha weights: {final_alpha_weights}")

    loss_fct = FocalLoss(alpha=final_alpha_weights, gamma=2.0, reduction='mean').to(device)

    train_dataset = ContextDataset(train_df, label_map)
    val_dataset = ContextDataset(val_df, label_map)
    
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=lambda b: collate_fn(b, pad_token_id), num_workers=args.num_workers, pin_memory=device.type == 'cuda')
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=lambda b: collate_fn(b, pad_token_id), num_workers=args.num_workers, pin_memory=device.type == 'cuda')

    emo_dim = sum(1 for c in df.columns if c.startswith("emo_"))
    model = ContextRiskModel(
        encoder_name=args.encoder_name, emo_feat_dim=emo_dim, num_labels=len(LABEL_ORDER),
        lstm_hidden_size=args.lstm_hidden_size, use_attention=args.use_attention
    ).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(len(train_loader) * args.epochs * 0.1), num_training_steps=len(train_loader) * args.epochs)
    # AMP(Automatic Mixed Precision) 사용 시, 그래디언트 스케일러 초기화
    scaler = torch.amp.GradScaler() if args.use_amp and device.type == 'cuda' else None

    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(args.epochs):
        model.train()
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} | Training"):
            optimizer.zero_grad()
            labels = batch.pop("labels").to(device)
            inputs = {k: v.to(device) for k, v in batch.items()}
            
            if scaler: # AMP 사용
                with torch.amp.autocast(device_type=device.type):
                    loss = loss_fct(model(**inputs), labels)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else: # AMP 미사용
                loss = loss_fct(model(**inputs), labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            scheduler.step()

        # --- 검증 단계 ---
        model.eval()
        total_eval_loss = 0
        for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.epochs} | Validation"):
            with torch.no_grad():
                labels = batch.pop("labels").to(device)
                inputs = {k: v.to(device) for k, v in batch.items()}
                
                if scaler:
                    with torch.amp.autocast(device_type=device.type):
                        logits = model(**inputs)
                else:
                    logits = model(**inputs)
                
                loss = loss_fct(logits, labels)
                total_eval_loss += loss.item()
        
        avg_val_loss = total_eval_loss / len(val_loader)
        print(f"Epoch {epoch+1} | Validation Loss: {avg_val_loss:.4f}")

        # --- 조기 종료(Early Stopping) 및 모델 저장 ---
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            print(f"New best model found! Saving to {args.output_dir}")
            model.save_pretrained(args.output_dir)
            tokenizer.save_pretrained(args.output_dir)
        else:
            patience_counter += 1
            print(f"Validation loss did not improve. Patience: {patience_counter}/{args.early_stopping_patience}")
        
        if patience_counter >= args.early_stopping_patience:
            print("Early stopping triggered.")
            break

    print(f"Training complete. Best model saved with validation loss: {best_val_loss:.4f}")

# 스크립트 실행을 위한 ArgumentParser 설정
parser = argparse.ArgumentParser(description="ContextRiskModel 학습 스크립트")
parser.add_argument("--train_preprocessed_path", type=str, default="../../data/label/preprocessed_train_data.csv", help="전처리된 학습 데이터 파일 경로 (CSV)")
parser.add_argument("--val_preprocessed_path", type=str, default="../../data/label/preprocessed_val_data.csv", help="전처리된 검증 데이터 파일 경로 (CSV)")
parser.add_argument("--output_dir", type=str, default="../../model/label", help="학습된 모델이 저장될 디렉토리")
parser.add_argument("--tokenizer_name", type=str, default="klue/roberta-base", help="사전 학습된 토크나이저 이름")
parser.add_argument("--encoder_name", type=str, default="klue/roberta-base", help="사전 학습된 인코더 모델 이름")
parser.add_argument("--epochs", type=int, default=10, help="총 학습 에폭 수")
parser.add_argument("--batch_size", type=int, default=64, help="배치 크기")
parser.add_argument("--learning_rate", type=float, default=2e-5, help="학습률")
parser.add_argument("--lstm_hidden_size", type=int, default=256, help="LSTM 은닉층 크기")
parser.add_argument("--num_workers", type=int, default=0, help="DataLoader를 위한 워커 수")
parser.add_argument("--early_stopping_patience", type=int, default=5, help="조기 중단을 위한 patience 값")
parser.add_argument("--use_amp", action='store_true', help="Automatic Mixed Precision 사용 여부")
parser.add_argument("--force_cpu", action='store_true', help="CUDA 사용 가능 시에도 CPU 강제 사용")
parser.add_argument("--use_attention", action=argparse.BooleanOptionalAction, default=True, help="모델에 어텐션 메커니즘 사용 여부")

if __name__ == "__main__":
    args = parser.parse_args()
    run_train(args)
