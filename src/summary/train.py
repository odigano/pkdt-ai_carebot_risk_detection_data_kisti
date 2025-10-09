import argparse
import os
import torch
import ast
from transformers import (
    DataCollatorForSeq2Seq,
    T5ForConditionalGeneration,
    T5TokenizerFast,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
)
from datasets import load_dataset
from argparse import Namespace

def load_dataset_from_csv(csv_path: str):
    """
    Hugging Face의 'datasets' 라이브러리를 사용하여 CSV 파일을 효율적으로 로드하고,
    문자열로 저장된 토큰 ID 리스트를 실제 리스트 객체로 변환합니다.

    Args:
        csv_path (str): 로드할 CSV 파일 경로.

    Returns:
        Dataset: 'datasets' 라이브러리 형식의 데이터셋 객체.
    """
    print(f"Loading dataset from {csv_path}...")
    # 'datasets' 라이브러리는 대용량 데이터를 메모리에 효율적으로 로드하고 처리하는 데 유용합니다.
    dataset = load_dataset('csv', data_files=csv_path, split='train')
    
    # CSV에 저장되면서 '[1, 2, 3]'과 같이 문자열로 변환된 컬럼들을
    # ast.literal_eval을 사용하여 실제 파이썬 리스트 객체로 변환합니다.
    # num_proc=4는 이 변환 작업을 병렬로 처리하여 속도를 높입니다.
    dataset = dataset.map(
        lambda x: {
            'input_ids': ast.literal_eval(x['input_ids']), 
            'attention_mask': ast.literal_eval(x['attention_mask']), 
            'labels': ast.literal_eval(x['labels'])
        }, 
        num_proc=4
    )
    return dataset

def run_train(args: Namespace):
    """
    전처리된 CSV 데이터셋을 불러와 T5 요약 모델을 파인튜닝합니다.

    Args:
        args (Namespace): 스크립트 실행 시 전달된 인자.
    """
    # 1. 데이터셋 경로 설정 및 로드
    train_csv_path = os.path.join(args.data_dir, "train_data.csv")
    val_csv_path = os.path.join(args.data_dir, "val_data.csv")
    
    train_dataset = load_dataset_from_csv(train_csv_path)
    val_dataset = load_dataset_from_csv(val_csv_path)

    # 2. 모델 및 토크나이저 로드
    print(f"Loading model and tokenizer from '{args.model_name}'...")
    model = T5ForConditionalGeneration.from_pretrained(args.model_name)
    tokenizer = T5TokenizerFast.from_pretrained(args.model_name)

    # 3. TrainingArguments 설정
    # Hugging Face Trainer를 사용하기 위한 학습 관련 모든 설정을 정의합니다.
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        logging_dir=f"{args.output_dir}/logs",
        logging_steps=100,
        eval_strategy="epoch",  # 매 에폭마다 검증 수행
        save_strategy="epoch",        # 매 에폭마다 모델 저장
        load_best_model_at_end=True,  # 학습 종료 후 가장 좋은 성능의 모델을 로드
        metric_for_best_model="eval_loss", # 최고 성능 모델 선택의 기준 (검증 손실)
        greater_is_better=False,      # metric_for_best_model의 값이 낮을수록 좋음 (손실이므로)
        gradient_accumulation_steps=4, # 그래디언트를 여러 스텝에 걸쳐 누적하여, 메모리 부족 시 배치 크기를 늘리는 효과
        fp16=torch.cuda.is_available(), # GPU 사용 가능 시, 혼합 정밀도(Mixed Precision) 학습으로 속도 향상 및 메모리 절약
        prediction_loss_only=True,    # 검증 시 손실(loss)만 계산하여 메모리 사용량 줄임 (OOM 방지)
    )

    # DataCollator: 배치 내에서 동적으로 패딩을 처리해주는 역할
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    # 4. Trainer 초기화 및 학습 시작
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience)], # 조기 종료 콜백 추가
    )

    print("Starting model training...")
    trainer.train()
    print("Training complete.")

    # 5. 최종 모델 저장
    # load_best_model_at_end=True로 설정했기 때문에, 검증 손실이 가장 낮았던 시점의 모델이 저장됩니다.
    print(f"Saving best model to '{args.output_dir}'...")
    trainer.save_model()
    tokenizer.save_pretrained(args.output_dir)
    print("Model and tokenizer saved.")

# 스크립트 실행을 위한 ArgumentParser 설정
parser = argparse.ArgumentParser(description="요약 모델 학습 스크립트")
parser.add_argument("--data_dir", type=str, default="../../data/summary", help="전처리된 훈련 및 검증 데이터셋 CSV 파일이 포함된 디렉토리")
parser.add_argument("--model_name", type=str, default="paust/pko-t5-small", help="학습에 사용할 기반 모델 이름")
parser.add_argument("--output_dir", type=str, default="../../model/summary", help="파인튜닝된 모델과 결과가 저장될 디렉토리")
parser.add_argument("--epochs", type=int, default=10, help="총 학습 에폭 수")
parser.add_argument("--train_batch_size", type=int, default=4, help="학습용 배치 크기")
parser.add_argument("--eval_batch_size", type=int, default=4, help="평가용 배치 크기")
parser.add_argument("--warmup_steps", type=int, default=1000, help="학습률 스케줄러의 워밍업 스텝 수")
parser.add_argument("--weight_decay", type=float, default=0.005, help="가중치 감쇠(Weight Decay) 값")
parser.add_argument("--early_stopping_patience", type=int, default=5, help="조기 종료를 위한 patience 값 (검증 성능이 개선되지 않는 에폭 수)")

if __name__ == "__main__":
    args = parser.parse_args()
    run_train(args)
