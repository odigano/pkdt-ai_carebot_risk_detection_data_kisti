# 시니어 발화 데이터 분석을 통한 위험도 분류 및 대화 요약 모델

본 프로젝트는 KISTI DATA/AI 경진대회 참가를 기본 목적으로 합니다.

-   [KISTI 2025 DATA/AI 경진대회 - 고독사 예방을 위한 시니어케어 돌봄로봇(돌봄인형) 데이터 분석](https://aida.kisti.re.kr/competition/main/problem/PROB_000000000002824/detail.do)

시니어 이용자와 AI 돌봄 인형과의 대화 데이터를 기반으로 사용자의 위험도를 분류하고, 대화 내용을 요약하는 두 가지 AI 모델을 제공합니다.

-   **위험도 분류**
    -   대화의 문맥, 감정, 시간적 특성을 종합적으로 분석하여 사용자의 위험도를 다음 4단계로 분류합니다.
        -   `positive`, `danger`, `critical`, `emergency`
-   **대화 요약**
    -   긴 대화 세션의 핵심 내용을 간결하게 요약하여 제공합니다.

## 목차

-   [프로젝트 구조](#프로젝트-구조)
-   [설치 및 준비](#설치-및-준비)
-   [실행](#실행)
-   [모델 설명](#모델-설명)
-   [참고 자료](#참고-자료)
-   [라이선스](#라이선스)

## 프로젝트 구조

```
.
├── data/                  # 모델의 학습, 전처리, 테스트 데이터
├── demo/                  # 테스트 실행 데모 영상
├── figures/               # 모델의 설명, 실험 결과 설명에 사용된 그림 및 결과 파일
├── model/                 # 학습된 모델 파일
├── notebooks/             # 노트북 파일 (.ipynb)
├── scripts/               # 스크립트 파일 (미사용)
├── src/                   # 파이선 파일 (.py)
│   ├── label/
│   │   ├── preprocess.py  # 위험도 분류 모델 전처리 코드
│   │   ├── train.py       # 위험도 분류 모델 학습 코드
│   │   ├── predict.py     # 위험도 분류 모델 예측 및 평가 코드
│   │   └── run.py         # 위험도 분류 모델 실행 코드
│   └── summary/
│       ├── preprocess.py  # 요약 모델 전처리 코드
│       ├── train.py       # 요약 모델 학습 코드
│       ├── predict.py     # 요약 모델 예측 및 평가 코드
│       └── run.py         # 요약 모델 모델 실행 코드
├── requirements.txt
└── README.md
```

> 데이터 파일은 경진대회 규정에 따라 외부 공개 및 공유할 수 없으며 저장소에 포함되지 않습니다.

## 설치 및 준비

### 요구사항

`requirements.txt` 참고

#### 파이선 버전 및 주요 의존성

-   Python == 3.13
-   PyTorch == 2.8
-   Pandas == 2.3.3
-   Scikit-learn == 1.7.2
-   Transformers == 4.57.0
-   Sentence-transformers == 5.1.1
-   Konlpy (Okt) == 0.6.0
    -   Java(JDK) 실행 환경 필요

##### 개발 환경

기본 개발 환경은 다음과 같습니다.

-   Driver Version: 581.42
-   CUDA Version: 13.0

### 설치

저장소를 클론하고 필요한 라이브러리를 설치합니다.

```bash
git clone https://github.com/odigano/pkdt-ai_carebot_risk_detection_data_kisti.git
cd pkdt-ai_carebot_risk_detection_data_kisti
pip install -r requirements.txt
```

#### 모델 준비

다음 압축된 모델 파일을 `model/label`, `model/summary` 디렉터리에 각각 압축 해제해주세요.

##### 위험도 분류 모델

-   ~~[TODO label_model]()~~

##### 요약 모델

-   ~~[TODO summary_model]()~~

## 실행

위험도 분류 모델 및 대화 요약 모델 소스 코드는 유사한 형태로 실행 가능합니다.

```bash
cd ./src/label # 또는 cd ./src/summary
```

세부 파라미터 및 실행 결과는 다르니 해당 소스 코드 또는 --help 옵션을 참고해주세요.

### 단계별 실행

#### 전처리

```bash
python preprocess.py

# usage: preprocess.py [--input_csv INPUT_CSV] [--output_path OUTPUT_PATH] [--tokenizer_name TOKENIZER_NAME]
#                      [--k_context K_CONTEXT] [--session_gap_seconds SESSION_GAP_SECONDS] [--max_seq_len MAX_SEQ_LEN]
# options:
#   --input_csv INPUT_CSV
#                         원본 데이터 CSV 파일 경로
#   --output_path OUTPUT_PATH
#                         전처리된 데이터(CSV)가 저장될 경로
#   --tokenizer_name TOKENIZER_NAME
#                         토크나이저로 사용할 모델 이름
#   --k_context K_CONTEXT
#                         문맥으로 사용할 이전 발화의 수
#   --session_gap_seconds SESSION_GAP_SECONDS
#                         새로운 세션을 정의하기 위한 시간 간격(초)
#   --max_seq_len MAX_SEQ_LEN
#                         입력 시퀀스의 최대 토큰 길이
```

#### 학습

```bash
python train.py

# usage: train.py [--train_preprocessed_path TRAIN_PREPROCESSED_PATH] [--val_preprocessed_path VAL_PREPROCESSED_PATH]
#                 [--output_dir OUTPUT_DIR] [--tokenizer_name TOKENIZER_NAME] [--encoder_name ENCODER_NAME] [--epochs EPOCHS]
#                 [--batch_size BATCH_SIZE] [--learning_rate LEARNING_RATE] [--lstm_hidden_size LSTM_HIDDEN_SIZE]
#                 [--num_workers NUM_WORKERS] [--early_stopping_patience EARLY_STOPPING_PATIENCE] [--use_amp] [--force_cpu]
#                 [--use_attention | --no-use_attention]
# options:
#   --train_preprocessed_path TRAIN_PREPROCESSED_PATH
#                         전처리된 학습 데이터 파일 경로 (CSV)
#   --val_preprocessed_path VAL_PREPROCESSED_PATH
#                         전처리된 검증 데이터 파일 경로 (CSV)
#   --output_dir OUTPUT_DIR
#                         학습된 모델이 저장될 디렉토리
#   --tokenizer_name TOKENIZER_NAME
#                         사전 학습된 토크나이저 이름
#   --encoder_name ENCODER_NAME
#                         사전 학습된 인코더 모델 이름
#   --epochs EPOCHS       총 학습 에폭 수
#   --batch_size BATCH_SIZE
#                         배치 크기
#   --learning_rate LEARNING_RATE
#                         학습률
#   --lstm_hidden_size LSTM_HIDDEN_SIZE
#                         LSTM 은닉층 크기
#   --num_workers NUM_WORKERS
#                         DataLoader를 위한 워커 수
#   --early_stopping_patience EARLY_STOPPING_PATIENCE
#                         조기 중단을 위한 patience 값
#   --use_amp             Automatic Mixed Precision 사용 여부
#   --force_cpu           CUDA 사용 가능 시에도 CPU 강제 사용
#   --use_attention, --no-use_attention
#                         모델에 어텐션 메커니즘 사용 여부
```

#### 예측 및 평가

```bash
# 데이터셋 전체 평가
python predict.py

# 단일 텍스트 요약 (요약 모델의 경우)
python predict.py --mode inference --text "요약할 텍스트입니다."

# usage: predict.py [--val_preprocessed_path VAL_PREPROCESSED_PATH] [--model_dir MODEL_DIR] [--output_dir OUTPUT_DIR]
#                   [--mode {inference,evaluate}] [--batch_size BATCH_SIZE] [--num_workers NUM_WORKERS] [--force_cpu]
# options:
#   --val_preprocessed_path VAL_PREPROCESSED_PATH
#                         전처리된 검증 데이터 경로 (CSV)
#   --model_dir MODEL_DIR
#                         학습된 모델(pytorch_model.bin, config.json 등)이 저장된 디렉토리
#   --output_dir OUTPUT_DIR
#                         평가 결과(CSV, 이미지 등)를 저장할 디렉토리
#   --mode {inference,evaluate}
#                         'inference': 단순 추론, 'evaluate': 정답과 비교하여 성능 평가
#   --batch_size BATCH_SIZE
#                         예측 시 사용할 배치 크기
#   --num_workers NUM_WORKERS
#                         DataLoader를 위한 워커 수
#   --force_cpu           CUDA 사용 가능 시에도 CPU를 강제로 사용
```

### 전체 파이프라인 실행

```bash
python run.py all

# usage: run.py {all,preprocess,train,predict} ...
# positional arguments:
#   {all,preprocess,train,predict}
#                         실행할 명령어:
#                           all         - 전체 파이프라인 (전처리, 학습, 평가) 실행
#                           preprocess  - 데이터 전처리만 실행
#                           train       - 모델 학습만 실행
#                           predict     - 모델 예측 및 평가만 실행
#     all                 전체 파이프라인(전처리, 학습, 평가)을 실행합니다.
#     preprocess          데이터 전처리만 실행합니다.
#     train               모델 학습만 실행합니다.
#     predict             학습된 모델의 평가만 실행합니다.
```

## 모델 설명

본 프로젝트의 모델 출력을 기반으로 별도의 [FastAPI 프로젝트](https://github.com/odigano/pkdt-ai_carebot_risk_detection_python) 서버를 통해 아래 예시와 같은 응답을 서비스로 제공합니다.

```json
{
    "overall_result": {
        "doll_id": "1",
        "dialogue_count": 3,
        "char_length": 32,
        "label": "positive",
        "confidence_scores": {
            "positive": "0.9995",
            "danger": "0.0002",
            "critical": "0.0001",
            "emergency": "0.0000"
        },
        "full_text": "오늘 너무 덥네 지금 몇 시야 조금 있다가 밥 먹어야 겠다",
        "reason": {
            "evidence": [
                {
                    "seq": 1,
                    "text": "지금 몇 시야",
                    "score": "1.0000"
                },
                {
                    "seq": 0,
                    "text": "오늘 너무 덥네",
                    "score": "0.9995"
                }
            ],
            "summary": "오늘 너무 덥다고 말하며 조금 있다가 밥을 먹어야겠다고 함"
        }
    },
    "dialogue_result": [
        {
            "seq": 0,
            "doll_id": "1",
            "text": "오늘 너무 덥네",
            "uttered_at": "2025-09-22T10:20:30",
            "label": "positive",
            "confidence_scores": {
                "positive": "0.9995",
                "danger": "0.0002",
                "critical": "0.0001",
                "emergency": "0.0000"
            }
        },
        {
            "seq": 1,
            "doll_id": "1",
            "text": "지금 몇 시야",
            "uttered_at": "2025-09-22T10:20:40",
            "label": "positive",
            "confidence_scores": {
                "positive": "1.0000",
                "danger": "0.0001",
                "critical": "0.0001",
                "emergency": "0.0001"
            }
        },
        {
            "seq": 2,
            "doll_id": "1",
            "text": "조금 있다가 밥 먹어야 겠다",
            "uttered_at": "2025-09-22T10:20:50",
            "label": "positive",
            "confidence_scores": {
                "positive": "0.9995",
                "danger": "0.0002",
                "critical": "0.0001",
                "emergency": "0.0000"
            }
        }
    ]
}
```

### 위험도 분류 모델

사용자 발화의 위험도를 positive, danger, critical, emergency 4단계로 분류하는 모델입니다.

단일 발화보다 가능한 대화의 문맥을 종합적으로 이해하여 위험 상황을 탐지하는 것을 목표로 합니다.

#### 주요 특징

##### 학습 데이터

-   전체 발화 수
    -   168,476건
-   위험도 별 발화 개수 및 비중:
    -   positive: 48,687건 (28.90%)
    -   danger: 42,224건 (25.06%)
    -   critical: 37,270건 (22.12%)
    -   emergency: 40,295건 (23.92%)

자체적으로 위험도 분류 수기 라벨링 작업이 완료된 KISTI DATA/AI 경진대회 제공 데이터를 사용하였습니다. (총 발화 32,973건)

부족한 데이터 및 클래스 불균형 이슈를 해결하기 위해 LLM을 이용해 데이터를 증강하였습니다.

##### 다중 특성 활용

-   **텍스트 (Text)**: 사전 학습된 **klue/roberta-base** 한국어 모델을 기반으로 텍스트의 의미적 정보를 추출합니다.
-   **시간 (Time)**: 발화 시간(hour), 이전 발화와의 시간 간격(delta_t)을 특성으로 사용하여 사용자의 생활 패턴 및 대화의 긴급성을 모델링합니다.
-   **감정 (Emotion)**: 사전에 정의된 위험도별 키워드('도와줘', '아파', '힘들어' 등)의 등장 횟수와 가중치를 기반으로 '감정 점수'를 추출하여 특성으로 활용합니다.
-   **문맥 위험도 (Contextual Risk)**: 현재 세션 내 이전 발화들의 '감정 점수'를 시간 경과에 따라 감쇠(decay)시켜 누적한 '문맥 위험도' 특성을 새롭게 설계하여 사용합니다. 이를 통해 최근에 위험한 대화가 집중되었는지를 모델이 학습할 수 있습니다.

##### 하이브리드 모델 아키텍처

-   **Transformer Encoder (klue/roberta-base)**: 입력된 대화 문맥의 깊은 언어적 의미를 파악합니다.
-   **Bi-LSTM (양방향 LSTM)**: Transformer가 추출한 텍스트 임베딩의 순차적 흐름을 학습합니다.
-   **Multi-head Attention**: LSTM이 처리한 시퀀스에서 위험도 판단에 중요한 부분에 더 높은 가중치를 부여하여 핵심 정보를 강조합니다.
-   **분류기 (Classifier)**: 위에서 추출된 모든 특성(텍스트, 시간, 감정, 문맥 위험도)을 결합하여 최종적으로 위험도를 분류합니다.

##### 클래스 불균형 해소

-   Focal Loss를 손실 함수로 사용하여, 데이터 수가 적은 critical, emergency와 같은 소수 클래스의 학습 가중치를 높였습니다.
-   또한, 클래스별 데이터 수의 역빈도와 위험도 수준에 따른 수동 가중치를 조합하여 손실 함수에 적용함으로써, 모델이 높은 위험도 클래스를 탐지하는 데 더 집중하도록 유도했습니다.

#### 평가

##### Confusion Matrix Score

| Label            | Precision | Recall | F1-Score | Support |
| ---------------- | --------- | ------ | -------- | ------- |
| Positive         | 1.00      | 0.98   | 0.99     | 32,111  |
| Danger           | 0.56      | 0.79   | 0.65     | 642     |
| Critical         | 0.62      | 0.79   | 0.69     | 150     |
| Emergency        | 0.49      | 0.89   | 0.63     | 70      |
|                  |           |        |          |         |
| **Accuracy**     |           |        | 0.98     | 32,973  |
| **Macro Avg**    | 0.67      | 0.86   | 0.74     | 32,973  |
| **Weighted Avg** | 0.98      | 0.98   | 0.98     | 32,973  |

![위험도 분류 모델 평가 지표 이미지](./figures/label/evaluation-confusion_matrix_report_plot.png)

##### Confusion Matrix Heatmap

![위험도 분류 모델 평가 지표 이미지](./figures/label/evaluation-confusion_matrix_heatmap.png)

### 요약 모델

긴 대화 세션의 내용을 압축하여 핵심을 파악할 수 있는 간결한 요약문을 생성하는 모델입니다.

한국어에 특화된 T5 모델을 파인튜닝하여 사용합니다.

#### 주요 특징

##### 학습 데이터

-   전체 요약 데이터 수
    -   7,700건

KISTI DATA/AI 경진대회 제공 데이터에 대해 인형 별 발화 간격이 10분 이내인 경우를 하나의 대화 세션으로 상정하였습니다. (총 발화 32,973건)

하나의 대화 세션 내용을 LLM을 통해 요약 라벨링 데이터를 생성하였습니다.

##### T5 기반의 Seq2Seq 모델

-   사전 학습된 **paust/pko-t5-small** 한국어 T5 모델을 기반으로 합니다.
-   추출 요약이 아닌 추상 요약을 목표로 합니다.

#### 평가

![요약 모델 평가 지표 이미지](./figures/summary/evaluation-metrics_plot.png)

-   **ROUGE-1**: 생성된 요약과 정답 요약 간에 겹치는 단어(unigram)의 비율을 측정합니다.
-   **ROUGE-2**: 생성된 요약과 정답 요약 간에 겹치는 연속된 두 단어(bigram)의 비율을 측정합니다.
-   **ROUGE-L**: 생성된 요약과 정답 요약 간의 최장 공통 부분 서열(LCS)을 기반으로 문장 구조의 유사성을 측정합니다.
-   **BLEU**: 생성된 문장이 정답 문장과 얼마나 유사한지를 n-gram의 정밀도(precision)를 통해 측정합니다.
-   **METEOR**: 단어의 동의어, 형태소 등을 고려하여 정밀도와 재현율의 조화 평균으로 문장 유사도를 측정합니다.
-   **BERT Score (F1)**: BERT 임베딩을 사용하여 생성된 문장과 정답 문장 간의 의미적 유사도를 F1 점수로 평가합니다.
-   **SBERT Similarity**: 문장 전체를 벡터로 임베딩하여 문장 간의 코사인 유사도로 의미적 유사성을 측정합니다.

성능 평가 지표에서 ROUGE 등 형태론적 접근 지표는 떨어지지만 BERTScore 등 의미론적 접근 지표에서는 비교적 높은 성능을 보여줍니다.

## 참고 자료

-   [Hugging Face - klue/roberta-base](https://huggingface.co/klue/roberta-base)
-   [Hugging Face - paust/pko-t5-small](https://huggingface.co/paust/pko-t5-small)

## 라이선스

MIT 라이선스를 따릅니다.
