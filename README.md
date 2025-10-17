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
├── scripts/               # 스크립트 파일 (미사용)
├── notebooks/             # 노트북 파일 (.ipynb)
├── src/                   # 파이선 파일 (.py)
│   ├── predict.py         # 예측 코드
│   ├── label/
│   │   ├── preprocess.py  # 위험도 분류 모델 전처리 코드
│   │   ├── train.py       # 위험도 분류 모델 학습 코드
│   │   ├── evaluation.py  # 위험도 분류 모델 평가 코드
│   │   └── run.py         # 위험도 분류 모델 실행 코드
│   └── summary/
│       ├── preprocess.py  # 요약 모델 전처리 코드
│       ├── train.py       # 요약 모델 학습 코드
│       ├── evaluation.py  # 요약 모델 평가 코드
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
    -   Java(JDK) 1.8 이상 필요

##### 개발 환경

기본 개발 환경은 다음과 같습니다.

-   OS: Windows 10
-   GPU: NVIDIA GeForce RTX
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

-   [위험도 분류 모델](https://drive.google.com/file/d/1k-kUJ2VDCJYR5nvihfhwjfgZa2arTCZy/view?usp=sharing)
-   [요약 모델](https://drive.google.com/file/d/1LTsyHJ29-r0CF1P1_uCNYK9MIXjvVcnE/view?usp=sharing)

## 실행

### 예측

분석할 CSV 파일(UTF-8)을 준비합니다. 파일 형태 및 예시는 다음을 참고해주세요.

| doll_id | text         | uttered_at          |
|---------|--------------|---------------------|
| 1       | 오늘 너무 덥네 | 2025-09-22 10:20:30 |
| 1       | 지금 몇 시야   | 2025-09-22 10:20:40 |

파일이 준비되었다면 다음 코드를 실행합니다.

```bash
cd src
python predict.py

# 기본 값으로 적용되는 경로 파일은 ‘../data/prediction/dialogue.csv’ 입니다.
# 다른 경로 파일을 지정할 경우 --input_csv 옵션을 사용해주세요.
python predict.py --input_csv “../data/prediction/dialogue_another.csv”

# 각 개발 환경마다 리소스가 다르기 때문에 분석 가능한 최대 글자수를 제한하고 있습니다.
# 기본 값은 최대 10,000개의 문자수가 적용됩니다.
# 해당 값 수정이 필요하다면 --max_chars 옵션을 사용해주세요.
python predict.py –-max_chars 5000
```

### 단계별 실행

위험도 분류 모델과 대화 요약 모델의 소스 코드 파일은 기본적으로 유사한 형태로 실행 가능합니다.

세부 파라미터는 각 소스 코드 또는 --help 옵션을 참고해주세요

세부 파라미터를 생략하더라도 적용된 기본 값으로 코드가 실행됩니다.

```bash
cd src/label # 또는 cd src/summary
```

#### 전처리

```bash
python preprocess.py
```

#### 학습

```bash
python train.py
```

#### 평가

```bash
python evaluation.py
```

### 전체 파이프라인 실행

```bash
python run.py all
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
            "positive": "0.9897",
            "danger": "0.0079",
            "critical": "0.0008",
            "emergency": "0.0017"
        },
        "treatment_plan": "특별한 위험 징후는 없습니다. 지속적으로 모니터링해 주세요.",
        "full_text": "오늘 너무 덥네 지금 몇 시야 조금 있다가 밥 먹어야 겠다",
        "reason": {
            "evidence": [
                {
                    "seq": 1,
                    "text": "지금 몇 시야",
                    "score": "0.9937"
                },
                {
                    "seq": 2,
                    "text": "조금 있다가 밥 먹어야 겠다",
                    "score": "0.9937"
                }
            ],
            "summary": "오늘 너무 덥다고 말하며 밥을 먹어야겠다고 함"
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
                "positive": "0.9897",
                "danger": "0.0079",
                "critical": "0.0008",
                "emergency": "0.0017"
            }
        },
        {
            "seq": 1,
            "doll_id": "1",
            "text": "지금 몇 시야",
            "uttered_at": "2025-09-22T10:20:40",
            "label": "positive",
            "confidence_scores": {
                "positive": "0.9937",
                "danger": "0.0030",
                "critical": "0.0012",
                "emergency": "0.0021"
            }
        },
        {
            "seq": 2,
            "doll_id": "1",
            "text": "조금 있다가 밥 먹어야 겠다",
            "uttered_at": "2025-09-22T10:20:50",
            "label": "positive",
            "confidence_scores": {
                "positive": "0.9937",
                "danger": "0.0034",
                "critical": "0.0014",
                "emergency": "0.0014"
            }
        }
    ]
}
```

### 위험도 분류 모델

사용자 발화의 위험도를 `positive`, `danger`, `critical`, `emergency` 4단계로 분류하는 모델입니다. 단일 발화의 표면적 의미를 넘어, 대화의 연속적인 흐름과 다양한 비언어적 신호(시간, 감정, 누적 위험도)를 종합적으로 분석하여 사용자의 상태를 깊이 있게 추론하도록 설계되었습니다.

#### 주요 특징

##### 학습 데이터

KISTI DATA/AI 경진대회에서 제공된 32,973건의 원본 데이터와, 클래스 불균형 문제 해소를 위해 LLM으로 증강한 70,000건의 데이터를 포함하여 총 **102,973건**의 데이터로 학습되었습니다.

| 데이터 종류 | positive | danger | critical | emergency | 합계 |
|:---:|:---:|:---:|:---:|:---:|:---:|
| 최종 학습 데이터 | 42,111 (40.9%) | 15,642 (15.2%) | 20,150 (19.6%) | 25,070 (24.4%) | 102,973 |

##### 하이브리드 모델 아키텍처

사전 학습된 언어 모델과 순환 신경망, 어텐션 메커니즘을 결합한 하이브리드 구조를 채택했습니다.

-   **Transformer Encoder (`klue/roberta-base`)**: 입력된 대화 문맥의 깊은 의미적 표현(Embedding)을 추출하여 한국어의 복잡한 뉘앙스를 이해합니다.
-   **Bi-LSTM (양방향 LSTM)**: 텍스트 임베딩 시퀀스의 순방향 및 역방향 흐름을 모두 학습하여, 발화 순서에 따른 시간적 맥락과 대화의 동적인 변화를 포착합니다.
-   **Multi-head Attention**: LSTM이 처리한 시퀀스 내에서 위험도 판단에 가장 결정적인 영향을 미치는 단어나 구절에 더 높은 가중치를 부여하여 핵심 정보에 집중하도록 돕습니다.
-   **이중 구조 분류기 (Dual-Component Classifier)**:
    -   **Base Classifier**: 텍스트, 시간, 감정 특징을 종합하여 기본적인 위험도를 예측합니다.
    -   **Risk Bias Generator**: 과거 대화에서 누적된 위험도를 기반으로 최종 예측에 대한 '편향(Bias)'을 생성합니다.
    -   두 결과를 합산하여, "최근 위험 대화가 많았다면 현재 발화의 위험도를 더 높게 예측하라"는 지침을 명시적으로 모델에 전달합니다.

##### 처리 방식 - 다중 특성 활용

단순 텍스트를 넘어 사용자의 상태를 입체적으로 파악하기 위해, 4가지 종류의 특성을 추출하여 모델의 입력으로 활용합니다.

1.  **텍스트 특징**: 현재 발화를 기준으로 이전 발화들을 최대 20개까지 묶어 하나의 시퀀스로 구성하고, `klue/roberta-base` 토크나이저로 변환합니다. 모델은 이를 통해 대화의 의미적, 순차적 맥락을 학습합니다.
2.  **시간 특징**:
    -   **발화 간 시간 간격**: 이전 발화와의 시간 차이를 계산하여, 과거 위험 발화의 영향력을 시간에 따라 감쇠시키는 핵심 요소로 사용합니다.
    -   **발화 시각**: 시간의 순환적 특성을 `sin/cos` 변환으로 인코딩하여, 특정 시간대에 반복되는 위험 패턴을 학습할 수 있도록 합니다.
3.  **감정 특징**: 자체 구축한 위험도별 키워드 사전(`'도와줘', '아파' 등`)을 기반으로, 발화에 포함된 키워드의 빈도와 가중치를 곱해 감정 점수를 계산합니다. 이를 통해 명시적인 위험 신호를 모델이 놓치지 않도록 합니다.
4.  **문맥 위험도 특징**: 과거 대화의 위험도가 현재에 미치는 누적된 영향을 수치화한 동적 특성입니다. 과거 발화들의 위험도 점수에 **시간 감쇠(Time Decay)**를 적용하여 합산하며, "최근에 위험한 대화가 많이 쌓였다면, 현재 발화의 위험도를 더 높게 예측하라"는 강력한 지침을 모델에 전달합니다.

##### 클래스 불균형 해소

데이터 불균형 문제를 완화하고 고위험군 탐지 성능을 높이기 위해 **Focal Loss**와 **맞춤형 이중 가중치 전략**을 적용했습니다.

-   **Focal Loss**: 모델이 이미 잘 맞추는 쉬운 샘플(e.g., `positive`)의 손실은 줄이고, 맞추기 어려운 소수 클래스 샘플(e.g., `emergency`)의 학습에 더 집중하도록 유도합니다.
-   **이중 가중치(Dual-Weighting)**: Focal Loss의 `alpha` 파라미터를 ①클래스별 역빈도 가중치와 ②위험도 수준에 따른 수동 가중치(`[1.0, 4.0, 8.0, 12.0]`)를 곱하여 계산했습니다. 이를 통해 "데이터가 적으면서 동시에 위험도가 높은 클래스"의 학습 중요도를 극대화했습니다.

#### 평가

원본 데이터셋(32,973건) 전체에 대한 평가 결과, 모델은 프로젝트의 최우선 목표인 '실제 위험 상황을 놓치지 않는 것'에 집중한 것을 확인할 수 있습니다.

##### Confusion Matrix Score

| Label            | Precision | Recall | F1-Score | Support |
| ---------------- | --------- | ------ | -------- | ------- |
| positive         |    0.9999 | 0.9887 |   0.9943 |   32111 |
| danger           |    0.6944 | 0.9766 |   0.8117 |     642 |
| critical         |    0.6863 | 0.9333 |   0.7910 |     150 |
| emergency        |    0.5826 | 0.9571 |   0.7243 |      70 |
|                  |           |        |          |         |
| **accuracy**     |           |        |   0.9881 |   32973 |
| **macro avg**    |    0.7408 | 0.9640 |   0.8303 |   32973 |
| **weighted avg** |    0.9916 | 0.9881 |   0.9892 |   32973 |

![위험도 분류 모델 평가 지표 이미지](./figures/label/evaluation-confusion_matrix_report_plot.png)

##### Confusion Matrix Heatmap

![위험도 분류 모델 평가 지표 이미지](./figures/label/evaluation-confusion_matrix_heatmap.png)

### 요약 모델

긴 대화 내용을 한두 문장의 간결하고 핵심적인 요약문으로 자동 생성하는 **추상적 요약(Abstractive Summarization)** 모델입니다. 위험 상황이 감지되었을 때, 관제센터 담당자가 해당 대화 세션의 전체 맥락을 빠르게 파악할 수 있도록 지원합니다.

#### 주요 특징

##### 학습 데이터

원본 발화 로그를 시간 간격(10분 이내) 기준으로 그룹화하여 **7,700건**의 대화 세션을 구성했습니다. 각 세션에 대해 LLM을 활용, "담당자가 시니어의 위험도를 판별하는 목적"에 맞게 요약문을 생성하여 (대화 세션, 요약문) 쌍의 학습 데이터를 구축했습니다.

##### T5 기반의 Seq2Seq 모델

-   **모델**: 한국어 데이터로 사전 학습된 **`paust/pko-t5-small`** 모델을 기반으로, "summarize: [대화 내용]" 형태의 프롬프트를 사용하여 요약 과업을 수행하도록 파인튜닝했습니다.
-   **추상적 요약**: 단순히 원문에서 중요한 문장을 추출하는 것을 넘어, 전체 대화의 의미를 종합적으로 이해하고 새로운 문장을 생성함으로써 사용자의 상황과 의도를 효과적으로 전달합니다.
-   **한국어 특화 ROUGE 평가자**: 표준 ROUGE 평가 도구가 한국어 형태소 분석을 제대로 지원하지 못하는 문제를 해결하기 위해, `Okt` 형태소 분석기를 기반으로 하는 `KORougeEvaluator`를 직접 구현하여 모델 성능을 신뢰도 높게 평가했습니다.

#### 평가

| Metric | Score |
|:---:|:---:|
| ROUGE-1 | 46.60 |
| ROUGE-2 | 28.92 |
| ROUGE-L | 44.86 |
| BLEU | 8.26 |
| METEOR | 28.45 |
| BERT Score (F1) | 82.57 |
| SBERT Similarity | 74.45 |

![요약 모델 평가 지표 이미지](./figures/summary/evaluation-metrics_plot.png)

-   **ROUGE-1**: 생성된 요약과 정답 요약 간에 겹치는 단어(unigram)의 비율을 측정합니다.
-   **ROUGE-2**: 생성된 요약과 정답 요약 간에 겹치는 연속된 두 단어(bigram)의 비율을 측정합니다.
-   **ROUGE-L**: 생성된 요약과 정답 요약 간의 최장 공통 부분 서열(LCS)을 기반으로 문장 구조의 유사성을 측정합니다.
-   **BLEU**: 생성된 문장이 정답 문장과 얼마나 유사한지를 n-gram의 정밀도(precision)를 통해 측정합니다.
-   **METEOR**: 단어의 동의어, 형태소 등을 고려하여 정밀도와 재현율의 조화 평균으로 문장 유사도를 측정합니다.
-   **BERT Score (F1)**: BERT 임베딩을 사용하여 생성된 문장과 정답 문장 간의 의미적 유사도를 F1 점수로 평가합니다.
-   **SBERT Similarity**: 문장 전체를 벡터로 임베딩하여 문장 간의 코사인 유사도로 의미적 유사성을 측정합니다.

어휘 기반 지표(ROUGE)와 의미 기반 지표(BERTScore) 간의 점수 차이는, 본 모델이 원문에서 단순히 단어를 추출하는 것이 아니라, 본래 의도인 전체 대화의 의미를 이해하고 이를 바탕으로 새로운 문장을 생성하는 **추상적 요약** 모델로 학습되었음을 확인할 수 있습니다.

## 참고 자료

-   [Hugging Face - klue/roberta-base](https://huggingface.co/klue/roberta-base)
-   [Hugging Face - paust/pko-t5-small](https://huggingface.co/paust/pko-t5-small)
-   [Hugging Face - BM-K/KoSimCSE-roberta-multitaskl](https://huggingface.co/BM-K/KoSimCSE-roberta-multitask)

## 라이선스

MIT 라이선스를 따릅니다.
