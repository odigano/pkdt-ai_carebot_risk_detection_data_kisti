import argparse
from argparse import Namespace

# 각 모듈에서 실행 함수와 해당 스크립트의 ArgumentParser를 임포트합니다.
# 이를 통해 각 스크립트의 인자 설정을 재사용할 수 있습니다.
from preprocess import run_preprocess, parser as preprocess_parser
from train import run_train, parser as train_parser
from predict import run_predict, parser as predict_parser

def run_all(args: Namespace):
    """
    'all' 커맨드가 호출될 때 실행되는 함수.
    전처리, 학습, 평가 파이프라인 전체를 순차적으로 실행합니다.
    'all' 커맨드로 받은 인자들을 각 단계에 맞게 재구성하여 전달합니다.

    Args:
        args (Namespace): 'all' 커맨드 실행 시 전달된 모든 인자.
    """
    print("--- [1/3] Running Preprocessing ---")
    # 1. 전처리 단계에 필요한 인자들만 모아 새로운 Namespace 객체를 생성합니다.
    preprocess_args = Namespace(
        input_csv=args.input_csv,
        output_path=args.preprocessed_path,
        tokenizer_name=args.tokenizer_name,
        k_context=args.k_context,
        session_gap_seconds=args.session_gap_seconds,
        max_seq_len=args.max_seq_len
    )
    run_preprocess(preprocess_args)
    print("--- Preprocessing Complete ---\n")

    print("--- [2/3] Running Training ---")
    # 2. 학습 단계에 필요한 인자들로 Namespace 객체를 생성합니다.
    train_args = Namespace(
        preprocessed_path=args.preprocessed_path, # 전처리 결과물을 입력으로 사용
        output_dir=args.model_output_dir,
        tokenizer_name=args.tokenizer_name,
        encoder_name=args.encoder_name,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        lstm_hidden_size=args.lstm_hidden_size,
        num_workers=args.num_workers,
        seed=args.seed,
        use_amp=args.use_amp,
        force_cpu=args.force_cpu,
        use_attention=args.use_attention,
        early_stopping_patience=args.early_stopping_patience
    )
    run_train(train_args)
    print("--- Training Complete ---\n")

    print("--- [3/3] Running Evaluation ---")
    # 3. 평가 단계에 필요한 인자들로 Namespace 객체를 생성합니다.
    predict_args = Namespace(
        preprocessed_path=args.preprocessed_path, # 평가할 데이터
        model_dir=args.model_output_dir,         # 학습된 모델 경로를 입력으로 사용
        output_dir=args.eval_output_dir,
        mode='evaluate', # 'all' 파이프라인에서는 항상 평가 모드로 실행
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        force_cpu=args.force_cpu
    )
    run_predict(predict_args)
    print("--- Evaluation Complete ---")

def main():
    """
    프로젝트의 메인 진입점(Entrypoint).
    argparse를 사용하여 커맨드 라인 인터페이스(CLI)를 구성하고,
    입력된 커맨드('all', 'preprocess', 'train', 'predict')에 따라
    해당하는 함수를 실행합니다.
    """
    parser = argparse.ArgumentParser(
        description="위험도 탐지 모델 파이프라인 실행 스크립트",
        formatter_class=argparse.RawTextHelpFormatter # help 메시지 포맷 유지
    )
    # 서브파서: 'all', 'preprocess' 등 다양한 명령어를 관리합니다.
    subparsers = parser.add_subparsers(dest='command', required=True, help='실행할 명령어:\n'
                                                                        '  all         - 전체 파이프라인 (전처리, 학습, 평가) 실행\n'
                                                                        '  preprocess  - 데이터 전처리만 실행\n'
                                                                        '  train       - 모델 학습만 실행\n'
                                                                        '  predict     - 모델 예측 및 평가만 실행')

    # --- 'all' 커맨드 파서 정의 ---
    # 'all' 커맨드는 모든 파이프라인을 실행하며, 모든 단계의 인자를 포함합니다.
    # conflict_handler='resolve'는 부모 파서와 자식 파서 간의 인자 이름 충돌을 자동으로 해결합니다.
    all_parser = subparsers.add_parser('all', help='전체 파이프라인(전처리, 학습, 평가)을 실행합니다.', conflict_handler='resolve')

    # 각 모듈별로 인자를 그룹화하여 help 메시지의 가독성을 높입니다.
    preprocess_group = all_parser.add_argument_group('Preprocessing Arguments')
    train_group = all_parser.add_argument_group('Training Arguments')
    predict_group = all_parser.add_argument_group('Prediction/Evaluation Arguments')

    # 전처리 인자 추가
    preprocess_group.add_argument("--input_csv", type=str, default="../../data/label/train_data.csv", help="원본 데이터 CSV 파일 경로")
    preprocess_group.add_argument("--preprocessed_path", type=str, default="../../data/label/preprocessed.csv", help="전처리된 데이터가 저장될 경로")
    preprocess_group.add_argument("--k_context", type=int, default=20, help="문맥으로 사용할 이전 발화의 수")
    preprocess_group.add_argument("--session_gap_seconds", type=int, default=600, help="새로운 세션을 정의하기 위한 시간 간격(초)")
    preprocess_group.add_argument("--max_seq_len", type=int, default=128, help="입력 시퀀스의 최대 토큰 길이")

    # 여러 단계에서 공통으로 사용되는 인자는 all_parser에 직접 추가
    all_parser.add_argument("--tokenizer_name", type=str, default="klue/roberta-base", help="모든 단계에서 사용할 토크나이저 이름")
    all_parser.add_argument("--batch_size", type=int, default=64, help="학습 및 평가 시 배치 크기")
    all_parser.add_argument("--num_workers", type=int, default=0, help="DataLoader를 위한 워커 수")
    all_parser.add_argument("--force_cpu", action='store_true', help="CUDA 사용 가능 시에도 CPU 강제 사용")
    
    # 학습 인자 추가
    train_group.add_argument("--model_output_dir", type=str, default="../../model/label", help="학습된 모델이 저장될 디렉토리")
    train_group.add_argument("--encoder_name", type=str, default="klue/roberta-base", help="사전 학습된 인코더 모델 이름")
    train_group.add_argument("--epochs", type=int, default=10, help="총 학습 에폭 수")
    train_group.add_argument("--learning_rate", type=float, default=2e-5, help="학습률")
    train_group.add_argument("--lstm_hidden_size", type=int, default=256, help="LSTM 은닉층 크기")
    train_group.add_argument("--seed", type=int, default=42, help="재현성을 위한 시드 값")
    train_group.add_argument("--early_stopping_patience", type=int, default=5, help="조기 종료 patience")
    train_group.add_argument("--use_amp", action='store_true', help="Automatic Mixed Precision 사용 여부")
    train_group.add_argument("--use_attention", action=argparse.BooleanOptionalAction, default=True, help="모델에 어텐션 메커니즘 사용 여부")

    # 평가 인자 추가
    predict_group.add_argument("--eval_output_dir", type=str, default="../../figures/label", help="평가 결과가 저장될 디렉토리")

    # 'all' 커맨드가 입력되면 run_all 함수를 실행하도록 설정
    all_parser.set_defaults(func=run_all)

    # --- 개별 커맨드 파서 정의 ---
    # 'parents' 옵션을 사용하여 각 모듈에 이미 정의된 파서를 그대로 상속받습니다.
    # 이를 통해 인자 정의의 중복을 피하고 코드를 간결하게 유지할 수 있습니다.
    subparsers.add_parser('preprocess', help='데이터 전처리만 실행합니다.', parents=[preprocess_parser], add_help=False).set_defaults(func=run_preprocess)
    subparsers.add_parser('train', help='모델 학습만 실행합니다.', parents=[train_parser], add_help=False).set_defaults(func=run_train)
    subparsers.add_parser('predict', help='학습된 모델의 평가만 실행합니다.', parents=[predict_parser], add_help=False).set_defaults(func=run_predict)

    # 커맨드 라인에서 받은 인자를 파싱합니다.
    args = parser.parse_args()
    # 파싱된 인자에 따라 'func'에 할당된 함수(run_all, run_preprocess 등)를 실행합니다.
    if hasattr(args, 'func'):
        args.func(args)
    else:
        parser.print_help()

if __name__ == '__main__':
    main()
