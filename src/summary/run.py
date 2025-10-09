import argparse
import os
from argparse import Namespace

# 각 모듈에서 실행 함수와 해당 스크립트의 ArgumentParser를 임포트합니다.
# 이를 통해 각 스크립트의 인자 설정을 재사용하고, 이 파일(run.py)을
# 전체 파이프라인의 진입점(entrypoint)으로 사용할 수 있습니다.
from preprocess import run_preprocess, parser as preprocess_parser
from train import run_train, parser as train_parser
from predict import run_predict, parser as predict_parser

def run_all(args):
    """
    'all' 커맨드가 호출될 때 실행되는 함수.
    전처리, 학습, 평가 파이프라인 전체를 순차적으로 실행합니다.
    'all' 커맨드로 받은 인자들을 각 단계에 맞게 재구성하여 전달합니다.

    Args:
        args (Namespace): 'all' 커맨드 실행 시 전달된 모든 인자.
    """
    print("--- 1. Running Preprocessing ---")
    # 1. 전처리 단계에 필요한 인자들만 모아 새로운 Namespace 객체를 생성합니다.
    preprocess_args = Namespace(
        data_path=args.data_path,
        model_name=args.model_name,
        output_dir=args.preprocess_output_dir,
        max_input_length=args.max_input_length,
        max_target_length=args.max_target_length,
        seed=args.seed
    )
    run_preprocess(preprocess_args)
    print("--- Preprocessing Complete ---")

    print("--- 2. Running Training ---")
    # 2. 학습 단계에 필요한 인자들로 Namespace 객체를 생성합니다.
    train_args = Namespace(
        data_dir=args.preprocess_output_dir, # 전처리 결과 디렉토리를 학습 데이터 디렉토리로 사용
        model_name=args.model_name,
        output_dir=args.train_output_dir,
        epochs=args.epochs,
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        early_stopping_patience=args.early_stopping_patience
    )
    run_train(train_args)
    print("--- Training Complete ---")

    print("--- 3. Running Evaluation ---")
    # 3. 평가 단계에 필요한 인자들로 Namespace 객체를 생성합니다.
    predict_args = Namespace(
        model_path=args.train_output_dir, # 학습된 모델 경로를 평가 모델 경로로 사용
        val_csv_path=os.path.join(args.preprocess_output_dir, "val_data.csv"), # 전처리된 검증 데이터셋 사용
        mode='evaluate',
        text=None,
        sbert_model=args.sbert_model,
        output_dir=args.eval_output_dir,
        eval_batch_size=args.eval_batch_size,
        max_input_length=args.max_input_length
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
        description="요약 모델 파이프라인 실행 스크립트",
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
    all_parser = subparsers.add_parser('all', help='전체 파이프라인(전처리, 학습, 평가)을 실행합니다.', conflict_handler='resolve')

    # 각 단계별 인자를 그룹화하여 help 메시지의 가독성을 높입니다.
    all_parser.add_argument_group('Preprocessing Arguments')
    all_parser.add_argument("--data_path", type=str, default="../../data/summary/session_data.csv", help="원본 데이터 CSV 파일 경로")
    all_parser.add_argument("--preprocess_output_dir", type=str, default="../../data/summary", help="전처리된 데이터가 저장될 디렉토리")
    all_parser.add_argument("--max_input_length", type=int, default=1024, help="입력 텍스트의 최대 토큰 길이")
    all_parser.add_argument("--max_target_length", type=int, default=128, help="타겟 요약문의 최대 토큰 길이")
    all_parser.add_argument("--seed", type=int, default=42, help="데이터 분할 시 사용할 랜덤 시드")

    all_parser.add_argument_group('Training Arguments')
    all_parser.add_argument("--train_output_dir", type=str, default="../../model/summary", help="학습된 모델이 저장될 디렉토리")
    all_parser.add_argument("--model_name", type=str, default="paust/pko-t5-small", help="사전 학습된 모델 이름 (토크나이저 및 모델 로드에 사용)")
    all_parser.add_argument("--epochs", type=int, default=10, help="총 학습 에폭 수")
    all_parser.add_argument("--train_batch_size", type=int, default=4, help="학습용 배치 크기")
    all_parser.add_argument("--eval_batch_size", type=int, default=4, help="평가용 배치 크기")
    all_parser.add_argument("--warmup_steps", type=int, default=1000, help="학습률 스케줄러의 워밍업 스텝 수")
    all_parser.add_argument("--weight_decay", type=float, default=0.005, help="가중치 감쇠(Weight Decay) 값")
    all_parser.add_argument("--early_stopping_patience", type=int, default=5, help="조기 종료 patience")

    all_parser.add_argument_group('Prediction/Evaluation Arguments')
    all_parser.add_argument("--eval_output_dir", type=str, default="../../figures/summary", help="평가 결과가 저장될 디렉토리")
    all_parser.add_argument("--sbert_model", type=str, default="BM-K/KoSimCSE-roberta-multitask", help="SBERT 유사도 계산에 사용할 모델 이름")

    # 'all' 커맨드가 입력되면 run_all 함수를 실행하도록 설정합니다.
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
