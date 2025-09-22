import argparse
import json
from model_handler import ModelHandler
from evaluator import ModelEvaluator
from utils import load_data

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--model_name", type=str, required=True)
    args = parser.parse_args()
    
    # 모델 로딩
    model_handler = ModelHandler(
        model_path=args.model_path,
        data_path=args.data_path,
        model_name=args.model_name
    )
    model_handler.setup_model()
    
    # 데이터 로딩
    model_handler.data = load_data(args.data_path)
    
    # 평가 실행
    evaluator = ModelEvaluator(model_handler.model, model_handler.tokenizer)
    evaluator.evaluate_dataset(args.data_path, args.model_name)
    
if __name__ == "__main__":
    print("모델 평가")
    main()