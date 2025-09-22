from itertools import islice
import json
import os
import re
import torch
from tqdm import tqdm
from typing import List, Dict, Any

class ModelEvaluator:
    # 모델 평가를 위한 클래스
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.labels = ["A", "B", "C", "D"]
        self.correct_count = 0
        self.wrong_count = 0
    
    # 모델 출력에서 정답 추출
    def clean_prediction(self, raw_output: str) -> str:
        raw_output = raw_output.strip().upper()
        for pattern in [r"\b([A-D])\b", r"\b([A-D])[).:,\s]", r"\b([A-D])"]:
            match = re.search(pattern, raw_output)
            if match:
                return match.group(1)
        return "INVALID"
    # 프롬프트 생성
    def create_prompt(self, question: str, choices: List[str]) -> str:
        formatted_choices = "\n".join([f"{self.labels[i]}. {choice}" for i, choice in enumerate(choices)])
        
        prompt = f"""[지시]
선택지 중에서 가장 적절한 정답을 하나만 선택하세요.  
정답은 반드시 A, B, C, D 중 알파벳 한 글자만 출력하세요.  
절대 해설이나 설명은 쓰지 마세요.  
정답만 출력하세요.

[질문]: 
{question}

[선택지]
{formatted_choices}
### 정답:"""
        return prompt
    # 모델 추론 실행    
    def run_model(self, question: str, choices: List[str]) -> str:
        prompt = self.create_prompt(question, choices)
        
        # 모델 입력 구성
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        input_len = inputs["input_ids"].shape[-1]
        
        # EOS 토큰 체크
        if self.tokenizer.eos_token_id is not None:
            print("EOS 토큰 존재")
        else:
            print("EOS 토큰 없음")

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=5,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # 생성된 토큰 중 prompt 이후만 디코딩
        generated_only = outputs[0][input_len:]
        decoded = self.tokenizer.decode(generated_only, skip_special_tokens=True)
        
        print("🧩 모델 응답:", decoded)
        predicted = self.clean_prediction(decoded)
        print("🧼 최종 predicted:", predicted)
        
        return predicted
    
    def evaluate_single_item(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """단일 아이템 평가"""
        question = item["question"]
        choices = item["choices"]
        answer = item["answer"]
        candidate = item["candidate"]
        predicted = self.run_model(question, choices)
        print("필터링 모델 응답:", predicted)
        
        memory_result = "correct" if predicted == answer else "wrong"
        
        result = {
            "question": item["question"],
            "answer": item["answer"],
            "negative_answer": item["negative_answer"],
            "candidate": item["candidate"],
            "golden_context": item["golden_context"],            
            "negative_context": item["negative_context"],
            "choices": item["choices"],
            "memory_predicted": predicted,
            "memory_result": memory_result
        }
        
        # 통계 업데이트
        if memory_result == "correct":
            self.correct_count += 1
        else:
            self.wrong_count += 1
            
        return result
    
    def evaluate_dataset(self, input_file: str, model_name: str, max_samples: int = None):
        """전체 데이터셋 평가"""
        # 총 라인 수 계산
        with open(input_file, "r", encoding="utf-8") as f:
            total_lines = sum(1 for _ in f)
        
        if max_samples:
            total_lines = min(total_lines, max_samples)
        
        # 평가 실행
        with open(input_file, "r", encoding="utf-8") as f:
            for line in tqdm(f, total=total_lines, desc=f"Evaluating ({total_lines} samples)"):
            # for line in tqdm(islice(f, 20), total = 20, desc=f"Evaluating (20 samples)"):
                item = json.loads(line)
                result = self.evaluate_single_item(item)
                
                # 결과 저장 (현재 디렉토리에 저장)
                output_dir = f"results/{model_name}"
                os.makedirs(output_dir, exist_ok=True)
                output_file = f"{output_dir}/{model_name}_evaluation_results.jsonl"
                
                with open(output_file, "a", encoding="utf-8") as f:
                    f.write(json.dumps(result, ensure_ascii=False) + "\n")
        
        print(f"\n✅ 맞춘 개수: {self.correct_count} / 틀린 개수: {self.wrong_count}")
        print(f"📊 정확도: {self.correct_count / (self.correct_count + self.wrong_count) * 100:.2f}%")
