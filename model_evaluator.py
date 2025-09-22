from itertools import islice
import json
import os
import re
import torch
from tqdm import tqdm
from typing import Optional, List, Dict, Any
from utils import load_data

class ModelEvaluator:
    # 모델 평가를 위한 클래스
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.labels = ["A", "B", "C", "D"]
        self.correct_count = 0
        self.wrong_count = 0
        self.VR = 0
        self.RR = 0
        self.DMSS = 0

    # 모델 출력에서 정답 추출
    def clean_prediction(self, raw_output: str) -> str:
        raw_output = raw_output.strip().upper()
        for pattern in [r"\b([A-D])\b", r"\b([A-D])[).:,\s]", r"\b([A-D])"]:
            match = re.search(pattern, raw_output)
            if match:
                return match.group(1)
        return "INVALID"

    # 1. 모델 평가 프롬프트 생성 (질문만)
    def create_prompt_question(self, question: str, choices: List[str]) -> str:
        formatted_choices = "\n".join([f"{self.labels[i]}. {choice}" for i, choice in enumerate(choices)])
        
        prompt = f"""[지시]
질문을 읽고 선택지 중에서 가장 적절한 정답을 하나만 선택하세요.  
정답은 반드시 A, B, C, D 중 알파벳 한 글자만 출력하세요.  
절대 해설이나 설명은 쓰지 마세요.  
정답만 출력하세요.

[질문]: 
{question}

[선택지]
{formatted_choices}
### 정답:"""
        return prompt
    # 2. 모델 평가 프롬프트 생성 (질문 + 문맥)
    def create_prompt_question_with_context(self, question: str, choices: List[str], context: str) -> str:
        formatted_choices = "\n".join([f"{self.labels[i]}. {choice}" for i, choice in enumerate(choices)])
            
        prompt = f"""[지시]
질문을 읽고 문맥을 참고하여 선택지 중에서 가장 적절한 정답을 하나만 선택하세요.  
정답은 반드시 A, B, C, D 중 알파벳 한 글자만 출력하세요.  
절대 해설이나 설명은 쓰지 마세요.  
정답만 출력하세요.

[문맥]:
{context}

[질문]: 
{question}

[선택지]
{formatted_choices}
### 정답:"""
        return prompt

    # 모델 추론 실행    
    def run_model(self, question: str, choices: List[str], context: Optional[str] = None) -> str:
        if context is None:
            prompt = self.create_prompt_question(question, choices)
        else:
            prompt = self.create_prompt_question_with_context(question, choices, context)
        
        # 모델 입력 구성
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        input_len = inputs["input_ids"].shape[-1]
        
        # EOS 토큰 체크
        # if self.tokenizer.eos_token_id is not None:
        #     print("EOS 토큰 존재")
        # else:
        #     print("EOS 토큰 없음")

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
        print(f"prompt : \n{prompt}")
        print("--------------------------------"
        )
        return predicted
    
    def evaluate_question(self, item: Dict[str, Any]) -> Dict[str, Any]:
        question = item["question"]
        choices = item["choices"]
        answer = item["answer"]
        predicted = self.run_model(question, choices)
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
    
    def evaluate_question_with_context(self, question_result: Dict[str, Any]) -> Dict[str, Any]:
        question = question_result["question"]
        choices = question_result["choices"]
        answer = question_result["answer"]
        if question_result["memory_result"] == "correct":
            context = question_result["negative_context"]
        elif question_result["memory_result"] == "wrong":
            context = question_result["golden_context"]
        predicted = self.run_model(question, choices, context)
        return predicted

    
    def evaluate_metrics(self, data):
        D_plus = [item for item in data if item["memory_result"] == "correct"]
        D_minus = [item for item in data if item["memory_result"] == "wrong"]
        self.VR = sum(item["negative_context_predicted"] == item["answer"] for item in D_plus) / len(D_plus) if D_plus else 0.0
        self.RR = sum(item["golden_context_predicted"] == item["answer"] for item in D_minus) / len(D_minus) if D_minus else 0.0
        # DMSS의 1차 항
        term1 = sum(item["negative_context_predicted"] == item["answer"] for item in D_plus) + sum(item["golden_context_predicted"] == item["memory_predicted"] for item in D_minus)
        # DMSS의 2차 항
        term2 = sum(item["negative_context_predicted"] == item["candidate"] for item in D_plus) + sum(item["golden_context_predicted"] == item["answer"] for item in D_minus)
        self.DMSS = (term1 - term2) / len(data) if data else 0.0
        print(f"📊 VR 개수: {len(D_plus)}")
        print(f"📊 VR: {self.VR:.2f}%")
        print(f"📊 RR 개수: {len(D_minus)}")
        print(f"📊 RR: {self.RR:.2f}%")
        print(f"📊 DMSS 개수: {term1} + {term2} = {term1 + term2}")
        print(f"📊 DMSS: {self.DMSS:.2f}%")
        return {
            "PA": len(D_plus) / (len(D_plus) + len(D_minus)),
            "VR": self.VR, 
            "RR": self.RR, 
            "DMSS": self.DMSS,
            "counts" :{
                "data_len" : len(D_plus) + len(D_minus),
                "data_question_correct(VR_len)" : len(D_plus),
                "data_question_wrong(RR_len)" : len(D_minus),
                "DMSS_term1" : term1,
                "DMSS_term2" : term2,
                "DMSS_len" : term1 + term2
            }}

    def evaluate_dataset(self, data: List[Dict[str, Any]], model_name: str, max_samples: int = None):
        """전체 데이터셋 평가"""
        # 총 라인 수 계산
        total_lines = len(data)
        
        if max_samples:
            total_lines = min(total_lines, max_samples)

        # 결과 저장 (현재 디렉토리에 저장)
        output_dir = f"results/{model_name}"
        os.makedirs(output_dir, exist_ok=True)
        output_file = f"{output_dir}/{model_name}_evaluation_results.jsonl"

        # 평가 실행
        for item in tqdm(data, total=total_lines, desc=f"Evaluating ({total_lines} samples)"):
            question_result = self.evaluate_question(item)
            if question_result["memory_result"] == "correct":
                negative_context_predicted = self.evaluate_question_with_context(question_result)
            elif question_result["memory_result"] == "wrong":
                golden_context_predicted = self.evaluate_question_with_context(question_result)
            question_result.update({
                "negative_context_predicted": negative_context_predicted if question_result["memory_result"] == "correct" else None,
                "golden_context_predicted": golden_context_predicted if question_result["memory_result"] == "wrong" else None
            })            
            with open(output_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(question_result, ensure_ascii=False) + "\n")
        
        print(f"\n✅ 맞춘 개수: {self.correct_count} / 틀린 개수: {self.wrong_count}")
        print(f"📊 정확도: {self.correct_count / (self.correct_count + self.wrong_count) * 100:.2f}%")

        metrics_data = load_data(output_file)
        metrics = self.evaluate_metrics(metrics_data)
        metrics_path = f"{output_dir}/{model_name}_metrics.jsonl"
        with open(metrics_path, "w", encoding="utf-8") as mf:
            json.dump(metrics, mf, ensure_ascii=False, indent=2)
        print(f"📝 Saved metrics to {metrics_path}")