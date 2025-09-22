import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

class ModelHandler:
    def __init__(self, model_path: str, data_path: str, model_name: str):
        self.model_path = model_path
        self.data_path = data_path
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.data = None
    
    # 모델과 토크나이저 설정
    def setup_model(self):
        cache_dir = "/data"
        
        print(f"토크나이저 로딩 중: {self.model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, cache_dir=cache_dir)
        
        print(f"모델 로딩 중: {self.model_path}")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            cache_dir=cache_dir,
            trust_remote_code=True,
            device_map="auto",
            dtype=torch.bfloat16
        )
        print(f"모델 설정 완료: {self.model_name}")