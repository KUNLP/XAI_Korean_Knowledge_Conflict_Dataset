export HF_HOME=/data # 연구실 HDD 데이터 저장 경로
python eval.py \
    --model_path kakaocorp/kanana-1.5-2.1b-instruct-2505 \
    --data_path data/KoConQA.jsonl \
    --model_name kanana-1.5-2.1b-instruct-2505 \
    
