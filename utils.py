import json

# 데이터 로딩
def load_data(file_path):
    print(f"데이터 로딩 중: {file_path}")
    samples = []
    if file_path.endswith(".jsonl"):
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                samples.append(json.loads(line))
    elif file_path.endswith(".json"):
        with open(file_path, "r", encoding="utf-8") as f:
            data_obj = json.load(f)
            if isinstance(data_obj, list):
                samples = data_obj
            else:
                samples = [data_obj]
    else:
        raise ValueError("지원하지 않는 데이터 형식입니다. .jsonl 또는 .json 파일을 제공하세요.")
    print(f"데이터 로딩 완료: {len(samples)} samples")
    return samples