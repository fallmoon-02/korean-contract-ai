import os
import json
import random
import pandas as pd

# 1) 경로 설정
TEMPLATES_JSON_DIR = os.path.join(os.getcwd(), "data", "templates_json")
OUTPUT_CSV = os.path.join(os.getcwd(), "data", "clauses_sample.csv")

# 2) JSON 파일을 하나씩 읽어, clauses 리스트를 모두 모음
all_clauses = []  # (clause_id, text) 튜플을 담을 리스트
for fname in os.listdir(TEMPLATES_JSON_DIR):
    if not fname.endswith(".json"):
        continue
    path = os.path.join(TEMPLATES_JSON_DIR, fname)
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # data["clauses"]는 [{"clause_id": "...", "text": "..."}, ...] 형태
    for clause in data["clauses"]:
        all_clauses.append((clause["clause_id"], clause["text"]))

# 3) 전체 조항 수와 샘플 크기 결정
total = len(all_clauses)
sample_size = min(500, total)  # 전체 조항이 500개 미만이면 전부, 아니라면 500개 샘플링
print(f"[INFO] Total clauses available: {total}. Sampling {sample_size} clauses.")

# 4) 무작위로 섞어서 sample_size만큼 추출 (재현을 위해 시드 고정)
random.seed(2025)
sampled = random.sample(all_clauses, sample_size)

# 5) DataFrame으로 만들어 CSV 저장
df = pd.DataFrame(sampled, columns=["clause_id", "text"])
os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
print(f"[INFO] Saved {len(df)} clauses to {OUTPUT_CSV}")
