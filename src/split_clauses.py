import os
import re
import pandas as pd
import json

# 입력/출력 경로 설정
INPUT_CSV = os.path.join(os.getcwd(), "data", "templates_raw.csv")
OUTPUT_JSON_DIR = os.path.join(os.getcwd(), "data", "templates_json")
os.makedirs(OUTPUT_JSON_DIR, exist_ok=True)

# “제n조(…)” 패턴으로 조항 분리하는 정규표현식
CLAUSE_PATTERN = re.compile(r"(제\d+조\([^)]*\)\s*[^제]*)")

def split_into_clauses(text):
    matches = CLAUSE_PATTERN.findall(text)
    return [m.strip() for m in matches if m.strip()]

def main():
    df = pd.read_csv(INPUT_CSV, encoding="utf-8-sig")
    for idx, row in df.iterrows():
        tmpl_id = row["template_id"]
        full_text = row["full_text"]
        clauses = split_into_clauses(full_text)
        out = {"template_id": tmpl_id, "clauses": []}
        for i, clause_text in enumerate(clauses, start=1):
            clause_id = f"{tmpl_id}_cl_{i:03d}"
            out["clauses"].append({"clause_id": clause_id, "text": clause_text})
        out_path = os.path.join(OUTPUT_JSON_DIR, f"{tmpl_id}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        print(f"[INFO] Saved {len(clauses)} clauses to {out_path}")

if __name__ == "__main__":
    main()
