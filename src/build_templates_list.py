import os
import json

# 1) 입력 디렉터리 & 출력 경로 설정
INPUT_DIR = os.path.join(os.getcwd(), "data", "templates_json")
OUTPUT_PATH = os.path.join(os.getcwd(), "data", "templates_list.json")

# 2) templates_json 폴더 안의 JSON 파일들을 모아 리스트 생성
templates = []
for fname in os.listdir(INPUT_DIR):
    if fname.endswith(".json"):
        full_path = os.path.join(INPUT_DIR, fname)
        with open(full_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        # type 필드는 template_id에 “용역” 단어가 들어있으면 "용역", 
        # 아니면 "NDA" 단어가 들어있으면 "NDA", 그 외에는 "기타"
        ttype = (
            "용역"
            if "용역" in data["template_id"]
            else ("NDA" if "nda" in data["template_id"].lower() else "기타")
        )
        templates.append({
            "template_id": data["template_id"],
            "clauses": data["clauses"],
            "type": ttype
        })

# 3) 최종 리스트를 JSON 파일로 저장
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(templates, f, ensure_ascii=False, indent=2)

print(f"[INFO] {len(templates)} templates saved to {OUTPUT_PATH}")
