import os
import json
import pandas as pd

# 1) 위험 키워드 목록 정의
RISK_KEYWORDS = [
    # 계약 해지·위약
    "해지", "위약금", "계약 해제", "계약 해지", "계약 취소", "해약",

    # 손해배상·배상금
    "손해배상", "배상액", "배상금", "보상", "배상 책임", "위약 배상", "배상 청구",

    # 법적 책임·소송·분쟁
    "법적 책임", "법률 책임", "소송", "소송비용", "소송 청구", "소송 대응", "분쟁", "중재",
    "중재 재판", "중재 판정", "중재 조정", "중재 합의",

    # 벌금·과태료
    "벌금", "과태료",

    # 면책·면책조항
    "면책", "면책 조항", "면책사항", "면책 범위",

    # 기소·고소
    "기소", "고소", "고발", "고발장", "검사", "법정",

    # 연체·지체상금
    "연체", "지체상금", "지연 배상", "연체료", "지체 배상금", "연체 이자", "지체 손해",

    # 채무불이행·디폴트
    "채무 불이행", "디폴트", "채권 회수", "채무자", "채권자", "부도",

    # 보증·보장
    "보증", "보장", "연대 보증", "보증인", "보증 책임", "위임 보증", "보증 채무",

    # 위반·위법
    "위반", "위법", "법 위반", "법령 위반", "계약 위반", "약관 위반", "규정 위반",

    # 해약·중도 해지
    "해약", "중도 해지", "중도 해약", "계약 철회", "철회권",

    # 담보·저당
    "담보", "저당", "질권", "담보 설정", "담보 해지", "담보물",

    # 보증보험·보험
    "보험", "보험금", "보증보험", "보험 가입", "보험 혜택", "면책 보험",

    # 지체·연체 이자
    "연체 이자", "지체 이자", "지체 이자율", "연체 이자율",

    # 지체 배상·지연 배상
    "지연 배상", "지연 손해", "지체 손해",
    
    # 계약금·계약 보증금
    "계약금", "보증금", "계약 보증금", "예치금",

    # 손실·손해
    "손실", "손해", "손해액", "예상 손실", "현실 손해",

    # 불이행·불가항력
    "불이행", "불가항력", "천재지변", "면책 사유", "면책 사유", "불가항력 조항",

    # 비밀유지 위반·기밀 유출
    "비밀유지 위반", "기밀 유출", "비밀 정보", "비밀 보장", "기밀 보호",

    # 지체상금·지체상환
    "지체상금", "지연상환", "연체 상환",

    # 계약 위반 시 제재
    "제재", "시정 명령", "위반 시 과징금", "과징금", "시정 조치",

    # 기타 위험 단어
    "배임", "횡령", "기망", "사기", "부실", "위법 행위", "불법 행위"
]


# 2) JSON 폴더 경로 & 출력 CSV 경로 설정
TEMPLATES_JSON_DIR = os.path.join(os.getcwd(), "data", "templates_json")
OUTPUT_CSV = os.path.join(os.getcwd(), "data", "clauses_labeled_full.csv")

# 3) 키워드 기반 라벨링 함수
def rule_based_label(clause_text):
    lower_text = clause_text.lower()
    for kw in RISK_KEYWORDS:
        if kw in lower_text:
            return 1  # HighRisk
    return 0      # LowRisk

def main():
    rows = []  # (clause_id, text, label) 튜플을 담을 리스트

    # 4) templates_json 폴더 순회
    for fname in os.listdir(TEMPLATES_JSON_DIR):
        if not fname.endswith(".json"):
            continue
        full_path = os.path.join(TEMPLATES_JSON_DIR, fname)
        with open(full_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        # data["clauses"] 는 [{"clause_id": ..., "text": ...}, ...] 형태
        for clause in data["clauses"]:
            cid = clause["clause_id"]
            text = clause["text"]
            lbl = rule_based_label(text)
            rows.append((cid, text, lbl))

    # 5) DataFrame 생성 및 CSV로 저장
    df = pd.DataFrame(rows, columns=["clause_id", "text", "label"])
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    print(f"[INFO] Saved {len(df)} labeled clauses to {OUTPUT_CSV}")

    # 6) HighRisk/LowRisk 비율 출력
    high_count = df["label"].sum()
    total = len(df)
    print(f"[INFO] HighRisk: {high_count}/{total}, LowRisk: {total - high_count}/{total}")

if __name__ == "__main__":
    main()
