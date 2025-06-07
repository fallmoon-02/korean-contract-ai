import os
import pandas as pd

# 1) 위험 키워드 목록 정의
RISK_KEYWORDS = [
    "위약금", "해지", "손해배상", "법적 책임", "벌금", 
    "면책", "기소", "소송", "연체", "지체상금", "중재", 
    "배상액", "차임", "보증", "보장", "기한 내", "연체이자"
]

# 2) CSV 경로 설정 (현재 작업 디렉터리 기준)
INPUT_CSV = os.path.join(os.getcwd(), "data", "clauses_sample.csv")
OUTPUT_CSV = os.path.join(os.getcwd(), "data", "clauses_labeled.csv")

# 3) 간단한 룰 기반 라벨링 함수
def rule_based_label(clause_text):
    lower_text = clause_text.lower()
    for kw in RISK_KEYWORDS:
        if kw in lower_text:
            return 1  # HighRisk
    return 0      # LowRisk

def main():
    # 4) clauses_sample.csv 로드
    df = pd.read_csv(INPUT_CSV, encoding="utf-8-sig")
    # 5) 라벨 컬럼 추가
    df["label"] = df["text"].apply(rule_based_label)
    # 6) 결과 저장
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    print(f"[INFO] Saved labeled clauses to {OUTPUT_CSV}")
    # 7) 요약 정보 출력
    high_risk_count = df["label"].sum()
    total = len(df)
    print(f"[INFO] HighRisk: {high_risk_count}/{total}, LowRisk: {total - high_risk_count}/{total}")

if __name__ == "__main__":
    main()
