import streamlit as st
import requests

# ─────────────────────────────────────────────────────────────
# 1) 설정: 여러분 백엔드 서비스의 URL을 입력하세요
# ─────────────────────────────────────────────────────────────
BASE_URL = st.secrets.get("BASE_URL", "https://your-service.onrender.com")

st.set_page_config(page_title="AI 계약서 서비스", layout="centered")

st.title("🤝 AI 협업형 한국어 계약서 서비스")
st.write("계약서 초안 생성, 리스크 분석, 유사 템플릿 추천을 Streamlit으로 쉽게 사용해보세요.")

tabs = st.tabs(["계약서 초안", "리스크 분석", "유사 템플릿"])

# ─────────────────────────────────────────────────────────────
# 2) 계약서 초안 생성 탭
# ─────────────────────────────────────────────────────────────
with tabs[0]:
    st.header("1) 계약서 초안 생성")
    party_a = st.text_input("계약 당사자 A", "")
    party_b = st.text_input("계약 당사자 B", "")
    subject = st.text_input("계약 목적", "")
    date    = st.date_input("효력 발생일")
    if st.button("초안 생성"):
        payload = {
            "party_a": party_a,
            "party_b": party_b,
            "subject": subject,
            "date":    date.isoformat(),
        }
        with st.spinner("초안 생성 중…"):
            res = requests.post(f"{BASE_URL}/generate_draft", json=payload)
        if res.ok:
            st.success("생성 완료!")
            st.text_area("계약서 초안", res.json().get("draft", ""), height=300)
        else:
            st.error(f"오류: {res.status_code}")

# ─────────────────────────────────────────────────────────────
# 3) 리스크 분석 탭
# ─────────────────────────────────────────────────────────────
with tabs[1]:
    st.header("2) 조항별 리스크 분석")
    clause = st.text_area("조항 텍스트", height=100)
    if st.button("분석하기"):
        with st.spinner("분석 중…"):
            res = requests.post(f"{BASE_URL}/analyze_risk", json={"clause": clause})
        if res.ok:
            st.success("분석 완료!")
            st.metric("리스크 레이블", res.json().get("risk_label"))
        else:
            st.error(f"오류: {res.status_code}")

# ─────────────────────────────────────────────────────────────
# 4) 유사 템플릿 추천 탭
# ─────────────────────────────────────────────────────────────
with tabs[2]:
    st.header("3) 유사 템플릿 추천")
    clause2 = st.text_area("조항 텍스트", height=100)
    if st.button("추천받기"):
        with st.spinner("검색 중…"):
            res = requests.post(f"{BASE_URL}/recommend_templates", json={"clause": clause2})
        if res.ok:
            st.success("추천 완료!")
            templates = res.json().get("templates", [])
            for t in templates:
                st.markdown(f"**{t['title']}** (score: {t['score']:.3f})")
                st.write(t["snippet"])
                st.markdown("---")
        else:
            st.error(f"오류: {res.status_code}")
