import streamlit as st
import requests
import os

# Hugging Face API 설정
HF_TOKEN = st.secrets.get("HF_TOKEN", os.getenv("HF_TOKEN"))
API_URL = "https://api-inference.huggingface.co/models/5wqs/kobert-risk-final"
HEADERS = {"Authorization": f"Bearer {HF_TOKEN}"}

st.set_page_config(page_title="계약 리스크 분석기", layout="wide")
st.title("📜 계약 리스크 분석기")
st.markdown("KoBERT 기반으로 계약서 조항의 리스크를 분석합니다.")

clause = st.text_area("✍️ 계약 조항을 입력하세요", height=200)

if st.button("🔍 리스크 분석"):
    if not clause.strip():
        st.warning("조항을 입력해주세요.")
    else:
        with st.spinner("모델 분석 중..."):
            response = requests.post(API_URL, headers=HEADERS, json={"inputs": clause})
            if response.status_code == 200:
                result = response.json()
                label = result[0].get("label", "알 수 없음")
                score = result[0].get("score", 0)
                st.success(f"예측된 리스크 레이블: `{label}` (신뢰도: {score:.2f})")
            else:
                st.error(f"API 호출 실패: {response.status_code}")
