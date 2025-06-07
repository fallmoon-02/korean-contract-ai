
import os
import json
import re

import torch
import faiss
import openai
from flask import Flask, request, jsonify
from transformers import BertTokenizer, BertForSequenceClassification

app = Flask(__name__)

# ————————————————
# 1) KoBERT 리스크 분류 모델 로드
# ————————————————
MODEL_DIR = "kobert_risk_final"  # 파인튜닝한 모델 디렉토리
tokenizer = BertTokenizer.from_pretrained(MODEL_DIR)
model = BertForSequenceClassification.from_pretrained(MODEL_DIR)
model.eval()

# ————————————————
# 2) Faiss 인덱스 & 템플릿 ID 로드
# ————————————————
INDEX_PATH = os.path.join("data", "templates.faiss")
IDS_PATH   = os.path.join("data", "templates_ids.json")

index = faiss.read_index(INDEX_PATH)
with open(IDS_PATH, "r", encoding="utf-8") as f:
    template_ids = json.load(f)

# ————————————————
# 3) OpenAI GPT 클라이언트 초기화
# ————————————————
openai.api_key = os.getenv("OPENAI_API_KEY")  # 환경변수로 설정 필요

# ————————————————
# 헬퍼 함수: 텍스트 임베딩 (평균 풀링)
# ————————————————
def embed(text: str):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=256
    )
    with torch.no_grad():
        hidden = model.bert(**inputs, return_dict=True).last_hidden_state  # (1, L, D)
    vec = hidden.mean(dim=1).cpu().numpy().astype("float32")               # (1, D)
    faiss.normalize_L2(vec)
    return vec

# ————————————————
# 1) 계약서 초안 생성 엔드포인트
# ————————————————
@app.route("/generate_draft", methods=["POST"])
def generate_draft():
    data = request.get_json(silent=True)
    keywords = data.get("keywords") if data else None
    if not keywords:
        return jsonify({"error": "JSON body must contain 'keywords' field."}), 400

    prompt = f"""
아래 키워드를 참고하여, 한국어 표준 용역 계약서 초안을 조항별로 작성해 주세요.

{keywords}

• 각 조항은 “제1조(목적) … 제2조(용역 범위) …” 형태로 번호와 제목을 붙여 주세요.
"""
    try:
        res = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=1500
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    draft_text = res.choices[0].message.content.strip()
    return jsonify({"draft": draft_text})

# ————————————————
# 2) 조항별 리스크 예측 엔드포인트
# ————————————————
@app.route("/analyze_risk", methods=["POST"])
def analyze_risk():
    data = request.get_json(silent=True)
    clause = data.get("clause") if data else None
    if not clause:
        return jsonify({"error": "JSON body must contain 'clause' field."}), 400

    inputs = tokenizer(
        clause,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=256
    )
    with torch.no_grad():
        logits = model(**inputs).logits
    pred = torch.argmax(logits, dim=1).item()
    label = "HighRisk" if pred == 1 else "LowRisk"
    return jsonify({"label": label})

# ————————————————
# 3) 유사 템플릿 추천 엔드포인트
# ————————————————
@app.route("/recommend_templates", methods=["POST"])
def recommend_templates():
    data = request.get_json(silent=True)
    text = data.get("text") if data else None
    topk = data.get("topk", 3) if data else 3
    if not text:
        return jsonify({"error": "JSON body must contain 'text' field."}), 400
    try:
        topk = int(topk)
    except:
        return jsonify({"error": "'topk' must be an integer."}), 400

    vec = embed(text)
    D, I = index.search(vec, topk)
    recs = [template_ids[i] for i in I[0]]
    return jsonify({"recommendations": recs})

# ————————————————
# 서버 실행
# ————————————————
if __name__ == "__main__":
    # 디버그 모드 끔, 모든 인터페이스에서 접속 허용
    app.run(host="0.0.0.0", port=5000, debug=False)
