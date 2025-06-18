import os
import json
import requests
import torch
import faiss
import openai
from flask import Flask, request, jsonify
from transformers import BertTokenizer, BertModel, BertForSequenceClassification

# Flask 앱 초기화
app = Flask(__name__)

# 0. 환경변수
HF_TOKEN = os.getenv("HF_TOKEN")                 # Hugging Face 토큰
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")    # OpenAI API 키
MODEL_ID = "5wqs/kobert-risk-final"

# 1. Hugging Face 모델 및 토크나이저 로드
tokenizer = BertTokenizer.from_pretrained(MODEL_ID, token=HF_TOKEN)
model_cls = BertForSequenceClassification.from_pretrained(MODEL_ID, token=HF_TOKEN)
model_embed = BertModel.from_pretrained(MODEL_ID, token=HF_TOKEN)
model_cls.eval()
model_embed.eval()

# 2. OpenAI 세팅
openai.api_key = OPENAI_API_KEY

# 3. FAISS 템플릿 인덱스 로딩
INDEX_PATH = "data/templates.faiss"
IDS_PATH = "data/templates_ids.json"
index = faiss.read_index(INDEX_PATH)
with open(IDS_PATH, "r", encoding="utf-8") as f:
    template_ids = json.load(f)

# 4. health check
@app.route("/", methods=["GET"])
def home():
    return "✅ Korean Contract AI API is running!"

# 5. 초안 생성
@app.route("/generate_draft", methods=["POST"])
def generate_draft():
    data = request.get_json()
    keywords = data.get("keywords")
    if not keywords:
        return jsonify({"error": "Missing 'keywords' field"}), 400

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
        draft = res.choices[0].message.content.strip()
        return jsonify({"draft": draft})
    except Exception as e:
        return jsonify({"error": f"OpenAI error: {str(e)}"}), 502

# 6. 리스크 분석
@app.route("/analyze_risk", methods=["POST"])
def analyze_risk():
    data = request.get_json()
    clause = data.get("clause")
    if not clause:
        return jsonify({"error": "Missing 'clause' field"}), 400

    try:
        inputs = tokenizer(clause, return_tensors="pt", truncation=True, padding="max_length", max_length=256)
        with torch.no_grad():
            logits = model_cls(**inputs).logits
        pred = torch.argmax(logits, dim=1).item()
        label = "HighRisk" if pred == 1 else "LowRisk"
        return jsonify({"label": label})
    except Exception as e:
        return jsonify({"error": f"Risk analysis error: {str(e)}"}), 500

# 7. 임베딩 함수
def embed(text: str):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding="max_length", max_length=256)
    with torch.no_grad():
        last_hidden = model_embed(**inputs).last_hidden_state
        pooled = last_hidden.mean(dim=1).cpu().numpy().astype("float32")
    faiss.normalize_L2(pooled)
    return pooled

# 8. 유사 템플릿 추천
@app.route("/recommend_templates", methods=["POST"])
def recommend_templates():
    data = request.get_json()
    text = data.get("text")
    topk = int(data.get("topk", 3))

    if not text:
        return jsonify({"error": "Missing 'text' field"}), 400

    try:
        vec = embed(text)
        D, I = index.search(vec, topk)
        recs = [template_ids[i] for i in I[0]]
        return jsonify({"recommendations": recs})
    except Exception as e:
        return jsonify({"error": f"Recommendation error: {str(e)}"}), 502

# 9. 서버 실행
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
