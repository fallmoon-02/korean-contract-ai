import os
import json
import torch
import faiss
import openai
import requests
from flask import Flask, request, jsonify

from transformers import BertTokenizer, BertForSequenceClassification

app = Flask(__name__)

# 1) Load fine-tuned KoBERT model from Hugging Face Hub
MODEL_ID = "5wqs/kobert-risk-final"
HF_TOKEN = os.getenv("HF_TOKEN")  # Set in Render environment

tokenizer = BertTokenizer.from_pretrained(MODEL_ID, token=HF_TOKEN)
model = BertForSequenceClassification.from_pretrained(MODEL_ID, token=HF_TOKEN)
model.eval()

# 2) Load Faiss index & template ID list
INDEX_PATH = os.path.join("templates_index", "templates.faiss")
IDS_PATH = os.path.join("templates_index", "templates_ids.json")

index = faiss.read_index(INDEX_PATH)
with open(IDS_PATH, "r", encoding="utf-8") as f:
    template_ids = json.load(f)

# 3) OpenAI key
openai.api_key = os.getenv("OPENAI_API_KEY")  # Set in Render environment


# -------------------------------
# 유틸: 문장 임베딩
# -------------------------------
def embed(text: str):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=256
    )
    with torch.no_grad():
        hidden = model.bert(**inputs).last_hidden_state  # (1, L, D)
    vec = hidden.mean(dim=1).cpu().numpy().astype("float32")  # (1, D)
    faiss.normalize_L2(vec)
    return vec


# -------------------------------
# 루트 확인용 (Render health check)
# -------------------------------
@app.route("/", methods=["GET"])
def root():
    return "✅ Korean Contract AI API is running!"


# -------------------------------
# 1) 계약서 초안 생성 (GPT API)
# -------------------------------
@app.route("/generate_draft", methods=["POST"])
def generate_draft():
    data = request.get_json()
    keywords = data.get("keywords")
    if not keywords:
        return jsonify({"error": "Provide 'keywords' field."}), 400

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
        draft_text = res.choices[0].message.content.strip()
        return jsonify({"draft": draft_text})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# -------------------------------
# 2) 리스크 분석
# -------------------------------
@app.route("/analyze_risk", methods=["POST"])
def analyze_risk():
    data = request.get_json()
    clause = data.get("clause")
    if not clause:
        return jsonify({"error": "Provide 'clause' field."}), 400

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


# -------------------------------
# 3) 유사 템플릿 추천 (faiss)
# -------------------------------
@app.route("/recommend_templates", methods=["POST"])
def recommend_templates():
    data = request.get_json()
    text = data.get("text")
    topk = data.get("topk", 3)

    if not text:
        return jsonify({"error": "Provide 'text' field."}), 400

    try:
        topk = int(topk)
    except:
        return jsonify({"error": "'topk' must be an integer."}), 400

    try:
        vec = embed(text)
        D, I = index.search(vec, topk)
        recs = [template_ids[i] for i in I[0]]
        return jsonify({"recommendations": recs})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# -------------------------------
# Run app
# -------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
