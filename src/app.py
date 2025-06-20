import os
import json
import torch
import faiss
import openai
from flask import Flask, request, jsonify, render_template
from transformers import BertTokenizer, BertForSequenceClassification
from dotenv import load_dotenv

# 환경 변수 로딩
load_dotenv()

app = Flask(
    __name__,
    static_folder="static",
    template_folder="templates"
)

# Hugging Face 모델 로드
MODEL_ID = "5wqs/kobert-risk-final"
HF_TOKEN = os.getenv("HF_TOKEN")

tokenizer = BertTokenizer.from_pretrained(MODEL_ID, token=HF_TOKEN)
model = BertForSequenceClassification.from_pretrained(MODEL_ID, token=HF_TOKEN)
model.eval()

# Faiss 인덱스 로드
INDEX_PATH = os.path.join("templates_index", "templates.faiss")
IDS_PATH = os.path.join("templates_index", "templates_ids.json")

faiss_index = faiss.read_index(INDEX_PATH)
with open(IDS_PATH, "r", encoding="utf-8") as f:
    template_ids = json.load(f)

# OpenAI 키
openai.api_key = os.getenv("OPENAI_API_KEY")

# ---------------------- Routes ----------------------

@app.route("/", methods=["GET"])
def index_page():
    return render_template("index.html")


@app.route("/generate_draft", methods=["POST"])
def generate_draft():
    data = request.get_json()
    party_a = data.get("party_a", "")
    party_b = data.get("party_b", "")
    subject = data.get("subject", "")
    date = data.get("date", "")

    prompt = f"""
당사자 A: {party_a}
당사자 B: {party_b}
계약 목적: {subject}
효력 발생일: {date}

위 정보를 바탕으로 한국어 용역 계약서 초안을 조항별로 작성해 주세요. 각 조항은 번호와 제목을 포함해야 합니다.
"""

    try:
        res = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=1500
        )
        draft = res.choices[0].message.content.strip()
        return jsonify({"draft": draft})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/analyze_risk", methods=["POST"])
def analyze_risk():
    data = request.get_json()
    clause = data.get("clause")
    if not clause:
        return jsonify({"error": "clause is required"}), 400

    inputs = tokenizer(clause, return_tensors="pt", truncation=True, padding="max_length", max_length=256)
    with torch.no_grad():
        logits = model(**inputs).logits
    pred = torch.argmax(logits, dim=1).item()
    label = "HighRisk" if pred == 1 else "LowRisk"

    return jsonify({"label": label})


@app.route("/recommend_templates", methods=["POST"])
def recommend_templates():
    data = request.get_json()
    clause = data.get("clause", "")
    topk = int(data.get("topk", 3))

    inputs = tokenizer(clause, return_tensors="pt", truncation=True, padding="max_length", max_length=256)
    with torch.no_grad():
        vec = model.bert(**inputs).last_hidden_state.mean(dim=1).cpu().numpy().astype("float32")
    faiss.normalize_L2(vec)

    D, I = faiss_index.search(vec, topk)
    recs = [template_ids[i] for i in I[0]]

    return jsonify({"templates": recs})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
