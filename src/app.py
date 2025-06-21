from flask import Flask, request, jsonify, render_template
import json
import os
import torch
from transformers import BertTokenizer, BertForSequenceClassification

app = Flask(__name__)

# 모델 경로 (Hugging Face 또는 로컬 경로)
MODEL_PATH = "5wqs/kobert-risk-final"
tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
model = BertForSequenceClassification.from_pretrained(MODEL_PATH)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# 홈 페이지 렌더링
@app.route("/")
def index():
    return render_template("index.html")


# 1. 계약서 초안 생성 API
@app.route("/generate_draft", methods=["POST"])
def generate_draft():
    data = request.get_json()
    party_a = data.get("party_a", "")
    party_b = data.get("party_b", "")
    subject = data.get("subject", "")
    date = data.get("date", "")

    draft = f"""
본 계약은 {date}에 체결되며, 계약 당사자는 {party_a}와(과) {party_b}이다.

제1조 (목적)
본 계약은 {subject}와 관련한 제반 조건을 규정함을 목적으로 한다.

제2조 (계약 기간)
본 계약은 계약일로부터 1년간 유효하다.
    """.strip()

    return jsonify({"draft": draft})


# 2. 리스크 분석 API
@app.route("/analyze_risk", methods=["POST"])
def analyze_risk():
    clause = request.json.get("clause", "")
    if not clause:
        return jsonify({"risk_label": "Invalid Input"}), 400

    inputs = tokenizer(clause, return_tensors="pt", truncation=True, padding=True).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        pred = torch.argmax(outputs.logits, dim=1).item()

    label = "HighRisk" if pred == 1 else "LowRisk"
    return jsonify({"risk_label": label})


# 3. 유사 템플릿 추천 API
@app.route("/recommend_templates", methods=["POST"])
def recommend_templates():
    clause = request.json.get("clause", "")

    path = "templates_index/templates_ids.json"
    if not os.path.exists(path):
        return jsonify({"templates": [], "error": "템플릿 파일 없음"}), 500

    with open(path, "r", encoding="utf-8") as f:
        templates = json.load(f)

    # FAISS 없이 상위 5개 임시 반환
    top_k = templates[:5]
    return jsonify({"templates": top_k})


if __name__ == "__main__":
    app.run(debug=True)
