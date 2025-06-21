from flask import Flask, request, jsonify, render_template
import json
import os
import torch
from transformers import BertTokenizer, BertForSequenceClassification

app = Flask(__name__)

# 모델 불러오기
MODEL_PATH = "5wqs/kobert-risk-final"  # 또는 로컬 디렉토리 경로
tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
model = BertForSequenceClassification.from_pretrained(MODEL_PATH)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# 홈 페이지
@app.route("/")
def index():
    return render_template("index.html")


# 템플릿 추천 API
@app.route("/recommend_templates", methods=["POST"])
def recommend_templates():
    clause = request.json.get("clause", "")

    # templates_ids.json 로드
    path = "templates_index/templates_ids.json"
    if not os.path.exists(path):
        return jsonify({"templates": [], "error": "템플릿 파일 없음"}), 500

    with open(path, "r", encoding="utf-8") as f:
        templates = json.load(f)

    # FAISS가 없으므로 임시로 상위 5개 반환
    top_k = templates[:5]

    return jsonify({"templates": top_k})


# 리스크 분석 API
@app.route("/analyze_risk", methods=["POST"])
def analyze_risk():
    clause = request.json.get("clause", "")
    if not clause:
        return jsonify({"risk_label": "Invalid Input"}), 400

    # 입력 인코딩 및 모델 추론
    inputs = tokenizer(clause, return_tensors="pt", truncation=True, padding=True).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        pred = torch.argmax(outputs.logits, dim=1).item()

    label = "HighRisk" if pred == 1 else "LowRisk"
    return jsonify({"risk_label": label})


if __name__ == "__main__":
    app.run(debug=True)
