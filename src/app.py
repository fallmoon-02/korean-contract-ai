import os
import requests
from flask import Flask, request, jsonify, send_from_directory

app = Flask(__name__)

# Hugging Face Inference API
HF_TOKEN = os.getenv("HF_TOKEN")  # Render 환경 변수에 등록 필요
API_URL = "https://api-inference.huggingface.co/models/5wqs/kobert-risk-final"
HEADERS = {"Authorization": f"Bearer {HF_TOKEN}"}


# 기본 홈 페이지 (index_2.html 반환)
@app.route("/", methods=["GET"])
def home():
    return send_from_directory(directory="static", path="index.html")


# 리스크 분석 API
@app.route("/analyze_risk", methods=["POST"])
def analyze_risk():
    data = request.get_json()
    clause = data.get("clause")
    if not clause:
        return jsonify({"error": "clause is required"}), 400

    response = requests.post(API_URL, headers=HEADERS, json={"inputs": clause})
    if response.status_code != 200:
        return jsonify({"error": response.text}), response.status_code

    result = response.json()
    label = result[0]["label"]
    score = result[0]["score"]
    return jsonify({"label": label, "score": score})


# 필수: Render가 실행할 엔트리포인트
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
