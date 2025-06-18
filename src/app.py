import os
import requests
from flask import Flask, request, jsonify

app = Flask(__name__)

# 루트 헬스체크용
@app.route("/", methods=["GET"])
def home():
    return "✅ Korean Contract AI API is running!"

# 리스크 분석 API
HF_TOKEN = os.getenv("HF_TOKEN")
API_URL = "https://api-inference.huggingface.co/models/5wqs/kobert-risk-final"
HEADERS = {"Authorization": f"Bearer {HF_TOKEN}"}

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

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
