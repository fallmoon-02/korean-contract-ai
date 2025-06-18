import os
import json
import torch
import faiss
import openai
from flask import Flask, request, jsonify, render_template
from transformers import BertTokenizer, BertForSequenceClassification
from huggingface_hub import hf_hub_download

app = Flask(
    __name__,
    static_folder="static",
    template_folder="templates"
)

# 1) 허깅페이스 모델 로드
MODEL_ID = "5wqs/kobert-risk-final"
HF_TOKEN = os.getenv("HF_TOKEN")

tokenizer = BertTokenizer.from_pretrained(MODEL_ID, token=HF_TOKEN)
model = BertForSequenceClassification.from_pretrained(MODEL_ID, token=HF_TOKEN)
model.eval()

# 2) Faiss 인덱스 로드 (허깅페이스에서 직접 다운로드)
INDEX_PATH = hf_hub_download(
    repo_id=MODEL_ID,
    filename="templates.faiss",
    token=HF_TOKEN
)
IDS_PATH = hf_hub_download(
    repo_id=MODEL_ID,
    filename="templates_ids.json",
    token=HF_TOKEN
)

index = faiss.read_index(INDEX_PATH)
with open(IDS_PATH, "r", encoding="utf-8") as f:
    template_ids = json.load(f)

# 3) OpenAI API 키 로딩
openai.api_key = os.getenv("OPENAI_API_KEY")

# ——————————————————————————
# 헬퍼 함수: 텍스트 임베딩
# ——————————————————————————
def embed(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding="max_length", max_length=256)
    with torch.no_grad():
        hidden = model.bert(**inputs).last_hidden_state
    vec = hidden.mean(dim=1).cpu().numpy().astype("float32")
    faiss.normalize_L2(vec)
    return vec

# ——————————————————————————
# 루트: index.html 렌더링
# ——————————————————————————
@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

# 1) 계약서 초안 생성
@app.route("/generate_draft", methods=["POST"])
def generate_draft():
    data = request.get_json()
    party_a = data.get("party_a")
    party_b = data.get("party_b")
    subject = data.get("subject")
    date = data.get("date")

    if not all([party_a, party_b, subject, date]):
        return jsonify({"error": "모든 필드가 필요합니다."}), 400

    prompt = f"""
다음 조건에 따라 한국어 계약서 초안을 작성해 주세요.

계약 당사자 A: {party_a}
계약 당사자 B: {party_b}
계약 목적: {subject}
효력 발생일: {date}

조항 형식: “제1조(목적) … 제2조(용역 범위) …”
"""

    try:
        res = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-0125",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=1500
        )
        draft = res.choices[0].message.content.strip()
        return jsonify({"draft": draft})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# 2) 리스크 분석
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
    return jsonify({"risk_label": label})

# 3) 유사 템플릿 추천
@app.route("/recommend_templates", methods=["POST"])
def recommend_templates():
    data = request.get_json()
    clause = data.get("clause")
    if not clause:
        return jsonify({"error": "clause is required"}), 400

    vec = embed(clause)
    D, I = index.search(vec, 3)
    recs = [template_ids[i] for i in I[0]]

    return jsonify({"templates": recs})

# ——————————————————————————
# 실행
# ——————————————————————————
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
