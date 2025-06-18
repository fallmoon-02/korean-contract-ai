import os
import json
import torch
import faiss
import openai
from flask import Flask, request, jsonify, render_template
from transformers import BertTokenizer, BertForSequenceClassification

app = Flask(__name__, static_folder="static", template_folder="templates")

# 1. 모델 로드 (Hugging Face)
MODEL_ID = "5wqs/kobert-risk-final"
HF_TOKEN = os.getenv("HF_TOKEN")

tokenizer = BertTokenizer.from_pretrained(MODEL_ID, token=HF_TOKEN)
model = BertForSequenceClassification.from_pretrained(MODEL_ID, token=HF_TOKEN)
model.eval()

# 2. Faiss 인덱스 및 템플릿 ID 로드
INDEX_PATH = os.path.join("templates_index", "templates.faiss")
IDS_PATH = os.path.join("templates_index", "templates_ids.json")

faiss_index = faiss.read_index(INDEX_PATH)
with open(IDS_PATH, "r", encoding="utf-8") as f:
    template_ids = json.load(f)

# 3. OpenAI 키
openai.api_key = os.getenv("OPENAI_API_KEY")

# ------------------------- 라우트 -------------------------

# 🔹 홈 화면
@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")


# 🔹 계약서 초안 생성
@app.route("/generate_draft", methods=["POST"])
def generate_draft():
    data = request.get_json()
    a = data.get("party_a", "")
    b = data.get("party_b", "")
    purpose = data.get("subject", "")
    date = data.get("date", "")

    prompt = f"""
당사자 A: {a}
당사자 B: {b}
계약 목적: {purpose}
효력 발생일: {date}

위 정보를 바탕으로 한국어 용역 계약서 초안을 조항별로 작성해 주세요. 각 조항은 번호와 제목을 포함해야 합니다.
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
        return jsonify({"error": str(e)}), 500


# 🔹 리스크 분석
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


# 🔹 유사 템플릿 추천
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
    results = []
    for i, idx in enumerate(I[0]):
        template = template_ids[idx]
        results.append({
            "template_id": template.get("id", f"T{idx+1}"),
            "title": template.get("title", "제목 없음"),
            "snippet": template.get("text", "")[:100],
            "score": float(D[0][i])
        })

    return jsonify({"templates": results})


# 🔹 앱 실행
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
