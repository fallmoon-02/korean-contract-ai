import os
import json
import torch
import faiss
import openai
from flask import Flask, request, jsonify, render_template
from transformers import BertTokenizer, BertForSequenceClassification
from dotenv import load_dotenv

# 환경변수 로드
load_dotenv()

app = Flask(__name__, static_folder="static", template_folder="templates")

# 1. 모델 로드
MODEL_ID = "5wqs/kobert-risk-final"
HF_TOKEN = os.getenv("HF_TOKEN")

tokenizer = BertTokenizer.from_pretrained(MODEL_ID, token=HF_TOKEN)
model = BertForSequenceClassification.from_pretrained(MODEL_ID, token=HF_TOKEN)
model.eval()

# 2. FAISS 인덱스 로드
INDEX_PATH = os.path.join("templates_index", "templates.faiss")
IDS_PATH = os.path.join("templates_index", "templates_ids.json")

faiss_index = faiss.read_index(INDEX_PATH)
with open(IDS_PATH, "r", encoding="utf-8") as f:
    template_ids = json.load(f)

# 3. OpenAI 키 로딩
openai.api_key = os.getenv("OPENAI_API_KEY")

# -------------------------
# 임베딩 함수
# -------------------------
def embed(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding="max_length", max_length=256)
    with torch.no_grad():
        hidden = model.bert(**inputs).last_hidden_state
    vec = hidden.mean(dim=1).cpu().numpy().astype("float32")
    faiss.normalize_L2(vec)
    return vec

# -------------------------
# 라우트
# -------------------------
@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/generate_draft", methods=["POST"])
def generate_draft():
    data = request.get_json()
    party_a = data.get("party_a", "")
    party_b = data.get("party_b", "")
    subject = data.get("subject", "")
    date = data.get("date", "")

    prompt = f"""
너는 한국어 계약서를 작성하는 법률 비서야.

다음 정보를 바탕으로 계약서를 자연스럽고 조항별로 작성해줘:

- 계약 당사자 A: {party_a}
- 계약 당사자 B: {party_b}
- 계약 목적: {subject}
- 효력 발생일: {date}

제1조 (목적), 제2조 (계약 기간), 제3조 (권리 및 의무), 제4조 (비밀유지), 제5조 (계약 해지), 제6조 (기타사항) 등의 항목을 포함해줘.
법률적 문체를 사용하고, 각 조항은 실제 계약서처럼 구체적으로 작성해줘.
    """


    try:
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=1500
        )
        draft = response.choices[0].message.content.strip()
        return jsonify({"draft": draft})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/analyze_risk", methods=["POST"])
def analyze_risk():
    data = request.get_json()
    clause = data.get("clause", "")
    if not clause:
        return jsonify({"error": "clause is required"}), 400

    inputs = tokenizer(clause, return_tensors="pt", truncation=True, padding="max_length", max_length=256)
    with torch.no_grad():
        logits = model(**inputs).logits
    pred = torch.argmax(logits, dim=1).item()
    label = "HighRisk" if pred == 1 else "LowRisk"
    return jsonify({"risk_label": label})

@app.route("/recommend_templates", methods=["POST"])
def recommend_templates():
    data = request.get_json()
    clause = data.get("clause", "")
    if not clause:
        return jsonify({"error": "clause is required"}), 400

    vec = embed(clause)
    D, I = faiss_index.search(vec, 3)

    recs = []
    for idx in I[0]:
        template = template_ids[idx]
        if isinstance(template, dict):
            recs.append({
                "template_id": template.get("template_id", f"id_{idx}"),
                "title": template.get("title", ""),
                "score": float(D[0][list(I[0]).index(idx)]),
                "snippet": template.get("snippet", "")
            })
        else:
            # fallback in case template is a string
            recs.append({
                "template_id": f"id_{idx}",
                "title": str(template),
                "score": float(D[0][list(I[0]).index(idx)]),
                "snippet": ""
            })

    return jsonify({"templates": recs})

# -------------------------
# 실행
# -------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
