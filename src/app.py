from flask import Flask, request, jsonify, render_template
import os
import json
import torch
import torch.nn.functional as F
from transformers import BertTokenizer, BertForSequenceClassification
from sentence_transformers import SentenceTransformer, util
import openai

app = Flask(__name__)

# 🔐 OpenAI API 키
openai.api_key = os.getenv("OPENAI_API_KEY")

# ✅ 리스크 분석용 KoBERT 모델 초기화
MODEL_NAME = "5wqs/kobert-risk-final"
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
risk_model = BertForSequenceClassification.from_pretrained(MODEL_NAME)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
risk_model.to(device)
risk_model.eval()

# ✅ SBERT 임베딩 모델 초기화
embedder = SentenceTransformer("snunlp/KR-SBERT-V40K-klueNLI-augSTS")

# ✅ 템플릿 로딩 (id, title, file, snippet 포함)
TEMPLATE_PATH = "templates_index/templates_ids.json"
try:
    with open(TEMPLATE_PATH, "r", encoding="utf-8") as f:
        templates = json.load(f)
except Exception as e:
    print(f"[오류] 템플릿 로딩 실패: {e}")
    templates = []

# ✅ 템플릿 임베딩 사전 계산
template_snippets = [t.get("snippet", t["title"]) for t in templates]
template_embeddings = embedder.encode(template_snippets, convert_to_tensor=True)

# ✅ 홈
@app.route("/")
def index():
    return render_template("index.html")

# ✅ 계약서 초안 생성
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

제1조 (목적), 제2조 (계약 기간), 제3조 (권리 및 의무), 제4조 (비밀유지), 제5조 (계약 해지), 제6조 (기타사항) 항목 포함.
실제 계약서처럼 법률 문체로 구체적으로 작성해줘.
    """

    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=1500,
        )
        draft = response.choices[0].message["content"]
        return jsonify({"draft": draft})
    except Exception as e:
        print("초안 생성 오류:", e)
        return jsonify({"error": str(e)}), 500

# ✅ 리스크 분석
@app.route("/analyze_risk", methods=["POST"])
def analyze_risk():
    clause = request.json.get("clause", "")
    try:
        inputs = tokenizer(clause, return_tensors="pt", truncation=True, padding=True, max_length=512).to(device)
        with torch.no_grad():
            outputs = risk_model(**inputs)
            pred = torch.argmax(outputs.logits, dim=1).item()
        risk_label = "HighRisk" if pred == 1 else "LowRisk"
        return jsonify({"risk_label": risk_label})
    except Exception as e:
        print("리스크 분석 오류:", e)
        return jsonify({"error": str(e)}), 500

# ✅ 템플릿 추천 (SBERT + Cosine Similarity)
@app.route("/recommend_templates", methods=["POST"])
def recommend_templates():
    clause = request.json.get("clause", "")
    try:
        clause_embedding = embedder.encode(clause, convert_to_tensor=True)
        similarities = util.cos_sim(clause_embedding, template_embeddings)[0]
        top_indices = torch.topk(similarities, k=3).indices.tolist()
        top_templates = [templates[i] for i in top_indices]
        return jsonify({"templates": top_templates})
    except Exception as e:
        print("템플릿 추천 오류:", e)
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
