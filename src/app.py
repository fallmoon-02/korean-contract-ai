from flask import Flask, request, jsonify, render_template
import os
import json
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from sentence_transformers import SentenceTransformer, util
import openai

app = Flask(__name__)
openai.api_key = os.getenv("OPENAI_API_KEY")

# ✅ KoBERT 모델 (리스크 분석)
MODEL_NAME = "5wqs/kobert-risk-final"
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
model = BertForSequenceClassification.from_pretrained(MODEL_NAME)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# ✅ SBERT 모델 (유사도 기반 추천)
embedder = SentenceTransformer("snunlp/KR-SBERT-V40K-klueNLI-augSTS")

# ✅ 템플릿 로딩
TEMPLATE_PATH = "templates_index/templates_ids.json"
try:
    with open(TEMPLATE_PATH, "r", encoding="utf-8") as f:
        templates = json.load(f)
    template_titles = [t["title"] for t in templates]
    template_embeddings = embedder.encode(template_titles, convert_to_tensor=True)
except Exception as e:
    print(f"템플릿 파일 또는 임베딩 로딩 오류: {e}")
    templates, template_titles, template_embeddings = [], [], None

# ✅ 홈 페이지
@app.route("/")
def index():
    return render_template("index.html")

# ✅ 계약서 초안 생성 (GPT 3.5)
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
            outputs = model(**inputs)
            pred = torch.argmax(outputs.logits, dim=1).item()
        risk_label = "HighRisk" if pred == 1 else "LowRisk"
        return jsonify({"risk_label": risk_label})
    except Exception as e:
        print("리스크 분석 오류:", e)
        return jsonify({"error": str(e)}), 500

# ✅ SBERT 기반 템플릿 추천
@app.route("/recommend_templates", methods=["POST"])
def recommend_templates():
    clause = request.json.get("clause", "")
    try:
        if not template_embeddings:
            return jsonify({"templates": [], "error": "템플릿 임베딩이 없습니다."})

        clause_embedding = embedder.encode(clause, convert_to_tensor=True)
        cosine_scores = util.cos_sim(clause_embedding, template_embeddings)[0]
        top_indices = torch.topk(cosine_scores, k=3).indices.tolist()

        top_templates = [templates[i] for i in top_indices]
        return jsonify({"templates": top_templates})
    except Exception as e:
        print("템플릿 추천 오류:", e)
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
