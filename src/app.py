from flask import Flask, request, jsonify, render_template
import os
import json
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from sentence_transformers import SentenceTransformer, util
import openai

app = Flask(__name__)

# 🔐 OpenAI API Key 설정
openai.api_key = os.getenv("OPENAI_API_KEY")

# ✅ 리스크 분석 모델 로드 (KoBERT)
RISK_MODEL = "5wqs/kobert-risk-final"
tokenizer = BertTokenizer.from_pretrained(RISK_MODEL)
model = BertForSequenceClassification.from_pretrained(RISK_MODEL)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# ✅ 템플릿 데이터 로드
TEMPLATE_PATH = "templates_index/templates_ids.json"
with open(TEMPLATE_PATH, "r", encoding="utf-8") as f:
    templates = json.load(f)
template_titles = [t["title"] for t in templates]

# ✅ 문장 임베딩 모델
embedding_model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
template_embeddings = embedding_model.encode(template_titles, convert_to_tensor=True)

@app.route("/")
def index():
    return render_template("index.html")

# ✅ 계약서 초안 생성 (GPT-3 방식)
@app.route("/generate_draft", methods=["POST"])
def generate_draft():
    data = request.get_json()
    party_a = data.get("party_a", "")
    party_b = data.get("party_b", "")
    subject = data.get("subject", "")
    date = data.get("date", "")

    prompt = f"""
다음 정보를 바탕으로 자연스러운 한국어 계약서를 작성해줘:

- 계약 당사자 A: {party_a}
- 계약 당사자 B: {party_b}
- 계약 목적: {subject}
- 효력 발생일: {date}

제1조 (목적), 제2조 (계약 기간), 제3조 (권리 및 의무), 제4조 (비밀유지), 제5조 (계약 해지), 제6조 (기타사항)를 포함해서 작성해줘.
문장은 법률적이고, 형식은 조항별로 나눠서 써줘.
"""

    try:
        response = openai.Completion.create(
            model="text-davinci-003",
            prompt=prompt,
            temperature=0.7,
            max_tokens=1500
        )
        draft = response.choices[0].text.strip()
        return jsonify({"draft": draft})
    except Exception as e:
        print("초안 생성 오류:", e)
        return jsonify({"error": str(e)}), 500

# ✅ 리스크 분석
@app.route("/analyze_risk", methods=["POST"])
def analyze_risk():
    clause = request.json.get("clause", "")
    try:
        inputs = tokenizer(clause, return_tensors="pt", truncation=True, padding=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
            pred = torch.argmax(outputs.logits, dim=1).item()
        risk_label = "HighRisk" if pred == 1 else "LowRisk"
        return jsonify({"risk_label": risk_label})
    except Exception as e:
        print("리스크 분석 오류:", e)
        return jsonify({"error": str(e)}), 500

# ✅ 템플릿 추천
@app.route("/recommend_templates", methods=["POST"])
def recommend_templates():
    clause = request.json.get("clause", "")
    try:
        query_embedding = embedding_model.encode(clause, convert_to_tensor=True)
        cosine_scores = util.cos_sim(query_embedding, template_embeddings)[0]
        top_k = torch.topk(cosine_scores, k=3)

        results = []
        for score, idx in zip(top_k.values, top_k.indices):
            template = templates[int(idx)]
            results.append({
                "title": template.get("title", ""),
                "file": template.get("file", "")
            })

        return jsonify({"templates": results})
    except Exception as e:
        print("템플릿 추천 오류:", e)
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
