from flask import Flask, request, jsonify, render_template
import os
import json
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from sentence_transformers import SentenceTransformer, util
import openai

app = Flask(__name__)

# 🔐 OpenAI API 설정 (GPT 3.5 사용)
openai.api_key = os.getenv("OPENAI_API_KEY")

# ✅ KoBERT 리스크 분석 모델
MODEL_NAME = "5wqs/kobert-risk-final"
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
model = BertForSequenceClassification.from_pretrained(MODEL_NAME)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device).eval()

# ✅ SBERT 임베딩 모델
embedder = SentenceTransformer("jhgan/ko-sbert-nli")

# ✅ 템플릿 데이터 로드 및 임베딩
TEMPLATE_PATH = "templates_index/templates_ids.json"
templates = []
template_embeddings = []

if os.path.exists(TEMPLATE_PATH):
    with open(TEMPLATE_PATH, "r", encoding="utf-8") as f:
        templates = json.load(f)
    template_texts = [t.get("snippet", t["title"]) for t in templates]
    template_embeddings = embedder.encode(template_texts, convert_to_tensor=True)
else:
    print("❗템플릿 JSON 파일이 없습니다.")

# 🔹 홈페이지
@app.route("/")
def index():
    return render_template("index.html")

# 🔹 계약서 초안 생성 (GPT-3.5)
@app.route("/generate_draft", methods=["POST"])
def generate_draft():
    data = request.get_json()
    party_a = data.get("party_a", "")
    party_b = data.get("party_b", "")
    subject = data.get("subject", "")
    date = data.get("date", "")

    prompt = f"""
너는 계약서를 전문적으로 작성하는 법률 비서야.

다음 정보를 참고하여 한국어 계약서를 조항별로 구성해줘:

- 계약 당사자 A: {party_a}
- 계약 당사자 B: {party_b}
- 계약 목적: {subject}
- 효력 발생일: {date}

반드시 제1조 (목적), 제2조 (계약 기간), 제3조 (권리 및 의무), 제4조 (비밀유지), 제5조 (계약 해지), 제6조 (기타사항) 항목 포함.
법률 문체를 사용해서 자연스럽게 작성해줘.
    """

    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=1500
        )
        draft = response.choices[0].message["content"]
        return jsonify({"draft": draft})
    except Exception as e:
        print("초안 생성 오류:", e)
        return jsonify({"error": str(e)}), 500

# 🔹 리스크 분석 (KoBERT)
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

# 🔹 템플릿 추천 (SBERT 임베딩 + 코사인 유사도)
@app.route("/recommend_templates", methods=["POST"])
def recommend_templates():
    clause = request.json.get("clause", "")
    try:
        if not template_embeddings:
            return jsonify({"templates": [], "error": "템플릿 임베딩이 없습니다."})

        query_embedding = embedder.encode(clause, convert_to_tensor=True)
        scores = util.pytorch_cos_sim(query_embedding, template_embeddings)[0]
        top_k = torch.topk(scores, k=3)

        results = []
        for score, idx in zip(top_k.values, top_k.indices):
            t = templates[idx.item()]
            results.append({
                "title": t["title"],
                "file": t.get("file", "#"),
                "score": float(score)
            })

        return jsonify({"templates": results})
    except Exception as e:
        print("템플릿 추천 오류:", e)
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
