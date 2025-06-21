from flask import Flask, request, jsonify, render_template
import json
import os
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from sentence_transformers import SentenceTransformer, util
import openai

app = Flask(__name__)

# ✅ OpenAI API 초기화
openai.api_key = os.getenv("OPENAI_API_KEY")

# ✅ 계약 리스크 분석 모델 (KoBERT)
MODEL_NAME = "5wqs/kobert-risk-final"
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
model = BertForSequenceClassification.from_pretrained(MODEL_NAME)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# ✅ 문장 임베딩 모델 (SBERT)
embedder = SentenceTransformer("jhgan/ko-sbert-nli")

# ✅ 템플릿 불러오기 및 임베딩
template_path = "templates_index/templates_ids.json"
templates = []
template_embeddings = None

if os.path.exists(template_path):
    with open(template_path, encoding="utf-8") as f:
        templates = json.load(f)
    # snippet 없으면 title만 사용
    template_texts = [t.get("snippet") or t["title"] for t in templates]
    template_embeddings = embedder.encode(template_texts, convert_to_tensor=True)

# ✅ 라우팅

@app.route("/")
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
너는 계약서 전문 법률 비서야. 아래 정보를 참고해서 한국어 계약서 초안을 전문적으로 작성해줘.

- 계약 당사자 A: {party_a}
- 계약 당사자 B: {party_b}
- 계약 목적: {subject}
- 효력 발생일: {date}

제1조 (목적), 제2조 (계약 기간), 제3조 (권리 및 의무), 제4조 (비밀유지), 제5조 (계약 해지), 제6조 (기타사항)을 포함해서 실제 계약서처럼 구체적으로 작성해줘.
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


@app.route("/recommend_templates", methods=["POST"])
def recommend_templates():
    clause = request.json.get("clause", "")
    try:
        if template_embeddings is None or len(template_embeddings) == 0:
            return jsonify({"templates": [], "error": "템플릿 임베딩이 없습니다."}), 500

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
