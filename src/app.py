import os
import json
import torch
from flask import Flask, request, jsonify, render_template
from transformers import BertTokenizer, BertForSequenceClassification
from sentence_transformers import SentenceTransformer, util
import openai

# 환경 설정
os.environ["TOKENIZERS_PARALLELISM"] = "false"
openai.api_key = os.getenv("OPENAI_API_KEY")

app = Flask(__name__)

# 리스크 분석 모델 로드
RISK_MODEL_NAME = "5wqs/kobert-risk-final"
tokenizer = BertTokenizer.from_pretrained(RISK_MODEL_NAME)
model = BertForSequenceClassification.from_pretrained(RISK_MODEL_NAME)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# 문장 임베딩 모델 로드
embedder = SentenceTransformer("jhgan/ko-sbert-nli")

# 템플릿 불러오기 및 임베딩 계산
TEMPLATE_PATH = "templates_index/templates_ids.json"
if os.path.exists(TEMPLATE_PATH):
    with open(TEMPLATE_PATH, "r", encoding="utf-8") as f:
        templates = json.load(f)
    template_texts = [t["snippet"] if "snippet" in t else t.get("text", "") for t in templates]
    template_embeddings = embedder.encode(template_texts, convert_to_tensor=True)
else:
    templates = []
    template_embeddings = []

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
    user_clause = request.json.get("clause", "")
    try:
        user_embedding = embedder.encode(user_clause, convert_to_tensor=True)
        cos_scores = util.pytorch_cos_sim(user_embedding, template_embeddings)[0]
        top_results = torch.topk(cos_scores, k=min(3, len(templates)))

        results = []
        for score, idx in zip(top_results.values, top_results.indices):
            t = templates[idx.item()]
            results.append({
                "template_id": t.get("template_id", idx.item()),
                "title": t.get("title", "제목 없음"),
                "score": float(score),
                "snippet": t.get("snippet", ""),
                "file": t.get("file", "")
            })

        return jsonify({"templates": results})
    except Exception as e:
        print("템플릿 추천 오류:", e)
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
