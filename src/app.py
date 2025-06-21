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

# ✅ KoBERT 로드 (리스크 분석용)
MODEL_NAME = "5wqs/kobert-risk-final"
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
model = BertForSequenceClassification.from_pretrained(MODEL_NAME)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# ✅ SBERT 로드 (템플릿 추천용)
embedder = SentenceTransformer("jhgan/ko-sbert-nli")

# ✅ 템플릿 로딩 (최대 100개 제한)
template_path = "templates_index/templates_ids.json"
if os.path.exists(template_path):
    with open(template_path, "r", encoding="utf-8") as f:
        templates = json.load(f)[:100]
    template_texts = [t["snippet"] for t in templates if "snippet" in t]
    template_embeddings = embedder.encode(template_texts, convert_to_tensor=True)
else:
    templates = []
    template_embeddings = []

@app.route("/")
def index():
    return render_template("index.html")

# ✅ 계약서 초안 생성 (GPT-3.5)
@app.route("/generate_draft", methods=["POST"])
def generate_draft():
    data = request.get_json()
    prompt = f"""
너는 한국어 계약서를 작성하는 법률 비서야.
다음 정보를 바탕으로 계약서를 조항별로 자세히 작성해줘:
- 계약 당사자 A: {data.get("party_a", "")}
- 계약 당사자 B: {data.get("party_b", "")}
- 계약 목적: {data.get("subject", "")}
- 효력 발생일: {data.get("date", "")}

제1조 (목적), 제2조 (계약 기간), 제3조 (권리 및 의무), 제4조 (비밀유지), 제5조 (계약 해지), 제6조 (기타사항) 포함해서 작성해줘.
"""
    try:
        res = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=1500,
        )
        draft = res.choices[0].message.content
        return jsonify({"draft": draft})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ✅ 리스크 분석
@app.route("/analyze_risk", methods=["POST"])
def analyze_risk():
    clause = request.json.get("clause", "")
    try:
        inputs = tokenizer(clause, return_tensors="pt", truncation=True, padding=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            logits = model(**inputs).logits
        pred = torch.argmax(logits, dim=1).item()
        return jsonify({"risk_label": "HighRisk" if pred == 1 else "LowRisk"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ✅ 템플릿 추천
@app.route("/recommend_templates", methods=["POST"])
def recommend_templates():
    clause = request.json.get("clause", "")
    try:
        clause_embedding = embedder.encode(clause, convert_to_tensor=True)
        scores = util.cos_sim(clause_embedding, template_embeddings)[0]
        top_k_idx = torch.topk(scores, k=3).indices
        top_k_templates = [templates[idx] for idx in top_k_idx]
        return jsonify({"templates": top_k_templates})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
