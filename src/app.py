import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from flask import Flask, request, jsonify
from transformers import BertTokenizer, BertForSequenceClassification
from sentence_transformers import SentenceTransformer, util
import torch
import json

app = Flask(__name__)

# 디바이스 설정 (가능하면 GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 모델 로드 (지연 로딩용으로 None으로 시작)
tokenizer = None
risk_model = None
embedder = SentenceTransformer("jhgan/ko-sbert-nli")  # CPU friendly 모델

# 템플릿 데이터 로딩 (embedding 미리 생성된 버전 사용 권장)
with open("templates.json", encoding="utf-8") as f:
    templates = json.load(f)
    template_texts = [t["snippet"] for t in templates]
    template_embeddings = embedder.encode(template_texts, convert_to_tensor=True)


@app.route("/generate_draft", methods=["POST"])
def generate_draft():
    try:
        from openai import OpenAI
        openai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        data = request.json
        prompt = f"계약 당사자: {data['party_a']} 와 {data['party_b']}\n계약 목적: {data['subject']}\n시작일: {data['date']}\n위 정보를 바탕으로 한글 계약서 초안을 만들어줘."

        completion = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "당신은 계약서 초안 작성 전문가입니다."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1024,
        )
        return jsonify({"draft": completion.choices[0].message.content})

    except Exception as e:
        print("초안 생성 오류:", e)
        return jsonify({"error": str(e)}), 500


@app.route("/analyze_risk", methods=["POST"])
def analyze_risk():
    global tokenizer, risk_model

    try:
        clause = request.json.get("clause", "")

        # 필요할 때만 로드
        if tokenizer is None or risk_model is None:
            tokenizer = BertTokenizer.from_pretrained("beomi/kcbert-base")
            risk_model = BertForSequenceClassification.from_pretrained("monologg/koelectra-small-discriminator")
            risk_model.to(device)

        inputs = tokenizer(clause, return_tensors="pt", truncation=True, padding=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = risk_model(**inputs)
            pred = torch.argmax(outputs.logits, dim=1).item()

        risk_label = "HighRisk" if pred == 1 else "LowRisk"
        return jsonify({"risk_label": risk_label})

    except Exception as e:
        print("리스크 분석 오류:", e)
        return jsonify({"error": str(e)}), 500


@app.route("/recommend_templates", methods=["POST"])
def recommend_templates():
    try:
        clause = request.json.get("clause", "")
        query_embedding = embedder.encode(clause, convert_to_tensor=True)

        cos_scores = util.cos_sim(query_embedding, template_embeddings)[0]
        top_results = torch.topk(cos_scores, k=5)

        top_templates = []
        for score, idx in zip(top_results.values, top_results.indices):
            t = templates[idx.item()]
            top_templates.append({"title": t["title"], "file": t["file"]})

        return jsonify({"templates": top_templates})

    except Exception as e:
        print("템플릿 추천 오류:", e)
        return jsonify({"error": str(e)}), 500


@app.route("/")
def home():
    return "AI 계약서 서비스가 실행 중입니다."


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
