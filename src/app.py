import os
import json
from flask import Flask, request, jsonify, render_template
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from openai import OpenAI

app = Flask(__name__)

# ✅ GPT 클라이언트 초기화
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ✅ 리스크 분석 모델 초기화
MODEL_NAME = "5wqs/kobert-risk-final"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# ✅ 홈페이지 렌더링
@app.route("/")
def index():
    return render_template("index.html")


# ✅ 계약서 초안 생성 (GPT)
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
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=1500,
        )
        draft = response.choices[0].message.content
        return jsonify({"draft": draft})
    except Exception as e:
        print("초안 생성 오류:", e)
        return jsonify({"error": str(e)}), 500


# ✅ 리스크 분석 (KoBERT 모델 추론)
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


# ✅ 유사 템플릿 추천
@app.route("/recommend_templates", methods=["POST"])
def recommend_templates():
    user_clause = request.json.get("clause", "")
    template_path = "templates_index/templates_ids.json"

    if not os.path.exists(template_path):
        return jsonify({"templates": [], "error": "템플릿 파일이 없습니다."}), 500

    try:
        with open(template_path, "r", encoding="utf-8") as f:
            templates = json.load(f)
        top_k = templates[:3]
        return jsonify({"templates": top_k})
    except Exception as e:
        print("템플릿 추천 오류:", e)
        return jsonify({"templates": [], "error": str(e)}), 500


# ✅ 앱 실행
if __name__ == "__main__":
    app.run(debug=True)
