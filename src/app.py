

import os
import json

import torch
import faiss
import openai
from flask import Flask, request, jsonify
from transformers import BertTokenizer, BertForSequenceClassification

app = Flask(__name__)

# ——————————————————————————
# 1) 허깅페이스 허브에서 파인튜닝된 모델 로드
# ——————————————————————————
MODEL_ID = "5wqs/kobert-risk-final"               # 여러분의 HF 리포지터리 아이디
HF_TOKEN = os.getenv("HUGGINGFACE_TOKEN")         # Render 환경변수로 설정

tokenizer = BertTokenizer.from_pretrained(
    MODEL_ID,
    use_auth_token=HF_TOKEN
)
model = BertForSequenceClassification.from_pretrained(
    MODEL_ID,
    use_auth_token=HF_TOKEN
)
model.eval()

# ——————————————————————————
# 2) Faiss 인덱스 & 템플릿 ID 리스트 로드
# ——————————————————————————
INDEX_PATH = os.path.join("data", "templates.faiss")
IDS_PATH   = os.path.join("data", "templates_ids.json")

index = faiss.read_index(INDEX_PATH)
with open(IDS_PATH, "r", encoding="utf-8") as f:
    template_ids = json.load(f)

# ——————————————————————————
# 3) OpenAI 클라이언트 초기화
# ——————————————————————————
openai.api_key = os.getenv("OPENAI_API_KEY")  # Render 환경변수로 설정

# ——————————————————————————
# 헬퍼: 텍스트 임베딩 (평균 풀링)
# ——————————————————————————
def embed(text: str):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=256
    )
    with torch.no_grad():
        hidden = model.bert(**inputs, return_dict=True).last_hidden_state  # (1, L, D)
    vec = hidden.mean(dim=1).cpu().numpy().astype("float32")              # (1, D)
    faiss.normalize_L2(vec)
    return vec

# ——————————————————————————
# 1) 계약서 초안 생성
# ——————————————————————————
@app.route("/generate_draft", methods=["POST"])
def generate_draft():
    data = request.get_json(silent=True)
    keywords = data.get("keywords") if data else None
    if not keywords:
        return jsonify({"error": "Provide 'keywords' field."}), 400

    prompt = f"""
아래 키워드를 참고하여, 한국어 표준 용역 계약서 초안을 조항별로 작성해 주세요.

{keywords}

• 각 조항은 “제1조(목적) … 제2조(용역 범위) …” 형태로 번호와 제목을 붙여 주세요.
"""
    try:
        res = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=1500
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    draft_text = res.choices[0].message.content.strip()
    return jsonify({"draft": draft_text})

# ——————————————————————————
# 2) 조항별 리스크 예측
# ——————————————————————————
@app.route("/analyze_risk", methods=["POST"])
def analyze_risk():
    data = request.get_json(silent=True)
    clause = data.get("clause") if data else None
    if not clause:
        return jsonify({"error": "Provide 'clause' field."}), 400

    inputs = tokenizer(
        clause,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=256
    )
    with torch.no_grad():
        logits = model(**inputs).logits
    pred = torch.argmax(logits, dim=1).item()
    label = "HighRisk" if pred == 1 else "LowRisk"
    return jsonify({"label": label})

# ——————————————————————————
# 3) 유사 템플릿 추천
# ——————————————————————————
@app.route("/recommend_templates", methods=["POST"])
def recommend_templates():
    data = request.get_json(silent=True)
    text = data.get("text") if data else None
    topk = data.get("topk", 3) if data else 3
    if not text:
        return jsonify({"error": "Provide 'text' field."}), 400

    try:
        topk = int(topk)
    except:
        return jsonify({"error": "'topk' must be an integer."}), 400

    vec = embed(text)
    D, I = index.search(vec, topk)
    recs = [template_ids[i] for i in I[0]]
    return jsonify({"recommendations": recs})

# ——————————————————————————
# 서버 실행
# ——————————————————————————
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
