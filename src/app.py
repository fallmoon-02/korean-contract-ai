from flask import Flask, request, jsonify, render_template
import json
import os

app = Flask(__name__)

# 홈 페이지 렌더링
@app.route("/")
def index():
    return render_template("index.html")

# 추천 리스트 API
@app.route("/recommend", methods=["POST"])
def recommend():
    user_text = request.json.get("text", "")

    # templates_ids.json 읽기
    template_path = "templates_index/templates_ids.json"
    if not os.path.exists(template_path):
        return jsonify({"results": [], "error": "템플릿 파일 없음"}), 500

    with open(template_path, "r", encoding="utf-8") as f:
        templates = json.load(f)

    # 여기선 FAISS 없이 그냥 상위 5개 샘플 반환
    top_k = templates[:5]

    return jsonify({"results": top_k})


# 리스크 분석 예시용 (선택)
@app.route("/analyze-risk", methods=["POST"])
def analyze_risk():
    clause = request.json.get("text", "")
    # 실제 모델 추론 로직을 여기에 넣어야 함
    dummy_prediction = "HighRisk" if "해지" in clause else "LowRisk"
    return jsonify({"label": dummy_prediction})


if __name__ == "__main__":
    app.run(debug=True)
