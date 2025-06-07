# src/build_faiss_index.py

import os, json, faiss, numpy as np, torch
from transformers import BertTokenizer, BertForSequenceClassification

# 1) 경로
TEMPLATES_JSON = os.path.join("data", "templates_list.json")
INDEX_PATH      = os.path.join("data", "templates.faiss")
IDS_PATH        = os.path.join("data", "templates_ids.json")
MODEL_DIR       = "kobert_risk_final"

# 2) KoBERT 모델 & 토크나이저 로드
tokenizer = BertTokenizer.from_pretrained(MODEL_DIR)
model     = BertForSequenceClassification.from_pretrained(MODEL_DIR)
model.eval()

# 3) 임베딩 함수 (평균 풀링)
def embed_texts(texts):
    inputs = tokenizer(texts, return_tensors="pt",
                       truncation=True, padding="max_length", max_length=256)
    with torch.no_grad():
        out = model.bert(**inputs, return_dict=True).last_hidden_state  # (B, L, D)
    pooled = out.mean(dim=1).cpu().numpy().astype("float32")          # (B, D)
    return pooled

# 4) 템플릿 불러와서 전체 조항 합치기
with open(TEMPLATES_JSON, "r", encoding="utf-8") as f:
    templates = json.load(f)

texts, ids = [], []
for tmpl in templates:
    full = " ".join(cl["text"] for cl in tmpl["clauses"])
    texts.append(full)
    ids.append(tmpl["template_id"])

# 5) 일괄 임베딩
batch_size = 16
all_vecs = []
for i in range(0, len(texts), batch_size):
    batch = texts[i:i+batch_size]
    all_vecs.append(embed_texts(batch))
embeddings = np.vstack(all_vecs)  # shape (N, D)

# 6) Faiss 인덱스 생성 & 저장
dim = embeddings.shape[1]
faiss.normalize_L2(embeddings)
index = faiss.IndexFlatIP(dim)
index.add(embeddings)
faiss.write_index(index, INDEX_PATH)

# 7) ID 리스트 저장
with open(IDS_PATH, "w", encoding="utf-8") as f:
    json.dump(ids, f, ensure_ascii=False)

print(f"[INFO] Faiss index saved to {INDEX_PATH}")
print(f"[INFO] Template IDs saved to {IDS_PATH}")
