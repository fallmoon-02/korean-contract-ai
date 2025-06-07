import os
import torch
import pandas as pd
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score
from transformers import BertTokenizer, BertForSequenceClassification

def evaluate_risk_model():
    # 1) 라벨링된 전체 CSV 경로
    LABELED_CSV = os.path.join(os.getcwd(), "data", "clauses_labeled_full.csv")

    # 2) CSV 로드 → 🤗 Dataset 변환
    df = pd.read_csv(LABELED_CSV, encoding="utf-8-sig")
    ds = Dataset.from_pandas(df[["text", "label"]])

    # 3) 모델 & 토크나이저 로드
    MODEL_DIR = os.path.join(os.getcwd(), "kobert_risk_model")
    tokenizer = BertTokenizer.from_pretrained(MODEL_DIR)
    model = BertForSequenceClassification.from_pretrained(MODEL_DIR)
    model.eval()

    # 4) 토큰화 함수
    def tokenize_fn(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=128
        )

    tokenized = ds.map(tokenize_fn, batched=True)
    tokenized = tokenized.rename_column("label", "labels")
    tokenized.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    # 5) 80:20 비율로 train/test 분리 (seed 고정)
    split = tokenized.train_test_split(test_size=0.2, seed=2025)
    eval_dataset = split["test"]  # 검증용 데이터

    # 6) DataLoader 준비 (배치 사이즈 16)
    from torch.utils.data import DataLoader
    eval_loader = DataLoader(eval_dataset, batch_size=16)

    # 7) 예측 결과와 실제 레이블을 모을 리스트
    all_preds = []
    all_labels = []

    # 8) 평가 루프
    for batch in eval_loader:
        inputs = {
            "input_ids": batch["input_ids"],
            "attention_mask": batch["attention_mask"],
        }
        with torch.no_grad():
            outputs = model(**inputs)
        logits = outputs.logits
        preds = torch.argmax(logits, dim=1).cpu().numpy().tolist()
        labels = batch["labels"].cpu().numpy().tolist()

        all_preds.extend(preds)
        all_labels.extend(labels)

    # 9) 최종 스코어 계산 (scikit-learn)
    acc = accuracy_score(all_labels, all_preds)
    f1_weighted = f1_score(all_labels, all_preds, average="weighted")
    f1_macro = f1_score(all_labels, all_preds, average="macro")

    print(f"[RESULT] Accuracy: {acc:.4f}")
    print(f"[RESULT] F1 (weighted): {f1_weighted:.4f}")
    print(f"[RESULT] F1 (macro): {f1_macro:.4f}")

if __name__ == "__main__":
    evaluate_risk_model()
