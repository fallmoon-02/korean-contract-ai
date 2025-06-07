import os
import pandas as pd
import torch
from datasets import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments

# ——————————————————————————
# 1) 데이터 로드 및 전처리
# ——————————————————————————
df = pd.read_csv("data/clauses_labeled_full.csv", encoding="utf-8-sig")
df["text"] = df["text"].apply(
    lambda x: " ".join(x.replace("\n", " ").replace("\r", " ").split())
)

train_df, val_df = train_test_split(
    df, test_size=0.2, stratify=df["label"], random_state=2025
)
train_ds = Dataset.from_pandas(train_df[["text", "label"]])
val_ds = Dataset.from_pandas(val_df[["text", "label"]])

# ——————————————————————————
# 2) 토크나이저 & 모델 로드 (최적 dropout 반영)
# ——————————————————————————
MODEL_NAME = "monologg/kobert"
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
model = BertForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2,
    hidden_dropout_prob=0.14243969479652366,       # Optuna 최적값
    attention_probs_dropout_prob=0.14243969479652366
)

# ——————————————————————————
# 3) 데이터셋 토큰화
# ——————————————————————————
def tokenize_fn(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=256
    )

train_ds = train_ds.map(tokenize_fn, batched=True)
val_ds = val_ds.map(tokenize_fn, batched=True)
train_ds = train_ds.rename_column("label", "labels")
val_ds = val_ds.rename_column("label", "labels")
train_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
val_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

# ——————————————————————————
# 4) Trainer 인자 설정 (Optuna 최적값 반영)
# ——————————————————————————
training_args = TrainingArguments(
    output_dir="kobert_risk_final",               # 최종 모델 저장 경로
    num_train_epochs=5,                            # Optuna 최적값
    per_device_train_batch_size=16,                # Optuna 최적값
    per_device_eval_batch_size=32,                 # (batch_size * 2 권장)
    learning_rate=1.4697079226949323e-05,          # Optuna 최적값
    weight_decay=0.01,
    warmup_steps=100,
    do_eval=True,
    eval_steps=200,        # 검증/저장 주기: 200 스텝마다
    save_steps=200,
    save_total_limit=1,
    logging_steps=100,
    # GPU가 있는 경우 자동으로 GPU 사용, 없으면 CPU 사용
)

# ——————————————————————————
# 5) Trainer 생성 및 학습
# ——————————————————————————
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
)

trainer.train()
model.save_pretrained("kobert_risk_final")
tokenizer.save_pretrained("kobert_risk_final")

# ——————————————————————————
# 6) 검증 성능 수동 계산 (scikit-learn)
# ——————————————————————————
from torch.utils.data import DataLoader

val_loader = DataLoader(val_ds, batch_size=32)
all_preds, all_labels = [], []

for batch in val_loader:
    inputs = {
        "input_ids": batch["input_ids"],
        "attention_mask": batch["attention_mask"],
    }
    with torch.no_grad():
        outputs = model(**inputs)
    preds = torch.argmax(outputs.logits, dim=1).cpu().numpy().tolist()
    labels = batch["labels"].cpu().numpy().tolist()
    all_preds.extend(preds)
    all_labels.extend(labels)

acc = accuracy_score(all_labels, all_preds)
f1_w = f1_score(all_labels, all_preds, average="weighted")
f1_m = f1_score(all_labels, all_preds, average="macro")
print(f"[FINAL] Accuracy: {acc:.4f}  F1_w: {f1_w:.4f}  F1_m: {f1_m:.4f}")