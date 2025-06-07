import os
import numpy as np
import pandas as pd
import torch
import optuna

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from datasets import Dataset
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    Trainer,
    TrainingArguments,
)
from transformers.trainer_utils import EvalPrediction

# ——————————————————————————
# 1) 데이터 로드 및 train_ds/val_ds 생성
# ——————————————————————————

CSV_PATH = os.path.join(os.getcwd(), "data", "clauses_labeled_full.csv")
df = pd.read_csv(CSV_PATH, encoding="utf-8-sig")

# (선택) 특수문자/공백 정리
df["text"] = df["text"].apply(
    lambda x: " ".join(x.replace("\n", " ").replace("\r", " ").split())
)

train_df, val_df = train_test_split(
    df, test_size=0.2, stratify=df["label"], random_state=2025
)

train_ds = Dataset.from_pandas(train_df[["text", "label"]])
val_ds = Dataset.from_pandas(val_df[["text", "label"]])

# ——————————————————————————
# 2) 토크나이저 로드 및 데이터셋 토큰화 함수 정의
# ——————————————————————————
MODEL_NAME = "monologg/kobert"
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)

def tokenize_fn(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=256,
    )

train_ds = train_ds.map(tokenize_fn, batched=True)
val_ds = val_ds.map(tokenize_fn, batched=True)

train_ds = train_ds.rename_column("label", "labels")
val_ds = val_ds.rename_column("label", "labels")

train_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
val_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

# ——————————————————————————
# 3) Optuna용 콜백 함수, 메트릭 계산 함수 정의
# ——————————————————————————

def model_init(trial):
    dropout = trial.suggest_float("dropout", 0.1, 0.3)
    return BertForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2,
        hidden_dropout_prob=dropout,
        attention_probs_dropout_prob=dropout,
    )

def compute_metrics(p: EvalPrediction):
    preds = np.argmax(p.predictions, axis=1)
    f1 = f1_score(p.label_ids, preds, average="macro")
    return {"f1": f1}

def objective(trial):
    lr = trial.suggest_loguniform("learning_rate", 1e-5, 5e-5)
    bs = trial.suggest_categorical("batch_size", [8, 16])
    epochs = trial.suggest_int("num_train_epochs", 3, 6)

    args = TrainingArguments(
        output_dir="./optuna_output",
        do_eval=True,              # 검증 수행
        eval_steps=100,            # 100 스텝마다 검증
        save_steps=100,            # 100 스텝마다 체크포인트 저장
        logging_steps=50,
        save_total_limit=1,
        learning_rate=lr,
        per_device_train_batch_size=bs,
        per_device_eval_batch_size=bs * 2,
        num_train_epochs=epochs,
        weight_decay=0.01,
        disable_tqdm=True,
        report_to="none",
    )

    trainer = Trainer(
        model_init=lambda: model_init(trial),
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    eval_result = trainer.evaluate()
    return eval_result["eval_f1"]

# ——————————————————————————
# 4) Optuna 스터디 생성 및 튜닝 실행
# ——————————————————————————
if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=10)

    print("===== Best trial hyperparameters =====")
    print(study.best_trial.params)
