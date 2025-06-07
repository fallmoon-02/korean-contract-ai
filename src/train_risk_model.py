import os
import pandas as pd
from datasets import Dataset
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments

def main():
    # 1) 학습에 사용할 라벨링된 CSV 경로 설정
    LABELED_CSV = os.path.join(os.getcwd(), "data", "clauses_labeled_full.csv")

    # 2) 데이터 로드 (pandas → 🤗 datasets로 변환)
    df = pd.read_csv(LABELED_CSV, encoding="utf-8-sig")
    print(f"[INFO] 라벨링된 데이터 개수: {len(df)}")

    # 3) 🤗 datasets 형태로 변환
    ds = Dataset.from_pandas(df[["text", "label"]])

    # 4) KoBERT 토크나이저 & 모델 로드
    MODEL_NAME = "monologg/kobert"
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
    model = BertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

    # 5) 데이터셋 토큰화 함수 정의
    def tokenize_fn(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=128
        )

    tokenized_ds = ds.map(tokenize_fn, batched=True)
    tokenized_ds = tokenized_ds.rename_column("label", "labels")
    tokenized_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    # 6) 훈련/검증 데이터셋 분리 (80:20)
    split_ds = tokenized_ds.train_test_split(test_size=0.2, seed=2025)
    train_dataset = split_ds["train"]
    eval_dataset = split_ds["test"]

    # 7) Trainer 인자 설정 (evaluation_strategy 제거, do_eval + eval_steps 사용)
    training_args = TrainingArguments(
        output_dir="kobert_risk_model",
        num_train_epochs=5,               # 에폭 수 필요시 조정
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        do_eval=True,                     # 검증 수행
        eval_steps=500,                   # 500 스텝마다 검증
        save_total_limit=1,
        logging_steps=50,
    )

    # 8) Trainer 생성
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    # 9) 모델 훈련
    trainer.train()

    # 10) 훈련된 모델 및 토크나이저 저장
    model.save_pretrained("kobert_risk_model")
    tokenizer.save_pretrained("kobert_risk_model")
    print("[INFO] KoBERT 리스크 분류 모델 학습 및 저장 완료.")

if __name__ == "__main__":
    main()
