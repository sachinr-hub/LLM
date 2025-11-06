import argparse
import os
from dataclasses import dataclass
from typing import Dict, List

import pandas as pd
from datasets import Dataset
from transformers import (
    T5Tokenizer,
    T5ForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
)
import torch


def build_input(example: Dict[str, str]) -> str:
    parts = []
    if example.get("sentiment"):
        parts.append(f"Detected Sentiment: {example['sentiment']}")
    if example.get("tone"):
        parts.append(f"Tone: {example['tone']}")
    if example.get("context"):
        parts.append(f"Context/Subject: {example['context']}")
    parts.append("\nInstruction: Generate a response email using the specified Tone and Subject. Address the sender by name.\n")
    parts.append(f"Original email: {example.get('original_email','')}\n\nResponse:")
    return "\n".join(parts)


def load_dataset(csv_path: str) -> Dataset:
    df = pd.read_csv(csv_path)
    # expected columns: original_email, sentiment, tone, context, target_response
    df = df.fillna("")
    df["source_text"] = df.apply(build_input, axis=1)
    df = df.rename(columns={"target_response": "target_text"})
    return Dataset.from_pandas(df[["source_text", "target_text"]])


@dataclass
class Seq2SeqDataModule:
    tokenizer: T5Tokenizer
    source_max_length: int = 512
    target_max_length: int = 256

    def preprocess(self, examples: Dict[str, List[str]]):
        model_inputs = self.tokenizer(
            examples["source_text"],
            max_length=self.source_max_length,
            truncation=True,
        )
        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(
                examples["target_text"],
                max_length=self.target_max_length,
                truncation=True,
            )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_csv", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--model_name", default="google/flan-t5-small")
    parser.add_argument("--eval_csv", default=None)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--fp16", action="store_true")
    args = parser.parse_args()

    tokenizer = T5Tokenizer.from_pretrained(args.model_name)
    model = T5ForConditionalGeneration.from_pretrained(args.model_name)

    train_ds = load_dataset(args.train_csv)
    eval_ds = load_dataset(args.eval_csv) if args.eval_csv else None

    dm = Seq2SeqDataModule(tokenizer)
    tokenized_train = train_ds.map(dm.preprocess, batched=True, remove_columns=train_ds.column_names)
    tokenized_eval = None
    if eval_ds is not None:
        tokenized_eval = eval_ds.map(dm.preprocess, batched=True, remove_columns=eval_ds.column_names)

    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        evaluation_strategy="steps" if tokenized_eval is not None else "no",
        eval_steps=200,
        save_steps=200,
        logging_steps=50,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        predict_with_generate=True,
        fp16=args.fp16 and torch.cuda.is_available(),
        save_total_limit=2,
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    trainer.train()

    os.makedirs(args.output_dir, exist_ok=True)
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    print(f"Model saved to {args.output_dir}")


if __name__ == "__main__":
    main()
