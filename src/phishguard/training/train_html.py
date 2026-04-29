"""Fine-tune DistilBERT on raw HTML text.

Strategy:
    1. Strip <script>, <style>, comments. Keep visible text + a flattened token of tag names.
    2. Tokenize with DistilBERT tokenizer at max_length=512. Truncation tail-biased
       because phishing pages often hide credential forms below the fold.
    3. Standard HuggingFace Trainer; fp16 if CUDA available.
    4. Export final model to ONNX for serving parity.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from bs4 import BeautifulSoup
from datasets import Dataset
from sklearn.metrics import (
    average_precision_score, brier_score_loss, f1_score, roc_auc_score,
)
from transformers import (
    AutoModelForSequenceClassification, AutoTokenizer,
    DataCollatorWithPadding, Trainer, TrainingArguments,
)


def clean_html(html: str) -> str:
    if not isinstance(html, str) or not html:
        return ""
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript", "svg"]):
        tag.decompose()
    text = soup.get_text(separator=" ", strip=True)
    tag_seq = " ".join(t.name for t in soup.find_all() if t.name)
    return f"{tag_seq[:1024]} [SEP] {text[:8000]}"


def metrics_fn(eval_pred):
    logits, labels = eval_pred
    probs = torch.softmax(torch.tensor(logits), dim=-1)[:, 1].numpy()
    preds = (probs >= 0.5).astype(int)
    return {
        "auc": roc_auc_score(labels, probs),
        "ap": average_precision_score(labels, probs),
        "f1": f1_score(labels, preds),
        "brier": brier_score_loss(labels, probs),
    }


def train(cfg: dict) -> None:
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])

    text_col, label_col = cfg["data"]["text_col"], cfg["data"]["label_col"]

    def load_split(p: str) -> Dataset:
        df = pd.read_parquet(p)
        df = df.rename(columns={label_col: "label"})
        df["text"] = df[text_col].apply(clean_html)
        return Dataset.from_pandas(df[["text", "label"]], preserve_index=False)

    train_ds = load_split(cfg["data"]["train_path"])
    val_ds = load_split(cfg["data"]["val_path"])
    test_ds = load_split(cfg["data"]["test_path"])

    tok = AutoTokenizer.from_pretrained(cfg["model"]["base"])

    def tokenize(batch):
        return tok(batch["text"], truncation=True, max_length=cfg["model"]["max_length"])

    train_ds = train_ds.map(tokenize, batched=True, remove_columns=["text"])
    val_ds = val_ds.map(tokenize, batched=True, remove_columns=["text"])
    test_ds = test_ds.map(tokenize, batched=True, remove_columns=["text"])

    model = AutoModelForSequenceClassification.from_pretrained(
        cfg["model"]["base"], num_labels=cfg["model"]["num_labels"]
    )

    out_dir = cfg["artifacts"]["output_dir"]
    args = TrainingArguments(
        output_dir=out_dir,
        num_train_epochs=cfg["training"]["epochs"],
        per_device_train_batch_size=cfg["training"]["batch_size"],
        per_device_eval_batch_size=cfg["training"]["batch_size"],
        gradient_accumulation_steps=cfg["training"]["grad_accum_steps"],
        learning_rate=cfg["training"]["learning_rate"],
        weight_decay=cfg["training"]["weight_decay"],
        warmup_ratio=cfg["training"]["warmup_ratio"],
        fp16=cfg["training"]["fp16"] and torch.cuda.is_available(),
        eval_strategy=cfg["training"]["eval_strategy"],
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        report_to=["wandb"],
        logging_steps=50,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tok,
        data_collator=DataCollatorWithPadding(tok),
        compute_metrics=metrics_fn,
    )

    trainer.train()
    test_metrics = trainer.evaluate(test_ds)
    print("test:", test_metrics)

    trainer.save_model(out_dir)
    tok.save_pretrained(out_dir)

    # ONNX export
    onnx_path = Path(cfg["artifacts"]["onnx_path"])
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    dummy = tok("hello world", return_tensors="pt", truncation=True, max_length=cfg["model"]["max_length"])
    model.eval().cpu()
    torch.onnx.export(
        model,
        (dummy["input_ids"], dummy["attention_mask"]),
        str(onnx_path),
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "seq"},
            "attention_mask": {0: "batch", 1: "seq"},
            "logits": {0: "batch"},
        },
        opset_version=17,
    )
    print(f"ONNX saved -> {onnx_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    train(cfg)


if __name__ == "__main__":
    main()
