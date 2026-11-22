"""Fine-tune DistilBERT on raw HTML text.

Strategy:
    1. Strip <script>, <style>, comments. Keep visible text + a flattened token of tag names.
    2. Tokenize with DistilBERT tokenizer at max_length=512. Truncation is the tokenizer
       default: it keeps the HEAD of the sequence (tag-name sequence first) and drops the tail.
    3. Standard HuggingFace Trainer; fp16 if CUDA available. The device is chosen by Trainer
       (CUDA, else MPS, else CPU).
    4. Export final model to ONNX for serving parity (skipped when artifacts.onnx_path is null).
    5. Logging goes to wandb only if logging.report_to is "wandb". Default is off. A JSONL
       file log and a resident-memory guard (training.max_rss_gb) are available for the Mac.
"""

from __future__ import annotations

import argparse
import inspect
import json
import resource
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml
from bs4 import BeautifulSoup
from datasets import Dataset
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    roc_auc_score,
)
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)


class MemoryLimitExceededError(RuntimeError):
    """Peak resident memory of this process passed the configured limit."""


def peak_rss_gb() -> float:
    """Peak resident set size of this process in GB (ru_maxrss is bytes on macOS, KB on Linux)."""
    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return raw / 1024**3 if sys.platform == "darwin" else raw / 1024**2


def check_rss(limit_gb: float) -> float:
    rss = peak_rss_gb()
    if rss > limit_gb:
        raise MemoryLimitExceededError(f"peak RSS {rss:.2f} GB passed limit {limit_gb} GB")
    return rss


def warmup_kwargs(ratio: float) -> dict[str, float]:
    """transformers 5 dropped warmup_ratio (warmup_steps takes a float ratio); 4.x still has it."""
    if "warmup_ratio" in inspect.signature(TrainingArguments.__init__).parameters:
        return {"warmup_ratio": ratio}
    return {"warmup_steps": ratio}


def build_report_to(cfg: dict[str, Any]) -> list[str]:
    """wandb is opt-in: only when logging.report_to is explicitly "wandb"."""
    value = (cfg.get("logging") or {}).get("report_to", "none")
    return ["wandb"] if value == "wandb" else []


class LocalLogCallback(TrainerCallback):  # type: ignore[misc]
    """Append Trainer logs to a local JSONL file and enforce the memory limit each step."""

    def __init__(self, log_file: Path | None, max_rss_gb: float | None) -> None:
        self.log_file = log_file
        self.max_rss_gb = max_rss_gb
        if log_file:
            log_file.parent.mkdir(parents=True, exist_ok=True)

    def on_log(self, args: Any, state: Any, control: Any, logs: Any = None, **kw: Any) -> None:
        if self.log_file and logs:
            row = {"step": state.global_step, "peak_rss_gb": round(peak_rss_gb(), 3), **logs}
            with self.log_file.open("a") as fh:
                fh.write(json.dumps(row) + "\n")

    def on_step_end(self, args: Any, state: Any, control: Any, **kw: Any) -> None:
        if self.max_rss_gb:
            check_rss(self.max_rss_gb)


def clean_html(html: str) -> str:
    if not isinstance(html, str) or not html:
        return ""
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript", "svg"]):
        tag.decompose()
    text = soup.get_text(separator=" ", strip=True)
    tag_seq = " ".join(t.name for t in soup.find_all() if t.name)
    return f"{tag_seq[:1024]} [SEP] {text[:8000]}"


def metrics_fn(eval_pred: Any) -> dict[str, float]:
    logits, labels = eval_pred
    probs = torch.softmax(torch.tensor(logits), dim=-1)[:, 1].numpy()
    preds = (probs >= 0.5).astype(int)
    return {
        "auc": roc_auc_score(labels, probs),
        "ap": average_precision_score(labels, probs),
        "f1": f1_score(labels, preds),
        "brier": brier_score_loss(labels, probs),
    }


def train(cfg: dict[str, Any]) -> None:
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

    def tokenize(batch: dict[str, Any]) -> dict[str, Any]:
        return tok(batch["text"], truncation=True, max_length=cfg["model"]["max_length"])  # type: ignore[no-any-return]

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
        **warmup_kwargs(cfg["training"]["warmup_ratio"]),
        fp16=cfg["training"]["fp16"] and torch.cuda.is_available(),
        eval_strategy=cfg["training"]["eval_strategy"],
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        report_to=build_report_to(cfg),
        logging_steps=cfg["training"].get("logging_steps", 50),
    )
    log_file = (cfg.get("logging") or {}).get("log_file")
    max_rss = cfg["training"].get("max_rss_gb")

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tok,
        data_collator=DataCollatorWithPadding(tok),
        compute_metrics=metrics_fn,
        callbacks=[LocalLogCallback(Path(log_file) if log_file else None, max_rss)],
    )

    trainer.train()
    test_metrics = trainer.evaluate(test_ds)
    print("test:", test_metrics)

    trainer.save_model(out_dir)
    tok.save_pretrained(out_dir)

    # ONNX export
    if not cfg["artifacts"].get("onnx_path"):
        print(f"peak RSS {peak_rss_gb():.2f} GB (smoke run, not a result)")
        return
    onnx_path = Path(cfg["artifacts"]["onnx_path"])
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    dummy = tok(
        "hello world", return_tensors="pt", truncation=True, max_length=cfg["model"]["max_length"]
    )
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
