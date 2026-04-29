"""Train EfficientNet-B0 on phishing screenshots.

Screenshots are organized as:
    train/{phish, benign}/*.png
    val/{phish, benign}/*.png
    test/{phish, benign}/*.png

Augmentation tuned for phishing pages: NO horizontal flip (logos and forms have
meaningful left-right structure), aggressive random erasing to force the model
to attend to multiple regions instead of latching onto a single brand mark.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from tqdm import tqdm

from phishguard.models.screenshot_model import ScreenshotClassifier


def make_loaders(cfg: dict[str, Any]) -> tuple[DataLoader, DataLoader, DataLoader]:
    size = cfg["data"]["image_size"]
    aug = cfg["training"]["augmentation"]
    norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    train_tf = transforms.Compose(
        [
            transforms.RandomResizedCrop(size, scale=(0.85, 1.0))
            if aug["random_resized_crop"]
            else transforms.Resize((size, size)),
            transforms.ColorJitter(
                brightness=aug["color_jitter"],
                contrast=aug["color_jitter"],
                saturation=aug["color_jitter"],
            ),
            transforms.ToTensor(),
            norm,
            transforms.RandomErasing(p=aug["random_erasing"]),
        ]
    )
    eval_tf = transforms.Compose(
        [
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            norm,
        ]
    )

    train_ds = ImageFolder(cfg["data"]["train_dir"], transform=train_tf)
    val_ds = ImageFolder(cfg["data"]["val_dir"], transform=eval_tf)
    test_ds = ImageFolder(cfg["data"]["test_dir"], transform=eval_tf)

    bs = cfg["training"]["batch_size"]
    return (
        DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=4, pin_memory=True),
        DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=4, pin_memory=True),
        DataLoader(test_ds, batch_size=bs, shuffle=False, num_workers=4, pin_memory=True),
    )


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    raw_probs: list[float] = []
    raw_labels: list[int] = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            p = F.softmax(model(x), dim=-1)[:, 1].cpu().numpy()
            raw_probs.extend(p.tolist())
            raw_labels.extend(y.numpy().tolist())
    probs = np.array(raw_probs)
    labels = np.array(raw_labels)
    preds = (probs >= 0.5).astype(int)
    return {
        "auc": float(roc_auc_score(labels, probs)),
        "ap": float(average_precision_score(labels, probs)),
        "f1": float(f1_score(labels, preds)),
        "brier": float(brier_score_loss(labels, probs)),
    }


def cosine_lr(step: int, total: int, base_lr: float, warmup: int) -> float:
    if step < warmup:
        return base_lr * (step + 1) / max(warmup, 1)
    progress = (step - warmup) / max(total - warmup, 1)
    return 0.5 * base_lr * (1.0 + math.cos(math.pi * progress))


def train(cfg: dict[str, Any]) -> None:
    torch.manual_seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader, test_loader = make_loaders(cfg)
    model = ScreenshotClassifier(
        num_classes=cfg["model"]["num_classes"],
        dropout=cfg["model"]["dropout"],
        pretrained=cfg["model"]["pretrained"],
    ).to(device)

    base_lr = cfg["training"]["learning_rate"]
    optim = torch.optim.AdamW(
        model.parameters(), lr=base_lr, weight_decay=cfg["training"]["weight_decay"]
    )
    scaler = torch.cuda.amp.GradScaler(enabled=cfg["training"]["fp16"] and device.type == "cuda")

    total_steps = cfg["training"]["epochs"] * len(train_loader)
    warmup = cfg["training"]["warmup_epochs"] * len(train_loader)
    step = 0
    best_f1 = 0.0
    ckpt_path = Path(cfg["artifacts"]["ckpt_path"])
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(cfg["training"]["epochs"]):
        model.train()
        pbar = tqdm(train_loader, desc=f"epoch {epoch}")
        for x, y in pbar:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            for g in optim.param_groups:
                g["lr"] = cosine_lr(step, total_steps, base_lr, warmup)
            optim.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
                logits = model(x)
                loss = F.cross_entropy(logits, y)
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
            step += 1
            pbar.set_postfix(loss=f"{loss.item():.3f}")

        val_metrics = evaluate(model, val_loader, device)
        print(f"epoch {epoch} val: {val_metrics}")
        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]
            torch.save({"state_dict": model.state_dict(), "cfg": cfg}, ckpt_path)
            print(f"saved best -> {ckpt_path}")

    # Reload best and evaluate on test
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state["state_dict"])
    print("test:", evaluate(model, test_loader, device))

    # ONNX export
    model.eval().cpu()
    dummy = torch.randn(1, 3, cfg["data"]["image_size"], cfg["data"]["image_size"])
    torch.onnx.export(
        model,
        dummy,
        cfg["artifacts"]["onnx_path"],
        input_names=["pixel_values"],
        output_names=["logits"],
        dynamic_axes={"pixel_values": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
    )
    print(f"ONNX saved -> {cfg['artifacts']['onnx_path']}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    train(cfg)


if __name__ == "__main__":
    main()
