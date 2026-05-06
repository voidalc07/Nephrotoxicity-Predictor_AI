from __future__ import annotations

import json
import math
import os
import random
from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd
import torch
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

from .metrics import compute_metrics, find_best_threshold
from .trainer import EarlyStopping, get_device


class SmilesTextDataset(Dataset):
    def __init__(self, smiles_list, labels, tokenizer, max_length: int):
        self.smiles_list = list(smiles_list)
        self.labels = list(labels)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.smiles_list)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.smiles_list[idx],
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["labels"] = torch.tensor(float(self.labels[idx]), dtype=torch.float)
        item["smiles"] = self.smiles_list[idx]
        return item


@dataclass
class EpochResult:
    loss: float
    probs: np.ndarray
    labels: np.ndarray



def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)



def save_json(payload: Dict, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)



def run_epoch(model, loader, device, optimizer=None, pos_weight=None, grad_clip: float = 1.0) -> EpochResult:
    train_mode = optimizer is not None
    model.train() if train_mode else model.eval()
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    total_loss = 0.0
    total_count = 0
    all_probs, all_labels = [], []

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        with torch.set_grad_enabled(train_mode):
            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(logits, labels)
            if train_mode:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

        probs = torch.sigmoid(logits).detach().cpu().numpy()
        labels_np = labels.detach().cpu().numpy()
        bs = labels.size(0)
        total_loss += loss.item() * bs
        total_count += bs
        all_probs.append(probs)
        all_labels.append(labels_np)

    probs = np.concatenate(all_probs)
    labels = np.concatenate(all_labels)
    return EpochResult(loss=total_loss / max(total_count, 1), probs=probs, labels=labels)



def train_text_model(model, train_df, val_df, test_df, smiles_col: str, label_col: str, config, output_dir: str, model_name: str = "chemberta"):
    os.makedirs(output_dir, exist_ok=True)
    device = get_device(prefer_mps=config.prefer_mps)
    model = model.to(device)

    tokenizer = AutoTokenizer.from_pretrained(config.hf_model_name)
    train_ds = SmilesTextDataset(train_df[smiles_col].tolist(), train_df[label_col].tolist(), tokenizer, config.max_length)
    val_ds = SmilesTextDataset(val_df[smiles_col].tolist(), val_df[label_col].tolist(), tokenizer, config.max_length)
    test_ds = SmilesTextDataset(test_df[smiles_col].tolist(), test_df[label_col].tolist(), tokenizer, config.max_length)

    train_loader = DataLoader(train_ds, batch_size=min(config.batch_size, 32), shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=min(config.batch_size, 32), shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=min(config.batch_size, 32), shuffle=False)

    y_train = np.array(train_df[label_col].tolist())
    class_weights = compute_class_weight(class_weight="balanced", classes=np.array([0, 1]), y=y_train)
    pos_weight = torch.tensor([class_weights[1] / class_weights[0]], dtype=torch.float, device=device)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=max(config.lr * 0.75, 2e-5), weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)
    stopper = EarlyStopping(patience=config.patience, mode="max")

    history = []
    best_val_threshold = 0.5

    for epoch in range(1, config.epochs + 1):
        train_result = run_epoch(model, train_loader, device, optimizer=optimizer, pos_weight=pos_weight, grad_clip=config.grad_clip)
        val_result = run_epoch(model, val_loader, device, optimizer=None, pos_weight=pos_weight, grad_clip=config.grad_clip)

        val_threshold, _ = find_best_threshold(val_result.labels, val_result.probs)
        val_metrics = compute_metrics(val_result.labels, val_result.probs, threshold=val_threshold)
        train_metrics = compute_metrics(train_result.labels, train_result.probs, threshold=val_threshold)

        selection_score = (
            0.40 * val_metrics["f1"]
            + 0.20 * val_metrics["specificity"]
            + 0.20 * val_metrics["kappa"]
            + 0.20 * (0.0 if math.isnan(val_metrics["auroc"]) else val_metrics["auroc"])
        )
        scheduler.step(selection_score)
        improved = stopper.step(selection_score, model)
        if improved:
            best_val_threshold = val_threshold

        row = {
            "epoch": epoch,
            "lr": optimizer.param_groups[0]["lr"],
            "train_loss": train_result.loss,
            "val_loss": val_result.loss,
            "train_accuracy": train_metrics["accuracy"],
            "train_recall": train_metrics["recall"],
            "train_f1": train_metrics["f1"],
            "train_kappa": train_metrics["kappa"],
            "train_auroc": train_metrics["auroc"],
            "train_specificity": train_metrics["specificity"],
            "val_accuracy": val_metrics["accuracy"],
            "val_recall": val_metrics["recall"],
            "val_f1": val_metrics["f1"],
            "val_kappa": val_metrics["kappa"],
            "val_auroc": val_metrics["auroc"],
            "val_specificity": val_metrics["specificity"],
            "threshold": val_threshold,
        }
        history.append(row)

        print(
            f"Epoch {epoch:03d} | train_loss={train_result.loss:.4f} | val_loss={val_result.loss:.4f} | "
            f"val_f1={val_metrics['f1']:.4f} | val_spec={val_metrics['specificity']:.4f} | "
            f"val_auroc={val_metrics['auroc']:.4f} | thr={val_threshold:.2f}"
        )

        if stopper.should_stop:
            print("Early stopping triggered")
            break

    if stopper.best_state is None:
        raise RuntimeError("Training ended without a best checkpoint.")

    model.load_state_dict(stopper.best_state)
    val_result = run_epoch(model, val_loader, device, optimizer=None, pos_weight=pos_weight, grad_clip=config.grad_clip)
    test_result = run_epoch(model, test_loader, device, optimizer=None, pos_weight=pos_weight, grad_clip=config.grad_clip)

    val_metrics = compute_metrics(val_result.labels, val_result.probs, threshold=best_val_threshold)
    test_metrics = compute_metrics(test_result.labels, test_result.probs, threshold=best_val_threshold)

    ckpt_path = os.path.join(output_dir, "best_model.pt")
    torch.save(
        {
            "model_name": model_name,
            "model_state_dict": model.state_dict(),
            "config": config.to_dict(),
            "threshold": best_val_threshold,
            "val_metrics": val_metrics,
            "test_metrics": test_metrics,
            "hf_model_name": config.hf_model_name,
        },
        ckpt_path,
    )

    history_path = os.path.join(output_dir, "history.json")
    metrics_path = os.path.join(output_dir, "metrics.json")
    predictions_path = os.path.join(output_dir, "predictions.csv")

    pred_df = pd.DataFrame(
        {
            "y_true": test_result.labels.astype(int),
            "y_prob": test_result.probs,
            "y_pred": (test_result.probs >= best_val_threshold).astype(int),
        }
    )
    pred_df.to_csv(predictions_path, index=False)

    save_json({"history": history}, history_path)
    save_json(
        {
            "model_name": model_name,
            "device": str(device),
            "threshold": best_val_threshold,
            "val": val_metrics,
            "test": test_metrics,
            "n_train": len(train_ds),
            "n_val": len(val_ds),
            "n_test": len(test_ds),
            "hf_model_name": config.hf_model_name,
        },
        metrics_path,
    )

    print("\nValidation metrics:")
    for key, val in val_metrics.items():
        print(f"  {key}: {val:.4f}")

    print("\nExternal test metrics:")
    for key, val in test_metrics.items():
        print(f"  {key}: {val:.4f}")

    print(f"\nSaved model to: {ckpt_path}")
    print(f"Saved metrics to: {metrics_path}")
    print(f"Saved predictions to: {predictions_path}")
