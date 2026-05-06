from __future__ import annotations

import json
import math
import os
import random
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd
import torch
from torch_geometric.loader import DataLoader

from .metrics import compute_metrics, find_best_threshold



def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)



def get_device(prefer_mps: bool = True) -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if prefer_mps and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@dataclass
class EpochResult:
    loss: float
    probs: np.ndarray
    labels: np.ndarray


class EarlyStopping:
    def __init__(self, patience: int = 20, mode: str = "max"):
        self.patience = patience
        self.mode = mode
        self.best_score = None
        self.counter = 0
        self.best_state = None
        self.should_stop = False

    def step(self, score: float, model: torch.nn.Module) -> bool:
        if self.best_score is None:
            improved = True
        elif self.mode == "max":
            improved = score > self.best_score
        else:
            improved = score < self.best_score

        if improved:
            self.best_score = score
            self.counter = 0
            self.best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return improved



def run_epoch(model, loader, device, optimizer=None, pos_weight: Optional[torch.Tensor] = None) -> EpochResult:
    train_mode = optimizer is not None
    model.train() if train_mode else model.eval()
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    total_loss = 0.0
    total_count = 0
    all_probs, all_labels = [], []

    for batch in loader:
        batch = batch.to(device)
        with torch.set_grad_enabled(train_mode):
            descriptors = getattr(batch, "descriptors", None)
            logits = model(batch.x, batch.edge_index, batch.batch, descriptors=descriptors)
            labels = batch.y.view(-1).float()
            loss = criterion(logits, labels)
            if train_mode:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
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



def save_json(payload: Dict, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)



def train_model(model, train_dataset, val_dataset, test_dataset, config, output_dir: str, model_name: str = "gin", extra_artifacts: dict | None = None):
    os.makedirs(output_dir, exist_ok=True)
    device = get_device(prefer_mps=config.prefer_mps)
    model = model.to(device)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)

    train_labels = np.array([int(train_dataset[i].y.item()) for i in range(len(train_dataset))])
    n_pos = int((train_labels == 1).sum())
    n_neg = int((train_labels == 0).sum())
    pos_weight_val = n_neg / max(n_pos, 1)
    pos_weight = torch.tensor([pos_weight_val], dtype=torch.float, device=device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)
    stopper = EarlyStopping(patience=config.patience, mode="max")

    history = []
    best_val_threshold = 0.5

    for epoch in range(1, config.epochs + 1):
        train_result = run_epoch(model, train_loader, device, optimizer=optimizer, pos_weight=pos_weight)
        val_result = run_epoch(model, val_loader, device, optimizer=None, pos_weight=pos_weight)

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
    val_result = run_epoch(model, val_loader, device, optimizer=None, pos_weight=pos_weight)
    test_result = run_epoch(model, test_loader, device, optimizer=None, pos_weight=pos_weight)

    val_metrics = compute_metrics(val_result.labels, val_result.probs, threshold=best_val_threshold)
    test_metrics = compute_metrics(test_result.labels, test_result.probs, threshold=best_val_threshold)

    ckpt = {
        "model_name": model_name,
        "model_state_dict": model.state_dict(),
        "input_dim": train_dataset[0].x.shape[1],
        "config": config.to_dict(),
        "threshold": best_val_threshold,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
    }
    if extra_artifacts:
        ckpt.update(extra_artifacts)

    ckpt_path = os.path.join(output_dir, "best_model.pt")
    torch.save(ckpt, ckpt_path)

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
            "n_train": len(train_dataset),
            "n_val": len(val_dataset),
            "n_test": len(test_dataset),
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
