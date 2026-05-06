from __future__ import annotations

import argparse

import pandas as pd
import torch
from torch_geometric.loader import DataLoader
from transformers import AutoTokenizer

from models.registry import get_model_class, get_model_family
from utils.data_utils import MoleculeDataset, canonicalize_smiles, smiles_to_data
from utils.descriptors import DescriptorScaler, compute_descriptor_vector
from utils.trainer import get_device



def parse_args():
    parser = argparse.ArgumentParser(description="Run inference on SMILES using a saved checkpoint.")
    parser.add_argument("--model_path", type=str, default="outputs/best_model.pt")
    parser.add_argument("--smiles", nargs="+", required=True)
    parser.add_argument("--prefer_cpu", action="store_true")
    return parser.parse_args()



def predict_graph(ckpt, args, device):
    model_class = get_model_class(ckpt["model_name"])
    cfg = ckpt["config"]
    model = model_class(
        input_dim=ckpt["input_dim"],
        hidden_dim=cfg["hidden_dim"],
        num_layers=cfg["num_layers"],
        dropout=cfg["dropout"],
        descriptor_dim=cfg.get("descriptor_dim", 12),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    rows = []
    graph_items = []
    scaler_state = ckpt.get("descriptor_scaler")
    scaler = DescriptorScaler.from_state_dict(scaler_state) if scaler_state else None

    for smi in args.smiles:
        canon = canonicalize_smiles(smi)
        if canon is None:
            rows.append({"smiles": smi, "valid": False, "probability": None, "prediction": None})
            continue
        descriptors = None
        if scaler is not None:
            vec = compute_descriptor_vector(canon).reshape(1, -1)
            descriptors = scaler.transform(vec)[0].tolist()
        graph = smiles_to_data(canon, label=None, descriptors=descriptors)
        graph_items.append(graph)
        rows.append({"smiles": canon, "valid": True})

    if graph_items:
        ds = MoleculeDataset(graph_items)
        loader = DataLoader(ds, batch_size=64, shuffle=False)
        all_probs = []
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(device)
                descriptors = getattr(batch, "descriptors", None)
                logits = model(batch.x, batch.edge_index, batch.batch, descriptors=descriptors)
                probs = torch.sigmoid(logits).cpu().numpy().tolist()
                all_probs.extend(probs)

        idx = 0
        threshold = ckpt.get("threshold", 0.5)
        for row in rows:
            if row["valid"]:
                p = float(all_probs[idx])
                row["probability"] = p
                row["prediction"] = int(p >= threshold)
                idx += 1

    print(pd.DataFrame(rows).to_string(index=False))



def predict_text(ckpt, args, device):
    cfg = ckpt["config"]
    model_class = get_model_class(ckpt["model_name"])
    model = model_class(
        hf_model_name=ckpt.get("hf_model_name", cfg.get("hf_model_name", "seyonec/ChemBERTa-zinc-base-v1")),
        dropout=min(cfg.get("dropout", 0.25), 0.25),
        freeze_backbone=cfg.get("freeze_backbone", True),
        unfreeze_last_n_layers=cfg.get("unfreeze_last_n_layers", 2),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(ckpt.get("hf_model_name", cfg.get("hf_model_name", "seyonec/ChemBERTa-zinc-base-v1")))
    rows = []
    valid_smiles = []
    for smi in args.smiles:
        canon = canonicalize_smiles(smi)
        if canon is None:
            rows.append({"smiles": smi, "valid": False, "probability": None, "prediction": None})
        else:
            rows.append({"smiles": canon, "valid": True})
            valid_smiles.append(canon)

    threshold = ckpt.get("threshold", 0.5)
    if valid_smiles:
        all_probs = []
        with torch.no_grad():
            for start in range(0, len(valid_smiles), 32):
                batch_smiles = valid_smiles[start:start + 32]
                enc = tokenizer(batch_smiles, truncation=True, max_length=cfg.get("max_length", 128), padding=True, return_tensors="pt")
                logits = model(input_ids=enc["input_ids"].to(device), attention_mask=enc["attention_mask"].to(device))
                probs = torch.sigmoid(logits).cpu().numpy().tolist()
                all_probs.extend(probs)

        idx = 0
        for row in rows:
            if row["valid"]:
                p = float(all_probs[idx])
                row["probability"] = p
                row["prediction"] = int(p >= threshold)
                idx += 1

    print(pd.DataFrame(rows).to_string(index=False))



def main():
    args = parse_args()
    device = get_device(prefer_mps=not args.prefer_cpu)
    ckpt = torch.load(args.model_path, map_location=device)
    family = get_model_family(ckpt["model_name"])
    if family == "text":
        predict_text(ckpt, args, device)
    else:
        predict_graph(ckpt, args, device)


if __name__ == "__main__":
    main()
