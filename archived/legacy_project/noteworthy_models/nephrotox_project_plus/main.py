from __future__ import annotations

import argparse
import os

from configs.config import Config
from models.registry import get_model_class, get_model_family
from utils.data_utils import (
    MoleculeDataset,
    clean_dataframe,
    dataframe_to_graphs,
    dataframe_to_hybrid_graphs,
    set_rdkit_silent,
)
from utils.scaffold_split import scaffold_split
from utils.trainer import set_seed, train_model
from utils.trainer_text import train_text_model



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train nephrotoxicity predictor with scaffold split.")
    parser.add_argument("--train_csv", type=str, default="data/train.csv")
    parser.add_argument("--test_csv", type=str, default="data/test.csv")
    parser.add_argument("--model_name", type=str, default="gin", choices=["gin", "gin_hybrid", "chemberta"])
    parser.add_argument("--scaffold_type", type=str, default="murcko", choices=["murcko", "carbon"])
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--hidden_dim", type=int, default=96)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.35)
    parser.add_argument("--lr", type=float, default=8e-4)
    parser.add_argument("--weight_decay", type=float, default=2e-4)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--val_fraction", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prefer_cpu", action="store_true")
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--hf_model_name", type=str, default="seyonec/ChemBERTa-zinc-base-v1")
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--unfreeze_last_n_layers", type=int, default=2)
    parser.add_argument("--no_freeze_backbone", action="store_true")
    return parser.parse_args()



def main() -> None:
    args = parse_args()
    set_rdkit_silent()
    set_seed(args.seed)

    epochs = args.epochs
    if epochs is None:
        epochs = 45 if args.model_name == "chemberta" else 60

    config = Config(
        model_name=args.model_name,
        train_csv=args.train_csv,
        test_csv=args.test_csv,
        scaffold_type=args.scaffold_type,
        val_fraction=args.val_fraction,
        batch_size=args.batch_size,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        lr=args.lr,
        weight_decay=args.weight_decay,
        epochs=epochs,
        patience=args.patience,
        seed=args.seed,
        prefer_mps=not args.prefer_cpu,
        output_dir=args.output_dir,
        hf_model_name=args.hf_model_name,
        max_length=args.max_length,
        freeze_backbone=not args.no_freeze_backbone,
        unfreeze_last_n_layers=args.unfreeze_last_n_layers,
    )

    train_clean = clean_dataframe(config.train_csv, label_col=config.label_col)
    test_clean = clean_dataframe(config.test_csv, label_col=config.label_col)

    train_df, val_df = scaffold_split(
        train_clean.df,
        smiles_col=train_clean.smiles_col,
        label_col=train_clean.label_col,
        val_fraction=config.val_fraction,
        scaffold_type=config.scaffold_type,
    )

    print(f"Training molecules:   {len(train_df)}")
    print(f"Validation molecules: {len(val_df)}")
    print(f"External test:        {len(test_clean.df)}")

    model_class = get_model_class(config.model_name)
    model_family = get_model_family(config.model_name)
    os.makedirs(config.output_dir, exist_ok=True)

    if model_family == "text":
        model = model_class(
            hf_model_name=config.hf_model_name,
            dropout=min(config.dropout, 0.25),
            freeze_backbone=config.freeze_backbone,
            unfreeze_last_n_layers=config.unfreeze_last_n_layers,
        )
        train_text_model(
            model=model,
            train_df=train_df,
            val_df=val_df,
            test_df=test_clean.df,
            smiles_col=train_clean.smiles_col,
            label_col=train_clean.label_col,
            config=config,
            output_dir=config.output_dir,
            model_name=config.model_name,
        )
        return

    if config.model_name == "gin_hybrid":
        train_graphs, scaler = dataframe_to_hybrid_graphs(train_df, train_clean.smiles_col, train_clean.label_col, scaler=None, fit_scaler=True)
        val_graphs, _ = dataframe_to_hybrid_graphs(val_df, train_clean.smiles_col, train_clean.label_col, scaler=scaler, fit_scaler=False)
        test_graphs, _ = dataframe_to_hybrid_graphs(test_clean.df, test_clean.smiles_col, test_clean.label_col, scaler=scaler, fit_scaler=False)
        train_dataset = MoleculeDataset(train_graphs)
        val_dataset = MoleculeDataset(val_graphs)
        test_dataset = MoleculeDataset(test_graphs)
        input_dim = train_dataset[0].x.shape[1]
        model = model_class(
            input_dim=input_dim,
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
            dropout=config.dropout,
            descriptor_dim=config.descriptor_dim,
        )
        train_model(
            model=model,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            test_dataset=test_dataset,
            config=config,
            output_dir=config.output_dir,
            model_name=config.model_name,
            extra_artifacts={"descriptor_scaler": scaler.state_dict()},
        )
        return

    train_dataset = MoleculeDataset(dataframe_to_graphs(train_df, train_clean.smiles_col, train_clean.label_col))
    val_dataset = MoleculeDataset(dataframe_to_graphs(val_df, train_clean.smiles_col, train_clean.label_col))
    test_dataset = MoleculeDataset(dataframe_to_graphs(test_clean.df, test_clean.smiles_col, test_clean.label_col))
    input_dim = train_dataset[0].x.shape[1]
    model = model_class(
        input_dim=input_dim,
        hidden_dim=config.hidden_dim,
        num_layers=config.num_layers,
        dropout=config.dropout,
    )

    train_model(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        config=config,
        output_dir=config.output_dir,
        model_name=config.model_name,
    )


if __name__ == "__main__":
    main()
