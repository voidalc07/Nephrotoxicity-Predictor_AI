from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys



def parse_args():
    parser = argparse.ArgumentParser(description="Train multiple models and build a leaderboard.")
    parser.add_argument("--train_csv", type=str, default="data/train.csv")
    parser.add_argument("--test_csv", type=str, default="data/test.csv")
    parser.add_argument("--models", nargs="+", default=["gin", "gin_hybrid", "chemberta"])
    parser.add_argument("--base_output_dir", type=str, default="outputs_compare")
    parser.add_argument("--scaffold_type", type=str, default="murcko", choices=["murcko", "carbon"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prefer_cpu", action="store_true")
    return parser.parse_args()



def main():
    args = parse_args()
    os.makedirs(args.base_output_dir, exist_ok=True)
    leaderboard_rows = []

    for model_name in args.models:
        out_dir = os.path.join(args.base_output_dir, model_name)
        cmd = [
            sys.executable,
            "main.py",
            "--train_csv", args.train_csv,
            "--test_csv", args.test_csv,
            "--model_name", model_name,
            "--output_dir", out_dir,
            "--scaffold_type", args.scaffold_type,
            "--seed", str(args.seed),
        ]
        if args.prefer_cpu:
            cmd.append("--prefer_cpu")
        print("Running:", " ".join(cmd))
        subprocess.run(cmd, check=True)

        metrics_path = os.path.join(out_dir, "metrics.json")
        with open(metrics_path, "r", encoding="utf-8") as f:
            metrics = json.load(f)

        test_metrics = metrics["test"]
        leaderboard_rows.append(
            {
                "model_name": model_name,
                "accuracy": test_metrics["accuracy"],
                "recall": test_metrics["recall"],
                "f1": test_metrics["f1"],
                "kappa": test_metrics["kappa"],
                "auroc": test_metrics["auroc"],
                "specificity": test_metrics["specificity"],
                "mcc": test_metrics["mcc"],
            }
        )

    leaderboard_rows = sorted(leaderboard_rows, key=lambda x: (x["auroc"], x["f1"], x["mcc"]), reverse=True)
    leaderboard_csv = os.path.join(args.base_output_dir, "leaderboard.csv")
    with open(leaderboard_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(leaderboard_rows[0].keys()))
        writer.writeheader()
        writer.writerows(leaderboard_rows)

    print(f"Saved leaderboard to: {leaderboard_csv}")


if __name__ == "__main__":
    main()
