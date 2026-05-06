#!/usr/bin/env python3
"""Build consolidated score reports from model score CSV files."""

from __future__ import annotations

from pathlib import Path
import shutil
from typing import Iterable

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "reports" / "model_scores"
SOURCE_COPY_DIR = REPORT_DIR / "source_csvs"

SCORE_NAME_KEYWORDS = ("metric", "metrics", "summary", "leaderboard", "score", "scores")
EXCLUDED_PATH_PARTS = {"data", ".venv", "__pycache__", "reports"}

PRIMARY_METRIC_PRIORITY = [
    "auroc",
    "auc",
    "roc_auc",
    "auprc",
    "f1",
    "accuracy",
    "acc",
    "mcc",
    "kappa",
    "specificity",
    "recall",
    "se",
]

NON_METRIC_COLUMNS = {
    "fold",
    "split",
    "metric",
    "model_name",
    "mean",
    "std",
    "source_csv",
    "source_name",
    "model_group",
    "row_index",
}


def find_score_csvs(root: Path) -> list[Path]:
    files: list[Path] = []
    for path in root.rglob("*.csv"):
        rel = path.relative_to(root)
        parts = set(rel.parts)
        if parts & EXCLUDED_PATH_PARTS:
            continue
        name = path.name.lower()
        if any(keyword in name for keyword in SCORE_NAME_KEYWORDS):
            files.append(path)
    return sorted(files)


def infer_model_group(rel_path: Path) -> str:
    parts = rel_path.parts
    if not parts:
        return "unknown"
    if parts[0] == "noteworthy_models" and len(parts) >= 2:
        return parts[1]
    if parts[0] == "confirmed_models":
        return "confirmed_models"
    if parts[0] == "experiments":
        return "experiments"
    if parts[0] == "models":
        return "models"
    return parts[0]


def choose_primary_metric(columns: Iterable[str]) -> str | None:
    lowered = {c.lower(): c for c in columns}
    for metric in PRIMARY_METRIC_PRIORITY:
        if metric in lowered:
            return lowered[metric]
    return None


def discover_expected_model_groups(root: Path) -> list[str]:
    groups: set[str] = set()
    if (root / "confirmed_models").exists():
        groups.add("confirmed_models")
    noteworthy_root = root / "noteworthy_models"
    if noteworthy_root.exists():
        for child in noteworthy_root.iterdir():
            if child.is_dir():
                groups.add(child.name)
    return sorted(groups)


def main() -> None:
    score_csvs = find_score_csvs(ROOT)
    expected_groups = discover_expected_model_groups(ROOT)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    SOURCE_COPY_DIR.mkdir(parents=True, exist_ok=True)

    if not score_csvs:
        empty = pd.DataFrame(columns=["source_csv", "model_group", "primary_metric", "best_primary_score"])
        empty.to_csv(REPORT_DIR / "model_score_summary.csv", index=False)
        (REPORT_DIR / "README.md").write_text(
            "# Model Score Summary\n\nNo score CSV files were found.\n",
            encoding="utf-8",
        )
        print("No score CSV files found.")
        return

    raw_rows: list[pd.DataFrame] = []
    long_rows: list[pd.DataFrame] = []
    summary_rows: list[dict[str, object]] = []
    source_index_rows: list[dict[str, object]] = []

    for source_path in score_csvs:
        rel = source_path.relative_to(ROOT)
        model_group = infer_model_group(rel)
        source_name = rel.as_posix().replace("/", "__")

        destination_dir = SOURCE_COPY_DIR / model_group
        destination_dir.mkdir(parents=True, exist_ok=True)
        destination_path = destination_dir / source_name
        shutil.copy2(source_path, destination_path)

        df = pd.read_csv(source_path)
        df = df.copy()
        df["source_csv"] = rel.as_posix()
        df["source_name"] = source_name
        df["model_group"] = model_group
        df["row_index"] = range(len(df))
        raw_rows.append(df)

        numeric_columns = list(df.select_dtypes(include="number").columns)
        primary_metric = choose_primary_metric(numeric_columns)
        best_primary_score = float("nan")
        if primary_metric:
            best_primary_score = pd.to_numeric(df[primary_metric], errors="coerce").max()

        summary_rows.append(
            {
                "source_csv": rel.as_posix(),
                "model_group": model_group,
                "primary_metric": primary_metric or "",
                "best_primary_score": best_primary_score,
                "num_rows": len(df),
            }
        )

        source_index_rows.append(
            {
                "source_csv": rel.as_posix(),
                "model_group": model_group,
                "copied_to": destination_path.relative_to(ROOT).as_posix(),
            }
        )

        metric_columns = [c for c in numeric_columns if c.lower() not in NON_METRIC_COLUMNS]
        for metric_column in metric_columns:
            metric_df = df[["source_csv", "source_name", "model_group", "row_index"]].copy()
            metric_df["metric"] = metric_column
            metric_df["score"] = pd.to_numeric(df[metric_column], errors="coerce")
            long_rows.append(metric_df)

    raw_combined = pd.concat(raw_rows, ignore_index=True)
    long_combined = pd.concat(long_rows, ignore_index=True) if long_rows else pd.DataFrame(
        columns=["source_csv", "source_name", "model_group", "row_index", "metric", "score"]
    )
    source_summary = pd.DataFrame(summary_rows).sort_values(
        by=["model_group", "best_primary_score"], ascending=[True, False]
    )
    source_index = pd.DataFrame(source_index_rows).sort_values(by=["model_group", "source_csv"])

    model_summary = (
        source_summary.groupby("model_group", as_index=False)
        .agg(
            best_primary_score=("best_primary_score", "max"),
            avg_primary_score=("best_primary_score", "mean"),
            score_csv_count=("source_csv", "count"),
        )
    )
    existing_groups = set(model_summary["model_group"].tolist())
    missing_groups = [g for g in expected_groups if g not in existing_groups]
    if missing_groups:
        missing_df = pd.DataFrame(
            {
                "model_group": missing_groups,
                "best_primary_score": [float("nan")] * len(missing_groups),
                "avg_primary_score": [float("nan")] * len(missing_groups),
                "score_csv_count": [0] * len(missing_groups),
            }
        )
        model_summary = pd.concat([model_summary, missing_df], ignore_index=True)
    model_summary = model_summary.sort_values(by=["score_csv_count", "best_primary_score"], ascending=[False, False])

    raw_combined.to_csv(REPORT_DIR / "all_score_csv_rows.csv", index=False)
    long_combined.to_csv(REPORT_DIR / "all_scores_long.csv", index=False)
    source_summary.to_csv(REPORT_DIR / "score_csv_summary.csv", index=False)
    source_index.to_csv(REPORT_DIR / "score_csv_index.csv", index=False)
    model_summary.to_csv(REPORT_DIR / "model_score_summary.csv", index=False)

    top_lines = []
    for row in model_summary.head(10).itertuples(index=False):
        best_str = "nan" if pd.isna(row.best_primary_score) else f"{row.best_primary_score:.6f}"
        avg_str = "nan" if pd.isna(row.avg_primary_score) else f"{row.avg_primary_score:.6f}"
        top_lines.append(
            f"- {row.model_group}: best={best_str}, avg={avg_str}, score_csv_count={int(row.score_csv_count)}"
        )

    readme = "\n".join(
        [
            "# Model Score Summary",
            "",
            "This folder keeps model score outputs together and separate from utility code.",
            "",
            "## Generated Files",
            "- `all_score_csv_rows.csv`: Union of all rows from discovered score CSV files.",
            "- `all_scores_long.csv`: Long-format metric rows (`metric`, `score`) for analysis.",
            "- `score_csv_summary.csv`: One row per score CSV with inferred primary metric and best score.",
            "- `score_csv_index.csv`: Mapping of source score CSVs to copied files in `source_csvs/`.",
            "- `model_score_summary.csv`: Aggregated score summary by model group.",
            "",
            "## Model Score Snapshot",
            *(top_lines or ["- No numeric scores were available."]),
            "",
        ]
    )
    (REPORT_DIR / "README.md").write_text(readme, encoding="utf-8")

    print(f"Processed {len(score_csvs)} score CSV file(s).")
    print(f"Wrote reports to: {REPORT_DIR.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
