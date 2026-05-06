"""
compare_all.py
==============
Loads results from every individual runner and prints a single
side-by-side comparison table.

Run this AFTER all individual runners have completed:
    python run_lightgbm.py
    python run_histgb.py
    python run_tanimoto_gpc.py
    python run_gin_virtual.py
    python run_node.py
    python run_ensemble.py
    python compare_all.py          ← this file
"""
import json, logging
from pathlib import Path
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import config
from utils import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

# Paper benchmark values
PAPER_INTERNAL = config.PAPER_INTERNAL
PAPER_EXTERNAL = config.PAPER_EXTERNAL

# Display order
ALL_MODELS = [
    "LightGBM",
    "HistGradientBoosting",
    "TanimotoGPC",
    "GIN_VirtualNode",
    "NODE",
    "StackingEnsemble",
]

METRIC_COLS = ["auc", "acc", "recall", "specificity", "f1", "kappa", "mcc"]
METRIC_LABELS = {
    "auc":         "AUC",
    "acc":         "Accuracy",
    "recall":      "Recall",
    "specificity": "Specificity",
    "f1":          "F1",
    "kappa":       "Kappa",
    "mcc":         "MCC",
}


def _load(model_name: str) -> dict | None:
    path = config.RESULTS_DIR / model_name / "metrics.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def _get(data: dict, split: str, metric: str) -> float:
    """Retrieve mean value from nested results dict."""
    try:
        v = data[split][metric]
        return v["mean"] if isinstance(v, dict) else float(v)
    except (KeyError, TypeError):
        return float("nan")


def _beat_symbol(ours: float, paper: float) -> str:
    if np.isnan(ours):   return "  —  "
    d = ours - paper
    if d > 0.005:        return " ✓ +{:.3f}".format(d)
    elif d >= -0.005:    return " ≈ {:.3f}".format(d)
    else:                return " ✗ {:.3f}".format(d)


def print_table(results: dict) -> None:
    """Print a formatted comparison table to the console."""
    # ── Header ────────────────────────────────────────────────────────────────
    col_w = 10
    name_w = 22
    sep  = "─" * (name_w + len(METRIC_COLS) * col_w + 4)

    for split_label, split_key, paper_ref in [
        ("CROSS-VALIDATION (scaffold-stratified)", "cv", PAPER_INTERNAL),
        ("EXTERNAL TEST SET (304 novel compounds)", "ext", PAPER_EXTERNAL),
    ]:
        print()
        print("═" * len(sep))
        print(f"  {split_label}")
        print("═" * len(sep))

        # Column headers
        header = f"  {'Model':<{name_w}}"
        for m in METRIC_COLS:
            header += f"{METRIC_LABELS[m]:>{col_w}}"
        print(header)
        print("  " + sep)

        # Paper row
        row = f"  {'Paper D-MPNN+ChemoPy2d':<{name_w}}"
        for m in METRIC_COLS:
            pv = paper_ref.get(m, float("nan"))
            cell = f"{pv:.3f}" if not np.isnan(pv) else "  —  "
            row += f"{cell:>{col_w}}"
        print(row)
        print("  " + sep)

        # Each model row
        for mn in ALL_MODELS:
            data = results.get(mn)
            if data is None:
                row = f"  {mn:<{name_w}}"
                row += f"{'(not run yet)':>{col_w * len(METRIC_COLS)}}"
                print(row)
                continue
            row = f"  {mn:<{name_w}}"
            for m in METRIC_COLS:
                val = _get(data, split_key, m)
                cell = f"{val:.3f}" if not np.isnan(val) else "  —  "
                # Mark cells that beat the paper
                pv = paper_ref.get(m, float("nan"))
                if not np.isnan(val) and not np.isnan(pv) and val > pv + 0.005:
                    cell += "*"
                row += f"{cell:>{col_w}}"
            print(row)

        print("  " + sep)
        print("  * = beats paper benchmark for that metric")

        # Beat count summary
        print()
        print(f"  Beat count vs paper ({split_label[:10]}...):")
        for mn in ALL_MODELS:
            data = results.get(mn)
            if data is None: continue
            beats = 0
            for m in METRIC_COLS:
                val = _get(data, split_key, m)
                pv  = paper_ref.get(m, float("nan"))
                if not np.isnan(val) and not np.isnan(pv) and val > pv + 0.005:
                    beats += 1
            total = sum(1 for m in METRIC_COLS if not np.isnan(paper_ref.get(m, float("nan"))))
            print(f"    {mn:<25} {beats}/{total} metrics beat")

    # Runtime
    print()
    print("  Runtime:")
    for mn in ALL_MODELS:
        data = results.get(mn)
        if data is None: continue
        rt = data.get("runtime_seconds", 0)
        print(f"    {mn:<25} {rt/60:.1f} min")


def save_comparison_plot(results: dict) -> None:
    """Save a grouped bar chart comparing AUC across models on both splits."""
    models  = [mn for mn in ALL_MODELS if results.get(mn)]
    cv_aucs = [_get(results[mn], "cv",  "auc") for mn in models]
    ex_aucs = [_get(results[mn], "ext", "auc") for mn in models]

    x  = np.arange(len(models))
    w  = 0.35
    colors = ["#378ADD", "#1D9E75"]

    fig, ax = plt.subplots(figsize=(max(10, len(models) * 1.8), 6))
    ax.bar(x - w/2, cv_aucs, w, color=colors[0], alpha=0.85,
           edgecolor="white", label="CV AUC (scaffold-split)")
    ax.bar(x + w/2, ex_aucs, w, color=colors[1], alpha=0.85,
           edgecolor="white", label="External test AUC")

    # Paper reference lines
    ax.axhline(PAPER_INTERNAL["auc"], ls="--", lw=1.2, color="#D85A30", alpha=0.8,
               label=f"Paper CV = {PAPER_INTERNAL['auc']:.3f}")
    ax.axhline(PAPER_EXTERNAL["auc"], ls=":",  lw=1.2, color="#D85A30", alpha=0.8,
               label=f"Paper External = {PAPER_EXTERNAL['auc']:.3f}")

    for xi, (cv, ex) in enumerate(zip(cv_aucs, ex_aucs)):
        if not np.isnan(cv):
            ax.text(xi - w/2, cv + 0.005, f"{cv:.3f}", ha="center",
                    va="bottom", fontsize=8, color=colors[0])
        if not np.isnan(ex):
            ax.text(xi + w/2, ex + 0.005, f"{ex:.3f}", ha="center",
                    va="bottom", fontsize=8, color=colors[1])

    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=20, ha="right", fontsize=10)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("AUC")
    ax.set_title("Model comparison — AUC on CV and External test vs Liu et al. (2025) D-MPNN")
    ax.legend(loc="lower right", fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.3)

    out = config.RESULTS_DIR / "comparison_auc.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    logger.info(f"Comparison plot saved → {out}")


def save_full_metrics_plot(results: dict) -> None:
    """Heatmap of all metrics for all models on external test."""
    models = [mn for mn in ALL_MODELS if results.get(mn)]
    data   = np.zeros((len(models), len(METRIC_COLS)))
    for i, mn in enumerate(models):
        for j, m in enumerate(METRIC_COLS):
            data[i, j] = _get(results[mn], "ext", m)

    fig, ax = plt.subplots(figsize=(len(METRIC_COLS) * 1.4, len(models) * 0.9 + 1.5))
    im = ax.imshow(data, aspect="auto", cmap="YlGn", vmin=0.5, vmax=1.0)
    plt.colorbar(im, ax=ax, label="Score")

    ax.set_xticks(range(len(METRIC_COLS)))
    ax.set_xticklabels([METRIC_LABELS[m] for m in METRIC_COLS], rotation=30, ha="right")
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(models)
    ax.set_title("External test metrics heatmap (all models)")

    for i in range(len(models)):
        for j in range(len(METRIC_COLS)):
            v = data[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.3f}", ha="center", va="center",
                        fontsize=8.5, color="black")

    # Add paper row (dashed border)
    for j, m in enumerate(METRIC_COLS):
        pv = PAPER_EXTERNAL.get(m, float("nan"))
        if not np.isnan(pv):
            rect = plt.Rectangle((j - 0.5, -0.5), 1, len(models),
                                  fill=False, edgecolor="#D85A30",
                                  linewidth=0.5, linestyle="--")
            ax.add_patch(rect)
            ax.text(j, -0.8, f"P:{pv:.3f}", ha="center", va="center",
                    fontsize=7, color="#D85A30")

    plt.tight_layout()
    out = config.RESULTS_DIR / "comparison_heatmap.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Heatmap saved → {out}")


def main():
    logger.info("Loading results from all model runners ...")
    results = {}
    for mn in ALL_MODELS:
        data = _load(mn)
        if data:
            results[mn] = data
            logger.info(f"  Loaded: {mn}")
        else:
            logger.warning(f"  Not found: {mn} (run python run_{mn.lower().replace(' ','_')}.py first)")

    if not results:
        logger.error("No results found. Run the individual model scripts first.")
        return

    print_table(results)
    save_comparison_plot(results)
    save_full_metrics_plot(results)

    print()
    print(f"  Plots saved to: {config.RESULTS_DIR}/")
    print(f"    comparison_auc.png")
    print(f"    comparison_heatmap.png")
    print(f"    <model>/roc.png  (per-model ROC curves)")


if __name__ == "__main__":
    main()
