"""
evaluator.py
------------
Stratified K-fold CV and head-to-head evaluation against Liu et al. (2025).

Metrics exactly matching Table 1 and Table 3 of the paper:
    AUC    — area under ROC
    ACC    — accuracy
    SE     — sensitivity (recall for nephrotoxics)
    F1     — F1-measure
    Kappa  — Cohen's kappa  ← paper's primary ranking metric

Paper benchmarks loaded from config.py:
    PAPER_INTERNAL   — D-MPNN + ChemoPy2d on their 10% internal test set
    PAPER_EXTERNAL   — same model on 304-compound external set
    PAPER_EXTERNAL_UQ— same model + uncertainty filtering on external set
"""

import logging
from pathlib import Path
from typing import Any, Callable, Dict

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, average_precision_score, cohen_kappa_score,
    confusion_matrix, f1_score, matthews_corrcoef,
    recall_score, roc_auc_score, roc_curve, precision_recall_curve,
)

from config import PAPER_INTERNAL, PAPER_EXTERNAL, PAPER_EXTERNAL_UQ

logger = logging.getLogger(__name__)

_BLUE   = "#378ADD"
_GREEN  = "#639922"
_AMBER  = "#EF9F27"
_TEAL   = "#1D9E75"
_CORAL  = "#D85A30"
_PURPLE = "#7F77DD"
_GRAY   = "#888780"


class Evaluator:

    def __init__(self, output_dir) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._style()

    @staticmethod
    def _style():
        plt.rcParams.update({
            "figure.dpi": 130,
            "axes.spines.top": False, "axes.spines.right": False,
            "axes.grid": True, "grid.alpha": 0.3, "grid.linewidth": 0.5,
            "font.family": "sans-serif", "font.size": 11,
            "axes.titlesize": 12, "axes.labelsize": 11,
            "xtick.labelsize": 10, "ytick.labelsize": 10,
            "legend.fontsize": 10,
        })

    # ── Cross-validation ──────────────────────────────────────────────────────

    def cross_validate(
        self,
        model_factory: Callable,
        X_combined: np.ndarray,
        X_tabpfn:   np.ndarray,
        X_fp:       np.ndarray,
        y:          np.ndarray,
        n_folds:    int = 5,
    ) -> Dict[str, Any]:

        skf  = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        keys = ["auc", "acc", "se", "f1", "kappa", "specificity", "mcc"]
        fold_metrics: Dict[str, list] = {"fold": []}
        for k in keys:
            fold_metrics[k] = []
        all_y_true, all_y_prob = [], []

        logger.info(f"Starting {n_folds}-fold stratified CV ...")
        for fold, (tr, val) in enumerate(skf.split(X_combined, y), 1):
            logger.info(f"  Fold {fold}/{n_folds}  train={len(tr)}  val={len(val)}")
            m = model_factory()
            m.fit(X_combined[tr], X_tabpfn[tr], X_fp[tr], y[tr])
            prob = m.predict_proba(X_combined[val], X_tabpfn[val], X_fp[val])[:, 1]
            pred = (prob >= 0.5).astype(int)
            y_val = y[val]
            all_y_true.extend(y_val.tolist())
            all_y_prob.extend(prob.tolist())
            met = _metrics(y_val, pred, prob)
            fold_metrics["fold"].append(fold)
            for k in keys:
                fold_metrics[k].append(met[k])
            logger.info(
                f"    AUC={met['auc']:.3f}  ACC={met['acc']:.3f}  "
                f"SE={met['se']:.3f}  F1={met['f1']:.3f}  "
                f"Kappa={met['kappa']:.3f}"
            )

        agg = {}
        for k in keys:
            v = [x for x in fold_metrics[k] if not np.isnan(x)]
            agg[k] = {"mean": float(np.mean(v)), "std": float(np.std(v))}

        _log_summary(agg, "CV", logger)
        _compare(agg, PAPER_INTERNAL, "Paper internal test set", logger)

        return {
            **fold_metrics,
            "all_y_true": np.array(all_y_true),
            "all_y_prob":  np.array(all_y_prob),
            "agg": agg,
        }

    # ── External test evaluation ──────────────────────────────────────────────

    def evaluate_external(
        self,
        model,
        X_combined: np.ndarray,
        X_tabpfn:   np.ndarray,
        X_fp:       np.ndarray,
        y:          np.ndarray,
        threshold:  float = 0.5,
        unc_threshold: float = 0.2,
    ) -> Dict[str, Any]:
        """
        Full external test set evaluation — mirrors Table 3 of Liu et al.
        Returns metrics for both all predictions and high-confidence subset.
        """
        proba, unc = model.predict_with_uncertainty(X_combined, X_tabpfn, X_fp)
        prob = proba[:, 1]
        pred = (prob >= threshold).astype(int)

        all_metrics  = _metrics(y, pred, prob)
        high_conf    = unc <= unc_threshold
        hc_metrics   = _metrics(y[high_conf], pred[high_conf], prob[high_conf]) \
                       if high_conf.sum() > 5 else {}

        _log_summary({"all": {"mean": 0}}, "External test", logger)
        logger.info(
            f"  n={len(y)}  high-conf={high_conf.sum()} "
            f"({100*high_conf.mean():.1f}%)"
        )
        _log_summary({k: {"mean": v, "std": 0.0} for k, v in all_metrics.items()},
                     "External (all)", logger)
        _compare({k: {"mean": v, "std": 0} for k, v in all_metrics.items()},
                 PAPER_EXTERNAL, "Paper external (D-MPNN, all)", logger)
        if hc_metrics:
            _log_summary({k: {"mean": v, "std": 0.0} for k, v in hc_metrics.items()},
                         "External (high-conf)", logger)
            _compare({k: {"mean": v, "std": 0} for k, v in hc_metrics.items()},
                     PAPER_EXTERNAL_UQ, "Paper external + UQ", logger)

        return {
            "all_metrics": all_metrics,
            "hc_metrics":  hc_metrics,
            "y_true": y,
            "y_prob": prob,
            "y_pred": pred,
            "uncertainty": unc,
            "high_conf_mask": high_conf,
        }

    # ── Save all reports ──────────────────────────────────────────────────────

    def save_cv_report(self, cv_results: Dict) -> None:
        keys = ["fold", "auc", "acc", "se", "f1", "kappa", "specificity", "mcc"]
        pd.DataFrame({k: cv_results[k] for k in keys}).to_csv(
            self.output_dir / "cv_metrics.csv", index=False)
        agg = cv_results["agg"]
        pd.DataFrame([
            {"metric": k, "mean": v["mean"], "std": v["std"]}
            for k, v in agg.items()
        ]).to_csv(self.output_dir / "cv_summary.csv", index=False)

        yt = cv_results["all_y_true"]
        yp = cv_results["all_y_prob"]
        yc = (yp >= 0.5).astype(int)
        self._plot_roc(yt, yp, "CV pooled", "cv_roc.png", PAPER_INTERNAL["auc"])
        self._plot_pr(yt, yp, "CV pooled", "cv_pr.png")
        self._plot_confusion(yt, yc, "CV pooled", "cv_confusion.png")
        self._plot_metrics_bar(agg, "CV", "cv_metrics_bar.png")
        self._plot_calibration(yt, yp, "cv_calibration.png")
        self._plot_vs_paper(agg, PAPER_INTERNAL, "vs_paper_internal.png",
                            "CV mean vs paper internal test set")

    def save_external_report(self, ext_results: Dict) -> None:
        yt   = ext_results["y_true"]
        yp   = ext_results["y_prob"]
        yc   = ext_results["y_pred"]
        unc  = ext_results["uncertainty"]
        hc   = ext_results["high_conf_mask"]
        am   = ext_results["all_metrics"]
        hm   = ext_results.get("hc_metrics", {})

        self._plot_roc(yt, yp, "External test", "ext_roc.png",
                       PAPER_EXTERNAL["auc"])
        self._plot_pr(yt, yp, "External test", "ext_pr.png")
        self._plot_confusion(yt, yc, "External test (all)", "ext_confusion_all.png")
        if hc.sum() > 5:
            self._plot_confusion(yt[hc], yc[hc], "External (high-conf)",
                                 "ext_confusion_hc.png")
        self._plot_uncertainty(yt, yp, unc, "ext_uncertainty.png")

        # Head-to-head vs paper (external)
        self._plot_vs_paper(
            {k: {"mean": v, "std": 0} for k, v in am.items()},
            PAPER_EXTERNAL,
            "vs_paper_external.png",
            "External test: our model vs paper D-MPNN + ChemoPy2d",
        )

        # Save metrics CSVs
        pd.DataFrame([
            {"split": "all", **am},
            {"split": "high_confidence", **hm} if hm else {},
        ]).to_csv(self.output_dir / "external_metrics.csv", index=False)

        # Save predictions
        pd.DataFrame({
            "y_true": yt, "y_pred": yc, "prob_nephrotoxic": yp.round(4),
            "uncertainty": unc.round(4),
            "high_confidence": hc.astype(int),
            "flag_review": (unc > 0.2).astype(int),
        }).to_csv(self.output_dir / "external_predictions.csv", index=False)

        logger.info(f"External test reports → {self.output_dir.resolve()}/")

    # ── Plots ─────────────────────────────────────────────────────────────────

    def _plot_roc(self, yt, yp, title_suffix, fname, paper_auc=None):
        fpr, tpr, _ = roc_curve(yt, yp)
        auc = roc_auc_score(yt, yp)
        fig, ax = plt.subplots(figsize=(5.5, 5))
        ax.plot(fpr, tpr, color=_BLUE, lw=2, label=f"Our model AUC = {auc:.3f}")
        ax.fill_between(fpr, tpr, alpha=0.07, color=_BLUE)
        ax.axline((0, 0), slope=1, ls="--", color=_GRAY, lw=1, alpha=0.6,
                  label="Random (0.5)")
        if paper_auc:
            ax.axhline(paper_auc, ls=":", color=_CORAL, lw=1.2, alpha=0.8,
                       label=f"Paper D-MPNN = {paper_auc:.3f}")
        ax.set(xlabel="1 - Specificity", ylabel="Sensitivity",
               title=f"ROC — {title_suffix}")
        ax.legend(loc="lower right")
        ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.05)
        plt.tight_layout()
        plt.savefig(self.output_dir / fname, dpi=150)
        plt.close()

    def _plot_pr(self, yt, yp, title_suffix, fname):
        p, r, _ = precision_recall_curve(yt, yp)
        ap = average_precision_score(yt, yp)
        fig, ax = plt.subplots(figsize=(5.5, 5))
        ax.plot(r, p, color=_GREEN, lw=2, label=f"AP = {ap:.3f}")
        ax.fill_between(r, p, alpha=0.07, color=_GREEN)
        ax.axhline(yt.mean(), ls="--", color=_GRAY, lw=1, alpha=0.7,
                   label=f"No-skill = {yt.mean():.2f}")
        ax.set(xlabel="Recall", ylabel="Precision",
               title=f"Precision-Recall — {title_suffix}")
        ax.legend()
        plt.tight_layout()
        plt.savefig(self.output_dir / fname, dpi=150)
        plt.close()

    def _plot_confusion(self, yt, yc, title_suffix, fname):
        cm = confusion_matrix(yt, yc)
        fig, ax = plt.subplots(figsize=(4.5, 4))
        sns.heatmap(cm, annot=True, fmt="d",
                    cmap=sns.light_palette(_BLUE, as_cmap=True),
                    xticklabels=["Non-toxic (0)", "Nephrotoxic (1)"],
                    yticklabels=["Non-toxic (0)", "Nephrotoxic (1)"],
                    linewidths=0.5, linecolor="white", ax=ax)
        ax.set(xlabel="Predicted", ylabel="True",
               title=f"Confusion matrix — {title_suffix}")
        plt.tight_layout()
        plt.savefig(self.output_dir / fname, dpi=150)
        plt.close()

    def _plot_metrics_bar(self, agg, title_suffix, fname):
        keys  = list(agg.keys())
        means = [agg[k]["mean"] for k in keys]
        stds  = [agg[k]["std"]  for k in keys]
        cols  = [_BLUE, _GREEN, _CORAL, _AMBER, _TEAL, _PURPLE, _GRAY][:len(keys)]
        x = np.arange(len(keys))
        fig, ax = plt.subplots(figsize=(10, 4.5))
        bars = ax.bar(x, means, yerr=stds, color=cols, alpha=0.82,
                      edgecolor="white", capsize=4, width=0.55,
                      error_kw={"elinewidth": 1.2, "ecolor": _GRAY})
        for bar, val in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.025, f"{val:.3f}",
                    ha="center", va="bottom", fontsize=9)
        ax.set_xticks(x)
        ax.set_xticklabels([k.upper() for k in keys], rotation=15, ha="right")
        ax.set(ylim=(0, 1.18), ylabel="Score",
               title=f"Performance metrics — {title_suffix}")
        plt.tight_layout()
        plt.savefig(self.output_dir / fname, dpi=150)
        plt.close()

    def _plot_calibration(self, yt, yp, fname, n_bins: int = 10):
        bins = np.linspace(0, 1, n_bins + 1)
        mids = (bins[:-1] + bins[1:]) / 2
        obs  = [yt[(yp >= lo) & (yp < hi)].mean()
                if ((yp >= lo) & (yp < hi)).sum() > 0 else np.nan
                for lo, hi in zip(bins[:-1], bins[1:])]
        obs  = np.array(obs)
        v    = ~np.isnan(obs)
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.plot([0, 1], [0, 1], "--", color=_GRAY, lw=1, label="Perfect")
        ax.scatter(mids[v], obs[v], color=_PURPLE, s=60, zorder=5)
        ax.plot(mids[v], obs[v], color=_PURPLE, lw=1.5, label="Ensemble")
        ax.set(xlabel="Mean predicted probability",
               ylabel="Observed frequency",
               title="Reliability diagram")
        ax.legend()
        plt.tight_layout()
        plt.savefig(self.output_dir / fname, dpi=150)
        plt.close()

    def _plot_uncertainty(self, yt, yp, unc, fname):
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

        # Uncertainty distribution
        ax = axes[0]
        for lbl, col, nm in [(0, _BLUE, "Non-toxic"), (1, _CORAL, "Nephrotoxic")]:
            ax.hist(unc[yt == lbl], bins=30, alpha=0.6, color=col, label=nm,
                    density=True)
        ax.set(xlabel="Uncertainty", ylabel="Density",
               title="Uncertainty by true class")
        ax.legend()

        # Uncertainty vs prediction error
        ax = axes[1]
        correct = (yt == (yp >= 0.5).astype(int))
        ax.scatter(yp[correct],  unc[correct],  alpha=0.3, s=10,
                   color=_GREEN, label="Correct")
        ax.scatter(yp[~correct], unc[~correct], alpha=0.5, s=15,
                   color=_CORAL, label="Incorrect", marker="x")
        ax.set(xlabel="P(nephrotoxic)", ylabel="Uncertainty",
               title="Prediction confidence")
        ax.legend()

        plt.tight_layout()
        plt.savefig(self.output_dir / fname, dpi=150)
        plt.close()

    def _plot_vs_paper(self, agg, paper_ref, fname, title):
        metrics = [k for k in paper_ref if k in agg]
        ours_m  = [agg[k]["mean"] for k in metrics]
        ours_s  = [agg[k].get("std", 0) for k in metrics]
        paper_v = [paper_ref[k] for k in metrics]
        x = np.arange(len(metrics))
        w = 0.35
        fig, ax = plt.subplots(figsize=(9, 5))
        bars_p = ax.bar(x - w/2, paper_v, w,
                        label="Liu et al. (2025) D-MPNN + ChemoPy2d",
                        color=_GRAY, alpha=0.75, edgecolor="white")
        bars_o = ax.bar(x + w/2, ours_m,  w, yerr=ours_s,
                        label="Our model (ChemBERTa-2 + LightGBM + stacking)",
                        color=_PURPLE, alpha=0.82, edgecolor="white",
                        capsize=4, error_kw={"ecolor": _GRAY, "elinewidth": 1.2})
        for bar, val in zip(bars_p, paper_v):
            ax.text(bar.get_x() + bar.get_width()/2, val + 0.01,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=8.5)
        for bar, val, std in zip(bars_o, ours_m, ours_s):
            ax.text(bar.get_x() + bar.get_width()/2, val + std + 0.015,
                    f"{val:.3f}", ha="center", va="bottom",
                    fontsize=8.5, color=_PURPLE)
        ax.set_xticks(x)
        ax.set_xticklabels([m.upper() for m in metrics])
        ax.set(ylim=(0, 1.15), ylabel="Score", title=title)
        ax.legend()
        plt.tight_layout()
        plt.savefig(self.output_dir / fname, dpi=150)
        plt.close()
        logger.info(f"Comparison plot saved: {fname}")


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def _metrics(yt, yc, yp) -> Dict[str, float]:
    nc = len(np.unique(yt))
    auc   = float(roc_auc_score(yt, yp))              if nc > 1 else np.nan
    auprc = float(average_precision_score(yt, yp))    if nc > 1 else np.nan
    acc   = float(accuracy_score(yt, yc))
    se    = float(recall_score(yt, yc, pos_label=1, zero_division=0))
    f1    = float(f1_score(yt, yc, pos_label=1, zero_division=0))
    kappa = float(cohen_kappa_score(yt, yc))
    mcc   = float(matthews_corrcoef(yt, yc))
    cm    = confusion_matrix(yt, yc, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    spec  = tn / (tn + fp + 1e-9)
    return dict(auc=auc, acc=acc, se=se, f1=f1, kappa=kappa,
                specificity=spec, mcc=mcc, auprc=auprc)


def _log_summary(agg, label, logger_):
    logger_.info(f"  ── {label} ──")
    for k, v in agg.items():
        if isinstance(v, dict):
            logger_.info(f"    {k:>12} :  {v['mean']:.4f}  ± {v.get('std',0):.4f}")
        else:
            logger_.info(f"    {k:>12} :  {v:.4f}")


def _compare(agg, paper, label, logger_):
    logger_.info(f"\n  Comparison vs {label}")
    logger_.info(f"  {'Metric':>8}  {'Paper':>8}  {'Ours':>8}  {'Delta':>8}")
    for k, pv in paper.items():
        ov = agg.get(k, {}).get("mean", float("nan"))
        delta = ov - pv
        sign = "+" if delta >= 0 else ""
        flag = "BEAT" if delta > 0 else "behind"
        logger_.info(f"  {k.upper():>8}  {pv:>8.3f}  {ov:>8.3f}"
                     f"  {sign}{delta:.3f}  {flag}")
    logger_.info("")
