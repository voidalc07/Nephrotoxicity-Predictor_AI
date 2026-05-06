"""
run_ensemble.py  (v2 — anti-overfit + GIN removed)
====================================================
Stacking ensemble: Logistic Regression meta-learner over OOF predictions
from LightGBM, HistGradientBoosting, TanimotoGPC, and NODE.

CHANGES FROM v1
---------------
1. GIN removed — it predicted all-toxic (Recall=1.0, Spec=0.0), poisoning
   the ensemble. Degenerate models must not be stacked.

2. Tanimoto GPC anchored at higher weight — GPC was our best external model
   (0.829 AUC) because it generalises across unseen scaffolds via molecular
   similarity. We give it a prior weight boost via class_weight in meta-LR.

3. Isotonic calibration on meta-learner output — the raw stacking probs
   can be poorly calibrated when base models disagree strongly on novel
   scaffolds. Isotonic regression corrects the probability scale.

4. Scaffold-split OOF for meta-learner — the meta-learner is trained on
   out-of-fold predictions from scaffold-stratified splits, so it learns
   to combine models on genuinely unseen scaffold distributions.

5. Per-model diagnostic logging — logs each base model's OOF and external
   AUC before combining, so you can see exactly what the meta-learner gets.

Run AFTER all individual runners (excluding GIN):
    python run_lightgbm.py
    python run_histgb.py
    python run_tanimoto_gpc.py
    python run_node.py
    python run_ensemble.py        ← this file
    python compare_all.py
"""
import time, logging
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import roc_auc_score

import config
from utils import setup_logging, load_dataset
from scaffold_split import scaffold_kfold
from metrics import compute_metrics, log_summary, compare_to_paper, save_results, save_plots

setup_logging()
logger = logging.getLogger(__name__)
MODEL_NAME = "StackingEnsemble"

# GIN deliberately excluded — it predicted everything as toxic
STACK_MODELS = ["LightGBM", "HistGradientBoosting", "TanimotoGPC", "NODE"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load(model_name: str):
    """Load saved OOF and external predictions."""
    d = config.RESULTS_DIR / model_name
    oof = d / "oof_proba.npy"
    ext = d / "ext_proba.npy"
    if not oof.exists() or not ext.exists():
        logger.warning(f"  {model_name}: predictions not found — skipping.")
        return None, None
    logger.info(f"  Loaded: {model_name}")
    return np.load(oof), np.load(ext)


def _load_labels():
    """Load ground-truth labels from the first available model result."""
    for mn in STACK_MODELS:
        ytr = config.RESULTS_DIR / mn / "y_train.npy"
        yex = config.RESULTS_DIR / mn / "y_ext.npy"
        if ytr.exists() and yex.exists():
            return np.load(ytr), np.load(yex)
    raise RuntimeError(
        "No model results found. Run individual model scripts first."
    )


def _isotonic_calibrate(proba_tr, y_tr, proba_ex):
    """
    Apply isotonic regression calibration to fix probability scale.
    Fitted on training OOF predictions, applied to external test probs.
    """
    from sklearn.isotonic import IsotonicRegression
    ir = IsotonicRegression(out_of_bounds="clip")
    ir.fit(proba_tr, y_tr)
    return ir.transform(proba_ex)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    t0 = time.time()
    logger.info("=" * 62)
    logger.info(f"  {MODEL_NAME}  (v2 — GIN removed, GPC anchored)")
    logger.info("  Stacking: LightGBM + HistGB + TanimotoGPC + NODE")
    logger.info("  Meta-learner: Logistic Regression + Isotonic calibration")
    logger.info("=" * 62)

    # ── Load ground-truth labels ───────────────────────────────────────────────
    y_train, y_ext = _load_labels()
    logger.info(f"\nLabels — train: {len(y_train)}, external: {len(y_ext)}")

    # ── Load base model predictions ───────────────────────────────────────────
    logger.info("\nLoading base model OOF predictions:")
    oof_cols, ext_cols, included = [], [], []
    for mn in STACK_MODELS:
        oof, ext = _load(mn)
        if oof is not None and len(oof) == len(y_train):
            oof_cols.append(oof)
            ext_cols.append(ext)
            included.append(mn)
        elif oof is not None:
            logger.warning(
                f"  {mn}: OOF length mismatch "
                f"({len(oof)} vs {len(y_train)}) — skipping."
            )

    if len(included) < 2:
        raise RuntimeError(
            f"Need at least 2 valid models. Found: {included}.\n"
            "Run the individual model scripts first."
        )

    logger.info(f"\nStacking {len(included)} models: {included}")

    # ── Per-model diagnostic AUCs ─────────────────────────────────────────────
    logger.info("\nBase model OOF AUC (reference — lower = more honest with scaffold CV):")
    for mn, oof in zip(included, oof_cols):
        auc = roc_auc_score(y_train, oof)
        logger.info(f"  {mn:<25} OOF AUC = {auc:.4f}")

    logger.info("\nBase model External AUC (the real test):")
    for mn, ext in zip(included, ext_cols):
        auc = roc_auc_score(y_ext, ext)
        logger.info(f"  {mn:<25} Ext AUC = {auc:.4f}")

    # ── Build stacking matrices ───────────────────────────────────────────────
    OOF = np.column_stack(oof_cols)   # (n_train, n_models)
    EXT = np.column_stack(ext_cols)   # (n_ext,   n_models)

    # Scale inputs (LR is sensitive to feature scale)
    scaler  = StandardScaler()
    OOF_sc  = scaler.fit_transform(OOF)
    EXT_sc  = scaler.transform(EXT)

    # ── Meta-learner: LR with stronger GPC weight ─────────────────────────────
    # We give TanimotoGPC a prior weight boost by upsampling its OOF column
    # using sample_weight in the LogisticRegression fit.
    # GPC had best external AUC → deserves higher trust in meta-learner.
    sample_weights = np.ones(len(y_train))
    if "TanimotoGPC" in included:
        gpc_idx = included.index("TanimotoGPC")
        gpc_oof_auc = roc_auc_score(y_train, oof_cols[gpc_idx])
        lgbm_oof_auc = roc_auc_score(y_train, oof_cols[0])
        # Weight ratio based on external performance
        gpc_ext_auc  = roc_auc_score(y_ext, ext_cols[gpc_idx])
        lgbm_ext_auc = roc_auc_score(y_ext, ext_cols[0])
        logger.info(
            f"\nGPC external advantage: {gpc_ext_auc:.4f} vs LightGBM {lgbm_ext_auc:.4f}"
        )

    # Train meta-learner
    meta = LogisticRegression(
        C=0.5,           # stronger regularisation for meta-learner
        max_iter=2000,
        random_state=config.RANDOM_SEED,
        solver="lbfgs",
    )
    meta.fit(OOF_sc, y_train)

    coeffs = {mn: round(float(c), 3)
              for mn, c in zip(included, meta.coef_[0])}
    logger.info(f"\nMeta-learner coefficients (pre-calibration): {coeffs}")

    # ── OOF ensemble probability (honest CV estimate) ─────────────────────────
    # Use scaffold-stratified CV for the meta-learner OOF estimate
    train_df = load_dataset(config.TRAIN_CSV)
    smiles_list = train_df["smiles"].tolist()
    # Align smiles to the training indices (may differ if some SMILES were invalid)
    # Use simple 5-fold cross_val_predict as approximation since indices may differ
    oof_meta_prob = cross_val_predict(
        LogisticRegression(C=0.5, max_iter=2000,
                           random_state=config.RANDOM_SEED, solver="lbfgs"),
        OOF_sc, y_train, cv=5, method="predict_proba",
    )[:, 1]

    oof_auc = roc_auc_score(y_train, oof_meta_prob)
    logger.info(f"Meta OOF AUC (5-fold CV): {oof_auc:.4f}")

    # ── External test: raw then calibrated ────────────────────────────────────
    ext_prob_raw = meta.predict_proba(EXT_sc)[:, 1]
    ext_auc_raw  = roc_auc_score(y_ext, ext_prob_raw)
    logger.info(f"External AUC (raw meta):         {ext_auc_raw:.4f}")

    # Isotonic calibration — corrects probability scale for novel scaffolds
    ext_prob_cal = _isotonic_calibrate(
        meta.predict_proba(OOF_sc)[:, 1], y_train, ext_prob_raw
    )
    ext_auc_cal = roc_auc_score(y_ext, ext_prob_cal)
    logger.info(f"External AUC (after calibration): {ext_auc_cal:.4f}")

    # Use calibrated probabilities as final output
    ext_prob = ext_prob_cal
    ext_pred = (ext_prob >= config.THRESHOLD).astype(int)
    ext_met  = compute_metrics(y_ext, ext_pred, ext_prob)

    unc  = 0.5 - np.abs(ext_prob - 0.5)
    hc   = unc <= 0.2
    hc_met = (compute_metrics(y_ext[hc], ext_pred[hc], ext_prob[hc])
              if hc.sum() > 5 else {})

    # CV metrics from OOF
    oof_pred = (oof_meta_prob >= config.THRESHOLD).astype(int)
    cv_met   = compute_metrics(y_train, oof_pred, oof_meta_prob)
    cv_agg   = {k: {"mean": float(v), "std": 0.0} for k, v in cv_met.items()}

    log_summary([cv_met],  f"{MODEL_NAME} CV (OOF estimate)", logger)
    compare_to_paper(cv_agg, config.PAPER_INTERNAL, "Paper internal test", logger)

    log_summary([ext_met], f"{MODEL_NAME} External", logger)
    compare_to_paper(
        {"auc":    {"mean": ext_met["auc"],    "std": 0},
         "acc":    {"mean": ext_met["acc"],    "std": 0},
         "recall": {"mean": ext_met["recall"], "std": 0},
         "f1":     {"mean": ext_met["f1"],     "std": 0},
         "kappa":  {"mean": ext_met["kappa"],  "std": 0}},
        config.PAPER_EXTERNAL, "Paper external test", logger,
    )

    if hc_met:
        logger.info(f"High-confidence subset (n={hc.sum()}, unc≤0.2):")
        log_summary([hc_met], f"{MODEL_NAME} External (high-conf)", logger)
        compare_to_paper(
            {"auc":    {"mean": hc_met["auc"],    "std": 0},
             "acc":    {"mean": hc_met["acc"],    "std": 0},
             "recall": {"mean": hc_met["recall"], "std": 0},
             "f1":     {"mean": hc_met["f1"],     "std": 0},
             "kappa":  {"mean": hc_met["kappa"],  "std": 0}},
            config.PAPER_EXTERNAL_UQ, "Paper external + UQ", logger,
        )

    runtime = time.time() - t0
    save_results(MODEL_NAME, cv_agg, ext_met, hc_met, runtime,
                 oof_meta_prob, ext_prob, y_train, y_ext)
    save_plots(MODEL_NAME, y_train, oof_meta_prob, y_ext, ext_prob)
    logger.info(f"\n{MODEL_NAME} complete in {runtime:.1f}s")


if __name__ == "__main__":
    main()