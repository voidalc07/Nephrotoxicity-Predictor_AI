"""
main.py
=======
NephroTox Predictor v2 — Complete pipeline using the Liu et al. (2025) dataset.

HOW TO RUN IN PYCHARM
----------------------
Option A — Edit the RUN_CONFIG block below and press the green Run button.
Option B — Run > Edit Configurations > Python
             Script:     main.py
             Parameters: --mode train_eval
             Working dir: this folder (nephrotox_v2/)

MODES
-----
  train_eval   Full CV + train final model + evaluate on external test set
  predict      Score new SMILES using a saved model

TARGET
------
  Beat Liu et al. (2025) D-MPNN + ChemoPy2d:
    Internal test:  AUC 93.3%  Kappa 70.3%
    External test:  AUC 84.6%  Kappa 69.7%
    External + UQ:  AUC 86.8%  Kappa 75.6%
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import config
from utils import setup_logging, load_dataset, print_class_distribution, safe_n_folds
from featurizer import MolecularFeaturizer
from model import NephrotoxEnsemble
from evaluator import Evaluator

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Factory (creates a fresh model per CV fold)
# ---------------------------------------------------------------------------

def make_factory(args):
    def _factory():
        return NephrotoxEnsemble(
            tune_lgbm=not args.no_tune,
            lgbm_trials=args.lgbm_trials,
            device=args.device,
            seed=args.seed,
        )
    return _factory


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def build_parser():
    p = argparse.ArgumentParser(
        prog="nephrotox_v2",
        description="NephroTox Predictor v2 — Beat Liu et al. (2025)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    p.add_argument("--mode", choices=["train_eval", "predict"],
                   default="train_eval")

    # Data (defaults from config.py — no need to set these if using the provided CSVs)
    p.add_argument("--train_csv", default=str(config.TRAIN_CSV))
    p.add_argument("--ext_csv",   default=str(config.EXTTEST_CSV))
    p.add_argument("--predict_file", default=None,
                   help="CSV with 'smiles' column for --mode predict")
    p.add_argument("--output_dir",   default=str(config.RESULTS_DIR))

    # Training
    p.add_argument("--cv_folds",    type=int,   default=config.CV_FOLDS)
    p.add_argument("--lgbm_trials", type=int,   default=config.LGBM_TRIALS)
    p.add_argument("--no_tune",     action="store_true",
                   help="Skip Optuna HPO — faster but lower accuracy")
    p.add_argument("--device", default=config.DEVICE,
                   choices=["cpu", "mps", "cuda"],
                   help="'mps' for Apple Silicon M2/M3, else 'cpu'")
    p.add_argument("--threshold",   type=float, default=config.THRESHOLD)
    p.add_argument("--seed",        type=int,   default=config.RANDOM_SEED)

    # Extras
    p.add_argument("--explain", action="store_true",
                   help="Generate SHAP feature importance (adds ~3-5 min)")
    p.add_argument("--skip_external", action="store_true",
                   help="Skip external test set evaluation")
    p.add_argument("--verbose", action="store_true")

    return p


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args=None):
    parser = build_parser()
    args = parser.parse_args(args)
    setup_logging(logging.DEBUG if args.verbose else logging.INFO)

    logger.info("=" * 62)
    logger.info("  NephroTox Predictor v2")
    logger.info("  Target: beat Liu et al. (2025) D-MPNN + ChemoPy2d")
    logger.info(f"  Internal: AUC {config.PAPER_INTERNAL['auc']}  "
                f"Kappa {config.PAPER_INTERNAL['kappa']}")
    logger.info(f"  External: AUC {config.PAPER_EXTERNAL['auc']}  "
                f"Kappa {config.PAPER_EXTERNAL['kappa']}")
    logger.info("=" * 62)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load data ─────────────────────────────────────────────────────────────
    logger.info(f"\nLoading training data: {args.train_csv}")
    train_df = load_dataset(args.train_csv)
    print_class_distribution(train_df["label"].values, "Training set", logger)

    ext_df = None
    if not args.skip_external and Path(args.ext_csv).exists():
        logger.info(f"Loading external test data: {args.ext_csv}")
        ext_df = load_dataset(args.ext_csv)
        print_class_distribution(ext_df["label"].values, "External test", logger)

    # ── Featurise training data ───────────────────────────────────────────────
    logger.info("\nExtracting features ...")
    feat = MolecularFeaturizer(
        scaffold_csvs=config.SCAFFOLD_CSVs,
        fp_bits=config.FP_BITS,
        pca_components=config.PCA_COMPONENTS,
        bert_batch=config.BERT_BATCH,
        bert_model=config.BERT_MODEL,
        bert_fallback=config.BERT_FALLBACK,
        seed=args.seed,
    )

    Xc, Xt, Xf, vi = feat.fit_transform(train_df["smiles"])
    y_train = train_df["label"].values[vi]
    print_class_distribution(y_train, "After featurization", logger)

    # ── MODE: train_eval ──────────────────────────────────────────────────────
    if args.mode == "train_eval":

        evaluator = Evaluator(output_dir)
        n_folds   = safe_n_folds(y_train, args.cv_folds)

        # Cross-validation on training set
        if n_folds >= 2:
            factory    = make_factory(args)
            cv_results = evaluator.cross_validate(
                factory, Xc, Xt, Xf, y_train, n_folds=n_folds
            )
            evaluator.save_cv_report(cv_results)
        else:
            logger.warning("Too few samples for CV — training final model only.")

        # Final model on full training data
        logger.info("\nTraining final model on full training set ...")
        final_model = NephrotoxEnsemble(
            tune_lgbm=not args.no_tune,
            lgbm_trials=args.lgbm_trials,
            device=args.device,
            seed=args.seed,
        )
        final_model.fit(Xc, Xt, Xf, y_train)
        final_model.save(output_dir / "nephrotox_model.pkl")
        feat.save(output_dir / "featurizer.pkl")

        # Training predictions CSV
        proba_tr = final_model.predict_proba(Xc, Xt, Xf)
        pred_tr  = (proba_tr[:, 1] >= args.threshold).astype(int)
        out_tr   = train_df.iloc[vi].copy().reset_index(drop=True)
        out_tr["prob_nephrotoxic"]  = proba_tr[:, 1].round(4)
        out_tr["predicted_label"]   = pred_tr
        out_tr["predicted_class"]   = out_tr["predicted_label"].map(
            {0: "Non-nephrotoxic", 1: "Nephrotoxic"})
        out_tr.to_csv(output_dir / "train_predictions.csv", index=False)

        # External test evaluation
        if ext_df is not None:
            logger.info("\nEvaluating on external test set ...")
            Xce, Xte, Xfe, vie = feat.transform(ext_df["smiles"])
            y_ext = ext_df["label"].values[vie]
            ext_results = evaluator.evaluate_external(
                final_model, Xce, Xte, Xfe, y_ext,
                threshold=args.threshold,
                unc_threshold=0.2,
            )
            evaluator.save_external_report(ext_results)

        # SHAP
        if args.explain:
            run_shap(final_model, feat, Xc, y_train, output_dir)

        # Summary
        logger.info("\n" + "=" * 62)
        logger.info("  All outputs written to:")
        logger.info(f"  {output_dir.resolve()}/")
        logger.info("=" * 62)
        for f in sorted(output_dir.iterdir()):
            logger.info(f"  {f.name}")

    # ── MODE: predict ─────────────────────────────────────────────────────────
    elif args.mode == "predict":
        if not args.predict_file:
            logger.error(
                "--predict_file is required for predict mode.\n"
                "Example: --predict_file new_compounds.csv"
            )
            sys.exit(1)

        model_path = output_dir / "nephrotox_model.pkl"
        feat_path  = output_dir / "featurizer.pkl"
        if not model_path.exists():
            logger.error(
                f"No saved model at {model_path}.\n"
                "Run --mode train_eval first."
            )
            sys.exit(1)

        loaded_model = NephrotoxEnsemble.load(model_path)
        loaded_feat  = MolecularFeaturizer.load(feat_path)

        pred_df = load_dataset(args.predict_file)
        Xcp, Xtp, Xfp_p, vip = loaded_feat.transform(pred_df["smiles"])

        proba, unc = loaded_model.predict_with_uncertainty(Xcp, Xtp, Xfp_p)
        preds = (proba[:, 1] >= args.threshold).astype(int)

        result_df = pred_df.iloc[vip].copy().reset_index(drop=True)
        result_df["prob_non_nephrotoxic"] = proba[:, 0].round(4)
        result_df["prob_nephrotoxic"]     = proba[:, 1].round(4)
        result_df["predicted_label"]      = preds
        result_df["predicted_class"]      = result_df["predicted_label"].map(
            {0: "Non-nephrotoxic", 1: "Nephrotoxic"})
        result_df["uncertainty"]  = unc.round(4)
        result_df["flag_review"]  = (unc > 0.2).astype(int)

        out = output_dir / "predictions.csv"
        result_df.to_csv(out, index=False)
        logger.info(f"Predictions saved → {out}")
        logger.info(f"  Nephrotoxic: {preds.sum()} / {len(preds)}")
        logger.info(f"  Flagged for review: {(unc > 0.2).sum()}")


# ---------------------------------------------------------------------------
# SHAP
# ---------------------------------------------------------------------------

def run_shap(model, feat, X_combined, y, output_dir):
    try:
        import shap
    except ImportError:
        logger.warning("shap not installed — skipping. pip install shap")
        return

    logger.info("\nComputing SHAP values (2–5 min) ...")
    n_bg  = min(80, len(X_combined))
    n_exp = min(100, len(X_combined))
    bg    = shap.sample(X_combined, n_bg, random_state=42).astype("float32")
    X_ex  = X_combined[:n_exp].astype("float32")

    def _predict(X):
        return model._lgbm.predict_proba(X.astype("float32"))[:, 1]

    explainer   = shap.KernelExplainer(_predict, bg)
    shap_values = explainer.shap_values(X_ex, nsamples=128, l1_reg="aic")

    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_ex, feature_names=feat.feature_names,
                      show=False, max_display=20)
    plt.tight_layout()
    plt.savefig(output_dir / "shap_beeswarm.png", dpi=150, bbox_inches="tight")
    plt.close()

    fig = plt.figure(figsize=(8, 7))
    shap.summary_plot(shap_values, X_ex, feature_names=feat.feature_names,
                      plot_type="bar", show=False, max_display=20)
    plt.tight_layout()
    plt.savefig(output_dir / "shap_bar.png", dpi=150, bbox_inches="tight")
    plt.close()

    pd.DataFrame(shap_values, columns=feat.feature_names).to_csv(
        output_dir / "shap_values.csv", index=False)
    logger.info("SHAP outputs saved.")


# ---------------------------------------------------------------------------
# PyCharm run block
# Uncomment RUN_CONFIG lines and change main() to main(RUN_CONFIG) below
# ---------------------------------------------------------------------------

RUN_CONFIG = [
    "--mode",         "train_eval",
    # "--no_tune",                   # uncomment for quick test (~5 min total)
    "--lgbm_trials",  "50",
    "--cv_folds",     "5",
    "--device",       "cpu",         # change to "mps" on Apple Silicon M2/M3
    "--output_dir",   "./results",
    # "--explain",                   # uncomment for SHAP plots
    # "--skip_external",             # uncomment to skip external test
    # "--verbose",
]

if __name__ == "__main__":
    # ── To run directly from PyCharm without a terminal:
    #    1. Edit RUN_CONFIG above
    #    2. Change the next line to: main(RUN_CONFIG)
    #    3. Press the green Run button
    #main()
    main(RUN_CONFIG)   # ← uncomment this, comment out main() above
