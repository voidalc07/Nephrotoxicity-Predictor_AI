"""
run_histgb.py
=============
HistGradientBoostingClassifier — sklearn's histogram-based boosting.
Different regularisation and binning from LightGBM (complementary inductive bias).
No HPO required — sklearn's early stopping handles it internally.

Run: python run_histgb.py
"""
import time, logging
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

import config
from utils import setup_logging, load_dataset, print_dist
from featurizer import MolecularFeaturizer
from scaffold_split import scaffold_kfold
from metrics import compute_metrics, log_fold, log_summary, compare_to_paper, save_results, save_plots

setup_logging()
logger = logging.getLogger(__name__)
MODEL_NAME = "HistGradientBoosting"


def _build():
    return HistGradientBoostingClassifier(
        max_iter=800, learning_rate=0.05, max_leaf_nodes=63,
        max_depth=None, min_samples_leaf=20, l2_regularization=0.1,
        max_bins=255, class_weight="balanced",
        early_stopping=True, validation_fraction=0.1,
        n_iter_no_change=25, random_state=config.RANDOM_SEED,
    )


def main():
    t0 = time.time()
    logger.info(f"{'='*60}\n  {MODEL_NAME} — Scaffold-stratified CV\n{'='*60}")

    train_df = load_dataset(config.TRAIN_CSV)
    ext_df   = load_dataset(config.EXTTEST_CSV)
    print_dist(train_df["label"].values, "Train", logger)

    feat = MolecularFeaturizer(config.SCAFFOLD_CSVs, seed=config.RANDOM_SEED)
    Xc, _, _, vi = feat.fit_transform(train_df["smiles"])
    y = train_df["label"].values[vi]
    Xce, _, _, vie = feat.transform(ext_df["smiles"])
    y_ext = ext_df["label"].values[vie]

    smiles_list = train_df["smiles"].iloc[vi].tolist()
    fold_records = []
    oof_proba = np.zeros(len(y))

    for fold, (tr, val) in enumerate(
        scaffold_kfold(smiles_list, y, config.CV_FOLDS, config.RANDOM_SEED), 1
    ):
        logger.info(f"\n  Fold {fold}/{config.CV_FOLDS}  train={len(tr)}  val={len(val)}")
        clf = _build()
        clf.fit(Xc[tr], y[tr])
        prob = clf.predict_proba(Xc[val])[:, 1]
        oof_proba[val] = prob
        met = compute_metrics(y[val], (prob >= config.THRESHOLD).astype(int), prob)
        fold_records.append(met)
        log_fold(fold, config.CV_FOLDS, met, logger)

    cv_agg = log_summary(fold_records, f"{MODEL_NAME} CV", logger)
    compare_to_paper(cv_agg, config.PAPER_INTERNAL, "Paper internal test", logger)

    logger.info("Training final model ...")
    final_clf = _build()
    final_clf.fit(Xc, y)

    ext_prob = final_clf.predict_proba(Xce)[:, 1]
    ext_pred = (ext_prob >= config.THRESHOLD).astype(int)
    ext_met  = compute_metrics(y_ext, ext_pred, ext_prob)
    unc = 0.5 - np.abs(ext_prob - 0.5)
    hc  = unc <= 0.2
    hc_met = compute_metrics(y_ext[hc], ext_pred[hc], ext_prob[hc]) if hc.sum() > 5 else {}

    log_summary([ext_met], f"{MODEL_NAME} External", logger)
    compare_to_paper({"auc":  {"mean": ext_met["auc"],   "std": 0},
                      "acc":  {"mean": ext_met["acc"],   "std": 0},
                      "recall":{"mean": ext_met["recall"],"std": 0},
                      "f1":   {"mean": ext_met["f1"],    "std": 0},
                      "kappa":{"mean": ext_met["kappa"], "std": 0}},
                     config.PAPER_EXTERNAL, "Paper external test", logger)

    runtime = time.time() - t0
    save_results(MODEL_NAME, cv_agg, ext_met, hc_met, runtime, oof_proba, ext_prob, y, y_ext)
    save_plots(MODEL_NAME, y, oof_proba, y_ext, ext_prob)
    logger.info(f"\n{MODEL_NAME} complete in {runtime/60:.1f} min")


if __name__ == "__main__":
    main()
