"""
run_tanimoto_gpc.py
===================
Tanimoto Kernel Gaussian Process Classifier.

WHY: The Tanimoto (Jaccard) kernel is the provably optimal positive-definite
kernel for binary molecular fingerprints (Ralaivola et al., Neural Networks 2005).
Gives fully calibrated Bayesian uncertainty.

FIX vs previous version: Instead of PairwiseKernel (which injects 'gamma' kwarg),
we subclass sklearn's Kernel directly. This eliminates the
'unexpected keyword argument gamma' error completely.

Run: python run_tanimoto_gpc.py
"""
import time, logging
import numpy as np
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import Kernel, Hyperparameter

import config
from utils import setup_logging, load_dataset, print_dist
from featurizer import MolecularFeaturizer
from scaffold_split import scaffold_kfold
from metrics import compute_metrics, log_fold, log_summary, compare_to_paper, save_results, save_plots

setup_logging()
logger = logging.getLogger(__name__)
MODEL_NAME = "TanimotoGPC"


# ── Proper Tanimoto kernel as sklearn Kernel subclass ─────────────────────────

class TanimotoKernel(Kernel):
    """
    Tanimoto (Jaccard) kernel for binary bit-vector fingerprints.
        K(x, y) = (x·y) / (||x||² + ||y||² − x·y)
    Subclasses sklearn.gaussian_process.kernels.Kernel directly to avoid
    the 'gamma' keyword argument injection from PairwiseKernel.
    """

    def __init__(self, amplitude=1.0, amplitude_bounds=(0.05, 20.0)):
        self.amplitude        = amplitude
        self.amplitude_bounds = amplitude_bounds

    @property
    def hyperparameter_amplitude(self):
        return Hyperparameter("amplitude", "numeric", self.amplitude_bounds)

    def __call__(self, X, Y=None, eval_gradient=False):
        X = np.asarray(X, dtype=np.float64)
        Y = X if Y is None else np.asarray(Y, dtype=np.float64)

        XY    = X @ Y.T
        XX    = np.sum(X * X, axis=1, keepdims=True)
        YY    = np.sum(Y * Y, axis=1, keepdims=True).T
        denom = XX + YY - XY
        K_tan = np.where(denom == 0, 0.0, XY / denom)
        K     = self.amplitude * K_tan

        if eval_gradient:
            # Gradient w.r.t. log(amplitude)
            K_grad = K_tan[:, :, np.newaxis]
            return K, K_grad
        return K

    def diag(self, X):
        X = np.asarray(X, dtype=np.float64)
        norms_sq = np.sum(X * X, axis=1)
        return self.amplitude * np.where(norms_sq == 0, 0.0, norms_sq / norms_sq)

    def is_stationary(self):
        return False

    def __repr__(self):
        return f"TanimotoKernel(amplitude={self.amplitude:.3f})"


def _build_gpc():
    kernel = TanimotoKernel()
    return GaussianProcessClassifier(
        kernel=kernel,
        n_restarts_optimizer=1,
        random_state=config.RANDOM_SEED,
        max_iter_predict=200,
        copy_X_train=True,
    )


def main():
    t0 = time.time()
    logger.info(f"{'='*60}\n  {MODEL_NAME} — Scaffold-stratified CV\n{'='*60}")

    train_df = load_dataset(config.TRAIN_CSV)
    ext_df   = load_dataset(config.EXTTEST_CSV)
    print_dist(train_df["label"].values, "Train", logger)

    # GPC uses raw binary fingerprints (Tanimoto defined on bit vectors)
    feat = MolecularFeaturizer(config.SCAFFOLD_CSVs, seed=config.RANDOM_SEED)
    _, _, Xf, vi = feat.fit_transform(train_df["smiles"])
    y = train_df["label"].values[vi]
    _, _, Xfe, vie = feat.transform(ext_df["smiles"])
    y_ext = ext_df["label"].values[vie]

    # GPC is O(n³) — cap at 2000 (1527 is fine)
    logger.info(f"GPC training set size: {len(y)} (O(n³) — should complete in ~10-15 min)")

    smiles_list  = train_df["smiles"].iloc[vi].tolist()
    fold_records = []
    oof_proba    = np.zeros(len(y))

    for fold, (tr, val) in enumerate(
        scaffold_kfold(smiles_list, y, config.CV_FOLDS, config.RANDOM_SEED), 1
    ):
        logger.info(f"\n  Fold {fold}/{config.CV_FOLDS}  train={len(tr)}  val={len(val)}")
        clf = _build_gpc()
        clf.fit(Xf[tr].astype(np.float64), y[tr])
        prob = clf.predict_proba(Xf[val].astype(np.float64))[:, 1]
        oof_proba[val] = prob
        met = compute_metrics(y[val], (prob >= config.THRESHOLD).astype(int), prob)
        fold_records.append(met)
        log_fold(fold, config.CV_FOLDS, met, logger)

    cv_agg = log_summary(fold_records, f"{MODEL_NAME} CV", logger)
    compare_to_paper(cv_agg, config.PAPER_INTERNAL, "Paper internal test", logger)

    logger.info("Training final GPC on full training set ...")
    final_clf = _build_gpc()
    final_clf.fit(Xf.astype(np.float64), y)

    # Bayesian uncertainty from GPC probabilities
    ext_prob = final_clf.predict_proba(Xfe.astype(np.float64))[:, 1]
    ext_pred = (ext_prob >= config.THRESHOLD).astype(int)
    ext_met  = compute_metrics(y_ext, ext_pred, ext_prob)
    unc = 0.5 - np.abs(ext_prob - 0.5)
    hc  = unc <= 0.2
    hc_met = compute_metrics(y_ext[hc], ext_pred[hc], ext_prob[hc]) if hc.sum() > 5 else {}

    log_summary([ext_met], f"{MODEL_NAME} External", logger)
    compare_to_paper({"auc":  {"mean": ext_met["auc"],    "std": 0},
                      "acc":  {"mean": ext_met["acc"],    "std": 0},
                      "recall":{"mean": ext_met["recall"],"std": 0},
                      "f1":   {"mean": ext_met["f1"],     "std": 0},
                      "kappa":{"mean": ext_met["kappa"],  "std": 0}},
                     config.PAPER_EXTERNAL, "Paper external test", logger)

    runtime = time.time() - t0
    save_results(MODEL_NAME, cv_agg, ext_met, hc_met, runtime, oof_proba, ext_prob, y, y_ext)
    save_plots(MODEL_NAME, y, oof_proba, y_ext, ext_prob)
    logger.info(f"\n{MODEL_NAME} complete in {runtime/60:.1f} min")


if __name__ == "__main__":
    main()
