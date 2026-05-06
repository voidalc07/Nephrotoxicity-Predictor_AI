"""
model.py
--------
NephrotoxEnsemble — three-model stacking ensemble.

Base learners
  1. LightGBM (Microsoft Research) — leaf-wise boosting, Optuna-tuned
  2. HistGradientBoostingClassifier (sklearn) — different regularisation/
     binning from LightGBM; ships with scikit-learn, no auth required.
  3. Tanimoto Kernel GPC — exact Bayesian, optimal kernel for binary FPs

Change from v1
  TabPFN v2.5 is a gated HuggingFace model requiring account login.
  HistGradientBoostingClassifier is a strong, diverse replacement that is
  always available, runs instantly, and provides complementary inductive bias.

Meta-learner
  Logistic Regression trained on out-of-fold probabilities from all three.
"""

import logging
import warnings
from typing import Optional, Tuple

import numpy as np
import joblib

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tanimoto kernel
# ---------------------------------------------------------------------------

def tanimoto_kernel(X: np.ndarray, Y: Optional[np.ndarray] = None) -> np.ndarray:
    if Y is None:
        Y = X
    X = X.astype(np.float64)
    Y = Y.astype(np.float64)
    XY    = X @ Y.T
    XX    = np.sum(X * X, axis=1, keepdims=True)
    YY    = np.sum(Y * Y, axis=1, keepdims=True).T
    denom = XX + YY - XY
    return np.where(denom == 0, 0.0, XY / denom)


# ---------------------------------------------------------------------------
# Model builders
# ---------------------------------------------------------------------------

def _build_lgbm(seed: int = 42):
    try:
        from lightgbm import LGBMClassifier
    except ImportError:
        raise ImportError("pip install lightgbm")
    return LGBMClassifier(
        n_estimators=500, learning_rate=0.05, num_leaves=63,
        max_depth=-1, min_child_samples=10,
        subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=0.1,
        class_weight="balanced",
        random_state=seed, verbose=-1, n_jobs=-1,
    )


def _tune_lgbm(X: np.ndarray, y: np.ndarray,
               seed: int = 42, n_trials: int = 50):
    """Optuna TPE HPO for LightGBM. Uses pure numpy arrays throughout to
    silence the 'X does not have valid feature names' sklearn warning."""
    try:
        import optuna
        from lightgbm import LGBMClassifier
        from sklearn.model_selection import StratifiedKFold, cross_val_score
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        raise ImportError("pip install optuna lightgbm")

    X_np = np.asarray(X, dtype=np.float32)  # pure numpy -> no feature-name warning

    def objective(trial):
        p = {
            "n_estimators":      trial.suggest_int("n_estimators", 200, 1200),
            "learning_rate":     trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
            "num_leaves":        trial.suggest_int("num_leaves", 15, 255),
            "max_depth":         trial.suggest_int("max_depth", 3, 14),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 60),
            "subsample":         trial.suggest_float("subsample", 0.4, 1.0),
            "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.3, 1.0),
            "reg_alpha":         trial.suggest_float("reg_alpha", 1e-5, 10.0, log=True),
            "reg_lambda":        trial.suggest_float("reg_lambda", 1e-5, 10.0, log=True),
            "min_split_gain":    trial.suggest_float("min_split_gain", 0.0, 1.0),
            "class_weight": "balanced",
            "random_state": seed, "verbose": -1, "n_jobs": -1,
        }
        clf = LGBMClassifier(**p)
        cv  = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            sc = cross_val_score(clf, X_np, y, cv=cv, scoring="roc_auc", n_jobs=1)
        return sc.mean()

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=seed),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    logger.info(f"LightGBM best CV-AUC: {study.best_value:.4f}")
    logger.info(f"Best params: {study.best_params}")

    from lightgbm import LGBMClassifier
    return LGBMClassifier(**study.best_params, class_weight="balanced",
                          verbose=-1, n_jobs=-1)


def _build_hist_gb(seed: int = 42) -> HistGradientBoostingClassifier:
    """
    HistGradientBoostingClassifier — second base learner.

    Why it complements LightGBM:
      • Different binning: up to 255 bins vs LightGBM default 63
      • Only L2 regularisation (no L1), different bias-variance trade-off
      • Native missing-value handling (no imputer needed)
      • sklearn's early_stopping monitors a held-out validation fraction
        internally, avoiding overfitting without HPO
    These structural differences ensure the two models disagree on
    different compounds, making their combination via stacking meaningful.
    """
    return HistGradientBoostingClassifier(
        max_iter=600,
        learning_rate=0.05,
        max_leaf_nodes=63,
        max_depth=None,
        min_samples_leaf=20,
        l2_regularization=0.1,
        max_bins=255,
        class_weight="balanced",
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=20,
        random_state=seed,
    )


def _build_gpc(seed: int = 42):
    from sklearn.gaussian_process import GaussianProcessClassifier
    from sklearn.gaussian_process.kernels import PairwiseKernel, ConstantKernel as C
    kernel = C(1.0, (0.05, 20.0)) * PairwiseKernel(metric=tanimoto_kernel)
    return GaussianProcessClassifier(
        kernel=kernel, n_restarts_optimizer=0,
        random_state=seed, max_iter_predict=200, copy_X_train=True,
    )


# ---------------------------------------------------------------------------
# Stacking ensemble
# ---------------------------------------------------------------------------

class NephrotoxEnsemble:
    """
    Three-model stacking ensemble with Logistic Regression meta-learner.

    Base learners : LightGBM + HistGradientBoosting + Tanimoto GPC
    Meta-learner  : Logistic Regression on out-of-fold probabilities

    Parameters
    ----------
    tune_lgbm   : run Optuna HPO on LightGBM
    lgbm_trials : number of Optuna trials (default 50)
    device      : unused, kept for API compatibility
    gpc_max_n   : skip GPC when n_train > this (O(n³))
    seed        : random seed
    """

    def __init__(
        self,
        tune_lgbm:   bool = True,
        lgbm_trials: int  = 50,
        device:      str  = "cpu",
        gpc_max_n:   int  = 2000,
        seed:        int  = 42,
    ):
        self.tune_lgbm   = tune_lgbm
        self.lgbm_trials = lgbm_trials
        self.device      = device
        self.gpc_max_n   = gpc_max_n
        self.seed        = seed

        self._lgbm           = None
        self._hist_gb        = _build_hist_gb(seed)
        self._gpc            = _build_gpc(seed)

        self._lgbm_fitted    = False
        self._hist_gb_fitted = False
        self._gpc_fitted     = False

        self._meta: LogisticRegression = LogisticRegression(
            C=1.0, max_iter=1000, random_state=seed
        )
        self._meta_fitted = False
        self._meta_cols: list = []

    # ── Fit ───────────────────────────────────────────────────────────────────

    def fit(
        self,
        X_combined: np.ndarray,
        X_tabpfn:   np.ndarray,   # kept for API compat; HistGB uses X_combined
        X_fp:       np.ndarray,
        y:          np.ndarray,
    ) -> "NephrotoxEnsemble":

        n   = len(y)
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.seed)
        oof: dict = {}

        # Convert to pure numpy once — eliminates LightGBM feature-name warnings
        Xc = np.asarray(X_combined, dtype=np.float32)
        Xf = np.asarray(X_fp,       dtype=np.float32)

        # ── 1. LightGBM ──────────────────────────────────────────────────────
        logger.info("─" * 52)
        logger.info(f"[1/3] LightGBM  n={n}  tune={self.tune_lgbm}  trials={self.lgbm_trials}")
        self._lgbm = (
            _tune_lgbm(Xc, y, self.seed, self.lgbm_trials)
            if self.tune_lgbm else _build_lgbm(self.seed)
        )
        oof_lgbm = np.zeros(n)
        for tr, val in skf.split(Xc, y):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                self._lgbm.fit(Xc[tr], y[tr])
                oof_lgbm[val] = self._lgbm.predict_proba(Xc[val])[:, 1]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            self._lgbm.fit(Xc, y)
        self._lgbm_fitted = True
        oof["lgbm"] = oof_lgbm
        logger.info(f"  LightGBM OOF-AUC: {_auc(y, oof_lgbm):.4f}")

        # ── 2. HistGradientBoosting ───────────────────────────────────────────
        logger.info("[2/3] HistGradientBoosting (sklearn) ...")
        oof_hgb = np.zeros(n)
        for tr, val in skf.split(Xc, y):
            self._hist_gb.fit(Xc[tr], y[tr])
            oof_hgb[val] = self._hist_gb.predict_proba(Xc[val])[:, 1]
        self._hist_gb.fit(Xc, y)
        self._hist_gb_fitted = True
        oof["hist_gb"] = oof_hgb
        logger.info(f"  HistGB OOF-AUC: {_auc(y, oof_hgb):.4f}")

        # ── 3. Tanimoto GPC ──────────────────────────────────────────────────
        if n > self.gpc_max_n:
            logger.warning(
                f"[3/3] Tanimoto GPC skipped (n={n} > {self.gpc_max_n})."
            )
        else:
            logger.info(f"[3/3] Tanimoto GPC  n={n}")
            try:
                oof_gpc = np.zeros(n)
                for tr, val in skf.split(Xf, y):
                    self._gpc.fit(Xf[tr].astype(np.float64), y[tr])
                    oof_gpc[val] = self._gpc.predict_proba(
                        Xf[val].astype(np.float64))[:, 1]
                self._gpc.fit(Xf.astype(np.float64), y)
                self._gpc_fitted = True
                oof["gpc"] = oof_gpc
                logger.info(f"  GPC OOF-AUC: {_auc(y, oof_gpc):.4f}")
            except Exception as e:
                logger.warning(f"  GPC failed: {e}. Continuing without it.")

        # ── 4. Meta-learner ───────────────────────────────────────────────────
        logger.info("[4/4] Stacking meta-learner (Logistic Regression)")
        self._meta_cols = list(oof.keys())
        oof_mat = np.column_stack([oof[c] for c in self._meta_cols])
        self._meta.fit(oof_mat, y)
        self._meta_fitted = True
        coeff_str = "  ".join(
            f"{c}={v:.3f}" for c, v in zip(self._meta_cols, self._meta.coef_[0])
        )
        logger.info(f"  Coefficients: {coeff_str}")
        oof_ensemble = self._meta.predict_proba(oof_mat)[:, 1]
        logger.info(f"  Ensemble OOF-AUC: {_auc(y, oof_ensemble):.4f}")
        logger.info("─" * 52)
        return self

    # ── Predict ───────────────────────────────────────────────────────────────

    def _base_probas(self, X_combined, X_tabpfn, X_fp) -> np.ndarray:
        Xc = np.asarray(X_combined, dtype=np.float32)
        Xf = np.asarray(X_fp,       dtype=np.float32)
        cols = []
        if "lgbm" in self._meta_cols and self._lgbm_fitted:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                cols.append(self._lgbm.predict_proba(Xc)[:, 1])
        if "hist_gb" in self._meta_cols and self._hist_gb_fitted:
            cols.append(self._hist_gb.predict_proba(Xc)[:, 1])
        if "gpc" in self._meta_cols and self._gpc_fitted:
            cols.append(self._gpc.predict_proba(Xf.astype(np.float64))[:, 1])
        if not cols:
            raise RuntimeError("No fitted base model found.")
        return np.column_stack(cols)

    def predict_proba(self, X_combined, X_tabpfn, X_fp) -> np.ndarray:
        if not self._meta_fitted:
            raise RuntimeError("Call fit() first.")
        p = self._meta.predict_proba(
            self._base_probas(X_combined, X_tabpfn, X_fp))[:, 1]
        return np.column_stack([1 - p, p]).astype(np.float32)

    def predict(self, X_combined, X_tabpfn, X_fp,
                threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X_combined, X_tabpfn, X_fp)[:, 1]
                >= threshold).astype(int)

    def predict_with_uncertainty(
        self, X_combined, X_tabpfn, X_fp
    ) -> Tuple[np.ndarray, np.ndarray]:
        proba = self.predict_proba(X_combined, X_tabpfn, X_fp)
        unc   = 0.5 - np.abs(proba[:, 1] - 0.5)
        return proba, unc

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path) -> None:
        joblib.dump(self, path)
        logger.info(f"Model saved → {path}")

    @classmethod
    def load(cls, path) -> "NephrotoxEnsemble":
        model = joblib.load(path)
        logger.info(f"Model loaded ← {path}")
        return model


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _auc(y_true, y_prob) -> float:
    from sklearn.metrics import roc_auc_score
    try:
        return float(roc_auc_score(y_true, y_prob))
    except Exception:
        return float("nan")
