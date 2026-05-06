from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import optuna
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from utils.metrics import (
    compute_binary_classification_metrics,
    find_optimal_threshold,
    coverage_performance,
    calibration_scores,
)
from utils.logger import get_logger

logger = get_logger(__name__)


def _to_builtin(obj: Any) -> Any:
    """Convert numpy/pandas objects to JSON-serializable builtins."""
    if isinstance(obj, dict):
        return {k: _to_builtin(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_builtin(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_to_builtin(v) for v in obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


@dataclass
class TrainedModel:
    model: CatBoostClassifier
    calibrator: Optional["ProbabilityCalibrator"]
    threshold: float
    calibration_method: str

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        raw_prob = self.model.predict_proba(X)[:, 1]
        if self.calibrator is not None:
            raw_prob = self.calibrator.transform(raw_prob)
        return np.column_stack([1.0 - raw_prob, raw_prob])


@dataclass
class ProbabilityCalibrator:
    """Lightweight probability calibrator fit on validation predictions only."""

    method: str
    model: Any

    @staticmethod
    def _to_logits(prob_pos: np.ndarray) -> np.ndarray:
        prob_pos = np.clip(prob_pos, 1e-6, 1.0 - 1e-6)
        return np.log(prob_pos / (1.0 - prob_pos)).reshape(-1, 1)

    @classmethod
    def fit(cls, method: str, y_true: np.ndarray, prob_pos: np.ndarray) -> "ProbabilityCalibrator":
        if method == "sigmoid":
            lr = LogisticRegression(solver="lbfgs", max_iter=1000)
            lr.fit(cls._to_logits(prob_pos), y_true)
            return cls(method=method, model=lr)
        if method == "isotonic":
            iso = IsotonicRegression(out_of_bounds="clip")
            iso.fit(prob_pos, y_true)
            return cls(method=method, model=iso)
        raise ValueError(f"Unsupported calibration method: {method}")

    def transform(self, prob_pos: np.ndarray) -> np.ndarray:
        prob_pos = np.asarray(prob_pos, dtype=float)
        if self.method == "sigmoid":
            calibrated = self.model.predict_proba(self._to_logits(prob_pos))[:, 1]
        elif self.method == "isotonic":
            calibrated = self.model.predict(prob_pos)
        else:
            raise ValueError(f"Unsupported calibration method: {self.method}")
        return np.clip(calibrated, 1e-6, 1.0 - 1e-6)


class CatBoostPipeline:
    def __init__(
        self,
        feature_names: List[str],
        model_config: Dict[str, Any],
        thresholds_config: Dict[str, Any],
        coverage_thresholds: List[float],
        class_weights: Optional[List[float]] = None,
    ):
        self.feature_names = feature_names
        self.model_config = model_config
        self.thresholds_config = thresholds_config
        self.coverage_thresholds = coverage_thresholds
        self.class_weights = class_weights

    def _suggest_params(self, trial: optuna.trial.Trial) -> Dict[str, Any]:
        depth = trial.suggest_int("depth", int(self.model_config["depth_range"][0]), int(self.model_config["depth_range"][1]))
        learning_rate = trial.suggest_float(
            "learning_rate",
            float(self.model_config["learning_rate_loguniform"][0]),
            float(self.model_config["learning_rate_loguniform"][1]),
            log=True,
        )
        l2_leaf_reg = trial.suggest_float("l2_leaf_reg", float(self.model_config["l2_leaf_reg"][0]), float(self.model_config["l2_leaf_reg"][1]))
        subsample = trial.suggest_float("subsample", float(self.model_config["subsample"][0]), float(self.model_config["subsample"][1]))
        random_strength = trial.suggest_float("random_strength", float(self.model_config["random_strength"][0]), float(self.model_config["random_strength"][1]))
        bagging_temperature = trial.suggest_float("bagging_temperature", float(self.model_config["bagging_temperature"][0]), float(self.model_config["bagging_temperature"][1]))

        return {
            "depth": depth,
            "learning_rate": learning_rate,
            "l2_leaf_reg": l2_leaf_reg,
            "subsample": subsample,
            "random_strength": random_strength,
            "bagging_temperature": bagging_temperature,
        }

    def tune(self, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray) -> Dict[str, Any]:
        def objective(trial: optuna.trial.Trial) -> float:
            params = self._suggest_params(trial)
            model = CatBoostClassifier(
                iterations=int(self.model_config.get("iterations", 800)),
                eval_metric=self.model_config.get("eval_metric", "AUC"),
                loss_function=self.model_config.get("loss_function", "Logloss"),
                early_stopping_rounds=int(self.model_config.get("early_stopping_rounds", 50)),
                verbose=False,
                task_type="CPU",
                class_weights=self.class_weights if self.model_config.get("use_class_weights", True) else None,
                **params,
            )
            model.fit(X_train, y_train, eval_set=(X_val, y_val))
            proba = model.predict_proba(X_val)[:, 1]
            auc = compute_binary_classification_metrics(y_val, proba, 0.5)["auroc"]
            return auc

        study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42), pruner=optuna.pruners.MedianPruner())
        n_trials = int(self.model_config.get("n_trials", 20))
        timeout = int(self.model_config.get("timeout_minutes", 10) * 60)
        study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=False)
        best_params = study.best_params
        logger.info("Optuna best params: %s", best_params)
        return best_params

    def train_best_model(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        tuned_params: Dict[str, Any],
    ) -> CatBoostClassifier:
        params = tuned_params.copy()
        model = CatBoostClassifier(
            iterations=int(self.model_config.get("iterations", 800)),
            eval_metric=self.model_config.get("eval_metric", "AUC"),
            loss_function=self.model_config.get("loss_function", "Logloss"),
            early_stopping_rounds=int(self.model_config.get("early_stopping_rounds", 50)),
            verbose=100,
            task_type="CPU",
            class_weights=self.class_weights if self.model_config.get("use_class_weights", True) else None,
            **params,
        )
        model.fit(X_train, y_train, eval_set=(X_val, y_val))
        return model

    def calibrate(
        self,
        base_model: CatBoostClassifier,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> Tuple[Optional[ProbabilityCalibrator], str, Dict[str, Dict[str, float]]]:
        """Try none/sigmoid/isotonic; return best calibrator, method, and per-method scores."""
        methods = ["none", "sigmoid", "isotonic"]
        results: Dict[str, Dict[str, float]] = {}
        best_method = "none"
        best_score = -np.inf
        best_brier = np.inf
        best_calibrator: Optional[ProbabilityCalibrator] = None
        base_prob = base_model.predict_proba(X_val)[:, 1]

        for method in methods:
            if method == "none":
                proba = base_prob
                calibrator = None
            else:
                calibrator = ProbabilityCalibrator.fit(method=method, y_true=y_val, prob_pos=base_prob)
                proba = calibrator.transform(base_prob)

            scores = calibration_scores(y_val, proba)
            results[method] = scores

            # select by AUROC, tie-break by lower Brier
            if (scores["auroc"] > best_score) or (
                np.isclose(scores["auroc"], best_score) and scores["brier"] < best_brier
            ):
                best_score = scores["auroc"]
                best_brier = scores["brier"]
                best_method = method
                best_calibrator = calibrator

        logger.info("Selected calibration: %s (val AUROC=%.4f, Brier=%.4f)", best_method, best_score, best_brier)
        return best_calibrator, best_method, results

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> Tuple[TrainedModel, Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Dict[str, float]]]:
        start = time.time()
        tuned_params = self.tune(X_train, y_train, X_val, y_val)
        model = self.train_best_model(X_train, y_train, X_val, y_val, tuned_params)
        calibrator, cal_method, calibration_results = self.calibrate(model, X_val, y_val)

        trained = TrainedModel(model=model, calibrator=calibrator, threshold=0.5, calibration_method=cal_method)
        proba_val = trained.predict_proba(X_val)[:, 1]
        grid = np.arange(
            float(self.thresholds_config.get("grid_start", 0.1)),
            float(self.thresholds_config.get("grid_end", 0.9)) + 1e-9,
            float(self.thresholds_config.get("grid_step", 0.01)),
        )
        threshold, _ = find_optimal_threshold(y_val, proba_val, grid)

        trained.threshold = threshold
        proba_test = trained.predict_proba(X_test)[:, 1]
        test_metrics = compute_binary_classification_metrics(y_test, proba_test, threshold)
        coverage_df = coverage_performance(y_test, proba_test, threshold, self.coverage_thresholds)
        elapsed = time.time() - start
        logger.info("Training + eval finished in %.1f s", elapsed)
        return trained, tuned_params, test_metrics, coverage_df.to_dict(orient="records"), calibration_results

    @staticmethod
    def save_artifacts(
        run_dir: str,
        trained: TrainedModel,
        featurizer: Any,
        feature_names: List[str],
        tuned_params: Dict[str, Any],
        metrics_dict: Dict[str, Any],
        coverage_records: List[Dict[str, Any]],
        val_threshold: float,
        calibration_method: str,
        calibration_results: Dict[str, Dict[str, float]],
        external_metrics: Optional[Dict[str, Any]] = None,
        external_coverage: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        os.makedirs(run_dir, exist_ok=True)
        joblib.dump(
            {
                "model": trained.model,
                "calibrator": trained.calibrator,
                "threshold": trained.threshold,
                "featurizer": featurizer,
                "feature_names": feature_names,
                "calibration_method": calibration_method,
            },
            os.path.join(run_dir, "model.pkl"),
        )
        with open(os.path.join(run_dir, "params.json"), "w") as f:
            json.dump(_to_builtin(tuned_params), f, indent=2)
        with open(os.path.join(run_dir, "metrics_internal.json"), "w") as f:
            json.dump(
                {
                    "metrics": _to_builtin(metrics_dict),
                    "threshold": _to_builtin(val_threshold),
                    "calibration": calibration_method,
                    "calibration_results": _to_builtin(calibration_results),
                },
                f,
                indent=2,
            )
        with open(os.path.join(run_dir, "coverage_internal.json"), "w") as f:
            json.dump(_to_builtin(coverage_records), f, indent=2)
        # calibration table CSV
        calib_df = pd.DataFrame(
            [
                {
                    "method": m,
                    "auroc": vals.get("auroc"),
                    "brier": vals.get("brier"),
                    "logloss": vals.get("logloss"),
                    "selected": m == calibration_method,
                }
                for m, vals in calibration_results.items()
            ]
        )
        calib_df.to_csv(os.path.join(run_dir, "calibration_results.csv"), index=False)
        if external_metrics is not None:
            with open(os.path.join(run_dir, "metrics_external.json"), "w") as f:
                json.dump(_to_builtin(external_metrics), f, indent=2)
        if external_coverage is not None:
            with open(os.path.join(run_dir, "coverage_external.json"), "w") as f:
                json.dump(_to_builtin(external_coverage), f, indent=2)

    def feature_importances(self, model: CatBoostClassifier) -> List[Tuple[str, float]]:
        importances = model.get_feature_importance(type="FeatureImportance")
        return list(zip(self.feature_names, importances))
