from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Callable

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from rdkit.Chem import Descriptors
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

KNN_NEIGHBORS = 5
LGBM_PARAMS = {
    "n_estimators": 300,
    "learning_rate": 0.05,
    "max_depth": 6,
    "num_leaves": 31,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
    "n_jobs": -1,
    "verbose": -1,
}


def _load_module(module_name: str, path: Path):
    # -------------------------------------------------------------------------
    # Dynamic Import of Legacy Training Utilities
    # The serializer reuses the exact feature-construction functions from the
    # archived confirmed-model scripts so live deployment artefacts stay aligned
    # with the research pipeline rather than silently drifting into a new
    # featurisation regime.
    # -------------------------------------------------------------------------
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _sha256(path: Path) -> str:
    # Checksums are written into the manifest so dissertation artefacts and
    # deployment bundles can be audited for provenance and integrity.
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _pick_smiles_column(frame: pd.DataFrame) -> str:
    # Historical CSV exports use small header variants for canonical SMILES.
    normalized = {col.strip().lower().replace("_", " "): col for col in frame.columns}
    for key in ("canonical smiles", "canonical_smiles", "smiles", "smiles clean"):
        if key in normalized:
            return normalized[key]
    raise ValueError("Could not find a SMILES column in the training dataset.")


def _feature_frame(
    smiles: list[str],
    *,
    fp_builder: Callable[[list[str], int, int], np.ndarray],
    descriptor_builder: Callable[[list[str]], np.ndarray],
    descriptor_names: list[str],
    radius: int,
    bits: int,
) -> pd.DataFrame:
    # -------------------------------------------------------------------------
    # Deployment-Time Feature Matrix Reconstruction
    # This helper rebuilds the fused descriptor + Morgan fingerprint design
    # matrix using the same RDKit-based builders as the archived experiments.
    # The feature names are preserved explicitly because runtime inference must
    # reproduce the exact training column order.
    # -------------------------------------------------------------------------
    fp = fp_builder(smiles, radius, bits)
    desc = descriptor_builder(smiles)
    fp_columns = {f"fp_{idx}": fp[:, idx].astype(float) for idx in range(fp.shape[1])}
    desc_columns = {name: desc[:, idx].astype(float) for idx, name in enumerate(descriptor_names)}
    return pd.DataFrame({**desc_columns, **fp_columns})


def _fit_stacking_bundle(
    frame: pd.DataFrame,
    y: np.ndarray,
    *,
    out_dir: Path,
    engine_name: str,
    display_name: str,
    n_splits: int = 5,
    knn_neighbors: int = KNN_NEIGHBORS,
) -> dict[str, Any]:
    # -------------------------------------------------------------------------
    # Portable Stacking Bundle
    # The runtime bundle mirrors the main ensemble architecture in a simplified
    # deployable form: median imputation, standardisation, a KNN base learner, a
    # LightGBM base learner, and a logistic meta-learner trained on out-of-fold
    # base probabilities. This preserves the ensemble-learning rationale while
    # producing self-contained artefacts that can be loaded quickly by the live
    # analysis API.
    #
    # NOTE:
    # This serializer exports deployment-ready approximations of the shortlisted
    # ensemble families rather than every historical artefact emitted by the
    # full research scripts.
    # -------------------------------------------------------------------------
    feature_order = frame.columns.tolist()
    X = frame.to_numpy(dtype=float)

    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    X_imputed = imputer.fit_transform(X)
    X_scaled = scaler.fit_transform(X_imputed)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    oof_knn = np.zeros(len(y), dtype=float)
    oof_lgbm = np.zeros(len(y), dtype=float)

    # Out-of-fold base predictions are used so the meta-learner observes
    # approximately unbiased inputs rather than in-sample base-model scores.
    for train_idx, val_idx in skf.split(X_scaled, y):
        xtr = X_scaled[train_idx]
        ytr = y[train_idx]
        xva = X_scaled[val_idx]

        knn = KNeighborsClassifier(n_neighbors=knn_neighbors, metric="minkowski", n_jobs=-1)
        knn.fit(xtr, ytr)
        oof_knn[val_idx] = knn.predict_proba(xva)[:, 1]

        lgbm = LGBMClassifier(**LGBM_PARAMS)
        lgbm.fit(xtr.astype(np.float32), ytr)
        oof_lgbm[val_idx] = lgbm.predict_proba(xva.astype(np.float32))[:, 1]

    meta_X = np.column_stack([oof_knn, oof_lgbm])
    meta_model = LogisticRegression(max_iter=2000, solver="lbfgs")
    meta_model.fit(meta_X, y)

    knn_model = KNeighborsClassifier(n_neighbors=knn_neighbors, metric="minkowski", n_jobs=-1)
    knn_model.fit(X_scaled, y)

    lgbm_model = LGBMClassifier(**LGBM_PARAMS)
    lgbm_model.fit(X_scaled.astype(np.float32), y)

    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(imputer, out_dir / "imputer.joblib")
    joblib.dump(scaler, out_dir / "scaler.joblib")
    joblib.dump(knn_model, out_dir / "knn_model.joblib")
    joblib.dump(lgbm_model, out_dir / "lgbm_model.joblib")
    joblib.dump(meta_model, out_dir / "meta_model.joblib")
    calibrator_path = out_dir / "calibrator.joblib"
    if calibrator_path.exists():
        calibrator_path.unlink()

    # The engine manifest records feature order, artefact names, and threshold
    # assumptions so the runtime loader can validate availability explicitly.
    manifest = {
        "engine_name": engine_name,
        "display_name": display_name,
        "feature_order": feature_order,
        "meta_feature_order": ["knn_model", "lgbm_model"],
        "threshold": 0.5,
        "knn_neighbors": knn_neighbors,
        "calibration": "none",
        "lgbm_params": LGBM_PARAMS,
        "artifacts": {
            "imputer": "imputer.joblib",
            "scaler": "scaler.joblib",
            "knn_model": "knn_model.joblib",
            "lgbm_model": "lgbm_model.joblib",
            "meta_model": "meta_model.joblib",
        },
        "training_rows": int(len(frame)),
        "training_features": int(len(feature_order)),
    }
    (out_dir / "engine.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main() -> int:
    # -------------------------------------------------------------------------
    # One-Time Deployment Artefact Generation
    # This script is intended to be run once in the full training environment.
    # It serialises the two shortlisted tabular ensemble families into the
    # portable `models/` directory together with a manifest and SHA-256 index.
    # -------------------------------------------------------------------------
    parser = argparse.ArgumentParser(description="Serialize live model bundles for KV6013 portable deployment.")
    parser.add_argument("--train-csv", required=True)
    parser.add_argument("--models-dir", required=True)
    parser.add_argument("--full-build-root", required=True)
    args = parser.parse_args()

    train_csv = Path(args.train_csv).expanduser().resolve()
    models_dir = Path(args.models_dir).expanduser().resolve()
    full_build_root = Path(args.full_build_root).expanduser().resolve()
    models_dir.mkdir(parents=True, exist_ok=True)

    descriptor_script = _load_module(
        "descriptor_training",
        full_build_root / "archived" / "legacy_project" / "confirmed_models" / "run_full_coverage_descriptor_fp_meta_v4.py",
    )
    improvements_script = _load_module(
        "improvements_training",
        full_build_root / "archived" / "legacy_project" / "confirmed_models" / "run_full_coverage_improvements.py",
    )

    train_df = pd.read_csv(train_csv)
    smiles_col = _pick_smiles_column(train_df)
    smiles = train_df[smiles_col].astype(str).tolist()
    y = train_df["label"].astype(int).to_numpy()

    # Curated descriptors correspond to the reduced descriptor subset used by
    # the descriptor-focused ensemble family.
    curated_descriptor_names = [
        "MolWt",
        "TPSA",
        "MolLogP",
        "NumHDonors",
        "NumHAcceptors",
        "NumRotatableBonds",
        "RingCount",
        "FractionCSP3",
        "HeavyAtomCount",
        "NHOHCount",
        "NOCount",
        "NumAliphaticRings",
        "NumAromaticRings",
        "NumSaturatedRings",
    ]
    # Full RDKit descriptor enumeration is retained for the broader coverage
    # ensemble, which intentionally searches a larger physicochemical space.
    rdkit_descriptor_names = [name for name, _ in Descriptors._descList]

    descriptor_frame = _feature_frame(
        smiles,
        fp_builder=descriptor_script.build_morgan_matrix,
        descriptor_builder=descriptor_script.build_curated_descriptors,
        descriptor_names=curated_descriptor_names,
        radius=2,
        bits=2048,
    )
    descriptor_manifest = _fit_stacking_bundle(
        descriptor_frame,
        y,
        out_dir=models_dir / "descriptor_fp_ensemble",
        engine_name="descriptor_fp_ensemble",
        display_name="Descriptor + Fingerprint Ensemble",
    )

    improvements_frame = _feature_frame(
        smiles,
        fp_builder=descriptor_script.build_morgan_matrix,
        descriptor_builder=improvements_script.build_descriptors,
        descriptor_names=rdkit_descriptor_names,
        radius=2,
        bits=2048,
    )
    improvements_manifest = _fit_stacking_bundle(
        improvements_frame,
        y,
        out_dir=models_dir / "full_coverage_ensemble",
        engine_name="full_coverage_ensemble",
        display_name="Full Coverage Ensemble",
    )

    artifact_index: list[dict[str, Any]] = []
    for path in sorted(models_dir.rglob("*")):
        if path.is_file():
            artifact_index.append(
                {
                    "path": str(path.relative_to(models_dir)),
                    "sha256": _sha256(path),
                    "size_bytes": path.stat().st_size,
                }
            )

    manifest = {
        "generated_at": pd.Timestamp.now("UTC").isoformat(),
        "train_csv": str(train_csv),
        "full_build_root": str(full_build_root),
        "engines": [descriptor_manifest, improvements_manifest],
        "artifacts": artifact_index,
    }
    (models_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
