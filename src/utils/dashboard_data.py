from __future__ import annotations

import math
import os
import re
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from src.config.settings import DEFAULT_EXTERNAL_CSV, DEPLOYMENT_MODE, DETAILED_PREDICTIONS_DIR, FINAL_REPORTS_DIR, PROJECT_ROOT
from src.utils.io import safe_read_csv

try:
    from rdkit import Chem, DataStructs
    from rdkit import RDLogger
    from rdkit.Chem import AllChem

    RDLogger.DisableLog("rdApp.*")
except Exception:  # pragma: no cover
    Chem = None
    DataStructs = None
    AllChem = None


LOCAL_SUMMARY_CSV = FINAL_REPORTS_DIR / "overall_evaluation.csv"
LOCAL_PREDICTIONS_CSV = DETAILED_PREDICTIONS_DIR / "all_model_predictions.csv"

MAIN_MODEL_ORDER = [
    "full_coverage_improvements",
    "full_coverage_descriptor_fp_meta_v4",
    "nephro_unsupervised",
    "nephro_chemberta",
    "modular_tanimoto_gpc",
]

MODEL_LABELS = {
    "full_coverage_improvements": "Full Coverage Ensemble",
    "full_coverage_descriptor_fp_meta_v4": "Descriptor + Fingerprint Ensemble",
    "nephro_unsupervised": "Autoencoder Model",
    "nephro_chemberta": "ChemBERTa Hybrid",
    "modular_tanimoto_gpc": "Similarity Model",
}

VARIANT_LABELS = {
    "meta_logreg_knn_plus_lgbm__cal_none__thr_f1": "KNN + LightGBM Ensemble",
    "challenger_proxy_meta_logreg_knn_plus_lgbm__cal_none__thr_f1": "Descriptor-Fingerprint Ensemble",
    "Autoencoder": "Autoencoder",
    "chemberta_hybrid_cb_lgbm": "ChemBERTa + CatBoost + LightGBM",
    "TanimotoGPC": "Tanimoto Gaussian Process",
}

MEDICINE_MAPPING = {
    "cyclosporine": "C/C=C/C[C@@H](C)[C@@H](O)[C@@H]1C(=O)N[C@@H](CC)C(=O)N(C)CC(=O)N(C)[C@@H](CC(C)C)C(=O)N[C@@H](C(C)C)C(=O)N(C)[C@@H](CC(C)C)C(=O)N[C@@H](C)C(=O)N[C@@H](C)C(=O)N(C)[C@@H](CC(C)C)C(=O)N(C)[C@@H](CC(C)C)C(=O)N(C)[C@@H](C(C)C)C(=O)N1C",
    "ibuprofen": "CC(C)Cc1ccc(cc1)[C@@H](C)C(=O)O",
    "acyclovir": "Nc1nc2n(COCCOC(=O)CCCCCCC/C=C\\CCCCCCC)c(O)nc2[nH]1",
    "vancomycin": "C[C@H]1O[C@H](O[C@@H]2[C@@H](O)[C@H](O[C@H]3[C@H]4NC(=O)[C@H](NC(=O)[C@@H](NC(=O)[C@H]5NC(=O)[C@H](NC(=O)[C@@H](NC4=O)c6cc(Cl)c(O)c(c6)Oc7cc5cc(c7O)Oc8ccc(cc8Cl)[C@@H](O)[C@@H](NC(=O)CN)C(=O)N3)c9ccc(O)cc9)c2cc(O)cc2)c1ccc(O)c1)C(=O)O)[C@H](O)[C@@H](O)[C@H]1N",
    "paracetamol": "CC(=O)Nc1ccc(O)cc1",
    "gentamicin": "CN[C@@H]1[C@@H](O)[C@@H](O[C@H]2[C@@H](N)C[C@@H](N)[C@H](O[C@H]3[C@H](N)CC[C@H](N)[C@@H]3O)[C@@H]2O)OC[C@]1(C)O",
    "cisplatin": "N.N.Cl[Pt]Cl",
}

GENERIC_SAMPLE_RE = re.compile(r"^external_(\d+)$", re.IGNORECASE)
ANNOTATION_RE = re.compile(r"\s+\|.*\|$")

_CACHE: dict[str, Any] = {
    "key": None,
    "state": None,
}


def _report_candidates() -> list[tuple[Path, Path]]:
    # -------------------------------------------------------------------------
    # Evidence Source Prioritisation
    # The dashboard prefers dissertation-grade benchmark exports when they are
    # available, but can fall back to locally regenerated portable outputs.
    # This makes the interface robust while still privileging the archived
    # external-evaluation artefacts that underpin the reported results.
    # -------------------------------------------------------------------------
    roots = [
        PROJECT_ROOT.parent / "FINAL_ KV6013_NEPHROTOXICITY_PREDICTOR",
        PROJECT_ROOT.parent / "KV6013-Python-Project",
        PROJECT_ROOT,
    ]
    return [
        (root / "outputs" / "final_reports" / "overall_evaluation.csv", root / "outputs" / "detailed_predictions" / "all_model_predictions.csv")
        for root in roots
    ]


def _summary_is_complete(summary_df: pd.DataFrame) -> bool:
    # Require a complete external AUROC row for every main engine before
    # treating a report bundle as authoritative for analytics rendering.
    if summary_df.empty or "model_name" not in summary_df.columns or "dataset" not in summary_df.columns:
        return False
    for model_name in MAIN_MODEL_ORDER:
        model_rows = summary_df[summary_df["model_name"] == model_name]
        if model_rows.empty:
            return False
        external_rows = model_rows[model_rows["dataset"] == "external"]
        if external_rows.empty:
            return False
        if _to_float(external_rows.iloc[0].get("roc_auc")) is None:
            return False
    return True


def _merge_summary_frames(primary_df: pd.DataFrame, supplemental_df: pd.DataFrame) -> pd.DataFrame:
    # Merge without overwriting the primary benchmark rows so regenerated
    # portable results only fill gaps rather than silently replacing the
    # archived evidence used for dissertation reporting.
    if primary_df.empty:
        return supplemental_df.copy()
    if supplemental_df.empty:
        return primary_df.copy()

    key_columns = ["model_name", "variant", "dataset"]
    existing_keys = {
        tuple("" if pd.isna(row[col]) else str(row[col]) for col in key_columns)
        for _, row in primary_df.iterrows()
    }
    extra_rows = []
    for _, row in supplemental_df.iterrows():
        key = tuple("" if pd.isna(row[col]) else str(row[col]) for col in key_columns)
        if key not in existing_keys:
            extra_rows.append(row.to_dict())
    if not extra_rows:
        return primary_df.copy()
    return pd.concat([primary_df, pd.DataFrame(extra_rows)], ignore_index=True)


def _select_report_paths() -> tuple[Path, Path]:
    # Walk candidate output roots and select the first bundle that contains a
    # complete main-model benchmark table.
    for summary_path, predictions_path in _report_candidates():
        if not summary_path.exists() or not predictions_path.exists():
            continue
        summary_df = safe_read_csv(summary_path)
        if _summary_is_complete(summary_df):
            return summary_path, predictions_path
    return LOCAL_SUMMARY_CSV, LOCAL_PREDICTIONS_CSV


def _pick_smiles_column(frame: pd.DataFrame) -> str:
    # External benchmark files came from multiple historical pipelines, so the
    # loader tolerates common SMILES header variants.
    normalized = {col.strip().lower().replace("_", " "): col for col in frame.columns}
    for key in ("canonical smiles", "canonical_smiles", "smiles"):
        if key in normalized:
            return normalized[key]
    raise ValueError("Could not find a SMILES column in the external dataset.")


def _to_float(value: Any) -> float | None:
    # Dashboard JSON should expose missing metrics explicitly rather than
    # propagating NaN strings into the frontend.
    try:
        if value is None or value == "":
            return None
        parsed = float(value)
        if not math.isfinite(parsed):
            return None
        return parsed
    except (TypeError, ValueError):
        return None


def _to_int(value: Any) -> int | None:
    # Integer coercion is used for labels and avoids accidental truthiness bugs
    # when archived CSVs mix ints, floats, and blank cells.
    try:
        if value is None or value == "":
            return None
        parsed = float(value)
        if not math.isfinite(parsed):
            return None
        return int(parsed)
    except (TypeError, ValueError):
        return None


def _normalize_sample_id(sample_id: str, external_smiles: list[str]) -> str:
    # Some archived runners saved generic IDs such as `external_17`; this maps
    # them back onto canonical benchmark SMILES so search and comparison remain
    # chemically meaningful in the dashboard.
    clean = ANNOTATION_RE.sub("", str(sample_id).strip())
    match = GENERIC_SAMPLE_RE.match(clean)
    if match:
        index = int(match.group(1)) - 1
        if 0 <= index < len(external_smiles):
            return external_smiles[index]
    return clean


def _display_variant(raw_variant: str | None, model_name: str) -> str:
    # Variant labels are normalised for presentation so ensemble names remain
    # interpretable to readers outside the training scripts.
    if raw_variant:
        return VARIANT_LABELS.get(raw_variant, raw_variant.replace("_", " "))
    return MODEL_LABELS.get(model_name, model_name)


def _resolve_query(query: str) -> str:
    # Saved-mode search accepts either canonical SMILES or a small set of
    # medicine-name shortcuts used in demonstrations and screenshots.
    lowered = query.strip().lower()
    return MEDICINE_MAPPING.get(lowered, query.strip())


def _build_models(summary_df: pd.DataFrame) -> list[dict[str, Any]]:
    # -------------------------------------------------------------------------
    # Analytics Model Cards
    # Each model dictionary consolidates internal and external metrics into the
    # form required by the dashboard. The generalisation gap is reported as the
    # internal AUROC minus external AUROC, which surfaces the degree to which a
    # model's validation performance transfers to unseen chemistry.
    # -------------------------------------------------------------------------
    models: list[dict[str, Any]] = []
    for model_name in MAIN_MODEL_ORDER:
        model_rows = summary_df[summary_df["model_name"] == model_name] if not summary_df.empty else pd.DataFrame()
        external_row = (
            model_rows[model_rows["dataset"] == "external"].iloc[0].to_dict()
            if not model_rows.empty and (model_rows["dataset"] == "external").any()
            else {}
        )
        internal_row = (
            model_rows[model_rows["dataset"].isin(["internal_cv", "internal_validation"])].iloc[0].to_dict()
            if not model_rows.empty and model_rows["dataset"].isin(["internal_cv", "internal_validation"]).any()
            else {}
        )

        external = {
            "accuracy": _to_float(external_row.get("accuracy")),
            "precision": _to_float(external_row.get("precision")),
            "recall": _to_float(external_row.get("recall")),
            "f1": _to_float(external_row.get("f1")),
            "roc_auc": _to_float(external_row.get("roc_auc")),
            "pr_auc": _to_float(external_row.get("pr_auc")),
        }
        internal = {
            "accuracy": _to_float(internal_row.get("accuracy")),
            "precision": _to_float(internal_row.get("precision")),
            "recall": _to_float(internal_row.get("recall")),
            "f1": _to_float(internal_row.get("f1")),
            "roc_auc": _to_float(internal_row.get("roc_auc")),
            "pr_auc": _to_float(internal_row.get("pr_auc")),
        }

        internal_roc = internal["roc_auc"]
        external_roc = external["roc_auc"]
        generalization_pct = None
        gap_auc = None
        if internal_roc is not None and external_roc is not None and internal_roc != 0:
            generalization_pct = (external_roc / internal_roc) * 100
            gap_auc = internal_roc - external_roc

        raw_variant = external_row.get("variant") or internal_row.get("variant")
        models.append(
            {
                "model_name": model_name,
                "display_name": MODEL_LABELS.get(model_name, model_name),
                "variant": raw_variant or "",
                "variant_display": _display_variant(raw_variant, model_name),
                "external": external,
                "internal": internal,
                "generalization_pct": generalization_pct,
                "gap_auc": gap_auc,
            }
        )
    return models


def _smiles_token_set(smiles: str) -> set[str]:
    # Token fallback used only when RDKit is unavailable; it preserves a coarse
    # similarity view for the interface but is not chemically equivalent to
    # Morgan fingerprint Tanimoto similarity.
    normalized = smiles.strip()
    if not normalized:
        return set()
    if len(normalized) == 1:
        return {normalized}
    tokens = {normalized[index : index + 2] for index in range(len(normalized) - 1)}
    tokens.update(normalized)
    return tokens


def _select_heatmap_samples(
    external_df: pd.DataFrame,
    smiles_col: str,
    *,
    method: str,
) -> list[dict[str, Any]]:
    # Construct a balanced subset of toxic and non-toxic external compounds so
    # the similarity heatmap remains legible while still illustrating chemical
    # neighbourhood structure across both classes.
    use_rdkit = method == "morgan"
    if use_rdkit and (Chem is None or AllChem is None or DataStructs is None):
        return []

    selected: list[dict[str, Any]] = []
    per_label_limit = 5
    counters = {0: 0, 1: 0}

    for label in (0, 1):
        if "label" not in external_df.columns:
            continue
        label_rows = external_df[external_df["label"] == label]
        for _, row in label_rows.iterrows():
            smiles = str(row[smiles_col]).strip()
            if not smiles:
                continue
            similarity_repr: Any
            if use_rdkit:
                molecule = Chem.MolFromSmiles(smiles)
                if molecule is None:
                    continue
                similarity_repr = AllChem.GetMorganFingerprintAsBitVect(molecule, radius=2, nBits=2048)
            else:
                similarity_repr = _smiles_token_set(smiles)
                if not similarity_repr:
                    continue
            counters[label] += 1
            selected.append(
                {
                    "smiles": smiles,
                    "label": label,
                    "short_label": f"{'N' if label == 0 else 'T'}{counters[label]}",
                    "similarity_repr": similarity_repr,
                }
            )
            if counters[label] >= per_label_limit:
                break

    return selected


def _jaccard_similarity(left: set[str], right: set[str]) -> float:
    # NOTE:
    # This fallback is a resilience feature for the UI, not a scientifically
    # equivalent substitute for fingerprint-based Tanimoto similarity.
    union = left | right
    if not union:
        return 1.0
    return len(left & right) / len(union)


def _build_tanimoto_heatmap(external_df: pd.DataFrame, smiles_col: str) -> dict[str, Any] | None:
    # -------------------------------------------------------------------------
    # Similarity Visualisation
    # The analytics page uses RDKit Morgan fingerprints and Tanimoto similarity
    # to give a chemistry-native view of the external benchmark space. This is
    # complementary to the model metrics because it shows whether compounds are
    # densely clustered or structurally isolated across toxic and non-toxic
    # regions.
    # -------------------------------------------------------------------------
    method = "morgan"
    note = "Morgan fingerprint Tanimoto similarity computed with RDKit."
    samples = _select_heatmap_samples(external_df, smiles_col, method=method)
    if not samples:
        method = "smiles_jaccard"
        note = "RDKit was unavailable, so the dashboard used a SMILES-token Jaccard fallback for this similarity view."
        samples = _select_heatmap_samples(external_df, smiles_col, method=method)
    if not samples:
        return None

    matrix: list[list[float]] = []
    top_pairs: list[dict[str, Any]] = []
    for row_index, left in enumerate(samples):
        row_values: list[float] = []
        for col_index, right in enumerate(samples):
            if method == "morgan":
                similarity = float(DataStructs.TanimotoSimilarity(left["similarity_repr"], right["similarity_repr"]))
            else:
                similarity = _jaccard_similarity(left["similarity_repr"], right["similarity_repr"])
            rounded = round(similarity, 3)
            row_values.append(rounded)
            if col_index > row_index:
                top_pairs.append(
                    {
                        "left": left["short_label"],
                        "right": right["short_label"],
                        "left_smiles": left["smiles"],
                        "right_smiles": right["smiles"],
                        "similarity": rounded,
                    }
                )
        matrix.append(row_values)

    top_pairs.sort(key=lambda item: item["similarity"], reverse=True)
    labels = [
        {
            "short_label": sample["short_label"],
            "smiles": sample["smiles"],
            "true_label": sample["label"],
        }
        for sample in samples
    ]
    return {
        "method": method,
        "note": note,
        "labels": labels,
        "matrix": matrix,
        "top_pairs": top_pairs[:5],
    }


def _build_analytics(external_df: pd.DataFrame, smiles_col: str, models: list[dict[str, Any]]) -> dict[str, Any]:
    # Assemble the chart payloads used by the Deep Analytics page: dataset
    # balance, generalisation gap, and structural similarity summaries.
    labels = external_df["label"].dropna().astype(int) if "label" in external_df.columns else pd.Series(dtype=int)
    safe_count = int((labels == 0).sum())
    toxic_count = int((labels == 1).sum())

    gap_rows = [
        {
            "display_name": model["display_name"],
            "gap_auc": model["gap_auc"],
            "external_roc_auc": model["external"]["roc_auc"],
            "internal_roc_auc": model["internal"]["roc_auc"],
        }
        for model in models
        if model["gap_auc"] is not None
    ]

    return {
        "dataset_balance": {
            "labels": ["Non-toxic", "Toxic"],
            "counts": [safe_count, toxic_count],
        },
        "generalization_gap": gap_rows,
        "tanimoto_heatmap": _build_tanimoto_heatmap(external_df, smiles_col),
    }


def _load_state(force_reload: bool = False) -> dict[str, Any]:
    # -------------------------------------------------------------------------
    # Cached Dashboard State
    # The dashboard is effectively a read-mostly evidence browser, so a simple
    # mtime-keyed cache avoids repeated CSV parsing while remaining sensitive to
    # reruns of the analytical pipeline.
    # -------------------------------------------------------------------------
    summary_path, predictions_path = _select_report_paths()
    key = tuple(
        path.stat().st_mtime if path.exists() else None
        for path in (summary_path, predictions_path, DEFAULT_EXTERNAL_CSV)
    )
    if not force_reload and _CACHE["key"] == key and _CACHE["state"] is not None:
        return _CACHE["state"]

    summary_df = safe_read_csv(summary_path) if summary_path.exists() else pd.DataFrame()
    if summary_path != LOCAL_SUMMARY_CSV and LOCAL_SUMMARY_CSV.exists():
        local_summary_df = safe_read_csv(LOCAL_SUMMARY_CSV)
        summary_df = _merge_summary_frames(summary_df, local_summary_df)
    predictions_df = safe_read_csv(predictions_path) if predictions_path.exists() else pd.DataFrame()
    external_df = safe_read_csv(DEFAULT_EXTERNAL_CSV)

    smiles_col = _pick_smiles_column(external_df)
    external_smiles = external_df[smiles_col].fillna("").astype(str).tolist()
    labels = external_df["label"] if "label" in external_df.columns else pd.Series(dtype=float)
    label_lookup = {
        smiles: _to_int(label)
        for smiles, label in zip(external_df[smiles_col].astype(str), labels, strict=False)
    }

    if not summary_df.empty:
        summary_df = summary_df[summary_df["model_name"].isin(MAIN_MODEL_ORDER)].copy()

    if not predictions_df.empty:
        # Normalise archived identifiers and display labels once so all
        # downstream search and rendering paths operate on a common schema.
        predictions_df = predictions_df[predictions_df["model_name"].isin(MAIN_MODEL_ORDER)].copy()
        predictions_df["resolved_sample_id"] = predictions_df["sample_id"].apply(
            lambda sample_id: _normalize_sample_id(str(sample_id), external_smiles)
        )
        predictions_df["display_name"] = predictions_df["model_name"].map(MODEL_LABELS).fillna(predictions_df["model_name"])
        predictions_df["variant_display"] = predictions_df.apply(
            lambda row: _display_variant(row.get("variant"), row["model_name"]),
            axis=1,
        )
        predictions_df["model_sort"] = predictions_df["model_name"].apply(MAIN_MODEL_ORDER.index)

    models = _build_models(summary_df)
    ranked_models = [model for model in models if model["external"].get("roc_auc") is not None]
    best_model = max(ranked_models, key=lambda model: model["external"]["roc_auc"]) if ranked_models else None

    state = {
        "summary_df": summary_df,
        "predictions_df": predictions_df,
        "external_smiles": external_smiles,
        "label_lookup": label_lookup,
        "models": models,
        "best_model": best_model,
        "analytics": _build_analytics(external_df, smiles_col, models),
        "refreshed_at": datetime.now(UTC).isoformat(),
        "project_root": str(PROJECT_ROOT),
        "summary_path": str(summary_path),
        "predictions_path": str(predictions_path),
        "external_path": str(DEFAULT_EXTERNAL_CSV),
    }
    _CACHE["key"] = key
    _CACHE["state"] = state
    return state


def get_overview_payload(force_reload: bool = False) -> dict[str, Any]:
    # Public payload for `/api/overview`, used to populate the high-level
    # dashboard cards, charts, and quick-search sample chips.
    state = _load_state(force_reload=force_reload)
    resolved_ids = set(
        state["predictions_df"].get("resolved_sample_id", pd.Series(dtype=str)).astype(str).tolist()
    )
    verified_medicines = [
        name.title()
        for name, smiles in MEDICINE_MAPPING.items()
        if smiles in resolved_ids
    ]
    return {
        "refreshed_at": state["refreshed_at"],
        "deployment_mode": DEPLOYMENT_MODE,
        "project_root": state["project_root"],
        "summary_path": state["summary_path"],
        "predictions_path": state["predictions_path"],
        "external_path": state["external_path"],
        "models": state["models"],
        "best_model": state["best_model"],
        "example_queries": state["external_smiles"][:8],
        "medicine_names": verified_medicines,
        "analytics": state["analytics"],
    }


def search_predictions(query: str, limit: int = 8) -> dict[str, Any]:
    # -------------------------------------------------------------------------
    # Saved Prediction Retrieval
    # Search operates over archived external-test predictions so the user can
    # inspect how each shortlisted engine scored benchmark compounds. The
    # returned consensus is only a convenience summary for the UI; the per-model
    # rows remain the primary evidence.
    # -------------------------------------------------------------------------
    state = _load_state()
    predictions_df: pd.DataFrame = state["predictions_df"]
    clean_query = query.strip()
    if predictions_df.empty or not clean_query:
        return {"query": clean_query, "matches": [], "match_count": 0}

    resolved_query = _resolve_query(clean_query)
    resolved = predictions_df["resolved_sample_id"].fillna("").astype(str)
    lowered_query = resolved_query.lower()

    exact_ids = [sample_id for sample_id in resolved.unique().tolist() if sample_id.lower() == lowered_query]
    prefix_ids = [
        sample_id
        for sample_id in resolved.unique().tolist()
        if sample_id.lower().startswith(lowered_query) and sample_id not in exact_ids
    ]
    contains_ids = [
        sample_id
        for sample_id in resolved.unique().tolist()
        if lowered_query in sample_id.lower() and sample_id not in exact_ids and sample_id not in prefix_ids
    ]
    candidate_ids = (exact_ids + prefix_ids + contains_ids)[:limit]

    matches: list[dict[str, Any]] = []
    for sample_id in candidate_ids:
        sample_rows = predictions_df[predictions_df["resolved_sample_id"] == sample_id].sort_values("model_sort")
        predictions: list[dict[str, Any]] = []
        scores: list[float] = []
        true_label = None

        for _, row in sample_rows.iterrows():
            score = _to_float(row.get("predicted_score"))
            label = _to_int(row.get("predicted_label"))
            truth = _to_int(row.get("true_label"))
            if true_label is None and truth is not None:
                true_label = truth
            if score is not None:
                scores.append(score)
            predictions.append(
                {
                    "model_name": row["model_name"],
                    "display_name": row["display_name"],
                    "variant": row.get("variant", ""),
                    "variant_display": row["variant_display"],
                    "predicted_label": label,
                    "predicted_score": score,
                }
            )

        consensus_score = sum(scores) / len(scores) if scores else None
        consensus_label = int(consensus_score >= 0.5) if consensus_score is not None else None
        matches.append(
            {
                "canonical_smiles": sample_id,
                "true_label": true_label if true_label is not None else state["label_lookup"].get(sample_id),
                "consensus_score": consensus_score,
                "consensus_label": consensus_label,
                "model_predictions": predictions,
            }
        )

    return {
        "query": clean_query,
        "resolved_query": resolved_query,
        "matches": matches,
        "match_count": len(matches),
    }


def run_main_pipeline(*, force_rerun: bool = False, skip_data_prep: bool = True) -> dict[str, Any]:
    # Launch the consolidated CLI from within the dashboard process. The API
    # returns raw stdout/stderr so reruns remain auditable for the dissertation.
    python_executable = os.environ.get("KV6013_PYTHON", sys.executable)
    command = [python_executable, str(PROJECT_ROOT / "main.py")]
    if skip_data_prep:
        command.append("--skip-data-prep")
    if force_rerun:
        command.append("--force-rerun")

    completed = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    _load_state(force_reload=True)
    return {
        "command": command,
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "ran_at": datetime.now(UTC).isoformat(),
    }


def static_file_path(web_root: Path, requested_path: str) -> Path:
    # Resolve browser requests defensively so the lightweight server only serves
    # files inside the packaged web root.
    clean = requested_path.lstrip("/") or "index.html"
    candidate = (web_root / clean).resolve()
    try:
        candidate.relative_to(web_root.resolve())
    except ValueError as exc:
        raise FileNotFoundError("Requested file is outside the web root.") from exc
    if candidate.exists() and candidate.is_file():
        return candidate
    return web_root / "index.html"
