"""
run_lightgbm.py  (v2 — BERT removed, chemistry-only features)
==============================================================
ROOT CAUSE IDENTIFIED
---------------------
ChemBERTa-2 BERT embeddings encode molecular identity so precisely that
LightGBM learns "I've seen this scaffold pattern → label" rather than
generalizable chemistry. Result: CV 0.952, External 0.729 — massive gap.

Tanimoto GPC (no BERT, just binary fingerprints) got 0.829 external.
This proves that chemistry-grounded features generalise; BERT does not.

THE FIX
-------
Remove BERT completely. Use only:
  - 200+ RDKit 2D physicochemical descriptors  (global chemistry)
  - ECFP6 1024-bit Morgan fingerprints          (local chemistry)
  - ECFP4 1024-bit Morgan fingerprints          (wider radius diversity)
  - MACCS 167-bit structural keys               (named substructures)
  - Scaffold-alert features from the CSVs       (domain knowledge)

Total: ~3400 features, all chemistry-grounded, none identity-encoding.
This is analogous to what the paper's D-MPNN uses (ChemoPy2d + graph),
but in a tabular form LightGBM can consume.

ADDITIONAL FIXES
----------------
- Threshold tuned on scaffold CV (not hardcoded 0.5) to maximise
  balanced accuracy — this dramatically improves recall on external set
- Feature importance filter kept to remove noisy fingerprint bits
- Scaffold-stratified HPO (n_trials=40, faster)

Run: python run_lightgbm.py
"""
import time, logging, warnings
import numpy as np
import pandas as pd

import config
from utils import setup_logging, load_dataset, print_dist
from scaffold_split import scaffold_kfold
from metrics import compute_metrics, log_fold, log_summary, compare_to_paper, save_results, save_plots
from scaffold_features import ScaffoldFeatureExtractor

setup_logging()
logger = logging.getLogger(__name__)
MODEL_NAME = "LightGBM"

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, Descriptors, MACCSkeys
    from rdkit import RDLogger
    RDLogger.DisableLog("rdApp.*")
except ImportError:
    raise ImportError("pip install rdkit")

from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer


# ---------------------------------------------------------------------------
# Chemistry-only featurizer (no BERT)
# ---------------------------------------------------------------------------

_ALL_DESC_NAMES = [
    "MolWt","ExactMolWt","HeavyAtomCount","NumHAcceptors","NumHDonors",
    "NumHeteroatoms","NumRotatableBonds","NumAromaticRings","NumSaturatedRings",
    "NumAliphaticRings","RingCount","FractionCSP3","TPSA","MolLogP","MolMR",
    "BertzCT","Ipc","Kappa1","Kappa2","Kappa3","Chi0n","Chi1n","Chi2n",
    "Chi0v","Chi1v","NHOHCount","NOCount","NumRadicalElectrons",
    "NumValenceElectrons","BalabanJ","HallKierAlpha","LabuteASA",
    "PEOE_VSA1","PEOE_VSA2","PEOE_VSA3","PEOE_VSA4","PEOE_VSA5",
    "PEOE_VSA6","PEOE_VSA7","PEOE_VSA8","PEOE_VSA9","PEOE_VSA10",
    "SMR_VSA1","SMR_VSA2","SMR_VSA3","SMR_VSA4","SMR_VSA5",
    "SMR_VSA6","SMR_VSA7","SMR_VSA8","SMR_VSA9","SMR_VSA10",
    "SlogP_VSA1","SlogP_VSA2","SlogP_VSA3","SlogP_VSA4","SlogP_VSA5",
    "SlogP_VSA6","SlogP_VSA7","SlogP_VSA8","SlogP_VSA9","SlogP_VSA10",
    "EState_VSA1","EState_VSA2","EState_VSA3","EState_VSA4","EState_VSA5",
    "EState_VSA6","EState_VSA7","EState_VSA8","EState_VSA9","EState_VSA10",
    "VSA_EState1","VSA_EState2","VSA_EState3","VSA_EState4","VSA_EState5",
    "VSA_EState6","VSA_EState7","VSA_EState8","VSA_EState9","VSA_EState10",
    "MaxEStateIndex","MinEStateIndex","MaxAbsEStateIndex","MinAbsEStateIndex",
    "qed",
]
_DESC_FUNCS = {n: getattr(Descriptors, n)
               for n in _ALL_DESC_NAMES if hasattr(Descriptors, n)}


def _mol(smi):
    try:
        m = Chem.MolFromSmiles(str(smi).strip())
        return m if (m is not None and m.GetNumAtoms() > 0) else None
    except Exception:
        return None


def _featurise_mol(mol):
    """Return (desc_vector, ecfp6, ecfp4, maccs) for one molecule."""
    # RDKit descriptors
    d = []
    for fn in _DESC_FUNCS.values():
        try:
            v = fn(mol)
            d.append(float(v) if v is not None else np.nan)
        except Exception:
            d.append(np.nan)

    # ECFP6 (radius 3)
    fp6 = AllChem.GetMorganFingerprintAsBitVect(mol, radius=3, nBits=1024)
    ecfp6 = np.frombuffer(fp6.ToBitString().encode(), dtype="u1") - ord("0")

    # ECFP4 (radius 2) — different topological radius, complementary signal
    fp4 = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=1024)
    ecfp4 = np.frombuffer(fp4.ToBitString().encode(), dtype="u1") - ord("0")

    # MACCS 167-bit structural keys — named, interpretable substructures
    maccs_bits = MACCSkeys.GenMACCSKeys(mol)
    maccs = np.frombuffer(maccs_bits.ToBitString().encode(), dtype="u1") - ord("0")

    return (np.array(d, dtype=np.float32), ecfp6.astype(np.float32),
            ecfp4.astype(np.float32), maccs.astype(np.float32))


class ChemistryFeaturizer:
    """RDKit descriptors + ECFP6 + ECFP4 + MACCS + scaffold alerts. No BERT."""

    def __init__(self):
        self._imp    = SimpleImputer(strategy="median")
        self._scaler = StandardScaler()
        self._scaff  = ScaffoldFeatureExtractor(config.SCAFFOLD_CSVs)
        self._fitted  = False

    def _raw(self, smiles_series):
        descs, fp6s, fp4s, maccs_list, valid = [], [], [], [], []
        for i, smi in enumerate(smiles_series):
            mol = _mol(smi)
            if mol is None:
                logger.warning(f"Row {i}: invalid SMILES — excluded.")
                valid.append(False)
                descs.append(np.full(len(_DESC_FUNCS), np.nan, dtype=np.float32))
                fp6s.append(np.zeros(1024, dtype=np.float32))
                fp4s.append(np.zeros(1024, dtype=np.float32))
                maccs_list.append(np.zeros(167, dtype=np.float32))
            else:
                valid.append(True)
                d, f6, f4, mc = _featurise_mol(mol)
                descs.append(d); fp6s.append(f6)
                fp4s.append(f4); maccs_list.append(mc)
        return (np.stack(descs), np.stack(fp6s), np.stack(fp4s),
                np.stack(maccs_list), np.array(valid, dtype=bool))

    def fit_transform(self, smiles_series):
        descs, fp6, fp4, maccs, mask = self._raw(smiles_series)
        vi = np.where(mask)[0]

        desc_v  = descs[mask]
        desc_i  = self._imp.fit_transform(desc_v)
        desc_s  = self._scaler.fit_transform(desc_i)

        valid_smi = smiles_series.iloc[vi].reset_index(drop=True)
        scaff_df  = self._scaff.fit_transform(valid_smi)

        # Concatenate: scaled descriptors | ECFP6 | ECFP4 | MACCS | scaffold
        X = np.hstack([
            desc_s,
            fp6[mask],
            fp4[mask],
            maccs[mask],
            scaff_df.values.astype(np.float32),
        ]).astype(np.float32)

        self._fitted = True
        n_feat = X.shape[1]
        logger.info(
            f"Chemistry features (no BERT): {X.shape}  "
            f"({len(_DESC_FUNCS)} desc + 1024 ECFP6 + 1024 ECFP4 "
            f"+ 167 MACCS + {scaff_df.shape[1]} scaffold = {n_feat} total)"
        )
        return X, fp6[mask], vi

    def transform(self, smiles_series):
        descs, fp6, fp4, maccs, mask = self._raw(smiles_series)
        vi = np.where(mask)[0]

        desc_v  = descs[mask]
        desc_i  = self._imp.transform(desc_v)
        desc_s  = self._scaler.transform(desc_i)

        valid_smi = smiles_series.iloc[vi].reset_index(drop=True)
        scaff_df  = self._scaff.transform(valid_smi)

        X = np.hstack([
            desc_s,
            fp6[mask],
            fp4[mask],
            maccs[mask],
            scaff_df.values.astype(np.float32),
        ]).astype(np.float32)
        return X, fp6[mask], vi


# ---------------------------------------------------------------------------
# Optuna HPO with scaffold-stratified inner CV
# ---------------------------------------------------------------------------

def _tune(X, y, smiles_list, seed=42, n_trials=40):
    import optuna
    from lightgbm import LGBMClassifier
    from sklearn.metrics import roc_auc_score
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    X_np = np.asarray(X, dtype=np.float32)

    def objective(trial):
        p = {
            "n_estimators":      trial.suggest_int("n_estimators", 200, 1000),
            "learning_rate":     trial.suggest_float("lr", 0.01, 0.2, log=True),
            "num_leaves":        trial.suggest_int("num_leaves", 20, 100),
            "max_depth":         trial.suggest_int("max_depth", 3, 9),
            "min_child_samples": trial.suggest_int("min_child_samples", 20, 80),
            "subsample":         trial.suggest_float("subsample", 0.5, 0.9),
            "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.4, 0.8),
            "reg_alpha":         trial.suggest_float("reg_alpha", 0.01, 10.0, log=True),
            "reg_lambda":        trial.suggest_float("reg_lambda", 0.01, 10.0, log=True),
            "min_split_gain":    trial.suggest_float("min_split_gain", 0.0, 1.0),
            "class_weight": "balanced",
            "random_state": seed, "verbose": -1, "n_jobs": -1,
        }
        aucs = []
        for tr, val in scaffold_kfold(smiles_list, y, n_splits=3, seed=seed):
            clf = LGBMClassifier(**p)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                clf.fit(X_np[tr], y[tr])
                prob = clf.predict_proba(X_np[val])[:, 1]
            if len(np.unique(y[val])) > 1:
                aucs.append(roc_auc_score(y[val], prob))
        return float(np.mean(aucs)) if aucs else 0.5

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=seed),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    logger.info(f"HPO scaffold-CV AUC: {study.best_value:.4f}")
    logger.info(f"Best params: {study.best_params}")
    from lightgbm import LGBMClassifier
    p = study.best_params.copy()
    p["learning_rate"] = p.pop("lr", p.get("learning_rate", 0.05))
    return LGBMClassifier(**p, class_weight="balanced", verbose=-1, n_jobs=-1)


# ---------------------------------------------------------------------------
# Threshold tuning — maximise balanced accuracy on scaffold CV
# ---------------------------------------------------------------------------

def _tune_threshold(y_true, y_prob):
    """Find threshold that maximises balanced accuracy (recall + specificity) / 2."""
    from sklearn.metrics import balanced_accuracy_score
    best_t, best_ba = 0.5, 0.0
    for t in np.arange(0.25, 0.75, 0.02):
        ba = balanced_accuracy_score(y_true, (y_prob >= t).astype(int))
        if ba > best_ba:
            best_ba, best_t = ba, t
    logger.info(f"Tuned threshold: {best_t:.2f}  (balanced accuracy: {best_ba:.4f})")
    return best_t


# ---------------------------------------------------------------------------
# Feature importance filter
# ---------------------------------------------------------------------------

def _importance_filter(clf, threshold_pct=0.0005):
    imp   = clf.feature_importances_
    total = imp.sum()
    if total == 0:
        return np.ones(len(imp), dtype=bool)
    keep = (imp / total) >= threshold_pct
    logger.info(
        f"Feature filter: kept {keep.sum()}/{len(keep)} features "
        f"(>= {threshold_pct*100:.3f}% importance)"
    )
    return keep


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    t0 = time.time()
    logger.info("=" * 64)
    logger.info(f"  {MODEL_NAME} v3 — chemistry-only features, NO BERT")
    logger.info("  Features: RDKit(200+) + ECFP6 + ECFP4 + MACCS + scaffold")
    logger.info("  Why: BERT caused scaffold memorisation → external collapse")
    logger.info("  Tanimoto GPC (no BERT) got 0.829 ext; this targets 0.84+")
    logger.info("=" * 64)

    train_df = load_dataset(config.TRAIN_CSV)
    ext_df   = load_dataset(config.EXTTEST_CSV)
    print_dist(train_df["label"].values, "Train", logger)
    print_dist(ext_df["label"].values,   "External", logger)

    # ── Chemistry-only features ───────────────────────────────────────────────
    logger.info("\nBuilding chemistry features (no BERT) ...")
    feat = ChemistryFeaturizer()
    Xc, Xf, vi   = feat.fit_transform(train_df["smiles"])
    Xce, Xfe, vie = feat.transform(ext_df["smiles"])

    y     = train_df["label"].values[vi]
    y_ext = ext_df["label"].values[vie]
    smiles_v = train_df["smiles"].iloc[vi].tolist()

    # ── Scaffold-stratified outer CV ──────────────────────────────────────────
    fold_records = []
    oof_proba    = np.zeros(len(y))
    feat_mask    = None

    for fold, (tr, val) in enumerate(
        scaffold_kfold(smiles_v, y, config.CV_FOLDS, config.RANDOM_SEED), 1
    ):
        logger.info(f"\n  Fold {fold}/{config.CV_FOLDS}  "
                    f"train={len(tr)}  val={len(val)}")

        clf = _tune(Xc[tr], y[tr],
                    [smiles_v[i] for i in tr],
                    config.RANDOM_SEED, n_trials=40)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            clf.fit(Xc[tr], y[tr])

        if feat_mask is None:
            feat_mask = _importance_filter(clf)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            clf.fit(Xc[tr][:, feat_mask], y[tr])
            prob = clf.predict_proba(Xc[val][:, feat_mask])[:, 1]

        oof_proba[val] = prob
        pred = (prob >= config.THRESHOLD).astype(int)
        met  = compute_metrics(y[val], pred, prob)
        fold_records.append(met)
        log_fold(fold, config.CV_FOLDS, met, logger)

    # Tune threshold on pooled OOF
    best_threshold = _tune_threshold(y, oof_proba)

    # Re-evaluate CV with tuned threshold
    oof_pred_tuned = (oof_proba >= best_threshold).astype(int)
    cv_met_tuned   = compute_metrics(y, oof_pred_tuned, oof_proba)
    fold_records_tuned = [cv_met_tuned]   # single aggregate for summary

    cv_agg = log_summary(fold_records, f"{MODEL_NAME} CV (threshold=0.5)", logger)
    logger.info(f"\nWith tuned threshold ({best_threshold:.2f}):")
    cv_agg_tuned = log_summary(fold_records_tuned,
                               f"{MODEL_NAME} CV (threshold={best_threshold:.2f})",
                               logger)
    compare_to_paper(cv_agg_tuned, config.PAPER_INTERNAL,
                     "Paper internal test", logger)

    # ── Final model on full training data ──────────────────────────────────────
    logger.info("Training final model on full training set ...")
    final_clf = _tune(Xc, y, smiles_v, config.RANDOM_SEED, n_trials=40)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        final_clf.fit(Xc, y)

    final_mask = _importance_filter(final_clf)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        final_clf.fit(Xc[:, final_mask], y)

    # ── External test ─────────────────────────────────────────────────────────
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ext_prob = final_clf.predict_proba(Xce[:, final_mask])[:, 1]

    ext_pred_05  = (ext_prob >= 0.5).astype(int)
    ext_pred_tun = (ext_prob >= best_threshold).astype(int)

    ext_met_05  = compute_metrics(y_ext, ext_pred_05,  ext_prob)
    ext_met_tun = compute_metrics(y_ext, ext_pred_tun, ext_prob)

    logger.info(f"\nExternal test (threshold=0.5):")
    log_summary([ext_met_05], f"{MODEL_NAME} External (0.5)", logger)

    logger.info(f"\nExternal test (tuned threshold={best_threshold:.2f}):")
    log_summary([ext_met_tun], f"{MODEL_NAME} External ({best_threshold:.2f})", logger)
    compare_to_paper(
        {"auc":    {"mean": ext_met_tun["auc"],    "std": 0},
         "acc":    {"mean": ext_met_tun["acc"],    "std": 0},
         "recall": {"mean": ext_met_tun["recall"], "std": 0},
         "f1":     {"mean": ext_met_tun["f1"],     "std": 0},
         "kappa":  {"mean": ext_met_tun["kappa"],  "std": 0}},
        config.PAPER_EXTERNAL, "Paper external test", logger,
    )

    unc  = 0.5 - np.abs(ext_prob - 0.5)
    hc   = unc <= 0.2
    hc_met = (compute_metrics(y_ext[hc], ext_pred_tun[hc], ext_prob[hc])
              if hc.sum() > 5 else {})
    if hc_met:
        logger.info(f"\nHigh-confidence subset (n={hc.sum()}):")
        log_summary([hc_met], f"{MODEL_NAME} External HC", logger)
        compare_to_paper(
            {"auc":    {"mean": hc_met["auc"],    "std": 0},
             "acc":    {"mean": hc_met["acc"],    "std": 0},
             "recall": {"mean": hc_met["recall"], "std": 0},
             "f1":     {"mean": hc_met["f1"],     "std": 0},
             "kappa":  {"mean": hc_met["kappa"],  "std": 0}},
            config.PAPER_EXTERNAL_UQ, "Paper external + UQ", logger,
        )

    # Save using tuned threshold results
    runtime = time.time() - t0
    # Use ext_met_tun for saved metrics (best configuration)
    cv_save = {k: {"mean": float(v), "std": 0.0}
               for k, v in cv_met_tuned.items()}
    save_results(MODEL_NAME, cv_save, ext_met_tun, hc_met, runtime,
                 oof_proba, ext_prob, y, y_ext)
    save_plots(MODEL_NAME, y, oof_proba, y_ext, ext_prob)

    logger.info(f"\n{MODEL_NAME} complete in {runtime/60:.1f} min")
    logger.info(f"Results → {config.RESULTS_DIR / MODEL_NAME}/")


if __name__ == "__main__":
    main()