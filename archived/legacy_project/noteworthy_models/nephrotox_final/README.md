# NephroTox Predictor v2

**Goal: Beat Liu et al. (2025) J. Chem. Inf. Model. — D-MPNN + ChemoPy2d**

| Benchmark | Paper AUC | Paper Kappa |
|-----------|-----------|-------------|
| Internal test set | 93.3% | 70.3% |
| External test set (304 compounds) | 84.6% | 69.7% |
| External + uncertainty filtering | 86.8% | 75.6% |

---

## PyCharm Setup (3 steps)

**1. Open folder in PyCharm**
`File → Open → select the nephrotox_v2/ folder`

**2. Create interpreter + install**
```
Settings → Project → Python Interpreter → Add → Virtualenv → Python 3.10
```
Then in the PyCharm Terminal:
```bash
pip install -r requirements.txt
```
> If rdkit fails: `conda install -c conda-forge rdkit` first

**3. Run**
Open `main.py`, scroll to the bottom, uncomment `main(RUN_CONFIG)`, and press the green Run button.  
Or use Run → Edit Configurations with `--mode train_eval`.

---

## Project structure

```
nephrotox_v2/
├── main.py                   ← run this
├── config.py                 ← all settings (paths, hyperparams)
├── featurizer.py             ← ChemBERTa-2 + RDKit + ECFP6 + scaffold features
├── scaffold_features.py      ← novel scaffold-alert features from CSVs
├── model.py                  ← LightGBM + TabPFN + Tanimoto GP + stacking
├── evaluator.py              ← CV + external test + vs-paper plots
├── utils.py                  ← data loading, logging
├── requirements.txt
└── data/
    ├── model_construction_dataset.csv      (1527 compounds — training)
    ├── external_test_dataset.csv           (304 compounds — held-out test)
    ├── model_molecular_murcko_scaffolds.csv
    ├── model_molecular_carbon_scaffolds.csv
    ├── external_molecular_murcko_scaffolds.csv
    ├── external_molecular_carbon_scaffolds.csv
    ├── model_nephrotoxicity_molecular_murcko_scaffolds.csv
    └── model_nephrotoxicity_molecular_carbon_scaffolds.csv
```

---

## What's novel vs the paper

| Feature | Paper | This model |
|---------|-------|------------|
| Molecular representation | ChemoPy2d (2D descriptors only) | ChemBERTa-2 (pre-trained on 77M molecules) + RDKit + ECFP6 |
| Scaffold information | Used only for diversity analysis | Converted to 8 predictive features per molecule |
| Boosting | XGBoost | LightGBM + Optuna TPE (50 trials) |
| Uncertainty | Monte Carlo Dropout | Tanimoto GP (exact Bayesian) |
| Ensemble | Single model | 3-model stacking with meta-learner |
| Transfer learning | None | 77M molecules → ChemBERTa-2 |

---

## Output files (in results/)

| File | Description |
|------|-------------|
| `cv_metrics.csv` | Per-fold AUC / ACC / SE / F1 / Kappa |
| `cv_summary.csv` | Mean ± std across folds |
| `vs_paper_internal.png` | Bar chart: our CV vs paper internal test |
| `vs_paper_external.png` | Bar chart: our external test vs paper external |
| `external_metrics.csv` | External test metrics (all + high-confidence) |
| `external_predictions.csv` | Per-compound predictions + uncertainty |
| `ext_roc.png` | ROC with paper AUC reference line |
| `cv_roc.png` | CV ROC |
| `ext_uncertainty.png` | Uncertainty analysis plots |
| `nephrotox_model.pkl` | Saved final model |
| `featurizer.pkl` | Saved featurizer |

---

## Runtime on M2 Air (approx)

| Step | Time |
|------|------|
| ChemBERTa-2 download (once) | ~3 min |
| BERT embeddings (1527 molecules) | ~5 min |
| LightGBM Optuna (50 trials) | ~5 min |
| TabPFN v2 | ~2 min |
| Tanimoto GPC | ~3 min |
| 5-fold CV total | ~45–60 min |

**Quick test (no Optuna, 2 folds):** add `--no_tune --cv_folds 2` → ~10 min

---

## Predict new compounds

```
--mode predict --predict_file new_compounds.csv
```

Input needs only a `smiles` or `canonical SMILES` column.
Output columns: `prob_nephrotoxic`, `predicted_class`, `uncertainty`, `flag_review`.
