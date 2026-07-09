# KV6013 Nephrotoxicity Predictor

This repository contains a portable nephrotoxicity screening system for university reporting and local demonstration. It combines data preparation, model execution, consolidated evaluation, live SMILES inference, and a browser-based dashboard for reviewing saved and live predictions.

## Project Aim

The aim of the project is to compare several molecular toxicity modelling families and present their outputs in a consistent screening interface. The system is designed to support two workflows:

- saved screening against the already evaluated external dataset
- live screening of a user-supplied SMILES string with applicability-domain routing and explanation output

## Dataset Used

The pipeline uses the curated CSV files under `datasets/raw/`:

- `model construction dataset.csv` for training and internal fingerprint context
- `external test dataset.csv` for external evaluation and saved predictions
- Murcko scaffold and carbon scaffold files for scaffold-level summaries and cleaning checks

Running `python main.py` writes cleaned versions into `datasets/processed/` and stores a cleaning audit in `datasets/processed/dataset_cleaning_report.json`.

## Current Implementation

The current codebase is organised around a simple end-to-end flow:

1. `main.py` optionally cleans the raw datasets and runs the selected model families.
2. `src/pipelines/prepare_datasets.py` standardises label and scaffold CSV files, removes invalid rows, deduplicates records, and writes the cleaning report.
3. `src/models/` contains the main model families and the wrappers that recover archived outputs when those artifacts are available.
4. `src/evaluation/schema.py` normalises summary and prediction rows into the report schema used across the project.
5. `src/utils/dashboard_data.py` reads the output CSV files and builds the analytics payload used by the dashboard.
6. `serve_dashboard.py` exposes the dashboard and live prediction API.

## Features

- Screening Terminal with an explicit saved/live mode toggle.
- Saved mode searches the consolidated external prediction outputs and keeps the existing behaviour of the archived evaluation lookup.
- Live mode accepts a valid SMILES string, canonicalises it with RDKit, and returns a prediction dossier with three engine probabilities, consensus arbitration, applicability-domain routing, and a four-layer explanation stack.
- Deep Analytics page with AUROC vs F1 comparison, dataset balance, generalisation gap, and a Tanimoto similarity heatmap.
- Intelligence Core and Research Heritage pages that summarise the model families and the project lineage.
- Local feedback capture for live predictions.

## Models and Algorithms Implemented

The repository currently exposes five main model families through `src/models/registry.py`:

- `full_coverage_improvements`
- `full_coverage_descriptor_fp_meta_v4`
- `nephro_unsupervised`
- `nephro_chemberta`
- `modular_tanimoto_gpc`

The associated algorithms are implemented as follows:

- `nephro_chemberta` combines RDKit descriptors, Morgan fingerprints, structural alert flags, and ChemBERTa embeddings, then evaluates CatBoost and LightGBM components before writing the external metrics and per-sample predictions.
- `nephro_unsupervised` compares a PCA projection with an autoencoder-based latent representation, then evaluates both with CatBoost on the external test set.
- `full_coverage_improvements` and `full_coverage_descriptor_fp_meta_v4` are selection wrappers over broader full-coverage comparison outputs from the archived project families.
- `modular_tanimoto_gpc` loads the similarity-first Tanimoto GPC outputs from the historical modular nephrotoxicity run.

For live inference, `src/utils/live_analysis.py` builds the query context, computes nearest-neighbour similarity, flags structural alerts, derives scaffold context, loads serialized live engines, and routes the result through `src/utils/consensus.py`.

## Evaluation Metrics

The consolidated summary schema records:

- accuracy
- precision
- recall
- F1 score
- ROC-AUC
- PR-AUC
- training time where available
- inference time where available

The standalone model runners also compute auxiliary metrics such as Cohen's kappa, but the unified project reports focus on the metrics above.

## Results and Insights

The current workspace already contains a consolidated evaluation file in `outputs/final_reports/overall_evaluation.csv`. Ranked by external-test ROC-AUC (the held-out generalisation measure), the recorded runs are:

- `full_coverage_improvements` / `meta_logreg_knn_plus_lgbm`: accuracy 0.768, F1 0.762, ROC-AUC 0.860, PR-AUC 0.872
- `full_coverage_descriptor_fp_meta_v4` / `challenger_proxy_meta`: accuracy 0.755, F1 0.747, ROC-AUC 0.856, PR-AUC 0.868
- `modular_tanimoto_gpc` / `TanimotoGPC`: accuracy 0.768, F1 0.767, ROC-AUC 0.829, PR-AUC 0.846
- `nephro_unsupervised` / `Autoencoder`: accuracy 0.745, F1 0.735, ROC-AUC 0.826, PR-AUC 0.840
- `nephro_chemberta` / `chemberta_hybrid_cb_lgbm`: accuracy 0.722, F1 0.718, ROC-AUC 0.785, PR-AUC 0.759

These outputs indicate that the `full_coverage` meta-ensembles (logistic regression + kNN + LightGBM) are currently the strongest on external data, with the Tanimoto GPC and the unsupervised autoencoder close behind. The ChemBERTa hybrid is the strongest on internal cross-validation (ROC-AUC 0.921) but shows the largest internal-to-external gap (~0.14), which points to overfitting to the training chemistry and is worth discussing as a limitation rather than a headline result.

The dashboard's Deep Analytics view reflects this comparison through the AUROC/F1 chart, the generalisation-gap panel, and the Tanimoto similarity heatmap.

## Limitations

- Some historical model wrappers still depend on archived legacy project paths that are not present in this portable checkout, so those runners can fail if the external legacy artifacts are missing.
- Live prediction requires RDKit to parse the input SMILES string and load the local serialized live-inference bundles under `models/`.
- The live explanation stack currently exposes nearest neighbours, structural alerts, scaffold context, and a placeholder feature-importance layer until serialized attribution support is added.

## Future Improvements

- Restore or vendor the missing archived legacy runners so all historical wrappers are reproducible from this repository alone.
- Export real feature-attribution outputs for the live explanation stack.
- Add regression tests for the dashboard payload shape and the processed dataset cleaning report.
- Extend the feedback loop so confirmed live labels can be reused in retraining.

## Requirements

Install dependencies with:

```bash
pip install -r requirements.txt
```

Recommended packages include `pandas`, `scikit-learn`, `lightgbm`, `catboost`, `rdkit`, `torch`, and `transformers`.

Python 3.12 is recommended for the most stable experience in this workspace.

## Run The Project

Run the preprocessing and model pipeline:

```bash
python main.py
```

Start the local dashboard:

```bash
python serve_dashboard.py
```

If you need another port:

```bash
python serve_dashboard.py --port 8001
```

The dashboard is intentionally split into the main files below:

- `main.py` orchestrates preprocessing and model execution.
- `serve_dashboard.py` serves the API and static dashboard.
- `webapp/` contains the screening terminal and analytics interface.
- `src/models/` contains the model-family entry points.
- `src/utils/dashboard_data.py` shapes the data used by the UI.

Stop the dashboard:

```text
Ctrl+C
```

If port `8000` is already in use:

```bash
lsof -i :8000
lsof -ti :8000
kill $(lsof -ti :8000)
```

Or use another port:

```bash
python serve_dashboard.py --port 8001
```

### Common Useful Commands

List the available models:

```bash
python main.py --list-models
```

Skip dataset preparation:

```bash
python main.py --skip-data-prep
```

Force a rerun:

```bash
python main.py --force-rerun
```

Run the dashboard on a chosen host and port:

```bash
python serve_dashboard.py --host 127.0.0.1 --port 8000
```

Default dashboard URL:

```text
http://127.0.0.1:8000
```

## Deployment Options

This project can be deployed as a small web service.

Before deploying, make sure the dashboard data files already exist in the project:

- `datasets/processed/external_test_dataset.csv`
- `outputs/final_reports/overall_evaluation.csv`
- `outputs/detailed_predictions/all_model_predictions.csv`

If they do not exist yet, generate them locally first:

```bash
python main.py
```

The project now supports deployment-friendly host and port settings:

- `HOST`
- `PORT`

It also includes:

- `Dockerfile`
- `.dockerignore`

### Free Hosted Demo Option

The simplest free hosted options for a public demo are usually:

- Render Free Web Service
- Hugging Face Spaces with Docker

These free options are useful for demos and submissions, but they are not ideal for production. Platform limits and pricing can change, so always check the official documentation before deploying.

Official docs:

- Render free web services: `https://render.com/docs/free`
- Render web services: `https://render.com/docs/your-first-deploy`
- Render Docker deploys: `https://render.com/docs/docker`
- Hugging Face Spaces overview: `https://huggingface.co/docs/hub/spaces-overview`
- Hugging Face Docker Spaces: `https://huggingface.co/docs/hub/spaces-sdks-docker-first-demo`

#### Option A: Render Free Web Service

Good for:

- quick public demo
- easiest browser-based deployment flow

Important limitations:

- free services spin down after inactivity
- startup after sleep is slower
- free services are not recommended for production
- the filesystem is ephemeral

Recommended setup on Render:

1. Push this project to GitHub
2. Create a new Web Service on Render
3. Set the root directory to this project folder if the repository contains multiple folders
4. Choose either:
   - `Docker` runtime, which will use the included `Dockerfile`
   - `Python` runtime

If you use the Python runtime:

- Build Command:

```bash
pip install -r requirements.txt
```

- Start Command:

```bash
python serve_dashboard.py --host 0.0.0.0 --port $PORT
```

Recommended health check path:

```text
/api/health
```

#### Option B: Hugging Face Spaces With Docker

Good for:

- a free hosted demo
- a project showcase page

Important limitations:

- free hardware can sleep when idle
- the environment is not meant for long-running production traffic

Suggested setup:

1. Create a new Hugging Face Space
2. Choose `Docker` as the Space SDK
3. Push this project into that Space repository
4. Make sure the processed dataset and output CSV files are included
5. If needed for the Space repo, add the following YAML block at the top of that deployment repo's `README.md`:

```yaml
---
title: FINAL_ KV6013_NEPHROTOXICITY_PREDICTOR
sdk: docker
app_port: 8000
---
```

The included `Dockerfile` can then be used to build and run the dashboard service.

### Reliable Deployable Option

For a more stable deployment, use one of these:

- Render paid web service
- any VPS that can run Docker
- any Docker-compatible platform

This is the best option if you want:

- a more reliable always-on service
- better control over uptime
- a cleaner deployment path for demonstrations or assessment

#### Docker Deployment

Build the image:

```bash
docker build -t kv6013-dashboard .
```

Run it:

```bash
docker run --rm -p 8000:8000 kv6013-dashboard
```

Then open:

```text
http://127.0.0.1:8000
```

To run it on a server with a custom port:

```bash
docker run --rm -e HOST=0.0.0.0 -e PORT=8000 -p 8000:8000 kv6013-dashboard
```

If you deploy the Docker image to a platform such as Render, the platform can provide the `PORT` value automatically.

## Data Files Used By The Dashboard

The frontend reads these files from this same folder:

- `outputs/final_reports/overall_evaluation.csv`
- `outputs/detailed_predictions/all_model_predictions.csv`
- `datasets/processed/external_test_dataset.csv`

The dashboard can:

- search saved evaluation molecules by SMILES
- search a small set of supported medicine names
- compare the saved predictions across the five main models
- show analytics based on the processed external dataset
- trigger a refresh run of `python main.py --skip-data-prep`

The current dashboard does not yet run brand-new single-molecule inference from raw model weights.

## Dataset Workflow

1. Place source CSV files in `datasets/raw/`
2. Run `python main.py`
3. Cleaned files are written to `datasets/processed/`
4. Outputs are written to `outputs/`
5. Start `python serve_dashboard.py` to inspect them in the browser

## Quick Start

If everything is already installed and you just want to run it:

```bash
cd "/Users/rajee/Library/CloudStorage/OneDrive-NorthumbriaUniversity-ProductionAzureAD/YEAR 3/KV6013 DIRIL DATASET PROJECT/FINAL_ KV6013_NEPHROTOXICITY_PREDICTOR"
python serve_dashboard.py
```

If you want a full refresh first:

```bash
cd "/Users/rajee/Library/CloudStorage/OneDrive-NorthumbriaUniversity-ProductionAzureAD/YEAR 3/KV6013 DIRIL DATASET PROJECT/FINAL_ KV6013_NEPHROTOXICITY_PREDICTOR"
python main.py
python serve_dashboard.py
```
