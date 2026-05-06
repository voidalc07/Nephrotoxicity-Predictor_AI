# KV6013 Unified Python Project

This folder is the consolidated Python submission project for KV6013.

## What is included

- `confirmed_models/`
  - `nephro_sota.py`
  - `nephro_chemberta.py`
  - `nephro_unsupervised.py`
  - `run_full_coverage_improvements.py`
  - `run_full_coverage_descriptor_fp_meta_v4.py`
- `noteworthy_models/`
  - `nephrotox_final/`
  - `nephrotox_fixed/`
  - `nephrotox_modular/`
  - `nephrotox_project_plus/`
- Shared helper modules for full-coverage scripts:
  - `experiments/`
  - `models/`
  - `utils/`

## Frontend

The frontend is intentionally separate and remains here:

- `../nephrotox-local-fixed`

## One shared environment

Use the single environment at project root:

```bash
cd "/Users/rajee/Desktop/Codebase for KV6013/KV6013-Python-Project"
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements-unified.txt
```

## Data location used by some scripts

The following scripts still use an absolute data path:

- `confirmed_models/nephro_sota.py`
- `confirmed_models/nephro_chemberta.py`
- `confirmed_models/nephro_unsupervised.py`

Expected data folder:

`/Users/rajee/Desktop/KV6013-Induvidual Project/molecular data and scaffolds`

## Recommended run order (submission demo)

1. `noteworthy_models/nephrotox_fixed/main.py` (primary upgraded pipeline)
2. `noteworthy_models/nephrotox_project_plus/main.py` (GIN/Hybrid/ChemBERTa comparison)
3. `confirmed_models/run_full_coverage_improvements.py` (from this root, so shared modules resolve)

Example:

```bash
cd "/Users/rajee/Desktop/Codebase for KV6013/KV6013-Python-Project"
source .venv/bin/activate
python noteworthy_models/nephrotox_fixed/main.py --help
python noteworthy_models/nephrotox_project_plus/main.py --help
python confirmed_models/run_full_coverage_improvements.py --help
```
