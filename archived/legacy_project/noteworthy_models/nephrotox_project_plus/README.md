# Nephrotoxicity Project (PyCharm-ready, upgraded)

This project now supports three models with one metric pipeline:

- `gin` - graph-only baseline
- `gin_hybrid` - GIN + RDKit descriptors
- `chemberta` - ChemBERTa SMILES classifier

All three use:
- scaffold split (`murcko` or `carbon`)
- threshold tuning on validation data
- class imbalance handling
- early stopping
- external test evaluation
- the same metrics: accuracy, recall, precision, f1, kappa, auroc, specificity, sensitivity, mcc

## Exact folder structure

```text
nephrotox_project_plus/
├── configs/
│   ├── __init__.py
│   └── config.py
├── data/
│   ├── train.csv
│   └── test.csv
├── models/
│   ├── __init__.py
│   ├── base_model.py
│   ├── chemberta.py
│   ├── gin.py
│   ├── gin_hybrid.py
│   └── registry.py
├── outputs/
├── utils/
│   ├── __init__.py
│   ├── data_utils.py
│   ├── descriptors.py
│   ├── metrics.py
│   ├── scaffold_split.py
│   ├── trainer.py
│   └── trainer_text.py
├── compare_models.py
├── main.py
├── predict.py
├── requirements.txt
└── README.md
```

## Where to add files exactly

You do not need to add the upgraded model files manually if you use this rebuilt project. They are already in place.

If you later add another graph model:
- create a new file in `models/`, for example `models/graphsage.py`
- register it in `models/registry.py`
- then run it through `main.py --model_name your_model_name`

If you later add another text model:
- create a new file in `models/`, for example `models/molformer.py`
- register it in `models/registry.py`
- it will use the text trainer if you set family to `text`

## Your dataset location

Put your CSVs here:

- `data/train.csv`
- `data/test.csv`

Supported SMILES column names:
- `smiles`
- `canonical SMILES`
- `canonical_smiles`
- `canon_smiles`

Label column must be:
- `label`

With values:
- `1` = nephrotoxic
- `0` = non-nephrotoxic

## Run in PyCharm

Open the project folder in PyCharm and run `main.py`.

### GIN
```bash
python main.py --model_name gin --output_dir outputs/gin
```

### Hybrid GIN
```bash
python main.py --model_name gin_hybrid --output_dir outputs/gin_hybrid
```

### ChemBERTa
```bash
python main.py --model_name chemberta --output_dir outputs/chemberta
```

### Compare all three
```bash
python compare_models.py
```

## Recommended quick settings for M2 Air

### Fast and safe
```bash
python main.py --model_name gin --hidden_dim 96 --num_layers 4 --dropout 0.35 --epochs 60 --output_dir outputs/gin
python main.py --model_name gin_hybrid --hidden_dim 96 --num_layers 4 --dropout 0.35 --epochs 60 --output_dir outputs/gin_hybrid
python main.py --model_name chemberta --batch_size 16 --epochs 30 --patience 6 --output_dir outputs/chemberta
```

## Inference

### Graph or hybrid
```bash
python predict.py --model_path outputs/gin_hybrid/best_model.pt --smiles "CCO" "CCN(CC)CC"
```

### ChemBERTa
```bash
python predict.py --model_path outputs/chemberta/best_model.pt --smiles "CCO" "CCN(CC)CC"
```

## Notes

- `chemberta` downloads the Hugging Face model the first time you run it.
- `gin_hybrid` stores a descriptor scaler inside the checkpoint so inference stays compatible.
- `compare_models.py` builds a leaderboard CSV with the same metrics for every model.
