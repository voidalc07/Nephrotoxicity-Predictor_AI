# Datasets

This folder is where the project keeps its data files.

Think of it like this:

- `raw/` holds the original files we start with.
- `processed/` holds the cleaned files the project creates automatically.

Put replacement source CSVs into `raw/`.

When you run `python main.py`, the project cleans those files and writes the finished versions into `processed/`.

The original filenames in `raw/` should stay as:

- `model construction dataset.csv`
- `external test dataset.csv`
- `model molecular murcko scaffolds.csv`
- `model molecular carbon scaffolds.csv`
- `model nephrotoxicity molecular murcko scaffolds.csv`
- `model nephrotoxicity molecular carbon scaffolds.csv`
- `external molecular murcko scaffolds.csv`
- `external molecular carbon scaffolds.csv`

After processing, the cleaned files are saved in `processed/` with underscore-style names, along with `dataset_cleaning_report.json`.
