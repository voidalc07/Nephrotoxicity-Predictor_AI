# Model Score Summary

This folder keeps model score outputs together and separate from utility code.

## Generated Files
- `all_score_csv_rows.csv`: Union of all rows from discovered score CSV files.
- `all_scores_long.csv`: Long-format metric rows (`metric`, `score`) for analysis.
- `score_csv_summary.csv`: One row per score CSV with inferred primary metric and best score.
- `score_csv_index.csv`: Mapping of source score CSVs to copied files in `source_csvs/`.
- `model_score_summary.csv`: Aggregated score summary by model group.

## Model Score Snapshot
- confirmed_models: best=0.860351, avg=0.820773, score_csv_count=256
- nephrotox_fixed: best=0.987143, avg=0.873286, score_csv_count=3
- nephrotox_project_plus: best=0.727851, avg=0.727851, score_csv_count=1
- nephrotox_final: best=nan, avg=nan, score_csv_count=0
- nephrotox_modular: best=nan, avg=nan, score_csv_count=0
