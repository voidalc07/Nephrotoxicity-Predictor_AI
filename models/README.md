This directory stores serialized live-inference artifacts for the portable dashboard.

Expected layout after running `scripts/serialize_models.py`:

- `manifest.json`
- `descriptor_fp_ensemble/engine.json`
- `descriptor_fp_ensemble/*.joblib`
- `full_coverage_ensemble/engine.json`
- `full_coverage_ensemble/*.joblib`

The runtime loader in `src/utils/engine_loader.py` discovers these files at server startup and degrades gracefully when any bundle is missing.
