# Confidence-Routed Cascade with Consensus Arbitration

## Current Portable Build

The extracted `KV6013_PORTABLE` folder is a dashboard-first portable build.

- `serve_dashboard.py` exposes:
  - `GET /api/overview`
  - `GET /api/search`
  - `POST /api/run-main`
- `src/utils/dashboard_data.py` currently:
  - loads summary metrics and saved external-test predictions
  - builds analytics charts
  - searches saved predictions only
- `webapp/app.js` and `webapp/index.html` currently implement:
  - one predictor mode
  - one input box
  - a saved-predictions search workflow

## Important Constraint

The portable folder contains datasets and saved output CSVs, but it does not currently contain the archived model artifact directories referenced by the model wrappers for all live engines.

Practical implication:

- saved-evaluation search works now
- chemistry explanation layers can be implemented now
- true live multi-engine inference needs either:
  - model artifacts copied into the portable build, or
  - a fallback mode that marks some engines unavailable

## Best Fit Architecture

### 1. New backend live-analysis module

Add a new module under `src/utils/` or `src/live/` responsible for:

- parsing arbitrary SMILES
- computing applicability-domain signals
- running available engines
- assembling the final routed response

Suggested responsibilities:

- `build_query_context(smiles)`
- `compute_domain_gate(smiles)`
- `route_query(domain_metrics)`
- `run_available_engines(smiles, route)`
- `build_explanation_stack(smiles, engine_outputs)`
- `build_live_payload(smiles)`

### 2. Applicability-domain gate

The Tanimoto GPC should act as the routing gate conceptually, but the portable build needs a robust fallback when the true GPC runtime assets are missing.

Portable-safe gate signals:

- maximum Tanimoto similarity to training molecules
- top-k nearest neighbours
- optional GPC probability and posterior variance if live GPC assets are available

This gives a route decision surface:

- Route A: high similarity, low uncertainty
- Route B: moderate similarity, moderate uncertainty
- Route C: low similarity, high uncertainty

### 3. Engine execution policy

Implement engine execution as capability-aware, not assumption-aware.

For each engine:

- Descriptor + Fingerprint Ensemble
- Full Coverage Ensemble
- Tanimoto GPC
- ChemBERTa Hybrid

the backend should report:

- `available: true/false`
- `reason_unavailable`
- `predicted_score`
- `predicted_label`
- any engine-specific explanation fields

This keeps the live mode honest inside the portable build.

### 4. Explanation stack

This can be implemented largely from existing portable assets plus RDKit:

- Nearest-neighbour layer
  - top 5 training neighbours by Tanimoto similarity
  - include known labels and scores
- Structural alert layer
  - SMARTS matches from the existing alert definitions
  - attach human-readable names
- Scaffold context layer
  - compute Murcko scaffold for the query
  - compare against training scaffold families
  - report toxic/non-toxic ratio for matching family
- Feature-importance layer
  - only when the live engine exposes local contributions
  - otherwise return an explicit `not_available_in_portable_build`

### 5. Dashboard integration

The Screening Terminal should split into two explicit modes:

- `Search saved`
- `Predict live`

Saved mode:

- keep existing behaviour unchanged
- preserve the 302-molecule comparative evidence

Live mode:

- accept arbitrary SMILES
- show route taken
- show applicability-domain badge
- show consensus bar
- show per-engine results only for engines that ran
- show explanation stack
- show out-of-domain warning when needed

### 6. Feedback loop

Low-risk portable implementation:

- add `POST /api/feedback`
- append confirmed labels to a local CSV such as `outputs/feedback/confirmed_live_labels.csv`
- do not auto-retrain from the dashboard
- expose retraining as a separate later step

## Recommended Delivery Order

1. Add a new live-analysis backend endpoint without touching saved search behaviour.
2. Add neighbour, alert, and scaffold explanations first.
3. Add route selection and route badges.
4. Add engine-availability reporting.
5. Add the UI mode toggle and live result cards.
6. Add feedback capture last.

## Concrete Code Touchpoints

- `serve_dashboard.py`
  - add live-analysis and feedback endpoints
- `src/utils/dashboard_data.py`
  - keep saved-mode search logic
  - avoid overloading this file with all live-analysis logic
- new module, e.g. `src/utils/live_analysis.py`
  - routing + explanations + capability-aware engine execution
- `webapp/index.html`
  - add mode toggle and live explanation panels
- `webapp/app.js`
  - branch between saved-search and live-analysis flows
- `webapp/styles.css`
  - add route badge, domain badge, consensus bar, and explanation layout styles

## Main Risk To Resolve Before Full Live Inference

If we want true live probabilities from all intended engines, we need to decide whether this portable build should:

- remain self-contained and degrade gracefully when engine assets are absent, or
- absorb the missing trained artifacts into the portable package

That decision affects portability, size, and how much of the workflow can be made fully operational offline.
