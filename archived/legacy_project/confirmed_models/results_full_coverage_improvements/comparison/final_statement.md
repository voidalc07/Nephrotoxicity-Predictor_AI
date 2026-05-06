# Full-Coverage Recommendation

- Protected baseline: `ens_r2_b1024_dice_k31_25_15_isotonic_thr040`
- Best internally selected variant (fair default candidate): `lgbm_morgan_r2_only__cal_sigmoid__thr_f1`
- Best external-ranked variant (reporting-only): `blend_inverse_stacking__cal_none__thr_kappa`
- Beat baseline fairly (AUROC/ACC/F1 all higher): yes
- Closed gap to paper full-coverage benchmark (AUC 0.868, ACC 0.878, F1 0.877): no

Methodology checks:
- Best-improved model chosen by internal repeated-split metrics only.
- External ranking is reporting-only across pre-finalized variants.
- No external threshold tuning.
- No external calibration fitting.
- No external feature/parameter selection.
- Coverage fixed at 1.000 for all variants.