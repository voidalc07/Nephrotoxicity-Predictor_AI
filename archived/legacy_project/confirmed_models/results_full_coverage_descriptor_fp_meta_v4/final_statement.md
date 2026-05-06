# Full-Coverage Descriptor+FP Meta (v4)

- Best AUROC model: `knn_r2r3_concat_1024_mean__cal_none__thr_f1` (AUROC=0.8596, F1=0.7331, ACC=0.7517)
- Best F1/ACC model: `meta_logreg_knn_lgbm_descriptor_fp_v4__cal_none__thr_kappa` (AUROC=0.8447, F1=0.7767, ACC=0.7715)
- Best overall fair full-coverage model: `meta_logreg_knn_lgbm_descriptor_fp_v4__cal_none__thr_kappa`
- Overall selection decision: `no_challenger_reference_available`
- Beats protected baseline on AUROC+F1+ACC: no
- Beats current challenger on AUROC+F1+ACC: no

Priority targets (F1, ACC, then AUROC):
- F1 > 0.790: no
- ACC > 0.785: no
- AUROC > 0.860: no

Methodology checks:
- Full coverage only (coverage=1.000 for all variants).
- No external threshold tuning.
- No external calibration fitting.
- No external feature/parameter tuning.
- Selection/tuning done with repeated internal 80/10/10 only.