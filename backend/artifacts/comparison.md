# DistressNet voice models: comparison

## Within RAVDESS (speaker cross-validation)

6 folds by speaker. Each fold tests on one block of speakers, uses 4 speakers from the next block to choose the epoch or C, the calibration temperature and the decision threshold, and trains on the rest. 'summary' is raw (threshold 0.5, uncalibrated); 'summary_calibrated' uses the validation temperature and threshold. Values are mean and standard deviation over the test folds.

| model | balanced acc (raw) | balanced acc (calibrated) | macro F1 | AUROC | specificity | ECE raw → calibrated | F1 > always-distress |
|---|---|---|---|---|---|---|---|
| cnn | 0.838 ± 0.031 | 0.842 ± 0.033 | 0.806 ± 0.051 | 0.910 ± 0.020 | 0.927 ± 0.048 | 0.122 ± 0.062 → 0.117 ± 0.058 | 5/6 folds |
| egemaps | 0.840 ± 0.059 | 0.792 ± 0.093 | 0.771 ± 0.085 | 0.919 ± 0.031 | 0.774 ± 0.239 | 0.110 ± 0.018 → 0.110 ± 0.019 | 5/6 folds |
| wavlm | 0.871 ± 0.026 | 0.875 ± 0.018 | 0.861 ± 0.024 | 0.943 ± 0.019 | 0.885 ± 0.062 | 0.107 ± 0.028 → 0.084 ± 0.022 | 6/6 folds |
| wavlm/binary-mean | 0.898 ± 0.035 | 0.898 ± 0.030 | 0.890 ± 0.020 | 0.970 ± 0.013 | 0.892 ± 0.096 | 0.082 ± 0.019 → 0.081 ± 0.017 | 6/6 folds |

Paired tests on calibrated balanced accuracy, same folds (a minus b):

| a | b | mean difference | paired t p | Wilcoxon p |
|---|---|---|---|---|
| cnn | egemaps | +0.050 | 0.2919 | 0.3125 |
| cnn | wavlm | -0.033 | 0.049 | 0.0312 |
| cnn | wavlm/binary-mean | -0.056 | 0.0291 | 0.0312 |
| egemaps | wavlm | -0.083 | 0.0767 | 0.125 |
| egemaps | wavlm/binary-mean | -0.106 | 0.0234 | 0.0312 |
| wavlm | wavlm/binary-mean | -0.023 | 0.0895 | 0.0312 |

With five or six folds the smallest possible two-sided Wilcoxon p is 0.0625 or 0.031, so read it with the t-test.

## Within CREMA-D (speaker cross-validation)

5 folds by speaker. Each fold tests on one block of speakers, uses 9 speakers from the next block to choose the epoch or C, the calibration temperature and the decision threshold, and trains on the rest. 'summary' is raw (threshold 0.5, uncalibrated); 'summary_calibrated' uses the validation temperature and threshold. Values are mean and standard deviation over the test folds.

| model | balanced acc (raw) | balanced acc (calibrated) | macro F1 | AUROC | specificity | ECE raw → calibrated | F1 > always-distress |
|---|---|---|---|---|---|---|---|
| cnn | 0.770 ± 0.025 | 0.758 ± 0.035 | 0.687 ± 0.062 | 0.862 ± 0.022 | 0.805 ± 0.134 | 0.156 ± 0.056 → 0.160 ± 0.055 | 1/5 folds |
| egemaps | 0.789 ± 0.028 | 0.782 ± 0.017 | 0.697 ± 0.038 | 0.864 ± 0.021 | 0.870 ± 0.071 | 0.149 ± 0.021 → 0.151 ± 0.021 | 0/5 folds |
| wavlm | 0.894 ± 0.009 | 0.892 ± 0.008 | 0.853 ± 0.026 | 0.962 ± 0.004 | 0.901 ± 0.042 | 0.077 ± 0.014 → 0.045 ± 0.020 | 5/5 folds |
| wavlm/binary-mean | 0.920 ± 0.009 | 0.912 ± 0.014 | 0.863 ± 0.038 | 0.976 ± 0.005 | 0.947 ± 0.029 | 0.060 ± 0.017 → 0.052 ± 0.016 | 5/5 folds |

Paired tests on calibrated balanced accuracy, same folds (a minus b):

| a | b | mean difference | paired t p | Wilcoxon p |
|---|---|---|---|---|
| cnn | egemaps | -0.024 | 0.2378 | 0.3125 |
| cnn | wavlm | -0.134 | 0.0008 | 0.0625 |
| cnn | wavlm/binary-mean | -0.154 | 0.0003 | 0.0625 |
| egemaps | wavlm | -0.110 | 0.0001 | 0.0625 |
| egemaps | wavlm/binary-mean | -0.129 | 0.0003 | 0.0625 |
| wavlm | wavlm/binary-mean | -0.020 | 0.0604 | 0.0625 |

With five or six folds the smallest possible two-sided Wilcoxon p is 0.0625 or 0.031, so read it with the t-test.

## Across corpora

Balanced accuracy, calibrated with a validation threshold, with the speaker-bootstrap 95% interval and specificity.

| setting | cnn | egemaps | wavlm | egemaps+corpus | wavlm+corpus | egemaps+speaker | wavlm+speaker | wavlm/binary-mean | wavlm/binary-mean+corpus | wavlm/binary-mean+speaker |
|---|---|---|---|---|---|---|---|---|---|---|
| RAVDESS speakers held out | 0.758 [0.59, 0.87] (spec 0.77) | 0.850 [0.79, 0.90] (spec 0.98) | 0.838 [0.73, 0.94] (spec 0.78) | 0.850 [0.79, 0.90] (spec 0.98) | 0.833 [0.66, 0.94] (spec 0.77) | 0.842 [0.75, 0.90] (spec 0.82) | 0.858 [0.74, 0.94] (spec 0.82) | 0.929 [0.90, 0.96] (spec 0.95) | 0.929 [0.90, 0.96] (spec 0.95) | 0.975 [0.95, 1.00] (spec 1.00) |
| CREMA-D speakers held out | 0.723 [0.68, 0.77] (spec 0.88) | 0.808 [0.77, 0.85] (spec 0.89) | 0.899 [0.85, 0.94] (spec 0.86) | 0.799 [0.75, 0.84] (spec 0.83) | 0.898 [0.85, 0.94] (spec 0.85) | 0.832 [0.80, 0.86] (spec 0.90) | 0.923 [0.88, 0.96] (spec 0.89) | 0.917 [0.88, 0.95] (spec 0.89) | 0.925 [0.89, 0.96] (spec 0.90) | 0.958 [0.94, 0.98] (spec 0.96) |
| Train RAVDESS, test CREMA-D | 0.475 [0.46, 0.49] (spec 0.15) | 0.651 [0.64, 0.66] (spec 0.98) | 0.522 [0.51, 0.53] (spec 0.05) | 0.587 [0.56, 0.61] (spec 0.60) | 0.576 [0.56, 0.59] (spec 0.24) | 0.568 [0.55, 0.59] (spec 0.37) | 0.628 [0.61, 0.64] (spec 0.41) | 0.520 [0.51, 0.53] (spec 0.06) | 0.692 [0.67, 0.72] (spec 0.56) | 0.748 [0.73, 0.76] (spec 0.64) |
| Train CREMA-D, test RAVDESS | 0.645 [0.59, 0.70] (spec 0.51) | 0.688 [0.66, 0.71] (spec 1.00) | 0.503 [0.50, 0.51] (spec 0.01) | 0.735 [0.68, 0.78] (spec 0.83) | 0.643 [0.60, 0.69] (spec 0.34) | 0.730 [0.68, 0.78] (spec 0.83) | 0.620 [0.59, 0.65] (spec 0.30) | 0.516 [0.50, 0.53] (spec 0.03) | 0.744 [0.70, 0.79] (spec 0.59) | 0.736 [0.70, 0.77] (spec 0.56) |

## Proposal targets

| target | result | value |
|---|---|---|
| RAVDESS wavlm balanced accuracy ≥ 0.89 | not met | 0.875 calibrated, 0.871 raw |
| RAVDESS wavlm F1 above always-distress in every fold | met | 6/6 |
| RAVDESS wavlm ECE ≤ 0.1 after temperature scaling | met | 0.084 (raw 0.107) |
| RAVDESS wavlm/binary-mean balanced accuracy ≥ 0.89 | met | 0.898 calibrated, 0.898 raw |
| RAVDESS wavlm/binary-mean F1 above always-distress in every fold | met | 6/6 |
| RAVDESS wavlm/binary-mean ECE ≤ 0.1 after temperature scaling | met | 0.081 (raw 0.082) |
| CREMA-D wavlm balanced accuracy ≥ 0.8 | met | 0.892 calibrated, 0.894 raw |
| CREMA-D wavlm F1 above always-distress in every fold | met | 5/5 |
| CREMA-D wavlm ECE ≤ 0.1 after temperature scaling | met | 0.045 (raw 0.077) |
| CREMA-D wavlm/binary-mean balanced accuracy ≥ 0.8 | met | 0.912 calibrated, 0.920 raw |
| CREMA-D wavlm/binary-mean F1 above always-distress in every fold | met | 5/5 |
| CREMA-D wavlm/binary-mean ECE ≤ 0.1 after temperature scaling | met | 0.052 (raw 0.060) |
| Train RAVDESS, test CREMA-D, wavlm (no normalization): ≥ 0.65, spec > 0.5, above eGeMAPS | not met | 0.522, spec 0.05, eGeMAPS 0.651 |
| Train CREMA-D, test RAVDESS, wavlm (no normalization): ≥ 0.65, spec > 0.5, above eGeMAPS | not met | 0.503, spec 0.01, eGeMAPS 0.688 |
| Train RAVDESS, test CREMA-D, wavlm+corpus (corpus): ≥ 0.65, spec > 0.5, above eGeMAPS | not met | 0.576, spec 0.24, eGeMAPS 0.587 |
| Train CREMA-D, test RAVDESS, wavlm+corpus (corpus): ≥ 0.65, spec > 0.5, above eGeMAPS | not met | 0.643, spec 0.34, eGeMAPS 0.735 |
| Train RAVDESS, test CREMA-D, wavlm+speaker (speaker): ≥ 0.65, spec > 0.5, above eGeMAPS | not met | 0.628, spec 0.41, eGeMAPS 0.568 |
| Train CREMA-D, test RAVDESS, wavlm+speaker (speaker): ≥ 0.65, spec > 0.5, above eGeMAPS | not met | 0.620, spec 0.30, eGeMAPS 0.730 |
| Train RAVDESS, test CREMA-D, wavlm/binary-mean (no normalization): ≥ 0.65, spec > 0.5, above eGeMAPS | not met | 0.520, spec 0.06, eGeMAPS 0.651 |
| Train CREMA-D, test RAVDESS, wavlm/binary-mean (no normalization): ≥ 0.65, spec > 0.5, above eGeMAPS | not met | 0.516, spec 0.03, eGeMAPS 0.688 |
| Train RAVDESS, test CREMA-D, wavlm/binary-mean+corpus (corpus): ≥ 0.65, spec > 0.5, above eGeMAPS | met | 0.692, spec 0.56, eGeMAPS 0.587 |
| Train CREMA-D, test RAVDESS, wavlm/binary-mean+corpus (corpus): ≥ 0.65, spec > 0.5, above eGeMAPS | met | 0.744, spec 0.59, eGeMAPS 0.735 |
| Train RAVDESS, test CREMA-D, wavlm/binary-mean+speaker (speaker): ≥ 0.65, spec > 0.5, above eGeMAPS | met | 0.748, spec 0.64, eGeMAPS 0.568 |
| Train CREMA-D, test RAVDESS, wavlm/binary-mean+speaker (speaker): ≥ 0.65, spec > 0.5, above eGeMAPS | met | 0.736, spec 0.56, eGeMAPS 0.730 |

Normalized cross-corpus runs standardize features with unlabeled audio from the test corpus or speaker; say so wherever those numbers are quoted.
