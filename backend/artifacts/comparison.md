# DistressNet voice models: comparison

## Within RAVDESS (speaker cross-validation)

6 folds by speaker. Each fold tests on one block of speakers, uses 4 speakers from the next block to choose the epoch or C, the calibration temperature and the decision threshold, and trains on the rest. 'summary' is raw (threshold 0.5, uncalibrated); 'summary_calibrated' uses the validation temperature and threshold. Values are mean and standard deviation over the test folds.

| model | balanced acc (raw) | balanced acc (calibrated) | macro F1 | AUROC | specificity | ECE raw → calibrated | F1 > always-distress |
|---|---|---|---|---|---|---|---|
| cnn | 0.842 ± 0.025 | 0.835 ± 0.023 | 0.802 ± 0.037 | 0.911 ± 0.017 | 0.906 ± 0.083 | 0.101 ± 0.056 → 0.098 ± 0.056 | 5/6 folds |
| egemaps | 0.840 ± 0.059 | 0.792 ± 0.093 | 0.771 ± 0.085 | 0.919 ± 0.031 | 0.774 ± 0.239 | 0.110 ± 0.018 → 0.110 ± 0.019 | 5/6 folds |
| wavlm | 0.867 ± 0.045 | 0.864 ± 0.052 | 0.859 ± 0.046 | 0.942 ± 0.026 | 0.833 ± 0.110 | 0.095 ± 0.029 → 0.084 ± 0.011 | 6/6 folds |
| wavlm/binary-mean | 0.893 ± 0.038 | 0.893 ± 0.035 | 0.885 ± 0.024 | 0.969 ± 0.014 | 0.882 ± 0.105 | 0.084 ± 0.022 → 0.085 ± 0.017 | 6/6 folds |

Paired tests on calibrated balanced accuracy, same folds (a minus b):

| a | b | mean difference | paired t p | Wilcoxon p |
|---|---|---|---|---|
| cnn | egemaps | +0.043 | 0.3261 | 0.4375 |
| cnn | wavlm | -0.029 | 0.2876 | 0.3125 |
| cnn | wavlm/binary-mean | -0.058 | 0.0068 | 0.0312 |
| egemaps | wavlm | -0.071 | 0.0171 | 0.0312 |
| egemaps | wavlm/binary-mean | -0.101 | 0.0274 | 0.0312 |
| wavlm | wavlm/binary-mean | -0.029 | 0.1736 | 0.1875 |

With five or six folds the smallest possible two-sided Wilcoxon p is 0.0625 or 0.031, so read it with the t-test.

## Within CREMA-D (speaker cross-validation)

5 folds by speaker. Each fold tests on one block of speakers, uses 9 speakers from the next block to choose the epoch or C, the calibration temperature and the decision threshold, and trains on the rest. 'summary' is raw (threshold 0.5, uncalibrated); 'summary_calibrated' uses the validation temperature and threshold. Values are mean and standard deviation over the test folds.

| model | balanced acc (raw) | balanced acc (calibrated) | macro F1 | AUROC | specificity | ECE raw → calibrated | F1 > always-distress |
|---|---|---|---|---|---|---|---|
| cnn | 0.777 ± 0.029 | 0.767 ± 0.027 | 0.708 ± 0.046 | 0.857 ± 0.023 | 0.790 ± 0.038 | 0.168 ± 0.020 → 0.175 ± 0.021 | 0/5 folds |
| egemaps | 0.789 ± 0.028 | 0.782 ± 0.017 | 0.697 ± 0.038 | 0.864 ± 0.021 | 0.870 ± 0.071 | 0.149 ± 0.021 → 0.151 ± 0.021 | 0/5 folds |
| wavlm | 0.887 ± 0.010 | 0.887 ± 0.006 | 0.841 ± 0.028 | 0.958 ± 0.006 | 0.907 ± 0.040 | 0.069 ± 0.027 → 0.039 ± 0.021 | 4/5 folds |
| wavlm/binary-mean | 0.918 ± 0.006 | 0.916 ± 0.009 | 0.874 ± 0.030 | 0.975 ± 0.005 | 0.940 ± 0.031 | 0.059 ± 0.017 → 0.050 ± 0.017 | 5/5 folds |

Paired tests on calibrated balanced accuracy, same folds (a minus b):

| a | b | mean difference | paired t p | Wilcoxon p |
|---|---|---|---|---|
| cnn | egemaps | -0.015 | 0.2862 | 0.1875 |
| cnn | wavlm | -0.120 | 0.0005 | 0.0625 |
| cnn | wavlm/binary-mean | -0.149 | 0.0001 | 0.0625 |
| egemaps | wavlm | -0.105 | 0.0001 | 0.0625 |
| egemaps | wavlm/binary-mean | -0.134 | 0.0001 | 0.0625 |
| wavlm | wavlm/binary-mean | -0.029 | 0.0004 | 0.0625 |

With five or six folds the smallest possible two-sided Wilcoxon p is 0.0625 or 0.031, so read it with the t-test.

## Across corpora

Balanced accuracy, calibrated with a validation threshold, with the speaker-bootstrap 95% interval and specificity.

| setting | cnn | egemaps | wavlm | egemaps+corpus | wavlm+corpus | egemaps+speaker | wavlm+speaker | wavlm/binary-mean | wavlm/binary-mean+corpus | wavlm/binary-mean+speaker |
|---|---|---|---|---|---|---|---|---|---|---|
| RAVDESS speakers held out | 0.846 [0.81, 0.88] (spec 0.87) | 0.850 [0.79, 0.90] (spec 0.98) | 0.829 [0.71, 0.94] (spec 0.75) | 0.850 [0.79, 0.90] (spec 0.98) | 0.825 [0.72, 0.93] (spec 0.75) | 0.842 [0.75, 0.90] (spec 0.82) | 0.842 [0.72, 0.94] (spec 0.78) | 0.929 [0.90, 0.96] (spec 0.95) | 0.929 [0.90, 0.96] (spec 0.95) | 0.975 [0.95, 1.00] (spec 1.00) |
| CREMA-D speakers held out | 0.722 [0.68, 0.76] (spec 0.87) | 0.808 [0.77, 0.85] (spec 0.89) | 0.897 [0.85, 0.94] (spec 0.87) | 0.799 [0.75, 0.84] (spec 0.83) | 0.895 [0.85, 0.94] (spec 0.84) | 0.832 [0.80, 0.86] (spec 0.90) | 0.921 [0.88, 0.96] (spec 0.88) | 0.916 [0.87, 0.95] (spec 0.87) | 0.934 [0.90, 0.97] (spec 0.92) | 0.952 [0.92, 0.98] (spec 0.94) |
| Train RAVDESS, test CREMA-D | 0.480 [0.46, 0.50] (spec 0.21) | 0.651 [0.64, 0.66] (spec 0.98) | 0.508 [0.50, 0.51] (spec 0.03) | 0.587 [0.56, 0.61] (spec 0.60) | 0.558 [0.55, 0.57] (spec 0.18) | 0.568 [0.55, 0.59] (spec 0.37) | 0.591 [0.58, 0.61] (spec 0.23) | 0.520 [0.51, 0.53] (spec 0.06) | 0.693 [0.67, 0.72] (spec 0.56) | 0.749 [0.73, 0.77] (spec 0.64) |
| Train CREMA-D, test RAVDESS | 0.618 [0.56, 0.68] (spec 0.34) | 0.688 [0.66, 0.71] (spec 1.00) | 0.500 [0.50, 0.50] (spec 0.00) | 0.735 [0.68, 0.78] (spec 0.83) | 0.636 [0.59, 0.68] (spec 0.33) | 0.730 [0.68, 0.78] (spec 0.83) | 0.621 [0.59, 0.65] (spec 0.30) | 0.516 [0.50, 0.53] (spec 0.03) | 0.740 [0.70, 0.78] (spec 0.57) | 0.736 [0.71, 0.77] (spec 0.55) |

## Proposal targets

| target | result | value |
|---|---|---|
| RAVDESS wavlm balanced accuracy ≥ 0.89 | not met | 0.864 calibrated, 0.867 raw |
| RAVDESS wavlm F1 above always-distress in every fold | met | 6/6 |
| RAVDESS wavlm ECE ≤ 0.1 after temperature scaling | met | 0.084 (raw 0.095) |
| RAVDESS wavlm/binary-mean balanced accuracy ≥ 0.89 | met | 0.893 calibrated, 0.893 raw |
| RAVDESS wavlm/binary-mean F1 above always-distress in every fold | met | 6/6 |
| RAVDESS wavlm/binary-mean ECE ≤ 0.1 after temperature scaling | met | 0.085 (raw 0.084) |
| CREMA-D wavlm balanced accuracy ≥ 0.8 | met | 0.887 calibrated, 0.887 raw |
| CREMA-D wavlm F1 above always-distress in every fold | not met | 4/5 |
| CREMA-D wavlm ECE ≤ 0.1 after temperature scaling | met | 0.039 (raw 0.069) |
| CREMA-D wavlm/binary-mean balanced accuracy ≥ 0.8 | met | 0.916 calibrated, 0.918 raw |
| CREMA-D wavlm/binary-mean F1 above always-distress in every fold | met | 5/5 |
| CREMA-D wavlm/binary-mean ECE ≤ 0.1 after temperature scaling | met | 0.050 (raw 0.059) |
| Train RAVDESS, test CREMA-D, wavlm (no normalization): ≥ 0.65, spec > 0.5, above eGeMAPS | not met | 0.508, spec 0.03, eGeMAPS 0.651 |
| Train CREMA-D, test RAVDESS, wavlm (no normalization): ≥ 0.65, spec > 0.5, above eGeMAPS | not met | 0.500, spec 0.00, eGeMAPS 0.688 |
| Train RAVDESS, test CREMA-D, wavlm+corpus (corpus): ≥ 0.65, spec > 0.5, above eGeMAPS | not met | 0.558, spec 0.18, eGeMAPS 0.587 |
| Train CREMA-D, test RAVDESS, wavlm+corpus (corpus): ≥ 0.65, spec > 0.5, above eGeMAPS | not met | 0.636, spec 0.33, eGeMAPS 0.735 |
| Train RAVDESS, test CREMA-D, wavlm+speaker (speaker): ≥ 0.65, spec > 0.5, above eGeMAPS | not met | 0.591, spec 0.23, eGeMAPS 0.568 |
| Train CREMA-D, test RAVDESS, wavlm+speaker (speaker): ≥ 0.65, spec > 0.5, above eGeMAPS | not met | 0.621, spec 0.30, eGeMAPS 0.730 |
| Train RAVDESS, test CREMA-D, wavlm/binary-mean (no normalization): ≥ 0.65, spec > 0.5, above eGeMAPS | not met | 0.520, spec 0.06, eGeMAPS 0.651 |
| Train CREMA-D, test RAVDESS, wavlm/binary-mean (no normalization): ≥ 0.65, spec > 0.5, above eGeMAPS | not met | 0.516, spec 0.03, eGeMAPS 0.688 |
| Train RAVDESS, test CREMA-D, wavlm/binary-mean+corpus (corpus): ≥ 0.65, spec > 0.5, above eGeMAPS | met | 0.693, spec 0.56, eGeMAPS 0.587 |
| Train CREMA-D, test RAVDESS, wavlm/binary-mean+corpus (corpus): ≥ 0.65, spec > 0.5, above eGeMAPS | met | 0.740, spec 0.57, eGeMAPS 0.735 |
| Train RAVDESS, test CREMA-D, wavlm/binary-mean+speaker (speaker): ≥ 0.65, spec > 0.5, above eGeMAPS | met | 0.749, spec 0.64, eGeMAPS 0.568 |
| Train CREMA-D, test RAVDESS, wavlm/binary-mean+speaker (speaker): ≥ 0.65, spec > 0.5, above eGeMAPS | met | 0.736, spec 0.55, eGeMAPS 0.730 |

Normalized cross-corpus runs standardize features with unlabeled audio from the test corpus or speaker; say so wherever those numbers are quoted.
