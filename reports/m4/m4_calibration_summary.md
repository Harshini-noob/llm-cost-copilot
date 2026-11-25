# Phase M4.3 — Probability Calibration and Reliability Analysis

**Project:** LLM Cost Copilot  
**Milestone:** M4.3 (Probability Calibration, ECE, Brier Score, and Reliability Analysis)  
**Date:** November 24, 2026  
**Status:** COMPLETE (Leakage-Safe OOF Calibration Verified)  

---

## 1. Executive Summary & Calibration Verdict

This evaluation investigates whether the correctness probabilities predicted by the learned routers
(kNN and regularized Logistic Regression) are statistically reliable and well-calibrated against observed test outcomes.

### Key Empirical Findings
- **Learned Router (Logistic Regression) is inherently well-calibrated:**
  - **Raw Brier Score:** `0.1734`
  - **Raw ECE:** `0.0389` (error is only ~3.9%)
- **kNN Router is less calibrated and under-confident:**
  - **Raw Brier Score:** `0.2173`
  - **Raw ECE:** `0.1123`
- **Impact of Calibration Correction (Platt Scaling fitted on Training OOF):**
  - Refines Logistic Regression ECE to `0.0283` and Brier Score to `0.1727`.
- **Downstream Routing Invariance:** Probability calibration maintains monotonic ranking and leaves model selections
  virtually unchanged across the Pareto sweep, demonstrating that uncalibrated logistic probabilities already provide
  optimal utility-based ranking.

---

## 2. Overall Calibration Performance Table

| Router Model | Calibration Method | Brier Score (Lower is Better) | Expected Calibration Error (ECE) | Maximum Calibration Error (MCE) | Test Observations |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **kNN (k=5)** | `Raw (Uncalibrated)` | 0.2173 | 0.1123 | 0.2095 | 67,122 |
| **kNN (k=5)** | `Platt Scaling` | 0.2112 | 0.0831 | 0.2983 | 67,122 |
| **kNN (k=5)** | `Isotonic Regression` | 0.2023 | 0.0490 | 0.0613 | 67,122 |
| **Learned Router (LR)** | `Raw (Uncalibrated)` | 0.1734 | 0.0389 | 0.0594 | 67,122 |
| **Learned Router (LR)** | `Platt Scaling` | 0.1727 | 0.0283 | 0.0434 | 67,122 |
| **Learned Router (LR)** | `Isotonic Regression` | 0.1740 | 0.0396 | 0.0785 | 67,122 |

---

## 3. Reliability Diagrams (10 Uniform Bins in [0, 1])

### A. Learned Router (Logistic Regression, Uncalibrated)
```
Bin Range        | Samples | Conf   | Acc    | Gap    | Reliability Histogram (Observed vs Perfect)
-----------------|---------|--------|--------|--------|-------------------------------------------
[0.00, 0.10) |    2335 | 0.0573 | 0.0531 | 0.0042 | [  #                            ]
[0.10, 0.20) |    2742 | 0.1526 | 0.1360 | 0.0166 | [    #|                         ]
[0.20, 0.30) |    3585 | 0.2505 | 0.2214 | 0.0291 | [       #|                      ]
[0.30, 0.40) |    3476 | 0.3500 | 0.3094 | 0.0406 | [         # |                   ]
[0.40, 0.50) |    3958 | 0.4528 | 0.4100 | 0.0428 | [            # |                ]
[0.50, 0.60) |    5670 | 0.5532 | 0.4938 | 0.0594 | [               # |             ]
[0.60, 0.70) |    8183 | 0.6532 | 0.6093 | 0.0439 | [                  # |          ]
[0.70, 0.80) |   11710 | 0.7525 | 0.7015 | 0.0510 | [                     # |       ]
[0.80, 0.90) |   14925 | 0.8518 | 0.8095 | 0.0423 | [                        # |    ]
[0.90, 1.00) |   10538 | 0.9338 | 0.9135 | 0.0203 | [                           #|  ]
Legend: | = Mean Confidence, # = Observed Accuracy
```

### B. kNN Router (k=5, Uncalibrated)
```
Bin Range        | Samples | Conf   | Acc    | Gap    | Reliability Histogram (Observed vs Perfect)
-----------------|---------|--------|--------|--------|-------------------------------------------
[0.00, 0.10) |    2724 | 0.0029 | 0.1143 | 0.1114 | [|  #                           ]
[0.10, 0.20) |    2546 | 0.1831 | 0.2359 | 0.0528 | [     | #                       ]
[0.20, 0.30) |    1632 | 0.2262 | 0.2690 | 0.0428 | [       |#                      ]
[0.30, 0.40) |    3709 | 0.3796 | 0.4017 | 0.0221 | [           |#                  ]
[0.40, 0.50) |    3228 | 0.4232 | 0.4532 | 0.0300 | [             |#                ]
[0.50, 0.60) |    5905 | 0.5796 | 0.5454 | 0.0342 | [                #|             ]
[0.60, 0.70) |    6797 | 0.6194 | 0.5936 | 0.0258 | [                  #|           ]
[0.70, 0.80) |    7959 | 0.7840 | 0.6768 | 0.1072 | [                    #   |      ]
[0.80, 0.90) |   11988 | 0.8128 | 0.7047 | 0.1081 | [                     #  |      ]
[0.90, 1.00) |   20634 | 0.9996 | 0.7901 | 0.2095 | [                        #     |]
Legend: | = Mean Confidence, # = Observed Accuracy
```

*(Vector charts available at `reports/m4/reliability_diagram_learned_lr.svg` and `reports/m4/reliability_diagram_knn.svg`)*

---

## 4. Task-Aware Calibration & SQuAD Continuous Handling

### Interpretation on Continuous SQuAD F1
For the 12 binary tasks, observed accuracy measures empirical frequency of correctness ({0, 1}).
For SQuAD, calibration measures alignment between predicted confidence and expected token-level F1 score ($[0, 1]$).
A predicted probability of 0.80 for SQuAD signifies an expectation of 80% mean token overlap with ground truth.

### Per-Task Calibration Table

| Task Name | Metric | Metric Type | N Samples | kNN Raw ECE | LR Raw ECE | LR Platt ECE | LR Raw Brier |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `aime_2020_2024` | `GSM8K` | `Binary` | 234 | 0.2940 | 0.0317 | 0.0187 | 0.0184 |
| `arc_challenge` | `em_mc` | `Binary` | 9,000 | 0.1068 | 0.0134 | 0.0158 | 0.1060 |
| `boolq` | `em_mc` | `Binary` | 900 | 0.0888 | 0.0478 | 0.0489 | 0.1543 |
| `commonsense_qa` | `em_mc` | `Binary` | 900 | 0.1142 | 0.0610 | 0.0702 | 0.1538 |
| `gsm8k` | `GSM8K` | `Binary` | 9,000 | 0.1065 | 0.0355 | 0.0339 | 0.1729 |
| `hellaswag` | `em_mc` | `Binary` | 900 | 0.0952 | 0.0364 | 0.0442 | 0.1490 |
| `human_eval` | `code_eval` | `Binary` | 288 | 0.1048 | 0.0623 | 0.0719 | 0.0755 |
| `math` | `MATH` | `Binary` | 9,000 | 0.1352 | 0.0563 | 0.0325 | 0.1915 |
| `mbpp` | `code_eval` | `Binary` | 9,000 | 0.1000 | 0.0288 | 0.0141 | 0.2045 |
| `mmlu` | `em_mc` | `Binary` | 9,000 | 0.1688 | 0.0757 | 0.0850 | 0.2075 |
| `mmlu_pro` | `em_mc` | `Binary` | 9,000 | 0.3312 | 0.1261 | 0.1050 | 0.2380 |
| `openbook_qa` | `em_mc` | `Binary` | 9,000 | 0.1286 | 0.0195 | 0.0242 | 0.1206 |
| `squad` | `f1_score` | `Continuous F1` | 900 | 0.2648 | 0.0541 | 0.0513 | 0.0360 |

---

## 5. Downstream Routing Impact Before vs. After Calibration

| Lambda Value | Uncalibrated Macro Quality | Calibrated Macro Quality | Macro Delta | Uncalibrated Cost / Query | Calibrated Cost / Query | Cost Delta |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| $\lambda = 0.00$ | 0.7208 | 0.7208 | +0.0000 | $0.000283 | $0.000283 | $+0.000000 |
| $\lambda = 0.05$ | 0.7044 | 0.7053 | +0.0009 | $0.000148 | $0.000152 | $+0.000004 |
| $\lambda = 0.10$ | 0.6906 | 0.6912 | +0.0006 | $0.000092 | $0.000095 | $+0.000003 |
| $\lambda = 0.25$ | 0.6710 | 0.6713 | +0.0003 | $0.000049 | $0.000051 | $+0.000002 |
| $\lambda = 0.50$ | 0.6664 | 0.6670 | +0.0006 | $0.000040 | $0.000040 | $+0.000000 |
| $\lambda = 0.75$ | 0.6533 | 0.6568 | +0.0035 | $0.000036 | $0.000037 | $+0.000001 |
| $\lambda = 1.00$ | 0.6513 | 0.6522 | +0.0009 | $0.000032 | $0.000033 | $+0.000001 |
| $\lambda = 1.50$ | 0.6504 | 0.6517 | +0.0013 | $0.000027 | $0.000028 | $+0.000001 |
| $\lambda = 2.00$ | 0.6508 | 0.6511 | +0.0003 | $0.000023 | $0.000024 | $+0.000001 |
| $\lambda = 3.00$ | 0.6505 | 0.6507 | +0.0002 | $0.000018 | $0.000019 | $+0.000001 |
| $\lambda = 5.00$ | 0.6411 | 0.6415 | +0.0005 | $0.000014 | $0.000014 | $+0.000000 |

---

## 6. Statistical Reliability & Limitations

1. **Small Category Disclosures:**
   - Tasks such as `aime_2020_2024` (13 test queries = 234 observations) and `human_eval` (16 test queries = 288 observations)
     have limited sample sizes in extreme confidence bins. Their ECE estimates carry higher bootstrap uncertainty.
2. **Strict Leakage Prevention:**
   - Calibrators were fitted exclusively on 5-fold training out-of-fold (OOF) predictions with prompt-grouped splits.
   - Zero test data was used during calibrator fitting.