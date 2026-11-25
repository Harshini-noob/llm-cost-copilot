# Phase M4.2 — Leakage-Safe Router Evaluation on xRouteBench

**Project:** LLM Cost Copilot  
**Milestone:** M4.2 (Comparative Evaluation and Pareto Sweeps)  
**Date:** November 24, 2026  
**Status:** COMPLETE (Zero Spending, Strict Leakage Prevention Verified)  

---

## 1. Executive Summary

This evaluation investigates whether learned routing policies (kNN and regularized Logistic Regression)
can outperform established baseline policies on unseen test prompts across the **xRouteBench** benchmark.

### Key Evaluation Findings
- **Strict Leakage Prevention:** Routers selected models using only prompt embeddings and features
  available before model invocation. Future output tokens, latencies, and ground truths were never accessed.
- **Pre-Inference Cost Normalization:** Candidate costs were estimated strictly from training rate cards
  and expected model expenditures, preserving candidate-set relativity.
- **Task-Aware Quality:** Preserved binary accuracy across 12 tasks while maintaining continuous token F1
  scores for SQuAD without loss of granular partial credit.
- **Primary Metric:** Unweighted Macro-average across all 13 benchmark tasks prevents disproportionately large
  benchmarks (e.g. math with 500 test queries) from drowning out small benchmarks (e.g. human_eval with 16 queries).

---

## 2. Comparative Strategy Performance Table

| Strategy Name | Strategy Type | Macro Quality (95% CI) | Micro Quality (95% CI) | Mean Cost / Query | Cost / 1k Queries | Historical LLM Latency* |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Always-Cheapest** | `deployable_policy` | 0.5806 [0.5627, 0.5965] | 0.6976 [0.6810, 0.7113] | $0.000014 | $0.014 | 3.21s |
| **Best Single Model (Test-Set Reference)** | `post_hoc_reference` | 0.6536 [0.6368, 0.6650] | 0.8054 [0.7931, 0.8188] | $0.000458 | $0.458 | 2.52s |
| **Random-Uniform** | `deployable_policy` | 0.5109 [0.4897, 0.5292] | 0.6173 [0.6004, 0.6343] | $0.000181 | $0.181 | 2.76s |
| **Static-Task-Table** | `deployable_policy` | 0.7321 [0.7041, 0.7570] | 0.8137 [0.8020, 0.8258] | $0.000315 | $0.315 | 3.15s |
| **Oracle (lambda=0.00)** | `theoretical_reference` | 0.8533 [0.8325, 0.8760] | 0.9293 [0.9218, 0.9377] | $0.000032 | $0.032 | 2.64s |
| **KNN (lambda=0.00)** | `deployable_policy` | 0.6508 [0.6244, 0.6753] | 0.7115 [0.6959, 0.7251] | $0.000079 | $0.079 | 2.92s |
| **KNN (lambda=0.05)** | `deployable_policy` | 0.6520 [0.6237, 0.6730] | 0.7104 [0.6946, 0.7238] | $0.000073 | $0.073 | 2.95s |
| **KNN (lambda=0.10)** | `deployable_policy` | 0.6529 [0.6251, 0.6739] | 0.7115 [0.6956, 0.7246] | $0.000072 | $0.072 | 2.95s |
| **KNN (lambda=0.25)** | `deployable_policy` | 0.6495 [0.6221, 0.6704] | 0.7056 [0.6898, 0.7187] | $0.000060 | $0.060 | 2.95s |
| **KNN (lambda=0.50)** | `deployable_policy` | 0.6365 [0.6113, 0.6579] | 0.6971 [0.6824, 0.7092] | $0.000034 | $0.034 | 2.88s |
| **KNN (lambda=0.75)** | `deployable_policy` | 0.6373 [0.6126, 0.6585] | 0.6982 [0.6835, 0.7116] | $0.000028 | $0.028 | 2.90s |
| **KNN (lambda=1.00)** | `deployable_policy` | 0.6397 [0.6130, 0.6596] | 0.6977 [0.6827, 0.7113] | $0.000026 | $0.026 | 2.91s |
| **KNN (lambda=1.50)** | `deployable_policy` | 0.6394 [0.6133, 0.6591] | 0.6974 [0.6827, 0.7106] | $0.000026 | $0.026 | 2.92s |
| **KNN (lambda=2.00)** | `deployable_policy` | 0.6393 [0.6123, 0.6591] | 0.6973 [0.6828, 0.7109] | $0.000025 | $0.025 | 2.93s |
| **KNN (lambda=3.00)** | `deployable_policy` | 0.6299 [0.6033, 0.6532] | 0.6965 [0.6810, 0.7092] | $0.000021 | $0.021 | 2.92s |
| **KNN (lambda=5.00)** | `deployable_policy` | 0.6152 [0.5875, 0.6406] | 0.6937 [0.6773, 0.7078] | $0.000017 | $0.017 | 2.96s |
| **LEARNED_LR (lambda=0.00)** | `deployable_policy` | 0.7208 [0.6994, 0.7478] | 0.7994 [0.7848, 0.8121] | $0.000283 | $0.283 | 3.44s |
| **LEARNED_LR (lambda=0.05)** | `deployable_policy` | 0.7044 [0.6808, 0.7286] | 0.7831 [0.7684, 0.7965] | $0.000148 | $0.148 | 3.44s |
| **LEARNED_LR (lambda=0.10)** | `deployable_policy` | 0.6906 [0.6691, 0.7168] | 0.7728 [0.7581, 0.7858] | $0.000092 | $0.092 | 3.44s |
| **LEARNED_LR (lambda=0.25)** | `deployable_policy` | 0.6710 [0.6472, 0.6942] | 0.7538 [0.7388, 0.7679] | $0.000049 | $0.049 | 3.37s |
| **LEARNED_LR (lambda=0.50)** | `deployable_policy` | 0.6664 [0.6432, 0.6895] | 0.7463 [0.7323, 0.7599] | $0.000040 | $0.040 | 3.28s |
| **LEARNED_LR (lambda=0.75)** | `deployable_policy` | 0.6533 [0.6305, 0.6761] | 0.7380 [0.7234, 0.7518] | $0.000036 | $0.036 | 3.22s |
| **LEARNED_LR (lambda=1.00)** | `deployable_policy` | 0.6513 [0.6274, 0.6756] | 0.7345 [0.7209, 0.7487] | $0.000032 | $0.032 | 3.19s |
| **LEARNED_LR (lambda=1.50)** | `deployable_policy` | 0.6504 [0.6276, 0.6737] | 0.7304 [0.7158, 0.7444] | $0.000027 | $0.027 | 3.18s |
| **LEARNED_LR (lambda=2.00)** | `deployable_policy` | 0.6508 [0.6286, 0.6725] | 0.7245 [0.7097, 0.7380] | $0.000023 | $0.023 | 3.20s |
| **LEARNED_LR (lambda=3.00)** | `deployable_policy` | 0.6505 [0.6272, 0.6723] | 0.7211 [0.7062, 0.7357] | $0.000018 | $0.018 | 3.23s |
| **LEARNED_LR (lambda=5.00)** | `deployable_policy` | 0.6411 [0.6191, 0.6638] | 0.7138 [0.6988, 0.7279] | $0.000014 | $0.014 | 3.21s |

\* *Note on Latency:* Latency represents downstream historical candidate LLM response execution time recorded in the benchmark. It is distinct from local router decision overhead (which is sub-millisecond on CPU).

---

## 3. Baseline Strategy Definitions

1. **Always-Cheapest (Deployable Policy):** Identifies the candidate model with lowest mean training cost
   (`gpt-oss-20b` at \$0.000015 / call) and routes all test queries to it.
2. **Best Single Model (Test-Set Reference):** Identifies the single candidate model with highest test set
   accuracy (`qwen3-next-80b-a3b-instruct` at 0.6775). Serves as a post-hoc reference for what a static model choice achieves with test-set hindsight; not deployable at test time and not a universal quality ceiling.
3. **Random-Uniform (Deployable Policy):** Uniform random candidate assignment across all 18 models with deterministic seed (`seed=42`),
   capturing stochastic baseline performance.
4. **Static Task-Type Table (Deployable Policy):** Maps each of the 13 benchmark tasks to the single candidate model
   that achieved the highest mean accuracy on that task during training.
5. **Oracle (Theoretical Upper Bound):** Selects the post-hoc optimal candidate model for every individual prompt
   under the utility objective $U(m, x) = Q(m, x) - \lambda \cdot c_{\text{norm}}(m)$ with $\lambda = 0.0$. Impossible to deploy.

---

## 4. Cost Estimation and Normalization Methodology

### Pre-Inference Decision Cost
Before selecting a model, the router cannot know future completion tokens. Pre-inference costs are estimated
strictly from candidate rate cards and historical training statistics:

$$\widehat{\text{cost}}(m, x) = \frac{T_{\text{in}}(x) \times \text{input\_price}(m) + \bar{T}_{\text{out}}(m) \times \text{output\_price}(m)}{1{,}000{,}000}$$

### Observed Evaluation Cost
Post-selection evaluation calculates actual expenditures using observed test tokens and published candidate rates:

$$\text{cost}_{\text{observed}} = \frac{\text{input\_tokens} \times \text{input\_price}(m) + \text{output\_tokens} \times \text{output\_price}(m)}{1{,}000{,}000}$$

---

## 5. Artifact Verification and Immutability

- All outputs written strictly to `reports/m4/` and `data/processed/xroutebench/`.
- Original M0–M2 historical matrix (`data/processed/matrix.parquet`) is unchanged.
- Raw Parquet files in `data/external/xroutebench/` remain cryptographically intact.
- All unit tests pass with zero warnings.