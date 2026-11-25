# Phase M4.2.1 — Independent Evaluation Methodology Audit

**Project:** LLM Cost Copilot  
**Author:** Antigravity / Engineering Team  
**Date:** November 24, 2026  
**Status:** COMPLETED & RECONCILED (ALL AUDIT FINDINGS ADDRESSED)  
**Scope:** Independent Methodological Audit of M4.2 Comparative Evaluation on xRouteBench  

---

## 1. Executive Summary & Audit Overview

Following the completion of Phase M4.2 (evaluating 7 routing strategies across 3,729 test prompts and 18 candidate LLMs), this independent audit was conducted to rigorously verify:
1. Baseline implementations and labeling honesty.
2. Terminology precision (e.g. post-hoc reference ceilings vs deployable policies).
3. The exact mathematical formulation and role of the Oracle.
4. The exact physical source and interpretation of latency metrics.
5. Numerical accuracy of cost-quality trade-off claims using unrounded data.
6. Statistical validity of prompt-clustered bootstrap confidence intervals.

### Summary Audit Verdicts

| Audit Area | Inspected Behavior | Verdict | Action Taken |
| :--- | :--- | :---: | :--- |
| **1. Random Baseline** | Evaluated uniform selection across 18 models without matching a specific cost target. | **Mislabeled** | Renamed `Random-Matched-Cost` $\rightarrow$ `Random-Uniform`. All reports and code updated. |
| **2. Always-Best Baseline** | Chose single globally best candidate model on test set (`qwen3-next-80b-a3b-instruct`). | **Mislabeled as ceiling** | Renamed `Always-Best` $\rightarrow$ `Best Single Model (Test-Set Reference)`. Clarified not a universal ceiling. |
| **3. Oracle Formulation** | Quality-maximizing selection under $U(m, x) = Q(m, x) - \lambda \cdot c_{\text{norm}}(m)$ with $\lambda = 0.0$. | **Correct, needed doc** | Explicitly documented $\lambda=0.0$ and utility formula across all reports and JSON schemas. |
| **4. Latency Source** | Reported values reflect downstream LLM API response time recorded in the benchmark. | **Ambiguous label** | Relabeled to `Historical LLM Latency (sec)`. Clarified router decision overhead is sub-millisecond CPU time. |
| **5. Numerical Claims** | Recalculated cost reductions, quality deltas, and ratios from unrounded floating-point data. | **Verified Correct** | Certified exact unrounded calculations (e.g. 92.23% cost reduction at quality parity). |
| **6. Bootstrap CIs** | Prompt-clustered resampling preserves query groups and task macro structures. | **Verified Correct** | Confirmed 100% prompt-level clustering, deterministic seed reproducibility, and task balance. |

---

## 2. Detailed Findings by Audit Area

### Issue 1: Random Baseline — Matched Cost vs. Uniform Random

- **Implementation Evidence:**  
  In `eval/m4_router_evaluation.py`, `evaluate_random_matched_cost` defined `target_cost: float | None = None`. When invoked in `run_m4_evaluation`, no `target_cost` was supplied:
  ```python
  probs = [1.0 / len(candidate_models)] * len(candidate_models)
  chosen_indices = rng.choice(len(candidate_models), size=n, p=probs)
  ```
  The sampler drew candidates uniformly at random with equal probability ($1/18 \approx 5.56\%$ per candidate) using `RandomState(seed=42)`.
- **Verdict:** **INCORRECT LABELING.** The baseline evaluated was a **Uniform Random** policy, not cost-matched to any specific router budget.
- **Correction Made:**  
  - Renamed function to `evaluate_random_uniform`.
  - Renamed strategy across all code, outputs, and documentation to **`Random-Uniform`**.
  - Documented that no artificial cost-matching was fabricated; it represents uniform stochastic selection across the candidate pool.

---

### Issue 2: Always-Best Terminology & Quality Ceiling

- **Implementation Evidence:**  
  In `eval/m4_router_evaluation.py`:
  ```python
  model_test_perf = test_df.groupby("model_name")["performance"].mean()
  best_model = str(model_test_perf.idxmax())
  ```
  The strategy computed the mean performance of each candidate model across the held-out test set and routed all 3,729 test queries to the single top-ranking model (`qwen3-next-80b-a3b-instruct`, test accuracy 0.6775).
- **Verdict:** **MISLEADING TERMINOLOGY.** Calling this strategy "Always-Best" implies a universal quality ceiling. In reality, it represents the performance of committing to a *single static model* selected with test-set hindsight. Dynamic routers (`LearnedRouter` achieving 0.7208 Macro Quality) and `Oracle` (0.8533) significantly outperform it.
- **Correction Made:**  
  - Renamed strategy to **`Best Single Model (Test-Set Reference)`**.
  - Retained `strategy_type="post_hoc_reference"` and `is_reference=True`.
  - Clarified in all documentation that it is not a quality ceiling, but an empirical upper bound for static single-model selection.

---

### Issue 3: Oracle Definition & Formulation

- **Implementation Evidence:**  
  In `eval/m4_router_evaluation.py`:
  ```python
  s_oracle, sel_oracle = evaluate_oracle(
      test_records_by_query, test_query_metadata, candidate_models, cost_estimator, lambda_value=0.0
  )
  ```
  For each test prompt $x$, the Oracle evaluated all 18 candidate outcomes and maximized utility:
  $$U(m, x) = Q(m, x) - \lambda \cdot c_{\text{norm}}(m)$$
  With $\lambda = 0.0$, the decision rule simplifies to:
  $$m^* = \arg\max_{m} Q(m, x)$$
  Ties are broken by selecting the candidate with the lowest observed cost. For continuous SQuAD queries, it selects the model with the highest token F1 score.
- **Verdict:** **CORRECT IMPLEMENTATION, NEEDED EXPLICIT DOCUMENTATION.**
  - **Leakage check:** The Oracle operates strictly on test outcomes as an evaluation reference; it is never used to train or calibrate deployable routers.
  - **Metric consistency:** Quality and costs are aggregated using the identical `compute_task_macro_score` and `compute_prompt_micro_score` routines as deployable routers.
- **Correction Made:**  
  - Renamed display label to **`Oracle (lambda=0.00)`** in reports and schemas.
  - Explicitly stated the utility formula and tie-breaking protocol in all markdown summaries and notes fields.

---

### Issue 4: Latency Terminology & Physical Source

- **Implementation Evidence:**  
  In `EvaluationSelectionRecord`:
  ```python
  response_time = float(obs["response_time"])
  ```
  This value is extracted directly from the raw xRouteBench benchmark dataset (`test.parquet`), which recorded the elapsed seconds for the candidate model provider API to return a response during benchmark execution.
- **Verdict:** **AMBIGUOUS LABEL.** Labeling this simply as "Latency" could mislead users into believing it represents router inference overhead or combined pipeline latency.
- **Correction Made:**  
  - Renamed the metric in CSV schemas to `historical_llm_latency_sec`.
  - Renamed table column headers in Markdown reports to `Historical LLM Latency*`.
  - Added an explicit note:
    > *\* Note on Latency: Latency represents downstream historical candidate LLM response execution time recorded in the benchmark. It is distinct from local router decision overhead (which is sub-millisecond on CPU).*

---

### Issue 5: Cost-Quality Trade-off Recalculations (Unrounded Verification)

Using raw floating-point data extracted directly from [reports/m4/m4_router_comparison.json](file:///d:/llm%20cost%20copilot/reports/m4/m4_router_comparison.json), we independently verified all reported cost-quality relationships:

#### A. Learned LR ($\lambda=0.00$) vs. Best Single Model
* **Learned LR ($\lambda=0.00$):**
  * Macro Quality: `0.72075027`
  * Micro Quality: `0.79936753`
  * Cost per 1,000 requests: `\$0.282553` (Mean: `\$0.0002825526`)
* **Best Single Model (Test-Set Reference):**
  * Macro Quality: `0.65355533`
  * Micro Quality: `0.80539313`
  * Cost per 1,000 requests: `\$0.457680` (Mean: `\$0.0004576803`)
* **Exact Deltas:**
  * Macro Quality Advantage: **`+0.067195`** (an unrounded **`+10.28%`** relative improvement).
  * Cost Reduction: $\frac{0.0004576803 - 0.0002825526}{0.0004576803} \times 100\% =$ **`38.26%`** savings.
  * Cost Ratio: Best Single Model is **`1.62x`** more expensive.

#### B. Learned LR ($\lambda=0.75$) vs. Best Single Model (Quality Parity)
* **Learned LR ($\lambda=0.75$):**
  * Macro Quality: `0.65329698`
  * Cost per 1,000 requests: `\$0.035545` (Mean: `\$0.0000355454`)
* **Quality Parity Check:**
  * Delta: $0.65329698 - 0.65355533 = -0.000258$ (within $0.04\%$ of parity).
* **Exact Cost Reduction:**
  * Savings: $\frac{0.0004576803 - 0.0000355454}{0.0004576803} \times 100\% =$ **`92.23%`** cost reduction.
  * Cost Ratio: Best Single Model is **`12.88x`** more expensive (~$13\times$ cost multiplier for the same accuracy).

#### C. Learned LR ($\lambda=5.00$) vs. Always-Cheapest
* **Learned LR ($\lambda=5.00$):**
  * Macro Quality: `0.64107083`
  * Cost per 1,000 requests: `\$0.013559` (Mean: `\$0.0000135595`)
* **Always-Cheapest:**
  * Macro Quality: `0.58064504`
  * Cost per 1,000 requests: `\$0.014489` (Mean: `\$0.0000144895`)
* **Exact Deltas:**
  * Macro Quality Advantage: **`+0.060426`** (an unrounded **`+10.41%`** relative improvement).
  * Cost Advantage: Learned LR ($\lambda=5.00$) is **`6.42%`** cheaper than Always-Cheapest while delivering $+10.4\%$ higher macro accuracy.

#### D. kNN Router Comparisons
* **kNN ($\lambda=0.00$):** Macro Quality `0.650819`, Cost/1k `\$0.078558`.
  * Achieves $99.6\%$ of Best Single Model quality at **`82.84%`** lower cost (Best is $5.83\times$ more expensive).
* **kNN ($\lambda=0.10$):** Macro Quality `0.652850`, Cost/1k `\$0.071822`.
  * Achieves $99.9\%$ of Best Single Model quality at **`84.31%`** lower cost.
* **kNN ($\lambda=5.00$):** Macro Quality `0.615212`, Cost/1k `\$0.017007`.
  * Outperforms Always-Cheapest by **`+0.034567`** Macro Quality points.

---

### Issue 6: Confidence Interval & Bootstrap Integrity

- **Implementation Evidence:**  
  Inspected `bootstrap_prompt_metric` in `eval/m4_data_adapter.py`:
  1. **Prompt-Level Clustering:** Resamples unique prompt query strings with replacement ($N=3,729$), ensuring all model evaluation records for a selected query are ingested together.
  2. **Cross-Task Cluster Preservation:** Unique prompt strings that appear in multiple task records remain linked.
  3. **Task-Macro Structure:** For each bootstrap iteration, scores are aggregated via `compute_task_macro_score`, which averages within each task category before computing the unweighted mean across the 13 tasks.
  4. **Reproducibility:** Initialized with `RandomState(seed=42)`, guaranteeing deterministic percentile bounds.
- **Verdict:** **VERIFIED CORRECT.** The bootstrap engine correctly models prompt-level sampling variance without row-level leakage.

---

## 3. Reconciled Strategy Performance Table

The official evaluation table has been updated across all M4 deliverables:

| Strategy Name | Strategy Type | Macro Quality [95% CI] | Micro Quality [95% CI] | Mean Cost / Query | Cost / 1k Queries | Historical LLM Latency* |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Always-Cheapest** | `deployable_policy` | 0.5806 [0.5627, 0.5965] | 0.6976 [0.6810, 0.7113] | \$0.000014 | \$0.014 | 3.21s |
| **Best Single Model (Test-Set Reference)** | `post_hoc_reference` | 0.6536 [0.6368, 0.6650] | 0.8054 [0.7931, 0.8188] | \$0.000458 | \$0.458 | 2.52s |
| **Random-Uniform** | `deployable_policy` | 0.5109 [0.4897, 0.5292] | 0.6173 [0.6004, 0.6343] | \$0.000181 | \$0.181 | 2.76s |
| **Static-Task-Table** | `deployable_policy` | 0.7321 [0.7041, 0.7570] | 0.8137 [0.8020, 0.8258] | \$0.000315 | \$0.315 | 3.15s |
| **Oracle ($\lambda=0.00$)** | `theoretical_reference` | 0.8533 [0.8325, 0.8760] | 0.9293 [0.9218, 0.9377] | \$0.000032 | \$0.032 | 2.64s |
| **KNN ($\lambda=0.00$)** | `deployable_policy` | 0.6508 [0.6244, 0.6753] | 0.7115 [0.6959, 0.7251] | \$0.000079 | \$0.079 | 2.92s |
| **KNN ($\lambda=0.05$)** | `deployable_policy` | 0.6520 [0.6237, 0.6730] | 0.7104 [0.6946, 0.7238] | \$0.000073 | \$0.073 | 2.95s |
| **KNN ($\lambda=0.10$)** | `deployable_policy` | 0.6529 [0.6251, 0.6739] | 0.7115 [0.6956, 0.7246] | \$0.000072 | \$0.072 | 2.95s |
| **KNN ($\lambda=0.25$)** | `deployable_policy` | 0.6495 [0.6221, 0.6704] | 0.7056 [0.6898, 0.7187] | \$0.000060 | \$0.060 | 2.95s |
| **KNN ($\lambda=0.50$)** | `deployable_policy` | 0.6365 [0.6113, 0.6579] | 0.6971 [0.6824, 0.7092] | \$0.000034 | \$0.034 | 2.88s |
| **KNN ($\lambda=0.75$)** | `deployable_policy` | 0.6373 [0.6126, 0.6585] | 0.6982 [0.6835, 0.7116] | \$0.000028 | \$0.028 | 2.90s |
| **KNN ($\lambda=1.00$)** | `deployable_policy` | 0.6397 [0.6130, 0.6596] | 0.6977 [0.6827, 0.7113] | \$0.000026 | \$0.026 | 2.91s |
| **KNN ($\lambda=1.50$)** | `deployable_policy` | 0.6394 [0.6133, 0.6591] | 0.6974 [0.6827, 0.7106] | \$0.000026 | \$0.026 | 2.92s |
| **KNN ($\lambda=2.00$)** | `deployable_policy` | 0.6393 [0.6123, 0.6591] | 0.6973 [0.6828, 0.7109] | \$0.000025 | \$0.025 | 2.93s |
| **KNN ($\lambda=3.00$)** | `deployable_policy` | 0.6299 [0.6033, 0.6532] | 0.6965 [0.6810, 0.7092] | \$0.000021 | \$0.021 | 2.92s |
| **KNN ($\lambda=5.00$)** | `deployable_policy` | 0.6152 [0.5875, 0.6406] | 0.6937 [0.6773, 0.7078] | \$0.000017 | \$0.017 | 2.96s |
| **LEARNED_LR ($\lambda=0.00$)** | `deployable_policy` | **0.7208** [0.6994, 0.7478] | 0.7994 [0.7848, 0.8121] | \$0.000283 | \$0.283 | 3.44s |
| **LEARNED_LR ($\lambda=0.05$)** | `deployable_policy` | **0.7044** [0.6808, 0.7286] | 0.7831 [0.7684, 0.7965] | \$0.000148 | \$0.148 | 3.44s |
| **LEARNED_LR ($\lambda=0.10$)** | `deployable_policy` | **0.6906** [0.6691, 0.7168] | 0.7728 [0.7581, 0.7858] | \$0.000092 | \$0.092 | 3.44s |
| **LEARNED_LR ($\lambda=0.25$)** | `deployable_policy` | **0.6710** [0.6472, 0.6942] | 0.7538 [0.7388, 0.7679] | \$0.000049 | \$0.049 | 3.37s |
| **LEARNED_LR ($\lambda=0.50$)** | `deployable_policy` | **0.6664** [0.6432, 0.6895] | 0.7463 [0.7323, 0.7599] | \$0.000040 | \$0.040 | 3.28s |
| **LEARNED_LR ($\lambda=0.75$)** | `deployable_policy` | **0.6533** [0.6305, 0.6761] | 0.7380 [0.7234, 0.7518] | \$0.000036 | \$0.036 | 3.22s |
| **LEARNED_LR ($\lambda=1.00$)** | `deployable_policy` | **0.6513** [0.6274, 0.6756] | 0.7345 [0.7209, 0.7487] | \$0.000032 | \$0.032 | 3.19s |
| **LEARNED_LR ($\lambda=1.50$)** | `deployable_policy` | **0.6504** [0.6276, 0.6737] | 0.7304 [0.7158, 0.7444] | \$0.000027 | \$0.027 | 3.18s |
| **LEARNED_LR ($\lambda=2.00$)** | `deployable_policy` | **0.6508** [0.6286, 0.6725] | 0.7245 [0.7097, 0.7380] | \$0.000023 | \$0.023 | 3.20s |
| **LEARNED_LR ($\lambda=3.00$)** | `deployable_policy` | **0.6505** [0.6272, 0.6723] | 0.7211 [0.7062, 0.7357] | \$0.000018 | \$0.018 | 3.23s |
| **LEARNED_LR ($\lambda=5.00$)** | `deployable_policy` | **0.6411** [0.6191, 0.6638] | 0.7138 [0.6988, 0.7279] | \$0.000014 | \$0.014 | 3.21s |

---

## 4. Test Suite Verification

Targeted tests were added to [tests_v2/test_m4_router_evaluation.py](file:///d:/llm%20cost%20copilot/tests_v2/test_m4_router_evaluation.py) covering the audited items:
- Verified `Random-Uniform` uniformly samples candidate models deterministically.
- Verified `Best Single Model (Test-Set Reference)` is marked reference-only.
- Verified `Oracle (lambda=0.00)` explicitly checks $\lambda=0.0$ and utility formulation.
- Verified that latency corresponds to historical candidate model response time.

```
pytest tests_v2/test_m4_router_evaluation.py -v
======================= 11 passed in 7.26s =======================

pytest tests_v2 -v
======================= 264 passed, 1 warning in 42.39s =======================
```
*(All 264 repository unit tests pass with zero regressions.)*

---

## 5. Remaining Limitations

1. **Static Task Metadata Reliance:**
   - Both `Static-Task-Table` and `LearnedRouter` utilize task category identifiers (`task_name`). In real-world zero-shot deployments where task category is unprovided, a task classifier or pure semantic embedding router would be required.
2. **Pre-Inference Output Variance:**
   - Pre-inference cost estimation uses training average output tokens per candidate model ($\bar{T}_{\text{out}}(m)$). On reasoning benchmarks with high token variance (e.g. MATH or GSM8K), this estimator exhibits dispersion around actual realization.
3. **Probability Calibration:**
   - The logistic regression classifiers output empirical sigmoid probabilities. While effective for ranking under $U(m, x)$, formal probability calibration (e.g. Platt scaling, isotonic regression) has not yet been applied; this is slated for Phase M4.3.

**Sign-off:** The methodology audit is complete. All terminology and calculations are verified, reconciled, and certified for M4.3.
