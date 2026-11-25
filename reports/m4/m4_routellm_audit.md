# M4.5.1 — RouteLLM & APGR Statistical and Reporting Integrity Audit Report

**Date**: 2026-11-25  
**Milestone**: M4.5.1 — RouteLLM Two-Model Baseline & APGR Methodological Audit  
**Target Repository**: `https://github.com/Harshini-noob/llm-cost-copilot`  
**Auditor**: LLM Cost Copilot Independent Audit Harness  
**Artifacts Audited**:
- `eval/m4_routellm_baseline.py`
- `reports/m4/m4_routellm_apgr.csv`
- `reports/m4/m4_routellm_results.csv`
- `reports/m4/m4_routellm_results.json`
- `reports/m4/m4_routellm_summary.md`
- `reports/m4/routellm_pgr_tradeoff.svg`
- `reports/m4/routellm_cost_quality_tradeoff.svg`
- `tests_v2/test_m4_routellm.py`

---

## 1. Executive Summary & Audit Verdict

**Audit Verdict**: **PASS WITH DOCUMENTATION CORRECTIONS**

This independent audit investigated seven potential methodological, mathematical, and reporting concerns in the completed M4.5 milestone (RouteLLM Two-Model Baseline & APGR Evaluation).

### Summary of Corrective Actions:
1. **Model Identity Clarified**: Verified that canonical model IDs `cogito-v2-1-671b` and `llama-3-8b-instruct-lite` were used identically across candidate metadata, training logic, test evaluation, pricing, bootstrap resampling, and SVG plots. Shortened names (`cogito-671b`, `llama-3-8b`) were colloquial chat display shorthand and have been replaced with canonical IDs across all tables and reports.
2. **Dataset Count Reconciliation**: Fully reconciled the 4,487 unique string queries reported in M4.1 with the 4,489 rows per model in training. The difference represents 1 cross-task query (`math` and `aime_2020_2024`) and 1 duplicate embedding record in `arc_challenge`. Zero prompts or rows were silently dropped.
3. **Quality Aggregation Clarified (Dual Reporting)**: Confirmed that original M4.5 tables reported **Micro Quality** (sample-weighted prompt mean across 3,729 test queries). Always-Strong micro quality (**0.8054**) matches M4.2's Best Single Model micro quality (**0.805393**) identically. We added explicit dual-metric reporting for both **Micro APGR** and **Macro APGR** (unweighted task mean across 13 tasks).
4. **CPT Continuous Interpolation**: Upgraded `calculate_cpt` from discrete nearest-threshold lookup to continuous linear interpolation between operating points, providing exact crossing fractions for $\text{CPT}(50\%)$ and $\text{CPT}(80\%)$.
5. **Formal Bootstrap Hypothesis Testing**: Replaced the informal assertion of $p < 0.05$ (and the initial M4.5.1 uncentered quantile check) with a null-centered prompt-cluster bootstrap hypothesis test of $H_0: \text{APGR} \le 0.5000$ vs $H_1: \text{APGR} > 0.5000$ using the Davison & Hinkley (1997) / Phipson & Smyth (2010) plus-one correction. Across $B = 1,000$ replicates, zero null-centered replicates exceeded $\hat{A}$, yielding $p = 1/1001 \approx 0.000999$ (rounded to `0.0010`).
6. **Cost Reconstructed Arithmetic Verified**: Reconstructed benchmark costs match theoretical token pricing arithmetic to 10 decimal places (`max_diff = 0.0000000000`).

---

## 2. Model-Pair Identity & Canonical Verification

### 2.1 Identity Across Data Files
We inspected the candidate metadata table `data/external/xroutebench/llm_candidates.parquet` and the processed datasets:

| Model Role | Canonical Model ID | Provider Service | API Model ID | Parameter Scale | Input Price / 1M | Output Price / 1M |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Strong Model** | `cogito-v2-1-671b` | Together | `cogito-v2-1-671b` | 671B | $1.25 | $1.25 |
| **Weak Model** | `llama-3-8b-instruct-lite` | Together | `meta-llama/Llama-3-8b-chat-hf` | 8B | $0.10 | $0.10 |

### 2.2 Investigation of Naming Discrepancies
- **Discrepancy Source**: In the chat summary table of the initial turn, the labels were truncated to `Always-Weak (llama-3-8b)` and `Always-Strong (cogito-671b)` for visual table brevity.
- **Underlying Code Audit**: Line-by-line inspection of `eval/m4_routellm_baseline.py` confirms that the selection logic (`select_model_pair_from_train`), test pivot indexing (`piv_test['cogito-v2-1-671b']`), token pricing maps, and JSON metadata strictly referenced the canonical IDs `cogito-v2-1-671b` and `llama-3-8b-instruct-lite`.
- **Correction Applied**: All report tables, CSV columns, JSON metadata, and test assertions now display the full canonical identifiers without truncation.

---

## 3. Dataset Count Reconciliation

### 3.1 Reconciling 4,487 vs 4,489 Training Counts

| Metric / Level | Training Set (`train_processed.parquet`) | Test Set (`test_processed.parquet`) | Explanation |
| :--- | :---: | :---: | :--- |
| **Total Raw Rows** | 80,802 | 67,122 | Exactly $4,489 \times 18$ models (train) and $3,729 \times 18$ models (test). |
| **Unique Query Strings** | 4,487 | 3,729 | `df['query'].nunique()`. |
| **Unique `(task_name, query)`** | 4,488 | 3,729 | Exactly 1 prompt is shared between `math` and `aime_2020_2024`. |
| **Unique `(task_name, query, embedding_id)`** | **4,489** | **3,729** | Exactly 1 prompt in `arc_challenge` contains two distinct `embedding_id` records. |
| **Rows per Candidate Model** | **4,489** | **3,729** | Every single model has complete evaluation coverage across all 4,489 instances. |
| **Bootstrap Clusters (Test)** | N/A | **3,729** | On the test set, unique queries and evaluation instances are strictly 1-to-1. |

### 3.2 Splitting & Leakage Verification
- In `create_prompt_grouped_validation_split`, prompts are grouped by normalized query string `df["query"].str.strip()`.
- Because grouping occurs on the raw query text, all 36 evaluation rows associated with the shared cross-task prompt are placed into the same split fold (zero leakage).
- On the test set, each of the 3,729 evaluation instances has a distinct query string, ensuring that prompt cluster resampling is unambiguous.

---

## 4. Quality Aggregation Verification (Micro vs Macro)

### 4.1 Evaluation Discrepancy Analysis
In benchmark evaluation across 13 unbalanced tasks, quality can be aggregated in two distinct ways:
1. **Micro Quality ($\text{Micro } Q$)**: Unweighted prompt-level mean across all $N = 3,729$ test queries. Tasks with more prompts (e.g., MMLU, GSM8K) have larger weight.
2. **Macro Quality ($\text{Macro } Q$)**: Unweighted mean of the 13 task-specific scores, giving equal weight to every benchmark task category regardless of prompt count.

### 4.2 Cross-Milestone Alignment with M4.2

| Strategy / Model | M4.5 Reported Quality | M4.2 Micro Score | M4.2 Macro Score | Audit Verdict |
| :--- | :---: | :---: | :---: | :--- |
| **Always-Strong (`cogito-v2-1-671b`)** | **0.8054** | **0.805393** | 0.653555 | **Exact Micro Match** ($0.8054 = 0.805393$) |
| **Always-Weak (`llama-3-8b-instruct-lite`)**| **0.4932** | N/A (Candidate) | 0.452296 | **Exact Micro Match** ($0.4932 = 0.493239$) |
| **Quality Gap ($\Delta Q$)** | **+0.3122** | N/A | +0.201259 | **Micro Gap**: +31.2%, **Macro Gap**: +20.1% |

### 4.3 Audited Micro vs Macro APGR Results

| Strategy | Micro APGR | Macro APGR | Difference | Conclusion |
| :--- | :---: | :---: | :---: | :--- |
| **Oracle Two-Model Ceiling** | 90.65% | 96.31% | +5.66% | Upper bound captures >90% gap under both metrics |
| **RouteLLM Hybrid Ensemble** | **59.06%** | **59.04%** | **-0.02%** | **Virtually identical recovery** across Micro & Macro |
| **RouteLLM-Inspired Learned LR** | 58.10% | 58.32% | +0.22% | Robust recovery across both aggregations |
| **RouteLLM-Inspired kNN Router** | 58.18% | 57.72% | -0.46% | Robust recovery across both aggregations |
| **Random Selection** | 50.00% | 50.00% | 0.00% | Exact theoretical diagonal expectation |

*Audit Finding*: The M4.5 code consistently computed Micro Quality for its main table. Both Micro and Macro metrics are now explicitly presented side-by-side in all deliverables.

---

## 5. Audit of APGR, PGR, and CPT Mathematical Formulas

### 5.1 Verification of Published Definitions
We verified the exact implementations against RouteLLM (*Ong et al., 2024*):
1. **Performance Gap Recovered (PGR)**:
   $$\text{PGR}(M_R) = \frac{r(M_R) - r(M_{\text{weak}})}{r(M_{\text{strong}}) - r(M_{\text{weak}})}$$
   - Boundary Check: When $r(M_R) = r(M_{\text{weak}})$, $\text{PGR} = 0.0$.
   - Boundary Check: When $r(M_R) = r(M_{\text{strong}})$, $\text{PGR} = 1.0$.
   - Zero-gap handling: If $r(M_{\text{strong}}) \le r(M_{\text{weak}})$, safely returns $0.0$.
2. **Average Performance Gap Recovered (APGR)**:
   $$\text{APGR}(M_R) = \int_{0}^{1} \text{PGR}(M_{R_c}) \, dc$$
   - Numerically verified via trapezoidal integration over monotonically ordered call fractions $c \in [0, 1]$.
   - Endpoint verification: Missing boundary endpoints $(0, 0)$ and $(1, 1)$ are injected prior to integration.
   - Tested on diagonal $y = x$: returns exactly $0.5000$ (Random routing).
3. **Call-Performance Threshold (CPT)**:
   $$\text{CPT}(P) = \min \{ c \in [0, 1] : \text{PGR}(c) \ge P \}$$
   - Continuous linear interpolation between adjacent operating points was implemented to eliminate discretization bias from the 101-threshold evaluation grid:
     $$\tilde{c} = c_{k-1} + \frac{P - \text{PGR}_{k-1}}{\text{PGR}_k - \text{PGR}_{k-1}} (c_k - c_{k-1})$$
4. **Oracle PGR Exceeding 1.0 Explained**:
   - On the test set, `llama-3-8b-instruct-lite` strictly outperforms `cogito-v2-1-671b` on 206 prompts (5.5%).
   - The Oracle router routes those prompts to the weak model, avoiding the errors made by the strong model.
   - As a result, Oracle quality reaches **0.8351** (exceeding Always-Strong's **0.8054**), yielding $\text{PGR} = \frac{0.8351 - 0.4932}{0.3122} = \mathbf{1.0951}$ (109.5%).

---

## 6. Audit of Bootstrap Inference & Significance Claims

### 6.1 Cluster Resampling Audit
- Verified that resampling operates on **prompt cluster indices** $i \in \{0, \dots, N-1\}$ ($N = 3,729$).
- When prompt $i$ is sampled, outcomes for both weak and strong models and router decisions are sampled simultaneously, strictly preserving query difficulty correlation.
- In every bootstrap replicate $b$, the complete call-performance curve is reconstructed from 101 thresholds, sorted by call rate, and integrated to compute $\text{APGR}^{(b)}$.

### 6.2 Audit of Previous Hypothesis Test vs Null-Centered Formulation (M4.5.2)

#### 6.2.1 Defect in Initial M4.5.1 Formulation
In the initial M4.5.1 audit, the reported $p$-value was computed as:
$$\hat{p}_{\text{initial}} = \frac{1}{B} \sum_{b=1}^B \mathbf{1}\{ \text{APGR}^{(b)} \le 0.5000 \}$$
A rigorous statistical audit reveals that this procedure **does not construct a null distribution**:
- $\text{APGR}^{(b)}$ are ordinary bootstrap replicates of the test statistic drawn from the sample empirical distribution, centered around the point estimate $\hat{A} \approx 0.59$.
- Counting how many replicates of this uncentered distribution fall below $0.5000$ is merely an **inverted quantile check** (checking whether the $(1 - \alpha)$ confidence interval covers 0.50).
- In classical and bootstrap hypothesis testing (Hall & Wilson 1991; Efron & Tibshirani 1993; Davison & Hinkley 1997), a valid $p$-value MUST evaluate the probability of the observed test statistic under a distribution that satisfies the null hypothesis $H_0$.

#### 6.2.2 Statistically Rigorous Null-Centered Bootstrap Test
To resolve this defect, Milestone M4.5.2 implements a formal **null-centered prompt-cluster bootstrap hypothesis test**:
1. **Hypothesis Formulation**:
   - $H_0: \text{APGR} \le 0.5000$ (router does not exceed random routing).
   - $H_1: \text{APGR} > 0.5000$ (one-sided superiority over random routing).
   - Boundary null: $H_0^*: \text{APGR} = 0.5000$.
2. **Centered Deviations**:
   For observed APGR $\hat{A}$ and $B = 1,000$ prompt-cluster bootstrap replicates $A_b^*$:
   $$D_b = A_b^* - \hat{A}$$
   By construction, $E[D_b] \approx 0$.
3. **Null Distribution Construction**:
   Shift the bootstrap distribution to the boundary null:
   $$A_{b, \text{null}} = 0.5000 + D_b = 0.5000 + (A_b^* - \hat{A})$$
   The resulting distribution $A_{b, \text{null}}$ is explicitly centered at $E[A_{\text{null}}] \approx 0.5000$, representing the sampling variability of the router under the null hypothesis.
4. **One-Sided Exceedances & Finite-Sample Correction**:
   Count the number of null replicates as extreme as or more extreme than the observed sample statistic:
   $$k = \sum_{b=1}^B \mathbf{1}\{ A_{b, \text{null}} \ge \hat{A} \}$$
   Under the finite-sample correction of Davison & Hinkley (1997) and Phipson & Smyth (2010):
   $$\hat{p}_{\text{DH}} = \frac{1 + k}{B + 1}$$
   - This formulation guarantees that $\hat{p} \in (0, 1]$, strictly eliminating impossible claims of $p = 0.0000$ from finite Monte Carlo sampling.
   - For $B = 1,000$ replicates and $k = 0$ exceedances, the exact test returns $p = 1/1001 \approx 0.000999$, rounded to **`0.0010`**—the minimum attainable plus-one Monte Carlo estimate with 1,000 replicates and zero exceedances.

#### 6.2.3 Empirical Audit Results Across Learned Routers
The null-centered bootstrap test was executed with $B = 1,000$ prompt-cluster resamples ($N = 3,729$ prompts):

| Strategy | Observed Micro APGR ($\hat{A}$) | 95% Bootstrap CI | $\max(A_{\text{null}})$ | Exceedances ($k$) | Null-Calibrated $p$-Value ($\hat{p}_{\text{DH}}$) | Verdict vs Random |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **RouteLLM Hybrid Ensemble** | **0.5906** | [0.5761, 0.6058] | 0.5270 | 0 / 1,000 | **0.0010** ($1/1001 \approx 0.000999$) | Rejects $H_0$ ($p \approx 0.0010$) |
| **RouteLLM-Inspired kNN Router** | **0.5818** | [0.5670, 0.5971] | 0.5256 | 0 / 1,000 | **0.0010** ($1/1001 \approx 0.000999$) | Rejects $H_0$ ($p \approx 0.0010$) |
| **RouteLLM-Inspired Learned LR** | **0.5810** | [0.5648, 0.5965] | 0.5234 | 0 / 1,000 | **0.0010** ($1/1001 \approx 0.000999$) | Rejects $H_0$ ($p \approx 0.0010$) |
| **Random Selection** | **0.5000** | [0.5000, 0.5000] | — | — | N/A (Reference) | Null Boundary |

**Statistical Conclusion**:
- **Versus Random Routing**: Across all 1,000 null-centered bootstrap replicates, zero replicates reached or exceeded the observed APGR ($\max(A_{\text{null}}) \le 0.5270 < \hat{A} \ge 0.5810$). We therefore reject $H_0: \text{APGR} \le 0.5000$ with $p = 1/1001 \approx 0.000999$ (rounded to `0.0010`), which is the minimum attainable plus-one Monte Carlo estimate with 1,000 replicates and zero exceedances.
- **Between Learned Routers**: Overlapping APGR confidence intervals across the RouteLLM Hybrid Ensemble (`[0.5761, 0.6058]`), kNN Router (`[0.5670, 0.5971]`), and Learned LR (`[0.5648, 0.5965]`) do **not** establish statistically significant differences between those three routers. The current test compares each learned router against the random-routing boundary ($0.5000$), not against one another.

---

## 7. Cost and Operating-Point Calculations

### 7.1 Reconstructed Expenditure Verification
Costs are reconstructed strictly from observed token usage and official candidate pricing tables:
$$\text{Cost} = \frac{\text{input\_tokens} \times \text{input\_price} + \text{output\_tokens} \times \text{output\_price}}{1,000,000}$$
- Reconstructed expenditures were audited against token usage on all 80,802 training rows and 67,122 test rows.
- Maximum absolute difference: **$0.0000000000$** (exact arithmetic match).
- **Disclaimer**: Documented explicitly as reconstructed benchmark expenditures based on official candidate pricing, NOT actual provider billing invoices.

### 7.2 Training vs Test Cost Reconciliation

| Model ID | Train In-Tokens | Train Out-Tokens | Train Mean Cost | Test In-Tokens | Test Out-Tokens | Test Mean Cost |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `cogito-v2-1-671b` | 160.1 | 202.9 | **$0.000454** | 128.4 | 237.8 | **$0.000458** |
| `llama-3-8b-instruct-lite` | 160.1 | 34.1 | **$0.000019** | 128.4 | 42.1 | **$0.000017** |
| **Cost Multiplier** | — | — | **23.9x** | — | — | **26.9x** |

*Note: Test costs differ slightly from training costs because test prompts have slightly shorter inputs (128 vs 160 tokens) and slightly longer outputs for `cogito-v2-1-671b` (238 vs 203 tokens). This validates why test evaluation computes exact expenditures on test tokens while decision-time routers rely on training estimates.*

---

## 8. Files Updated & Delivered

1. `eval/m4_routellm_baseline.py`:
   - Upgraded `bootstrap_two_model_metrics` to construct a rigorous null-centered prompt-cluster bootstrap distribution ($A_{b, \text{null}} = 0.5000 + (A_b^* - \hat{A})$).
   - Applied Davison & Hinkley (1997) / Phipson & Smyth (2010) finite-sample correction: $\hat{p}_{\text{DH}} = \frac{1 + k}{B + 1}$.
   - Added continuous linear interpolation to `calculate_cpt`.
   - Added Macro APGR and Operating Macro Quality calculations.
   - Replaced abbreviated names with canonical IDs.
2. `reports/m4/m4_routellm_apgr.csv`:
   - Updated schema to include `micro_apgr`, `macro_apgr`, `micro_apgr_ci_low`, `micro_apgr_ci_high`, `p_value_vs_random` ($0.0010$), `op_micro_quality`, `op_macro_quality`.
3. `reports/m4/m4_routellm_results.json`:
   - Added full dual-metric metadata for micro and macro scores, and finite-sample corrected null p-values.
4. `reports/m4/m4_routellm_summary.md`:
   - Updated executive summary, methodology sections, and evaluation tables with canonical model IDs, dual Micro/Macro metrics, and null-centered hypothesis test documentation.
5. `tests_v2/test_m4_routellm.py`:
   - Added 4 audit-focused unit tests (`test_model_pair_canonical_identity_audit`, `test_dataset_counts_audit_reconciliation`, `test_null_centered_bootstrap_hypothesis_testing`, `test_micro_macro_apgr_consistency`).
   - Validates that ordinary CI estimation and null hypothesis testing are cleanly decoupled.
6. `reports/m4/m4_routellm_audit.md`:
   - Comprehensive audit deliverable for M4.5.1 and M4.5.2.

---

## 9. Regression Test Suite Execution

- **Targeted RouteLLM Tests**: **13 passed** in 4.70s (`tests_v2/test_m4_routellm.py`).
- **Full Regression Suite (`tests_v2/`)**: Clean execution across all milestones (M0–M4.5.2). Zero regressions.
