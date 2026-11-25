# M4.5 — RouteLLM Two-Model Baseline & APGR Evaluation Report

**Milestone**: M4.5 — RouteLLM Two-Model Routing and APGR Benchmark  
**Dataset**: `ulab-ai/xRouteBench` (Held-Out Test Set, N = 3,729 Prompts)  
**Candidate Pool**: 18 Candidate LLMs (Two-Model Subspace Selected)  
**Evaluation Principle**: Zero live LLM API calls, strictly deterministic offline replay, prompt-level bootstrap resampling.  

---

## 1. Executive Summary & APGR Benchmark Results

Milestone M4.5 introduces a research-grounded two-model routing baseline inspired by **RouteLLM** (*Ong et al., 2024, LMSYS Org / UC Berkeley*). The router dynamically predicts whether a given prompt requires invoking the expensive, high-capacity flagship model or can be handled by the cost-effective compact model without performance loss.

### Key APGR Benchmark Findings:
1. **Strong/Weak Pair Selection (Training-Derived)**: Flagship frontier model `cogito-v2-1-671b` (671B parameters) vs compact model `llama-3-8b-instruct-lite` (8B parameters). On test, `cogito-v2-1-671b` achieves **0.8054 Micro / 0.6536 Macro** quality at **$0.000458/query**, while `llama-3-8b-instruct-lite` achieves **0.4932 Micro / 0.4523 Macro** quality at **$0.000017/query** (a **31.2% Micro / 20.1% Macro** performance gap and a **26.9x** cost multiplier).
2. **Superiority Over Random Routing (APGR > 50%)**: A random routing policy achieves a theoretical APGR of **50.00%**. All three learned two-model routing policies outperform random under both Micro and Macro aggregations under a null-centered bootstrap hypothesis test ($H_0: \text{APGR} \le 0.50$ vs $H_1: \text{APGR} > 0.50$, $B = 1,000$):
   - **RouteLLM Hybrid Ensemble**: **59.06% Micro APGR** (95% CI: [57.61%, 60.58%], null-calibrated $p = 1/1001 \approx 0.000999$, rounded to `0.0010`), **59.04% Macro APGR**
   - **RouteLLM-Inspired kNN Router**: **58.18% Micro APGR** (95% CI: [56.70%, 59.71%], null-calibrated $p = 1/1001 \approx 0.000999$, rounded to `0.0010`), **57.72% Macro APGR**
   - **RouteLLM-Inspired Learned LR**: **58.10% Micro APGR** (95% CI: [56.48%, 59.65%], null-calibrated $p = 1/1001 \approx 0.000999$, rounded to `0.0010`), **58.32% Macro APGR**
   - *Note*: The plus-one value $p = 1/1001 \approx 0.000999$ (rounded to `0.0010`) is the minimum attainable Monte Carlo estimate with $B = 1,000$ replicates and zero exceedances. Furthermore, overlapping APGR confidence intervals across Hybrid, kNN, and LR do not establish statistically significant differences between those three routers; the test evaluates each learned router against the random-routing boundary ($0.5000$), not against one another.
3. **Cost Savings at High Gap Recovery**: At operating threshold $\theta = 0.30$, the Hybrid Ensemble routes only **31.4%** of queries to the strong model while recovering **44.7%** of the performance gap, cutting overall inference costs by **60.1%** relative to Always-Strong.
4. **Oracle Two-Model Ceiling**: The theoretical upper bound for two-model routing achieves a Micro APGR of **90.65%** (Macro APGR: **89.52%**), reaching 100% gap recovery with only **25.0%** strong calls.

---

## 2. Model-Pair Selection Methodology & Training Data Justification

Strict leakage guidelines require the model pair to be chosen **exclusively from training data** (`train_processed.parquet`) without inspecting test labels.

| Attribute | Strong Model (`M_strong`) | Weak Model (`M_weak`) | Comparison / Ratio |
| :--- | :--- | :--- | :--- |
| **Model Identifier** | `cogito-v2-1-671b` | `llama-3-8b-instruct-lite` | Frontier Flagship vs Compact 8B |
| **Parameter Scale** | 671B | 8B | ~84x parameter ratio |
| **Input Price / 1M** | $1.25 | $0.10 | 12.5x input pricing ratio |
| **Output Price / 1M** | $1.25 | $0.10 | 12.5x output pricing ratio |
| **Train Quality (Micro)** | 0.7470 (Rank #1 / 18) | 0.5558 | +0.1912 (+19.1%) |
| **Train Mean Cost** | $0.000454 | $0.000019 | **23.4x** cost multiplier |
| **Task Coverage** | 100% (4,489 / 4,489 train, 3,729 / 3,729 test) | 100% (4,489 / 4,489 train, 3,729 / 3,729 test) | Zero missing observations |
| **Test Quality (Micro)** | 0.8054 | 0.4932 | +0.3122 (+31.2%) |
| **Test Quality (Macro)** | 0.6536 | 0.4523 | +0.2013 (+20.1%) |
| **Test Mean Cost** | $0.000458 | $0.000017 | **26.9x** cost multiplier |

**Literature Alignment**: In the published RouteLLM paper (*Ong et al., 2024*), the canonical evaluation benchmarks evaluate routing between a frontier flagship model (GPT-4) and an open compact model (Llama-3-8B-Instruct). Our selection directly mirrors this architecture.

---

## 3. Mathematical Formulation of APGR, PGR, and CPT

### 3.1 Performance Gap Recovered (PGR)
Given evaluation score $r(M)$ on a benchmark, PGR quantifies what fraction of the capability difference between $M_{\text{weak}}$ and $M_{\text{strong}}$ is captured by the router:
$$\text{PGR}(M_R) = \frac{r(M_R) - r(M_{\text{weak}})}{r(M_{\text{strong}}) - r(M_{\text{weak}})}$$
- $\text{PGR} = 0.0$: Router performs identically to the weak baseline.
- $\text{PGR} = 1.0$: Router completely bridges the performance gap to match the strong model.
- $\text{PGR} > 1.0$: Router outperforms the strong model by selectively keeping weak model successes where the strong model failed.

### 3.2 Average Performance Gap Recovered (APGR)
Because an unconstrained router can trivially achieve $\text{PGR} = 1.0$ by routing 100% of queries to the strong model (at maximum cost), RouteLLM sweeps the decision threshold $\theta \in [0, 1]$ to generate a **call-performance curve** mapping strong call fraction $c \in [0, 1]$ to $\text{PGR}(c)$:
$$\text{APGR}(M_R) = \int_{0}^{1} \text{PGR}(M_{R_c}) \, dc$$
Numerically, APGR is computed using exact trapezoidal integration over sorted empirical call points $(c_k, \text{PGR}_k)$:
$$\text{APGR} \approx \sum_{k=1}^{K-1} \frac{\text{PGR}_k + \text{PGR}_{k+1}}{2} (c_{k+1} - c_k)$$

### 3.3 Call-Performance Threshold (CPT)
$$\text{CPT}(P) = \min \{ c \in [0, 1] : \text{PGR}(c) \ge P \}$$
Linearly interpolated between adjacent threshold operating points to determine the continuous call rate required to recover at least $P\%$ of the quality gap.

---

## 4. Empirical Evaluation Table Across Routing Strategies

| Strategy | Model Type | Micro APGR | Macro APGR | 95% Bootstrap CI | p-value vs Random | CPT(50%) | CPT(80%) | Call Rate | Micro Quality | Macro Quality | Cost / Query | PGR | Cost Savings |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Oracle Two-Model Ceiling (Theoretical Reference)** | Theoretical | **0.9065** | **0.9631** | Reference | N/A (Reference) | 15.6% | 25.0% | 35.0% | 0.8351 | 0.6896 | $0.000181 | **109.5%** | **60.4%** |
| **RouteLLM Hybrid Ensemble** | Deployable | **0.5906** | **0.5904** | [0.5761, 0.6058] | 0.0010 | 36.7% | 72.0% | 31.4% | 0.6329 | 0.5416 | $0.000183 | **44.7%** | **60.1%** |
| **RouteLLM-Inspired kNN Router** | Deployable | **0.5818** | **0.5772** | [0.5670, 0.5971] | 0.0010 | 38.3% | 73.4% | 38.4% | 0.6494 | 0.5507 | $0.000213 | **50.0%** | **53.5%** |
| **RouteLLM-Inspired Learned LR** | Deployable | **0.5810** | **0.5832** | [0.5648, 0.5965] | 0.0010 | 37.8% | 73.5% | 36.3% | 0.6450 | 0.5518 | $0.000197 | **48.6%** | **57.0%** |
| **Random Selection** | Reference | **0.5000** | **0.5000** | [0.5000, 0.5000] | N/A (Ref) | 50.0% | 80.0% | 50.0% | 0.6493 | 0.5529 | $0.000237 | 50.0% | 48.1% |
| **Always-Weak (`llama-3-8b-instruct-lite`)** | Baseline | **0.0000** | **0.0000** | Baseline | N/A (Base) | 100.0% | 100.0% | 0.0% | 0.4932 | 0.4523 | $0.000017 | 0.0% | 96.3% |
| **Always-Strong (`cogito-v2-1-671b`)** | Baseline | **1.0000** | **1.0000** | Baseline | N/A (Base) | 0.0% | 0.0% | 100.0% | 0.8054 | 0.6536 | $0.000458 | 100.0% | 0.0% |

*(Visual call-performance curve saved as SVG artifact: `reports/m4/routellm_pgr_tradeoff.svg`. Note: `p-value vs Random = 0.0010` represents the rounded plus-one Monte Carlo estimate $p = 1/1001 \approx 0.000999$ with $B = 1,000$ replicates and 0 exceedances. Overlapping 95% CIs across Hybrid, kNN, and LR do not establish statistically significant differences among those three learned routers.)*

---

## 5. Per-Task Performance Breakdown (Operating Point $\theta = 0.30$)

| Task Category | Metric | Weak Quality | Strong Quality | Performance Gap | Hybrid Quality | Hybrid PGR | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `aime_2020_2024` | GSM8K | 0.0000 | 0.0000 | +0.0000 | **0.0000** | **0.0%** | Neutral |
| `arc_challenge` | em_mc | 0.6920 | 0.9640 | +0.2720 | **0.7580** | **24.3%** | Moderate |
| `boolq` | em_mc | 0.7600 | 0.8400 | +0.0800 | **0.7600** | **0.0%** | Neutral |
| `commonsense_qa` | em_mc | 0.7600 | 0.8600 | +0.1000 | **0.7800** | **20.0%** | Moderate |
| `gsm8k` | GSM8K | 0.6420 | 0.8200 | +0.1780 | **0.6560** | **7.9%** | Moderate |
| `hellaswag` | em_mc | 0.6400 | 0.9000 | +0.2600 | **0.7600** | **46.2%** | Moderate |
| `human_eval` | code_eval | 0.0000 | 0.0000 | +0.0000 | **0.0000** | **0.0%** | Neutral |
| `math` | MATH | 0.1140 | 0.6780 | +0.5640 | **0.6660** | **97.9%** | Strong Gain |
| `mbpp` | code_eval | 0.4800 | 0.7100 | +0.2300 | **0.5200** | **17.4%** | Moderate |
| `mmlu` | em_mc | 0.5100 | 0.8700 | +0.3600 | **0.6020** | **25.6%** | Moderate |
| `mmlu_pro` | em_mc | 0.2940 | 0.7400 | +0.4460 | **0.5000** | **46.2%** | Moderate |
| `openbook_qa` | em_mc | 0.7020 | 0.9480 | +0.2460 | **0.7600** | **23.6%** | Moderate |
| `squad` | f1_score | 0.2857 | 0.1662 | +-0.1195 | **0.2793** | **0.0%** | Neutral |

---

## 6. Answers to Research and Methodological Inquiries

### 1. Whether this is an exact RouteLLM reproduction or an adaptation
- **Classification**: This is a **RouteLLM-inspired adaptation** tailored to multi-task benchmark evaluation on xRouteBench.
- **Exact Preservations**: We preserve RouteLLM's mathematical formulation of PGR, APGR, call rates $c$, CPT thresholds, and the core preference-labeling convention (*'We consider tie as a win for the weaker models'*).
- **Adaptations**: While the original RouteLLM paper trained routers on Chatbot Arena pairwise human preference votes (LMSYS Arena logs), our router is trained on objective benchmark evaluation labels ({0, 1} accuracy and token F1) from xRouteBench, utilizing dense Sentence-Transformer MiniLM embeddings and regularized classifiers.

### 2. Model-Pair Identity & Canonical Identifiers
- Selected **strictly on training data** (`train_processed.parquet`).
- **Canonical Strong Model ID**: `cogito-v2-1-671b` (Together API, 671B parameters). Identified as the highest-performing model across all 18 candidates ({pair_selection.strong_train_perf:.4f} micro quality).
- **Canonical Weak Model ID**: `llama-3-8b-instruct-lite` (Together API, 8B parameters). Selected as the canonical 8B compact model with 100% task coverage, low unit price ($0.10/1M), and a 23.4x cost differential.
- **Note on Shorthand Aliases**: Any informal references such as `cogito-671b` and `llama-3-8b` are colloquial display shorthand referring strictly to the canonical IDs `cogito-v2-1-671b` and `llama-3-8b-instruct-lite`. No separate or modified model configurations exist.

### 3. Dataset Count Reconciliation
- **Training Partition**: Contains **80,802 raw rows** ($4,489 \times 18$ models). Exactly **4,487 unique string queries** exist. When preserving cross-task prompts (1 query shared between `math` and `aime_2020_2024`) and distinct embeddings (1 query in `arc_challenge` with two distinct embeddings), there are **4,489 unique evaluation instances**.
- **Test Partition**: Contains **67,122 raw rows** ($3,729 \times 18$ models). Exactly **3,729 unique queries** and evaluation instances exist with zero duplicate queries across tasks.
- **Prompt-Grouped Splitting**: Validation splitting groups strictly by query text string, ensuring all rows belonging to any shared prompt remain strictly in the same split fold (zero leakage).

### 4. Quality Aggregation (Micro vs Macro)
- **Dual Aggregation Reporting**:
  - **Micro Quality** (sample-weighted prompt mean): Strong = **0.8054**, Weak = **0.4932** (Gap = **+0.3122**). The Strong model micro quality matches the M4.2 Best Single Model reference score (0.8054) identically.
  - **Macro Quality** (unweighted task mean across 13 tasks): Strong = **0.6536**, Weak = **0.4523** (Gap = **+0.2013**).
- **APGR Consistency**: Under Micro aggregation, Hybrid Ensemble achieves **59.06% APGR**; under Macro aggregation, it achieves **59.04% APGR**, demonstrating that learned router recovery is consistent regardless of task-size weighting.

### 5. Statistical Uncertainty & Hypothesis Testing
- **Prompt-Level Resampling**: Bootstrap resamples 3,729 prompt clusters with replacement, preserving prompt difficulty correlation and joint model outcomes.
- **Null-Centered Bootstrap Hypothesis Test**: We explicitly tested the boundary null hypothesis $H_0: \text{APGR} \le 0.5000$ (random routing baseline) against $H_1: \text{APGR} > 0.5000$ using a null-centered bootstrap distribution (Hall & Wilson 1991; Davison & Hinkley 1997). For observed APGR $\hat{A}$ and bootstrap replicates $A_b^*$, we compute centered deviations $D_b = A_b^* - \hat{A}$ and construct the null distribution $A_{b, \text{null}} = 0.5000 + D_b$. Across $B = 1,000$ replicates, zero replicates exceeded $\hat{A}$ ($k = 0$, maximum null replicate was $\le 0.5270$, below $\hat{A} \ge 0.5810$).
- **Finite-Sample Correction**: Applying the standard Davison & Hinkley (1997) / Phipson & Smyth (2010) plus-one correction $\hat{p} = \frac{1 + k}{B + 1} = \frac{1 + 0}{1000 + 1} = \frac{1}{1001} \approx 0.000999$ (rounded to **`0.0010`**) strictly avoids reporting pseudo-exact $p = 0.0000$ or uncalibrated ordinary bootstrap boundary checks. Note that $1/1001 \approx 0.000999$ (rounded to `0.0010`) is the minimum attainable plus-one Monte Carlo $p$-value estimate with $B = 1,000$ replicates and zero exceedances.
  - Hybrid Ensemble 95% CI: [0.5761, 0.6058]
  - kNN Router 95% CI: [0.5670, 0.5971]
  - Learned LR 95% CI: [0.5648, 0.5965]
- **Between-Router Comparisons**: Overlapping APGR confidence intervals do not establish statistically significant differences between the Hybrid, kNN, and LR routers. The current hypothesis test compares each learned router individually against the random-routing boundary ($0.5000$), not against one another.
- **Fixed Reference vs Empirical Interval**: The 0.5000 APGR for Random selection is an exact mathematical expectation (integral of the diagonal $y = x$), not an empirical sample with variance.
- **Replicate Adequacy**: $B = 1,000$ replicates provides a Monte Carlo standard error of $\approx 0.0049$ on the 2.5% tail probability and yields a minimum attainable plus-one $p$-value of $1/1001 \approx 0.000999$ (rounded to `0.0010`).

### 6. Limitations and Comparability to Published Results
- Direct numerical comparison to published RouteLLM APGR figures (e.g. 65–75% reported on Arena-Hard/MT-Bench) is not valid because benchmark difficulty, evaluation metrics, and candidate pairs differ.
- Multi-task benchmarks (especially competition math like AIME and coding like HumanEval) feature high inherent difficulty where even frontier models struggle, limiting the maximum achievable routing efficiency compared to conversational chatbot evaluations.