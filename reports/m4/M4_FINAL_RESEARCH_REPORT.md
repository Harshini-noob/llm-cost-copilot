# Adaptive Cost-Aware LLM Routing: Multi-Model Evaluation, Probability Calibration, Out-of-Distribution Generalization, and Two-Model Performance Gap Recovery

**Project:** LLM Cost Copilot (`https://github.com/Harshini-noob/llm-cost-copilot`)  
**Milestone:** M4.6 — Final Research Report & M4 Milestone Closure  
**Date:** November 25, 2026  
**Evaluation Policy:** 100% Offline Deterministic Replay | Zero Live LLM API Calls | Frozen M0–M3 Artifacts Preserved  

---

## 1. Abstract

Routing queries dynamically across heterogeneous Large Language Models (LLMs) promises to balance response quality against inference expenditure, yet empirical routing studies frequently suffer from data leakage, uncalibrated confidence scores, overclaimed out-of-distribution (OOD) transfer, or uncalibrated statistical significance tests. This report consolidates Milestone M4 (*Evaluate Honestly*) of **LLM Cost Copilot**, presenting a rigorous, leakage-safe offline evaluation on **xRouteBench** (`ulab-ai/xRouteBench`, `llmrouter_generic` and `llm_candidates` configurations), spanning **18 candidate LLMs** (7B–671B parameters), **13 benchmark tasks**, **80,802 training** model-outcome rows (4,487 unique query strings; 4,489 evaluation records per model), and **67,122 held-out test** rows (3,729 unique prompts).

Across multi-model cost-aware routing (M4.2–M4.2.1), a regularized Logistic Regression router (`LEARNED_LR`, $\lambda = 0.00$) achieves **0.7208 Macro Quality** [0.6994, 0.7478] (**0.7994 Micro Quality**) at **\$0.283 per 1,000 queries**, outperforming the **Best Single Model (Test-Set Reference)** (`qwen3-next-80b-a3b-instruct`, 0.6536 Macro Quality at \$0.458/1k) by **+0.0672 Macro Quality** while reducing cost by **38.26%**. At $\lambda = 0.75$, `LEARNED_LR` matches single-model quality parity (**0.6533** vs. **0.6536** Macro Quality) at **\$0.036/1k**—a **92.23% cost reduction**. Under probability calibration (M4.3), raw Logistic Regression exhibits strong inherent calibration (**0.0389 ECE**, **0.1734 Brier**), improved to **0.0283 ECE** via out-of-fold Platt scaling, whereas kNN ($k = 5$) is poorly calibrated (**0.1123 raw ECE**, **0.2173 Brier**). Under leave-one-task-out OOD evaluation across six held-out tasks (M4.4–M4.4.1), neither learned router universally dominates: `LEARNED_LR` leads on reasoning/knowledge (`gsm8k`, `mmlu`, `aime_2020_2024`), whereas `kNN` leads on code/comprehension (`mbpp`, `human_eval`, `squad`), and calibration degrades sharply under novel task shifts (up to +0.3009 to +0.3174 ECE on code tasks). Finally, in a RouteLLM-inspired two-model cascade between `cogito-v2-1-671b` and `llama-3-8b-instruct-lite` (M4.5–M4.5.2.1), all three learned routers recover **58.10%–59.06% Micro APGR** (**57.72%–59.04% Macro APGR**), exceeding the 0.5000 random-routing boundary under a null-centered prompt-cluster bootstrap test ($B = 1,000$, $k = 0$ exceedances, $p = 1/1001 \approx 0.000999$, rounded to `0.0010`), though overlapping 95% CIs do not establish significant differences among the three learned routers. Key limitations include noncommercial dataset licensing (CC BY-NC-SA 4.0), reconstructed rather than billed costs, small OOD sample sizes on `human_eval` ($N = 16$) and `aime_2020_2024` ($N = 13$), and offline-only validation without live production latency or global budget coordination.

---

## 2. Introduction

Modern LLM deployments face a sharp tension between capability, inference cost, and response latency. Frontier models such as 671B-parameter reasoning engines deliver high accuracy on complex mathematics and multi-step reasoning, but at unit prices up to ~25–30× higher than compact 7B–20B models. Conversely, routing all traffic to the cheapest model sacrifices substantial accuracy on difficult queries, while committing to a single strong model wastes budget on straightforward prompts that smaller models already solve reliably.

Adaptive LLM routing addresses this trade-off by predicting, prior to inference, which candidate model maximizes a cost-aware utility objective for a given prompt. However, offline routing benchmarks can easily produce misleading conclusions if they leak future response tokens or test labels, conflate post-hoc model selection with deployable routing, ignore probability calibration, overlook out-of-distribution (OOD) brittleness, or rely on uncalibrated statistical tests.

Milestone M4 (**Evaluate Honestly**) was designed to audit and benchmark LLM Cost Copilot's routing strategies under strict zero-live-inference, leakage-safe offline replay on **xRouteBench**. Specifically, this milestone evaluates M2 baselines, M3 learned routers (kNN and Logistic Regression), probability calibration, OOD generalization, and RouteLLM-inspired two-model gap recovery.

---

## 3. Research Questions

Milestone M4 is organized around five empirical research questions:

- **RQ1 (Multi-Model Routing Effectiveness):** How do learned routers (M3 kNN and Logistic Regression) compare against Always-Cheapest, Random-Uniform, Static Task-Type Table, and post-hoc reference baselines (Best Single Model Test-Set Reference and Oracle) across cost-quality trade-offs?
- **RQ2 (Probability Calibration):** How accurately calibrated are the correctness probabilities predicted by kNN and Logistic Regression routers, and does post-hoc calibration alter downstream cost-aware routing decisions?
- **RQ3 (Out-of-Distribution Generalization):** How well do learned routing policies and their probability calibrators generalize when entire benchmark task categories are excluded from training?
- **RQ4 (Two-Model Performance Gap Recovery):** In a RouteLLM-inspired strong/weak routing setup, can learned routers recover a substantial fraction of the quality gap between a frontier model and a compact 8B model while reducing reconstructed cost?
- **RQ5 (Statistical Reliability):** How statistically reliable are observed routing gains and OOD degradations under prompt-cluster bootstrap confidence intervals, exact small-sample binomial intervals, and null-centered bootstrap hypothesis testing?

---

## 4. Dataset and Experimental Environment

### 4.1 xRouteBench Provenance and Licensing
All M4 evaluations use **xRouteBench** (`ulab-ai/xRouteBench`, configurations `llmrouter_generic` and `llm_candidates`), released alongside *LLMRouter: A Multi-Task Benchmark and Framework for LLM Routing* (Feng et al., 2026, `arXiv:2608.06867`).

> [!IMPORTANT]
> **Licensing Constraint (CC BY-NC-SA 4.0):** `ulab-ai/xRouteBench` is distributed under the **Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International** license. Accordingly, all dataset artifacts, derived splits, and trained benchmark models from xRouteBench are restricted to **noncommercial research and evaluation**; commercial deployment requires independent licensing or training on commercially licensed data.

### 4.2 Dataset Composition and Verified Counts
Raw Parquet files in `data/external/xroutebench/` were verified against SHA-256 digests (`m4_dataset_audit.md`, `m4_dataset_audit.json`):

| File | Size (Bytes) | SHA-256 Digest |
| :--- | :---: | :--- |
| `train.parquet` | 17,092,011 | `9307e40bd27737592f25677f3a7e58a8712181b08fe435b82c2943d7b49105c2` |
| `test.parquet` | 15,629,470 | `e7ce786054bcf1b81f4da32243d4d6414f4d94d9b5729ebd8b239f289b6dabe9` |
| `llm_candidates.parquet` | 8,081 | `5131134ed46c9ff68c0744fb90556714e99cbabd3ab0541778929bae9a91f23f` |

Verified structural counts:
- **Candidate Models:** **18** LLMs spanning 7B to 671B parameters (`gpt-oss-20b` at \$0.05/\$0.20 per 1M tokens up to `cogito-v2-1-671b` at \$1.25/\$1.25 per 1M tokens and `qwen3-next-80b-a3b-instruct` at \$0.15/\$1.50 per 1M tokens).
- **Benchmark Tasks:** **13** tasks across mathematics, coding, commonsense reasoning, and reading comprehension (`aime_2020_2024`, `arc_challenge`, `boolq`, `commonsense_qa`, `gsm8k`, `hellaswag`, `human_eval`, `math`, `mbpp`, `mmlu`, `mmlu_pro`, `openbook_qa`, `squad`).
- **Training Partition:** **80,802** model-outcome rows ($4,489 \times 18$ models), **4,487** unique query strings, and **4,489** evaluation records per model.
- **Test Partition:** **67,122** model-outcome rows ($3,729 \times 18$ models), **3,729** unique query strings, and **3,729** evaluation records per model.
- **Train/Test Overlap:** **0** shared query strings between `train.parquet` and `test.parquet`.

### 4.3 Query Strings vs. Task-Query Groups vs. Evaluation Records
In `train.parquet`, 4,487 unique query strings map to 4,489 evaluation records per model (80,802 total rows) because two queries appear twice:
1. **Cross-Task Shared Query:** One mathematics query appears in both `math` (18 model rows) and `aime_2020_2024` (18 model rows), yielding 4,488 unique `(task_name, query)` pairs.
2. **Dual-Embedding Record:** One query in `arc_challenge` appears with two distinct `embedding_id` values (320 and 4190), yielding 4,489 unique `(task_name, query, embedding_id)` evaluation records per model.

To prevent internal validation leakage, `create_prompt_grouped_validation_split` in `eval/m4_data_adapter.py` groups strictly by normalized query string (`df["query"].str.strip()`), keeping all 36 rows of any shared query in the same fold. On the test set, unique queries, `(task_name, query)` pairs, and evaluation records are strictly 1-to-1 ($N = 3,729$).

### 4.4 Cost Reconstruction, Metrics, and Local Environment
- **Reconstructed Cost:** Computed in `Decimal` fixed-point arithmetic from observed token counts and candidate rate cards:
  $$\text{cost}_{\text{USD}} = \frac{\text{input\_tokens} \times \text{input\_price\_per\_1m} + \text{output\_tokens} \times \text{output\_price\_per\_1m}}{1{,}000{,}000}$$
- **Evaluation Metrics:** 12 tasks use binary $\{0.0, 1.0\}$ scoring (`em_mc`, `GSM8K`, `MATH`, `code_eval`), while `squad` uses continuous token-level `f1_score` $\in [0.0, 1.0]$, preserved without binarization.
- **Local Embedding & Execution Environment:** Python 3.13.3 (`.venv`), `SentenceTransformerEncoder(local_files_only=True)` using 384-dimensional MiniLM embeddings, `scikit-learn` / `numpy` / `pandas` / `pyarrow`, and **zero live LLM inference or paid API calls**.

---

## 5. Methodology

The M4 experimental pipeline enforces strict separation between pre-inference routing inputs and post-hoc evaluation labels:

1. **Data Ingestion and Validation (`eval/m4_data_adapter.py`):** Verifies SHA-256 digests, token arithmetic ($\text{token\_num} = \text{input\_tokens} + \text{output\_tokens}$), non-negativity, metric bounds, and zero train/test overlap, exporting processed tables to `data/processed/xroutebench/`.
2. **Prompt-Level Splitting:** Validation splits group by normalized query string so shared prompts never span train and validation folds.
3. **Feature Representation:** Pre-inference features combine cached/local 384-d MiniLM prompt embeddings with prompt-level metadata available before model invocation (excluding future output tokens, latencies, and ground truths).
4. **M3 kNN Router & M3 Logistic Regression Router:**
   - **M3 kNN:** Distance-weighted nearest-neighbor quality estimator in embedding space ($k = 5$ in M4.2/M4.3/M4.4).
   - **M3 Logistic Regression (`LEARNED_LR`):** Per-candidate L2-regularized logistic regression classifiers trained on pre-inference features to predict $\hat{Q}(m, x) = P(\text{success} \mid m, x)$.
5. **Cost-Aware Decision Utility:** At test time, routers estimate pre-inference cost $\widehat{\text{cost}}(m, x)$ using input token count $T_{\text{in}}(x)$ and training mean output tokens $\bar{T}_{\text{out}}(m)$ from candidate rate cards, selecting:
   $$m^*(x) = \arg\max_{m} \left[ \hat{Q}(m, x) - \lambda \cdot c_{\text{norm}}(m, x) \right]$$
   swept across $\lambda \in \{0.00, 0.05, 0.10, 0.25, 0.50, 0.75, 1.00, 1.50, 2.00, 3.00, 5.00\}$.
6. **Baselines (`eval/m4_router_evaluation.py`):** Always-Cheapest (`gpt-oss-20b`), Random-Uniform (deterministic `seed=42` uniform $1/18$ sampling), Static Task-Type Table (best training model per `task_name`), Best Single Model (Test-Set Reference), and Oracle ($\lambda = 0.00$).
7. **Calibration (`eval/m4_calibration.py`):** Evaluates Brier score, 10-bin Expected Calibration Error (ECE), and Maximum Calibration Error (MCE) for raw scores, Platt scaling, and isotonic regression fitted strictly on 5-fold prompt-grouped training out-of-fold (OOF) predictions.
8. **OOD Evaluation (`eval/m4_ood_evaluation.py`):** Leave-one-task-out (LOTO) removes all training/validation rows of target task $T^*$ before fitting scalers, routers, and calibrators, evaluating on $\mathcal{D}_{\text{test}}^{(T^*)}$.
9. **Two-Model RouteLLM-Inspired Routing (`eval/m4_routellm_baseline.py`):** Adapts RouteLLM (*Ong et al., 2024*) from Chatbot Arena human preference logs to objective xRouteBench outcomes $\{0, 1\}$ / $[0, 1]$, labeling $y_i = 1$ if $Q(M_{\text{strong}}, x_i) > Q(M_{\text{weak}}, x_i)$ and $y_i = 0$ on ties or weak wins. Evaluates PGR, trapezoidal APGR, linearly interpolated CPT, and null-centered prompt-cluster bootstrap ($B = 1,000$).
10. **Prompt-Clustered Bootstrap:** All confidence intervals resample unique test prompt clusters ($N = 3,729$) with replacement, preserving joint multi-model outcomes and task-macro structure.

---

## 6. M4.1 — Dataset Integration and Validation

Milestone M4.1 (`eval/m4_data_adapter.py`, [m4_dataset_audit.md](file:///d:/llm%20cost%20copilot/reports/m4/m4_dataset_audit.md)) verified 100% dataset integrity before any router evaluation:
- **Cryptographic Integrity:** All three raw Parquet files matched their recorded SHA-256 digests.
- **Completeness & Coverage:** All 18 candidate models have 100% coverage across all 4,489 training records and 3,729 test prompts (zero missing model-prompt pairs).
- **Arithmetic & Bounds:** 100% of 147,924 combined rows satisfy $\text{token\_num} = \text{input\_tokens} + \text{output\_tokens}$, with zero negative values in tokens, latencies, or prices, and all `performance` values bounded in $[0.0, 1.0]$.
- **Pricing & Cost Validation:** Reconstructed `Decimal` costs matched token-pricing arithmetic to 10 decimal places (`max_diff = 0.0000000000`).
- **Tests:** All 234 pre-existing and M4.1 adapter tests passed without modifying frozen M0–M3 artifacts (`data/processed/matrix.parquet`).

---

## 7. M4.2 — Multi-Model Routing Evaluation

Milestone M4.2 evaluated seven strategy families across 3,729 held-out test prompts ([m4_evaluation_summary.md](file:///d:/llm%20cost%20copilot/reports/m4/m4_evaluation_summary.md), [m4_router_comparison.csv](file:///d:/llm%20cost%20copilot/reports/m4/m4_router_comparison.csv)). Because test task counts are unbalanced (500 prompts each for `gsm8k`, `math`, `mbpp`, `mmlu`, `mmlu_pro`, `arc_challenge`, `openbook_qa` vs. 50 for `boolq`, `commonsense_qa`, `hellaswag`, `squad`; 16 for `human_eval`; 13 for `aime_2020_2024`), **Task Macro Quality** (unweighted mean across 13 tasks) is reported alongside **Prompt Micro Quality**.

### Key Multi-Model Findings (`m4_router_comparison.json` / `m4_evaluation_summary.md`)
- **Always-Cheapest** (`gpt-oss-20b`, deployable): **0.5806** Macro [0.5627, 0.5965] | **0.6976** Micro | **\$0.014 / 1k** | 3.21s historical LLM latency.
- **Random-Uniform** (deployable, $1/18$ uniform sampling, `seed=42`): **0.5109** Macro [0.4897, 0.5292] | **0.6173** Micro | **\$0.181 / 1k** | 2.76s.
- **Best Single Model (Test-Set Reference)** (`qwen3-next-80b-a3b-instruct`, post-hoc reference): **0.6536** Macro [0.6368, 0.6650] | **0.8054** Micro [0.7931, 0.8188] | **\$0.458 / 1k** | 2.52s. Not deployable without test-set hindsight.
- **Static-Task-Table** (deployable when `task_name` is known): **0.7321** Macro [0.7041, 0.7570] | **0.8137** Micro | **\$0.315 / 1k** | 3.15s.
- **M3 `KNN` ($\lambda = 0.00 \rightarrow 5.00$):** Peaks at $\lambda = 0.10$ with **0.6529** Macro [0.6251, 0.6739] (**0.7115** Micro) at **\$0.072 / 1k** (84.31% cheaper than Best Single Model at 99.9% Macro quality). At $\lambda = 5.00$, drops to **0.6152** Macro at **\$0.017 / 1k**.
- **M3 `LEARNED_LR` ($\lambda = 0.00 \rightarrow 5.00$):**
  - At $\lambda = 0.00$: **0.7208** Macro [0.6994, 0.7478] (**0.7994** Micro) at **\$0.283 / 1k** (+0.0672 Macro over Best Single Model, 38.26% cost reduction).
  - At $\lambda = 0.75$ (quality parity): **0.6533** Macro [0.6305, 0.6761] (**0.7380** Micro) at **\$0.036 / 1k** (**92.23% cost reduction** vs. Best Single Model).
  - At $\lambda = 5.00$: **0.6411** Macro [0.6191, 0.6638] (**0.7138** Micro) at **\$0.014 / 1k** (6.42% cheaper than Always-Cheapest with +0.0604 higher Macro quality).
- **Oracle ($\lambda = 0.00$, theoretical reference):** **0.8533** Macro [0.8325, 0.8760] | **0.9293** Micro | **\$0.032 / 1k**. Impossible to deploy in practice.

---

## 8. M4.2.1 — Methodology Integrity Audit

An independent audit of M4.2 ([m4_methodology_audit.md](file:///d:/llm%20cost%20copilot/reports/m4/m4_methodology_audit.md)) identified and corrected four methodological and terminology issues:
1. **Random Baseline Naming:** `evaluate_random_matched_cost` sampled all 18 models uniformly ($p = 1/18$) without matching a target cost. Renamed to **`Random-Uniform`** (`evaluate_random_uniform`); no artificial cost-matched numbers were fabricated.
2. **Best Single Model Terminology:** `Always-Best` selected the single highest-scoring test model (`qwen3-next-80b-a3b-instruct`). Renamed to **`Best Single Model (Test-Set Reference)`** (`post_hoc_reference`) and clarified as a static hindsight reference, not a universal quality ceiling.
3. **Oracle Formulation:** Explicitly documented as `Oracle (lambda=0.00)` maximizing $U(m, x) = Q(m, x) - \lambda \cdot c_{\text{norm}}(m)$ with $\lambda = 0.00$ and lowest-cost tie-breaking, strictly excluded from training/calibration.
4. **Historical Latency Labeling:** Raw `response_time` records historical provider API execution latency from xRouteBench, not local router CPU overhead (sub-millisecond). Relabeled **`Historical LLM Latency (sec)`** across CSVs and reports.
5. **Bootstrap Integrity:** Verified `bootstrap_prompt_metric` resamples prompt clusters ($N = 3,729$, `seed=42`) while preserving task-macro aggregation. All 264 `tests_v2` tests passed after M4.2.1.

---

## 9. M4.3 — Probability Calibration

Milestone M4.3 ([m4_calibration_summary.md](file:///d:/llm%20cost%20copilot/reports/m4/m4_calibration_summary.md), [m4_calibration_metrics.csv](file:///d:/llm%20cost%20copilot/reports/m4/m4_calibration_metrics.csv)) evaluated 67,122 test predictions using 10 uniform bins in $[0, 1]$, fitting Platt scaling and isotonic regression strictly on 5-fold prompt-grouped training out-of-fold (OOF) predictions:

| Router Model | Calibration Method | Brier Score ($\downarrow$) | ECE ($\downarrow$) | MCE ($\downarrow$) | Test Observations |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **kNN ($k = 5$)** | Raw (Uncalibrated) | 0.2173 | 0.1123 | 0.2095 | 67,122 |
| **kNN ($k = 5$)** | Platt Scaling | 0.2112 | 0.0831 | 0.2983 | 67,122 |
| **kNN ($k = 5$)** | Isotonic Regression | 0.2023 | 0.0490 | 0.0613 | 67,122 |
| **Learned Router (LR)** | Raw (Uncalibrated) | 0.1734 | 0.0389 | 0.0594 | 67,122 |
| **Learned Router (LR)** | Platt Scaling | **0.1727** | **0.0283** | **0.0434** | 67,122 |
| **Learned Router (LR)** | Isotonic Regression | 0.1740 | 0.0396 | 0.0785 | 67,122 |

Key calibration findings:
- **LR vs. kNN:** Uncalibrated Logistic Regression is already well-calibrated (**0.0389 ECE**), and OOF Platt scaling reduces ECE to **0.0283** and Brier score to **0.1727**. Raw kNN is overconfident in high-probability bins (0.9996 mean confidence vs. 0.7901 observed accuracy in $[0.90, 1.00]$) and underconfident in $[0.00, 0.10)$ (0.0029 vs. 0.1143); isotonic regression reduces kNN ECE from **0.1123** to **0.0490**.
- **SQuAD Continuous F1 Interpretation:** For `squad`, predicted probabilities align with expected continuous token F1 ($[0, 1]$) rather than binary accuracy (LR Raw ECE `0.0541`, LR Platt ECE `0.0513`, LR Raw Brier `0.0360`).
- **Downstream Routing Invariance:** Across all 11 $\lambda$ values (`m4_calibration_downstream_impact.csv`), Platt calibration changes LR Macro Quality by at most `+0.0035` (at $\lambda = 0.75$) and mean cost by $\le \$0.000004$/query, confirming that uncalibrated LR probabilities already preserve utility ranking.
- **Visual Reliability Diagrams:** [reliability_diagram_learned_lr.svg](file:///d:/llm%20cost%20copilot/reports/m4/reliability_diagram_learned_lr.svg), [reliability_diagram_knn.svg](file:///d:/llm%20cost%20copilot/reports/m4/reliability_diagram_knn.svg), and [reliability_diagram_calibrated_lr.svg](file:///d:/llm%20cost%20copilot/reports/m4/reliability_diagram_calibrated_lr.svg).

---

## 10. M4.4 — Out-of-Distribution Generalization

Milestone M4.4 ([m4_ood_summary.md](file:///d:/llm%20cost%20copilot/reports/m4/m4_ood_summary.md), [m4_ood_results.csv](file:///d:/llm%20cost%20copilot/reports/m4/m4_ood_results.csv), [m4_ood_calibration.csv](file:///d:/llm%20cost%20copilot/reports/m4/m4_ood_calibration.csv)) evaluated leave-one-task-out (LOTO) transfer across six held-out tasks ($\lambda = 0.00$):

| Held-Out Task | Test $N$ | Metric | `LEARNED_LR` OOD | `kNN` OOD | `Static-Task-Table` (Fallback) | `Always-Cheapest` | Best Single Model (Ref) | Winner (LR vs. kNN) |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `gsm8k` | 500 | `GSM8K` | **0.8200** | 0.7060 | 0.8200 | 0.6500 | 0.9380 (`llama-4-maverick`) | **Learned-LR** (+0.1140) |
| `mmlu` | 500 | `em_mc` | **0.7720** | 0.6760 | 0.8700 | 0.7380 | 0.8700 (`cogito-v2-1-671b`) | **Learned-LR** (+0.0960) |
| `mbpp` | 500 | `code_eval` | 0.4820 | **0.5020** | 0.7100 | 0.6240 | 0.7100 (`cogito-v2-1-671b`) | **kNN** (+0.0200) |
| `human_eval` | 16 | `code_eval` | 0.0000 | **0.0625** | 0.0000 | 0.0000 | 0.7500 (`qwen3-coder-next`) | **kNN** (+0.0625) |
| `squad` | 50 | `f1_score` | 0.2153 | **0.2222** | 0.1662 | 0.2464 | 0.2857 (`llama-3-8b-instruct-lite`) | **kNN** (+0.0069) |
| `aime_2020_2024` | 13 | `GSM8K` | **0.0769** | 0.0000 | 0.0000 | 0.0000 | 0.1538 (`llama-3.3-70b-instruct-turbo`) | **Learned-LR** (+0.0769) |

Key OOD findings:
- **3–3 Domain Split:** Neither router dominates universally. `LEARNED_LR` wins on math/knowledge (`gsm8k`, `mmlu`, `aime_2020_2024`), whereas `kNN` wins on code synthesis and reading comprehension (`mbpp`, `human_eval`, `squad`).
- **Task-Aware vs. Task-Agnostic Features:** One-hot task indicators collapse to the zero vector $\mathbf{0}$ on unseen tasks, yielding virtually identical OOD performance (e.g., `0.7720` vs. `0.7720` on `mmlu`).
- **Calibration Degradation Under Shift:** While LR ECE shifts by only `+0.0201` on `mmlu` (0.0757 $\rightarrow$ 0.0958), ECE inflates severely on novel domains: `+0.2307` on `gsm8k` and `+0.3174` on `human_eval` for LR, and `+0.3009` on `mbpp` (`0.4009` OOD ECE) and `+0.3159` on `human_eval` (`0.4208` OOD ECE) for kNN.
- **Visual OOD Charts:** [ood_quality_degradation.svg](file:///d:/llm%20cost%20copilot/reports/m4/ood_quality_degradation.svg) and [ood_calibration_shift.svg](file:///d:/llm%20cost%20copilot/reports/m4/ood_calibration_shift.svg).

---

## 11. M4.4.1 — OOD Statistical Validity Audit

The M4.4.1 audit ([m4_ood_methodology_audit.md](file:///d:/llm%20cost%20copilot/reports/m4/m4_ood_methodology_audit.md)) corrected two statistical defects in the preliminary OOD pipeline:
1. **Paired Difference Bootstrap for $\Delta Q = Q_{\text{OOD}} - Q_{\text{ID}}$:** Replaced marginal OOD CIs with a paired prompt-level difference bootstrap ($B = 1,000$ in `eval/m4_ood_evaluation.py`), preserving within-prompt correlation. Statistically significant degradation ($p < 0.05$) was confirmed for `gsm8k` (`-0.1100` LR, CI `[-0.1420, -0.0819]` in `m4_ood_results.csv` / `m4_ood_summary.md`), `mbpp` (`-0.2200` LR, CI `[-0.2620, -0.1800]`; `-0.1320` kNN, CI `[-0.1720, -0.0980]`), and `mmlu` (`-0.0340` LR, CI `[-0.0640, -0.0040]`).
2. **Small-Sample & Zero-Success Binomial Intervals:** On `human_eval` ($N = 16$) and `aime_2020_2024` ($N = 13$), naive percentile bootstrap collapsed to `[0.0000, 0.0000]` when $k = 0$. Implemented **Wilson score** and **Clopper-Pearson exact** 95% intervals:
   - `human_eval` ($k = 0, N = 16$, LR OOD): Wilson `[0.0000, 0.1936]`, Clopper-Pearson `[0.0000, 0.2059]`.
   - `human_eval` ($k = 1, N = 16$, kNN OOD): Wilson `[0.0111, 0.2833]`, Clopper-Pearson `[0.0016, 0.3023]`.
   - `aime_2020_2024` ($k = 1, N = 13$, LR OOD): Wilson `[0.0137, 0.3331]`, Clopper-Pearson `[0.0019, 0.3603]`.
   - `aime_2020_2024` ($k = 0, N = 13$, kNN OOD): Wilson `[0.0000, 0.2281]`, Clopper-Pearson `[0.0000, 0.2471]`.
3. **Train-Only Preprocessing & Fallback:** Verified `StandardScaler`, classifiers, and `Static-Task-Table` global fallback fit strictly on $\mathcal{D}_{\text{train}}^{(-T^*)}$, with all 287 `tests_v2` tests passing after M4.4.1.

---

## 12. M4.5 — RouteLLM-Inspired Two-Model Routing

Milestone M4.5 ([m4_routellm_summary.md](file:///d:/llm%20cost%20copilot/reports/m4/m4_routellm_summary.md), [m4_routellm_apgr.csv](file:///d:/llm%20cost%20copilot/reports/m4/m4_routellm_apgr.csv), [m4_routellm_results.json](file:///d:/llm%20cost%20copilot/reports/m4/m4_routellm_results.json)) adapted RouteLLM (*Ong et al., 2024*) to xRouteBench using a training-only model pair:
- **Strong Model:** `cogito-v2-1-671b` (671B, \$1.25/\$1.25 per 1M tokens; train quality **0.7470**, test Micro **0.8054** / Macro **0.6536**, test mean cost **\$0.000458**/query).
- **Weak Model:** `llama-3-8b-instruct-lite` (8B, \$0.10/\$0.10 per 1M tokens; train quality **0.5558**, test Micro **0.4932** / Macro **0.4523**, test mean cost **\$0.000017**/query).
- **Pair Gap & Cost Multiplier:** +0.1912 train gap (+0.3122 test Micro / +0.2013 test Macro gap) and **23.4×** train (**26.9×** test) cost ratio.
- **Adaptation vs. Exact Reproduction:** Unlike published RouteLLM (trained on Chatbot Arena pairwise human votes), this adaptation trains on objective xRouteBench outcomes using 384-d MiniLM embeddings and L2-regularized LR ($C = 0.0050$ tuned on prompt-grouped validation), cosine kNN ($k = 20$), and their Hybrid Ensemble, with ties labeled as weak wins ($y = 0$). Direct numerical comparison to published RouteLLM APGR values on Arena-Hard/MT-Bench is therefore invalid.

### Verified Two-Model APGR & Operating-Point Results (`m4_routellm_apgr.csv`)

| Strategy | Type | Micro APGR | Macro APGR | 95% Micro CI | $p$-value vs. Random | CPT(50%) | CPT(80%) | Op. Call Rate ($\theta=0.30$) | Op. Micro Quality | Op. Cost / Query | Op. PGR | Cost Savings |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Oracle Two-Model Ceiling** | Theoretical | **0.9065** | **0.9631** | Reference | N/A (Reference) | 15.6% | 25.0% | 35.0% | 0.8351 | \$0.000181 | **109.5%** | **60.4%** |
| **RouteLLM Hybrid Ensemble** | Deployable | **0.5906** | **0.5904** | [0.5761, 0.6058] | `0.0010` | 36.7% | 72.0% | 31.4% | 0.6329 | \$0.000183 | **44.7%** | **60.1%** |
| **RouteLLM-Inspired kNN Router** | Deployable | **0.5818** | **0.5772** | [0.5670, 0.5971] | `0.0010` | 38.3% | 73.4% | 38.4% | 0.6494 | \$0.000213 | **50.0%** | **53.5%** |
| **RouteLLM-Inspired Learned LR** | Deployable | **0.5810** | **0.5832** | [0.5648, 0.5965] | `0.0010` | 37.8% | 73.5% | 36.3% | 0.6450 | \$0.000197 | **48.6%** | **57.0%** |
| **Random Selection** | Reference | **0.5000** | **0.5000** | [0.5000, 0.5000] | N/A (Reference) | 50.0% | 80.0% | 50.0% | 0.6493 | \$0.000237 | 50.0% | 48.1% |
| **Always-Weak (`llama-3-8b-instruct-lite`)** | Baseline | **0.0000** | **0.0000** | Baseline | N/A (Baseline) | 100.0% | 100.0% | 0.0% | 0.4932 | \$0.000017 | 0.0% | 96.3% |
| **Always-Strong (`cogito-v2-1-671b`)** | Baseline | **1.0000** | **1.0000** | Baseline | N/A (Baseline) | 0.0% | 0.0% | 100.0% | 0.8054 | \$0.000458 | 100.0% | 0.0% |

- **Why Oracle PGR Exceeds 1.0:** On 206 test prompts (5.5%), `llama-3-8b-instruct-lite` succeeds where `cogito-v2-1-671b` fails; the Oracle routes those prompts to the weak model, achieving **0.8351** Micro quality above Always-Strong's **0.8054** (**109.51% PGR**).
- **Operating Point ($\theta = 0.30$):** Hybrid Ensemble sends **31.4%** of queries to `cogito-v2-1-671b`, recovers **44.7%** of the performance gap (**0.6329** Micro / **0.5416** Macro quality), and cuts reconstructed cost by **60.1%** (\$0.000183 vs. \$0.000458/query).
- **Visual RouteLLM Charts:** [routellm_pgr_tradeoff.svg](file:///d:/llm%20cost%20copilot/reports/m4/routellm_pgr_tradeoff.svg) and [routellm_cost_quality_tradeoff.svg](file:///d:/llm%20cost%20copilot/reports/m4/routellm_cost_quality_tradeoff.svg).

---

## 13. M4.5.1, M4.5.2, and M4.5.2.1 — Statistical Audits

Three sequential audits hardened M4.5 ([m4_routellm_audit.md](file:///d:/llm%20cost%20copilot/reports/m4/m4_routellm_audit.md)):
1. **M4.5.1 (Identity, Counts, Aggregation, CPT):** Verified canonical IDs (`cogito-v2-1-671b`, `llama-3-8b-instruct-lite`), reconciled 4,487 unique train query strings vs. 4,489 evaluation records, added dual Micro/Macro APGR reporting, and upgraded `calculate_cpt` to continuous linear interpolation.
2. **M4.5.2 (Null-Centered Bootstrap Test):** Replaced M4.5.1's uncentered quantile check ($\frac{1}{B}\sum \mathbf{1}\{A_b^* \le 0.50\}$) with a **null-centered prompt-cluster bootstrap** ($B = 1,000$):
   $$D_b = A_b^* - \hat{A}, \qquad A_{b, \text{null}} = 0.5000 + D_b, \qquad k = \sum_{b=1}^{B} \mathbf{1}\{ A_{b, \text{null}} \ge \hat{A} \}$$
   with the Davison & Hinkley (1997) / Phipson & Smyth (2010) plus-one finite-sample correction:
   $$\hat{p}_{\text{DH}} = \frac{1 + k}{B + 1}$$
   Across $B = 1,000$ replicates, $k = 0$ exceedances occurred for all three learned routers ($\max(A_{\text{null}}) = 0.5270$ Hybrid, $0.5256$ kNN, $0.5234$ LR), yielding **$p = 1/1001 \approx 0.000999$, rounded to `0.0010`**—the minimum attainable plus-one Monte Carlo estimate with 1,000 replicates and zero exceedances.
3. **M4.5.2.1 (Reporting Cleanup):** Removed unsupported standard-error/standard-deviation threshold claims and explicitly documented that overlapping 95% CIs across Hybrid (`[0.5761, 0.6058]`), kNN (`[0.5670, 0.5971]`), and LR (`[0.5648, 0.5965]`) do **not** establish statistically significant differences among those three routers; the test evaluates each router against the `0.5000` random-routing boundary, not against one another.

---

## 14. Consolidated Results

Because multi-model routing (18 candidates, utility $U = Q - \lambda c_{\text{norm}}$) and two-model RouteLLM evaluation (2 candidates, threshold sweep $\theta \in [0, 1]$) use distinct protocols, results are reported in separate tables.

### Table 14.1: Multi-Model Routing, Calibration, and OOD Summary (18 Candidates, $N = 3,729$ Test Prompts)

| Strategy | Protocol / Role | Macro Quality [95% CI] | Micro Quality | Mean Cost / 1k | In-Dist Calibration (ECE / Brier) | OOD Behavior (6 Held-Out Tasks) |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **Always-Cheapest** (`gpt-oss-20b`) | Deployable Baseline | 0.5806 [0.5627, 0.5965] | 0.6976 | \$0.014 | N/A | Static fallback (`0.6500` gsm8k, `0.0000` human_eval) |
| **Random-Uniform** | Deployable Baseline | 0.5109 [0.4897, 0.5292] | 0.6173 | \$0.181 | N/A | Uniform $1/18$ baseline |
| **Best Single Model** (`qwen3-next-80b-a3b-instruct`) | Post-Hoc Test Reference | 0.6536 [0.6368, 0.6650] | 0.8054 | \$0.458 | N/A | Per-task hindsight reference only |
| **Static-Task-Table** | Deployable (Needs `task_name`) | 0.7321 [0.7041, 0.7570] | 0.8137 | \$0.315 | N/A | Falls back to global train winner (`cogito-v2-1-671b`) |
| **M3 `KNN` ($\lambda = 0.10$)** | Deployable Learned | 0.6529 [0.6251, 0.6739] | 0.7115 | \$0.072 | Raw: 0.1123 / 0.2173; Iso: 0.0490 / 0.2023 | Wins 3/6 OOD (`mbpp`, `human_eval`, `squad`); ECE >0.40 on code |
| **M3 `LEARNED_LR` ($\lambda = 0.00$)** | Deployable Learned | 0.7208 [0.6994, 0.7478] | 0.7994 | \$0.283 | Raw: 0.0389 / 0.1734; Platt: 0.0283 / 0.1727 | Wins 3/6 OOD (`gsm8k`, `mmlu`, `aime_2020_2024`); mild ECE shift on `mmlu` (+0.0201) |
| **M3 `LEARNED_LR` ($\lambda = 0.75$)** | Deployable (Parity Point) | 0.6533 [0.6305, 0.6761] | 0.7380 | \$0.036 | Same underlying LR probabilities | N/A (OOD audited at $\lambda \in \{0.0, 0.1\}$) |
| **Oracle ($\lambda = 0.00$)** | Theoretical Reference | 0.8533 [0.8325, 0.8760] | 0.9293 | \$0.032 | N/A | Theoretical upper bound |

### Table 14.2: Two-Model RouteLLM-Inspired APGR Summary (`cogito-v2-1-671b` vs. `llama-3-8b-instruct-lite`)

| Strategy | Protocol / Role | Micro APGR [95% CI] | Macro APGR | $p$-Value vs. Random (0.50) | CPT(50%) / CPT(80%) | Call Rate ($\theta=0.30$) | Cost / Query ($\theta=0.30$) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **RouteLLM Hybrid Ensemble** | Deployable Learned | **0.5906** [0.5761, 0.6058] | 0.5904 | `0.0010` ($1/1001$) | 36.7% / 72.0% | 31.4% | \$0.000183 |
| **RouteLLM-Inspired kNN** | Deployable Learned | **0.5818** [0.5670, 0.5971] | 0.5772 | `0.0010` ($1/1001$) | 38.3% / 73.4% | 38.4% | \$0.000213 |
| **RouteLLM-Inspired LR** | Deployable Learned | **0.5810** [0.5648, 0.5965] | 0.5832 | `0.0010` ($1/1001$) | 37.8% / 73.5% | 36.3% | \$0.000197 |
| **Oracle Two-Model Ceiling** | Theoretical Reference | 0.9065 | 0.9631 | N/A (Reference) | 15.6% / 25.0% | 35.0% (Peak) | \$0.000181 |
| **Random / Always-Weak / Always-Strong** | Theoretical / Baselines | 0.5000 / 0.0000 / 1.0000 | 0.5000 / 0.0000 / 1.0000 | N/A | 50%/80% / 100% / 0% | 50% / 0% / 100% | \$0.000237 / \$0.000017 / \$0.000458 |

---

## 15. Discussion

1. **Answers to RQ1–RQ5:**
   - **RQ1:** Learned multi-model routing (`LEARNED_LR`) matches the post-hoc Best Single Model at 92.23% lower reconstructed cost ($\lambda = 0.75$) and outperforms it by +0.0672 Macro Quality at 38.26% lower cost ($\lambda = 0.00$). However, when `task_name` metadata is available in-distribution, `Static-Task-Table` remains a strong baseline (0.7321 Macro Quality), showing that no learned router universally dominates every setting.
   - **RQ2:** `LEARNED_LR` is well-calibrated out-of-the-box (0.0389 ECE, improved to 0.0283 via Platt scaling), while kNN requires isotonic calibration (0.1123 $\rightarrow$ 0.0490 ECE). Calibration preserves utility ranking in-distribution.
   - **RQ3:** OOD generalization is domain-dependent: LR transfers better on reasoning/knowledge (`gsm8k`, `mmlu`, `aime_2020_2024`), while kNN transfers better on code/comprehension (`mbpp`, `human_eval`, `squad`). Both suffer substantial calibration inflation on novel task types.
   - **RQ4 & RQ5:** Two-model routing recovers 58.10%–59.06% Micro APGR and cuts strong-model cost by 60.1% at $\theta = 0.30$ (Hybrid), rejecting the 0.5000 random boundary ($p = 1/1001 \approx 0.000999$, rounded to `0.0010`), though overlapping CIs prevent ranking Hybrid, kNN, and LR as significantly different from each other.
2. **Benchmark Quality vs. Production Utility:** High offline accuracy on static academic benchmarks does not automatically translate to production reliability, where prompt distributions drift, task labels are absent, and real-time latency, rate limits, and budget caps govern user experience.

---

## 16. Limitations and Threats to Validity

1. **Benchmark Representativeness & Licensing:** xRouteBench comprises 13 academic tasks and carries a **CC BY-NC-SA 4.0** noncommercial license, limiting generalizability to enterprise traffic and prohibiting commercial reuse of xRouteBench-derived weights.
2. **Reconstructed Costs vs. Live Invoices:** Costs are reconstructed from token counts and published rate cards, excluding provider taxes, minimum billing units, failed retries, and dynamic pricing changes.
3. **Offline Replay vs. Live Inference:** Zero live API calls were executed; live stochastic decoding variance and provider outages are not captured.
4. **Historical Latency vs. Router Overhead:** Reported latencies reflect historical xRouteBench provider response times, not end-to-end production pipeline latency including MiniLM embedding extraction.
5. **Small OOD Sample Sizes:** `human_eval` ($N = 16$) and `aime_2020_2024` ($N = 13$) have wide Clopper-Pearson intervals (up to `[0.0000, 0.2059]` and `[0.0019, 0.3603]`), limiting statistical power.
6. **Calibration Degradation Under Shift:** OOD ECE rises above 0.20–0.40 on unseen code and reading tasks, making raw probability thresholds unreliable under domain shift.
7. **No Demonstrated Between-Router Superiority in APGR:** Overlapping 95% CIs among Hybrid, kNN, and LR mean the experiments do not establish statistically significant differences among those three routers.
8. **Model-Pair Dependence:** Two-model APGR results depend on the chosen pair (`cogito-v2-1-671b` vs. `llama-3-8b-instruct-lite`).
9. **No Global Production Budget Control or Production Readiness:** M4 evaluates per-query utility $\lambda$ offline; global/monthly budget pacing, multi-tenant rate limiting, and live SLA enforcement are not demonstrated here, so **production readiness is not claimed**.

---

## 17. Reproducibility

- **Environment:** Python $\ge 3.9$ (verified on Python 3.13.3 in `.venv`), dependencies in [pyproject.toml](file:///d:/llm%20cost%20copilot/pyproject.toml) (`pandas`, `pyarrow`, `sentence-transformers`, `datasets`, `scikit-learn`, `numpy`).
- **Offline Execution Commands (verified in M4.1–M4.5.2):**
  ```powershell
  $env:PYTHONPATH="."
  .venv\Scripts\python -m eval.m4_data_adapter
  .venv\Scripts\python -m eval.m4_router_evaluation
  .venv\Scripts\python -m eval.m4_calibration
  .venv\Scripts\python -m eval.m4_ood_evaluation
  .venv\Scripts\python -m eval.m4_routellm_baseline
  .venv\Scripts\pytest tests_v2/test_m4_routellm.py -v
  .venv\Scripts\pytest tests_v2
  ```
- **Recorded Regression Result (from M4.5.2 verification, not re-run in this documentation milestone):** **300 passed, 1 warning** across `tests_v2/` (including 13/13 in `tests_v2/test_m4_routellm.py`).
- **Artifact Locations:** All M4 outputs reside under [reports/m4/](file:///d:/llm%20cost%20copilot/reports/m4/) and `data/processed/xroutebench/`; M0–M3 artifacts remain untouched.

---

## 18. Conclusion and Future Work

Milestone M4 establishes an end-to-end, leakage-safe empirical benchmark for adaptive LLM routing on xRouteBench, demonstrating substantial cost-quality gains in-distribution, quantifying probability calibration and OOD domain boundaries, and verifying RouteLLM-style two-model gap recovery with null-calibrated bootstrap testing.

**Future Work (Not Yet Implemented):**
- Domain-adaptive and conformal OOD probability calibration.
- Online/shadow routing evaluation with live inference and latency/overhead profiling.
- Multi-user and global/monthly budget coordination controllers.
- Validation on commercially licensed, multi-turn, and tool-use enterprise workloads with continuous live cost and reliability monitoring.

---

## 19. References

1. Feng, T., Zheng, C., et al. (2026). *LLMRouter: A Multi-Task Benchmark and Framework for LLM Routing* (`ulab-ai/xRouteBench`). [arXiv:2608.06867](https://arxiv.org/abs/2608.06867).
2. Ong, I., Almahairi, A., Wu, V., Chiang, W.-L., Wu, T., Gonzalez, J. E., Kadous, M. W., & Stoica, I. (2024). *RouteLLM: Learning to Route LLMs with Preference Data*. LMSYS Org / UC Berkeley. [arXiv:2406.18665](https://arxiv.org/abs/2406.18665).
3. Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017). *On Calibration of Modern Neural Networks*. ICML 2017.
4. Platt, J. (1999). *Probabilistic Outputs for Support Vector Machines and Comparisons to Regularized Likelihood Methods*. Advances in Large Margin Classifiers.
5. Zadrozny, B., & Elkan, C. (2002). *Transforming Classifier Scores into Accurate Multiclass Probability Estimates*. KDD 2002.
6. Davison, A. C., & Hinkley, D. V. (1997). *Bootstrap Methods and Their Application*. Cambridge University Press.
7. Phipson, B., & Smyth, G. K. (2010). *Permutation P-values Should Never Be Zero: Calculating Exact P-values When Permutations Are Randomly Drawn*. Statistical Applications in Genetics and Molecular Biology, 9(1).
8. Hall, P., & Wilson, S. R. (1991). *Two Guidelines for Bootstrap Hypothesis Testing*. Biometrics, 47(2), 757–762.
9. Reimers, N., & Gurevych, I. (2019). *Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks*. EMNLP 2019 (`sentence-transformers`).

---

## 20. Appendix: Equations and Artifact Inventory

- **Key Equations:**
  - Reconstructed Cost: $\text{cost}_{\text{USD}} = (\text{input\_tokens} \cdot P_{\text{in}} + \text{output\_tokens} \cdot P_{\text{out}}) / 10^6$
  - Routing Utility: $U(m, x) = \hat{Q}(m, x) - \lambda \cdot c_{\text{norm}}(m, x)$
  - PGR & APGR: $\text{PGR}(M_R) = \frac{r(M_R) - r(M_{\text{weak}})}{r(M_{\text{strong}}) - r(M_{\text{weak}})}$, $\text{APGR}(M_R) = \int_0^1 \text{PGR}(M_{R_c})\,dc$
  - Null-Centered Bootstrap: $D_b = A_b^* - \hat{A}$, $A_{b,\text{null}} = 0.5000 + D_b$, $\hat{p}_{\text{DH}} = \frac{1 + \sum_{b=1}^B \mathbf{1}\{A_{b,\text{null}} \ge \hat{A}\}}{B + 1}$
- **Primary M4 Artifacts:**
  - Dataset Audit: [m4_dataset_audit.md](file:///d:/llm%20cost%20copilot/reports/m4/m4_dataset_audit.md), [m4_dataset_audit.json](file:///d:/llm%20cost%20copilot/reports/m4/m4_dataset_audit.json)
  - Multi-Model Routing: [m4_evaluation_summary.md](file:///d:/llm%20cost%20copilot/reports/m4/m4_evaluation_summary.md), [m4_methodology_audit.md](file:///d:/llm%20cost%20copilot/reports/m4/m4_methodology_audit.md), [m4_router_comparison.csv](file:///d:/llm%20cost%20copilot/reports/m4/m4_router_comparison.csv), [m4_cost_quality_results.csv](file:///d:/llm%20cost%20copilot/reports/m4/m4_cost_quality_results.csv), [m4_per_task_results.csv](file:///d:/llm%20cost%20copilot/reports/m4/m4_per_task_results.csv)
  - Calibration: [m4_calibration_summary.md](file:///d:/llm%20cost%20copilot/reports/m4/m4_calibration_summary.md), [m4_calibration_metrics.csv](file:///d:/llm%20cost%20copilot/reports/m4/m4_calibration_metrics.csv), [m4_calibration_task_breakdown.csv](file:///d:/llm%20cost%20copilot/reports/m4/m4_calibration_task_breakdown.csv), [m4_calibration_downstream_impact.csv](file:///d:/llm%20cost%20copilot/reports/m4/m4_calibration_downstream_impact.csv)
  - OOD Evaluation: [m4_ood_summary.md](file:///d:/llm%20cost%20copilot/reports/m4/m4_ood_summary.md), [m4_ood_methodology_audit.md](file:///d:/llm%20cost%20copilot/reports/m4/m4_ood_methodology_audit.md), [m4_ood_results.csv](file:///d:/llm%20cost%20copilot/reports/m4/m4_ood_results.csv), [m4_ood_calibration.csv](file:///d:/llm%20cost%20copilot/reports/m4/m4_ood_calibration.csv)
  - RouteLLM & Audits: [m4_routellm_summary.md](file:///d:/llm%20cost%20copilot/reports/m4/m4_routellm_summary.md), [m4_routellm_audit.md](file:///d:/llm%20cost%20copilot/reports/m4/m4_routellm_audit.md), [m4_routellm_apgr.csv](file:///d:/llm%20cost%20copilot/reports/m4/m4_routellm_apgr.csv), [m4_routellm_results.json](file:///d:/llm%20cost%20copilot/reports/m4/m4_routellm_results.json)
