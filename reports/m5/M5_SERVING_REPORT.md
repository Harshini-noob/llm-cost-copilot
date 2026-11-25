# M5 — Production Serving, Runtime Routing & Milestone Closure Report

**Project:** LLM Cost Copilot (`Harshini-noob/llm-cost-copilot`)  
**Milestone:** M5 — Serve It (Runtime Routing, OpenAI-Compatible API, Ranked Fallback, Logging & Lifecycle)  
**Date:** November 25, 2026  
**Branch:** `feat/m5-runtime-routing`  
**Spending Policy:** Zero live LLM API calls; zero fabricated labels; all integration tests use mocked providers and cached local MiniLM weights.

---

## 1. Architecture Overview

Milestone M5 bridges the M3/M4 research routers (`eval/routers/`) into a production-structured serving layer without substituting keyword-based heuristics:

```text
Client (OpenAI SDK / HTTP)
   │  POST /v1/chat/completions (model="auto" | "auto:quality" | "auto:balanced" | "auto:cheap" | explicit)
   ▼
serving/chat_completions.py
   ├── Convert messages -> prompt
   ├── RuntimeRouter.route(prompt, lam, eligible_models=live_model_ids())
   │      ├── KNNRuntimeRouter (routers/knn_runtime.py)  [manifest.json + knn_state.npz]
   │      └── LearnedLRRuntimeRouter (routers/lr_runtime.py) [manifest.json + lr_state.pkl]
   │      └── SentenceTransformerEncoder(local_files_only=True) [cached 384-d MiniLM]
   ├── Ranked Fallback Loop + BudgetTracker(max_cost_usd)
   │      └── Try ranked_candidates in order; catch timeout/rate-limit/API errors; propagate CancelledError
   └── telemetry/decision_log.py (SQLite)
          ├── routing_decisions (prompt_hash, derived features, embedding, scores, costs, latency, fallbacks)
          └── outcome_feedback (judge_score | user_feedback | downstream_task_success via POST /v1/feedback)
```

Legacy `POST /query` in `main.py` remains untouched and backwards-compatible.

---

## 2. Implemented Components

1. **Unified Runtime Router (`routers/base.py`, `routers/knn_runtime.py`, `routers/lr_runtime.py`):**
   - Exposes `route(prompt, lam) -> RouteDecision` returning `chosen_model`, `predicted_scores`, `expected_costs_usd`, `normalized_costs`, `utility_scores`, `ranked_candidates`, `router_version`, `lambda_value`, `router_latency_ms`.
   - Versioned, serializable bundles (`save_bundle` / `load_bundle`) with encoder dimension validation and incremental kNN new-model addition (`add_model_outcomes`).
2. **Lambda Presets:**
   - `auto` / `auto:balanced` $\rightarrow \lambda = 0.5$ (M4.2 xRouteBench: `0.6664` Macro, `\$0.040/1k`; provisional for live pool).
   - `auto:quality` $\rightarrow \lambda = 0.0$ (M4.2 xRouteBench: `0.7208` Macro, `\$0.283/1k`).
   - `auto:cheap` $\rightarrow \lambda = 2.0$ (M4.2 xRouteBench: `0.6508` Macro, `\$0.023/1k`; provisional for live pool).
3. **OpenAI-Compatible API & Ranked Fallback (`main.py`, `serving/chat_completions.py`):**
   - `POST /v1/chat/completions` validates live models in `model_registry.MODELS`, enforces `BudgetTracker` per attempt without double-counting failures, separates `expected_cost_usd`, `actual_cost_usd` (`None` if estimated), and `budget_accounted_cost_usd`, and reports `fallback_attempts` transparently.
   - `POST /v1/feedback` records `judge_score` (`1–5` $\rightarrow [0,1]$), `user_feedback`, and `downstream_task_success` without manufacturing labels for missing outcomes.
4. **Retraining, Shadow Evaluation & Atomic Promotion (`routers/lifecycle.py`, `routers/registry.py`):**
   - Strict train/val/test prompt ID and normalized text leakage checks (`LeakageDetectedError`).
   - `shadow_evaluate` computes routing agreement, selection mix, predicted quality/cost distributions, p50/p95 overhead, and (when independent labels exist) actual validation quality/cost.
   - `promote_router` fails closed on incomplete bundles, encoder/pool mismatches, $<50$ validation prompts, or quality/cost gate failures, updating `active_router.json` atomically with `rollback()` support.
5. **Cold-Start Guide (`docs/adding-a-model.md`) & 50-Prompt Demo (`scripts/demo_m5.py`).**

---

## 3. Metrics & Validation Results (Measured vs. Unavailable)

> [!IMPORTANT]
> **Live Provider Pool Calibration Gap:** The live `model_registry.py` pool (`openai/gpt-oss-20b`, `openai/gpt-oss-120b`, `gemini-2.5-flash`) does **not** currently have a genuine held-out calibration dataset (`data/processed/matrix.parquet` contains only 25 rows across 2 math prompts for V2 IDs). When no active bundle is loaded, `/v1/chat/completions` with `model="auto*"` returns HTTP `503` rather than fabricating scores.

| Metric | Live Provider Pool Status | Offline xRouteBench Measurement (M4 / M5 Demo) | Source |
| :--- | :--- | :--- | :--- |
| **Quality (Held-Out)** | **UNAVAILABLE** (no live calibration set) | `0.7208` Macro ($\lambda=0.00$), `0.6664` ($\lambda=0.50$) across 3,729 prompts; `0.7600` on 50-prompt demo (`auto:balanced`) | `m4_router_comparison.csv`, `scripts/demo_m5.py` |
| **Cost (Reconstructed)** | **UNAVAILABLE** (live); mocked in tests | `\$0.283/1k` ($\lambda=0.00$), `\$0.040/1k` ($\lambda=0.50$); `\$0.0349/1k` on 50-prompt demo | `m4_router_comparison.csv`, `scripts/demo_m5.py` |
| **Cost Savings at Parity** | **UNAVAILABLE** (live) | **92.23%** at parity (`0.6533` vs `0.6536` Best Single Model, $\lambda=0.75$); **91.70%** vs val-best `cogito-v2-1-671b` on 50-prompt demo (`-0.1000` quality delta) | `m4_methodology_audit.md`, `scripts/demo_m5.py` |
| **Quality at Parity vs Static** | **UNAVAILABLE** (live) | `Static-Task-Table` achieves `0.7321` Macro at `\$0.315/1k`; `LEARNED_LR` ($\lambda=0.00$) achieves `0.7208` at `\$0.283/1k` | `m4_router_comparison.csv` |
| **Oracle Gap Closed** | **UNAVAILABLE** (live) | Oracle (`\lambda=0.00`): `0.8533` Macro at `\$0.032/1k`; `LEARNED_LR` ($\lambda=0.00$) closes ~51% of `Always-Cheapest` (`0.5806`) $\rightarrow$ `Oracle` (`0.8533`) gap | `m4_router_comparison.csv` |
| **APGR (Two-Model)** | **UNAVAILABLE** (live) | `0.5906` Micro / `0.5904` Macro (Hybrid Ensemble, `cogito-v2-1-671b` vs `llama-3-8b-instruct-lite`, $p = 1/1001 \approx 0.0010$) | `m4_routellm_apgr.csv` |
| **OOD Quality (LOTO)** | **UNAVAILABLE** (live) | 3–3 split across 6 held-out tasks (`LR` leads `gsm8k`, `mmlu`, `aime_2020_2024`; `kNN` leads `mbpp`, `human_eval`, `squad`) | `m4_ood_results.csv` |
| **Calibration (ECE)** | **UNAVAILABLE** (live) | Raw LR `0.0389` ECE (Platt `0.0283`); Raw kNN `0.1123` (Isotonic `0.0490`) | `m4_calibration_metrics.csv` |
| **Router Overhead** | **Measured (offline CPU, 50-prompt demo)** | **p50 = 43.20 ms**, **p95 = 82.61 ms** (with live local MiniLM embedding) | `scripts/demo_m5.py` |
| **Routing Mix** | **Measured (50-prompt demo)** | `gpt-oss-20b`: 18, `gpt-oss-120b`: 13, `rnj-1-instruct`: 8, `mistral-small-3-24b-instruct`: 5, `qwen2.5-7b-instruct`: 3, `gemma-2-9b-it`: 3 | `scripts/demo_m5.py` |

---

## 4. Test Results & Deployment Status

- **Targeted M5 Tests (`tests_v2/test_m5_serving.py`):** **9 passed, 1 warning in 8.32s** (mocked providers, `DeterministicTestEncoder`, zero API spend).
- **Deployment Status:** **Engineering Prototype — Not Production-Calibrated for Live Pool.** All runtime serving, fallback, budget, logging, retraining, shadow-evaluation, and atomic promotion/rollback mechanics are implemented and tested. Production `auto*` routing on the live provider pool requires collecting genuine calibration outcomes as documented in `docs/adding-a-model.md`.
