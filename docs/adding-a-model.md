# Cold-Start Procedure: Adding a New Model to LLM Cost Copilot

This guide documents the exact, leakage-safe procedure for onboarding a new candidate LLM into the M5 runtime routing system without fabricating labels or breaking existing calibration.

> [!CAUTION]
> **Do not copy xRouteBench labels into the runtime calibration set** unless the exact provider, model ID, quantization/variant, system prompt, and evaluation semantics genuinely match. For example, Together's `gpt-oss-20b` in xRouteBench is not guaranteed to have identical latency, pricing, or behavior to Groq's `openai/gpt-oss-20b`.

---

## Step-by-Step Onboarding Workflow

### 1. Add the Provider and Model ID
- Verify the provider adapter exists in `providers/` (e.g., `groq_provider.py`, `gemini_provider.py`) and supports async `generate(prompt, model=..., max_tokens=...)` returning `answer`, `input_tokens`, `output_tokens`, and `latency_sec`.
- Confirm the exact upstream API model identifier (e.g., `openai/gpt-oss-20b`, `gemini-2.5-flash`).

### 2. Add Dated Input/Output Pricing to `config/models.yaml`
- Record the dated provider pricing (`pricing_last_verified`, `pricing_sources`, `input_price_per_1m`, `output_price_per_1m`, `context_window`, `enabled: true`) in `config/models.yaml`.

### 3. Register the Model in the Live Registry
- Add the model entry to `MODELS` in `model_registry.py` with `provider`, `model`, `input_cost_per_million`, `output_cost_per_million`, `context_window`, `avg_latency_sec`, and `available: True`.
- Confirm `get_model_by_id(model_id)` and `budget.token_cost(model_id, ...)` resolve cleanly.

### 4. Verify Provider Compatibility and Availability
- Run a single dry-run or mocked integration test verifying that `BudgetTracker` reserves worst-case cost, `call_model` dispatches to the registered provider, and `InvalidTokenUsageError` / `RateLimitError` / timeouts trigger ranked fallback cleanly.

### 5. Build a Calibration Prompt Set
- Select a stratified, leakage-free calibration prompt set (recommended minimum: $\ge 50$ prompts per validation gate, ideally 200+ across representative task categories) with fixed `prompt_id`, `prompt_text`, `task_category`, and deterministic `train` / `val` / `test` split assignments.
- Never share prompt IDs or normalized prompt texts across `train`, `val`, and `test`.

### 6. Evaluate the New Model Using Real Observed Outcomes
- Run the new model on the calibration prompts using authorized evaluation scripts (or import genuine observed runtime telemetry from `telemetry/decision_log.py` `outcome_feedback`).
- Record actual binary/continuous correctness labels $\in [0.0, 1.0]$ and actual token-based `cost_usd`. Do not manufacture labels for unscored or missing outcomes.

### 7. Generate Embeddings Locally
- Encode calibration prompts using the exact same cached embedding model used by the active router (`SentenceTransformerEncoder(model_name="all-MiniLM-L6-v2", local_files_only=True)`).
- Verify embedding dimension matches (`384`).

### 8. Train or Incrementally Update the Router
- **For kNN (`KNNRuntimeRouter`):**
  - Because kNN stores prompt embeddings and per-model outcome dictionaries independently, you can incorporate the new model **without recalibrating existing models** by calling `knn_runtime.add_model_outcomes(model_id, outcomes_by_prompt_id, expected_cost_usd)` on the existing training calibration prompt IDs, or by running `routers.lifecycle.retrain_router(..., router_type="knn")`.
- **For Logistic Regression (`LearnedLRRuntimeRouter`):**
  - `LearnedRouter` fits one L2-regularized `LogisticRegression` classifier per candidate model using shared `PromptFeatureExtractor` features (384-d embedding + z-scored length + code indicator + one-hot task category).
  - Adding a new model trains a new model-specific classifier on `(X_train, y_train_new_model)` via `routers.lifecycle.retrain_router(..., router_type="learned_lr")` and saves a new versioned bundle.

### 9. Validate New Model Coverage
- Check that the new bundle's `manifest.json` includes the new `model_id` and non-negative `model_costs[model_id]`.
- Verify `route(prompt, lam="auto:balanced", eligible_models=live_model_ids())` scores and ranks the new model without `None` predictions.

### 10. Shadow-Evaluate and Promote
- Run `routers.lifecycle.shadow_evaluate(candidate_router, active_router, val_prompts, labelled_outcomes=val_outcomes)` on $\ge 50$ held-out validation prompts.
- Promote via `routers.lifecycle.promote_router(candidate_version, manager, shadow_report=report, allow_pool_change=True)`.
- Promotion fails closed if the bundle is incomplete, the encoder mismatches, validation sample count $< 50$, or quality/cost gates fail. Keep the previous active version intact for instant `manager.rollback()`.
