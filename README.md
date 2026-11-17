# LLM Cost Autopilot

An adaptive, multi-provider LLM gateway that routes each prompt using estimated **quality, cost, and latency**, verifies generated answers with an LLM judge, and tracks request-level expenditure.

The project combines a FastAPI service, Groq and Google Gemini providers, a Supabase telemetry store, a Streamlit dashboard, and a Python client SDK.

> **Project status:** A feature-rich engineering prototype with budget-aware routing and per-request accounting. It is not an absolute provider-billing cap or a fully production-hardened gateway; see [Current limitations](#current-limitations).

## Why this project?

Different prompts have different reasoning requirements. Sending every request to the most capable model can waste tokens and increase latency. LLM Cost Autopilot estimates the requirements of a prompt, filters eligible models, and selects one according to the requested routing strategy.

It also records the complete request lifecycle—including generation, judge, retry, fallback, and escalation calls—so cost and quality can be inspected together.

## Features

### Adaptive model routing

- Supports Groq-hosted `openai/gpt-oss-20b` and `openai/gpt-oss-120b`, plus Google `gemini-2.5-flash`.
- Classifies prompts into simple, medium, and complex tiers using a deterministic rule-based classifier.
- Filters candidates by availability, context window, expected quality, and estimated cost.
- Supports four routing modes:
  - **balanced** — weighted quality, cost, and latency score.
  - **economy** — lowest estimated cost among eligible candidates.
  - **quality** — highest expected quality among eligible candidates.
  - **latency** — lowest configured average latency among eligible candidates.
- Uses live historical quality scores from Supabase when enough samples exist; otherwise falls back to registry quality scores.
- Keeps LLM-based and embedding-based classifiers as separate experimental alternatives; the production routing path currently uses the rule-based classifier.

The balanced score currently uses quality (0.50), normalized cost (0.35), and normalized latency (0.15). The quality floor is derived from the complexity tier or the caller's `min_quality`, with a safety margin.

### Quality verification and escalation

- Runs an LLM-as-a-Judge evaluation after generation, returning a score from 1–5.
- Uses a configurable pass threshold (currently 3/5).
- If verification fails, considers stronger eligible candidates and escalates only when the remaining request budget can accommodate the next call.
- Reports quality verification as unavailable when the judge cannot run or returns an unusable result.

### Per-request budget enforcement

- Accepts an optional `max_cost_usd` request limit.
- Uses `Decimal` for monetary calculations and keeps model pricing in the model registry.
- Estimates input-token cost and the configured maximum output-token cost before a provider call.
- Reserves estimated expenditure before dispatch; concurrent calls sharing one request tracker cannot each reserve the same remaining balance.
- Reconciles reservations against reported token usage after successful calls.
- Conservatively accounts for failed calls and calls with missing usage metadata using the pre-call estimate.
- Applies the same budget checks to generation, retries, fallbacks, judge calls, and escalations.
- Returns actual reported expenditure separately from conservative budget-accounted expenditure.

**Important:** budgets are per request, not shared monthly or account-wide quotas. Actual provider charges can exceed estimates; this is not an absolute billing guarantee. See [Budget model and limitations](#budget-model-and-limitations).

### Telemetry and dashboard

Supabase stores request-level totals and individual model-call records, including model, token usage, estimated/actual/accounted cost, latency, fallback/escalation flags, call type, success status, and error type.

The Streamlit dashboard currently provides:

- Request volume and spend summaries.
- Budget remaining and utilization for budgeted requests.
- Tier and model distributions.
- Cost contribution and cumulative accounted cost.
- Average provider-call latency.
- Fallback counts and recent-call details.

The dashboard reads the latest 500 model-call rows and caches the query for 10 seconds. Its latency metric is based on individual provider calls, not end-to-end request duration.

### API and client

- FastAPI `POST /query` endpoint protected by an `X-API-Key` header.
- Interactive API documentation at `/docs` when the server is running.
- Streamlit query console for trying routing modes and request constraints.
- Synchronous Python SDK returning typed dataclass results.

## Architecture

```text
Client / Query Console
        |
        | POST /query + X-API-Key
        v
   FastAPI Gateway
        |
        +--> Validate request constraints
        |
        +--> Rule-based complexity classification
        |
        +--> Read model metadata and recent quality scores
        |
        +--> Filter candidates by quality, context, and estimated cost
        |
        +--> Select model (balanced / economy / quality / latency)
        |
        v
   Per-request BudgetTracker
        |
        +--> Reserve estimated call cost
        |
        v
   Async Provider Dispatcher
        |
        +--> Groq
        |
        +--> Google Gemini
        |
        +--> One configured retry / fallback path where applicable
        |
        v
   LLM-as-a-Judge
        |
        +--> Pass: return answer
        |
        +--> Fail: check remaining budget, then optionally escalate
        |
        v
   Supabase telemetry
        |
        +--> Streamlit dashboard
        +--> Historical quality feedback for future routing
```

## Routing and model configuration

The model registry is the source of truth for provider mapping, model identifiers, configured prices, context windows, expected quality, availability, and average latency.

| Model | Provider | Registry quality tier | Default role |
|---|---|---:|---|
| `openai/gpt-oss-20b` | Groq | 0.75 | Lower-cost candidate |
| `openai/gpt-oss-120b` | Groq | 0.85 | Higher-quality candidate / judge |
| `gemini-2.5-flash` | Google Gemini | 0.90 | Higher-complexity candidate |

Default tier output limits are 250 tokens (simple), 600 (medium), and 1,200 (complex). A registry entry may define a lower per-tier output limit; the effective limit is used for routing estimates and provider calls.

Model prices and provider availability can change. Review `model_registry.py` before running cost comparisons or deploying.

## Budget model and limitations

For each provider attempt, the tracker estimates:

```text
estimated call cost =
    estimated input tokens × input price
  + configured maximum output tokens × output price
```

The estimate uses the current word-count heuristic (`1.3 tokens per whitespace-separated word`) for prompt input. It is not provider-tokenizer exact.

When usage metadata is available, the tracker calculates cost from reported input and output tokens. When a call fails or usage is missing, it retains the pre-call estimate as conservative accounted expenditure. Outstanding reservations are included when checking the remaining request budget.

The API distinguishes:

- `actual_total_cost_usd`: cost calculated from provider-reported token usage.
- `budget_accounted_cost_usd` / `total_cost_usd`: actual costs plus conservative estimates for calls whose usage is unknown or failed.
- `remaining_budget_usd`: request limit minus accounted expenditure and any outstanding reservations.

This design prevents the application from intentionally dispatching a call whose conservative estimate exceeds the currently available request budget. It cannot guarantee the final provider invoice because tokenization, hidden/provider-side usage, failed-request billing, reported usage, price changes, and provider enforcement may differ. If actual reported usage exceeds the reservation, the response marks `budget_exceeded` and further unaffordable calls are blocked.

When `max_cost_usd` is omitted, the request has no budget cap.

## API

### `POST /query`

The endpoint accepts query parameters:

| Parameter | Required | Description |
|---|---|---|
| `prompt` | Yes | User prompt |
| `routing_mode` | No | `balanced`, `economy`, `quality`, or `latency` |
| `max_cost_usd` | No | Positive per-request budget in USD |
| `min_quality` | No | Minimum expected quality from 0.0 to 1.0 |

Example:

```bash
curl -X POST "http://localhost:8000/query?prompt=Explain%20caching&routing_mode=balanced&max_cost_usd=0.01" \
  -H "X-API-Key: YOUR_API_KEY"
```

The response includes the answer, selected model, routing reason, candidate/rejection details, quality result, escalation status, individual `model_calls`, and cost fields such as:

- `estimated_cost_usd`
- `actual_total_cost_usd`
- `budget_accounted_cost_usd`
- `remaining_budget_usd`
- `budget_limit_usd`
- `budget_exceeded`
- `quality_verification_status`
- `escalation_attempted`
- `escalation_reason`

Invalid constraints return HTTP 400. If no eligible model fits the constraints/budget, the API returns HTTP 422. Provider failures can return HTTP 502.

## Tech stack

- **Backend:** Python, FastAPI, Uvicorn, AsyncIO
- **LLM providers:** Groq SDK, Google GenAI SDK
- **Routing and accounting:** custom Python policy engine, Decimal, asyncio synchronization
- **Database:** Supabase / PostgreSQL
- **Dashboard:** Streamlit, Pandas
- **Client:** Requests-based Python SDK
- **Packaging / deployment:** pip, Docker

Provider calls are asynchronous. The current authentication lookup and telemetry persistence use the synchronous Supabase client, so the whole request path should not be described as fully non-blocking.

## Repository layout

```text
.
├── main.py                         # FastAPI query endpoint and request orchestration
├── router.py                       # Complexity classifier and tier token limits
├── routing_policy.py               # Candidate filtering and model selection
├── model_registry.py               # Provider, pricing, quality, and context metadata
├── models.py                       # Provider dispatch, retry, and fallback
├── budget.py                       # Decimal budget tracker and reservations
├── cost_estimator.py               # Prompt token and request cost estimates
├── dynamic_quality.py              # Historical quality lookup from Supabase
├── providers/
│   ├── base.py
│   ├── groq_provider.py
│   └── gemini_provider.py
├── logger.py                       # Request and model-call telemetry persistence
├── dashboard.py                    # Streamlit telemetry dashboard
├── query_console.py                # Interactive routing console
├── llm_cost_autopilot_client/      # Python client SDK
├── migrations/
│   └── 001_hard_budget_enforcement.sql
├── test_budget_enforcement.py      # Mocked budget and SDK regression tests
├── compare_baseline.py             # Live-provider cost experiment
├── quality_check.py                # Live-provider judge experiment
└── classifier_accuracy.py          # Classifier evaluation script
```

## Setup

### 1. Prerequisites

- Python 3.11 recommended (the CI and Docker image use Python 3.11).
- A Supabase project with the baseline tables used by the application: `api_keys`, `requests`, `model_calls`, and `quality_scores`.
- API keys for Groq and Google Gemini.

The repository includes an additive hard-budget migration, but does not include a complete bootstrap schema for creating all baseline Supabase tables from scratch.

### 2. Configure environment variables

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key
GEMINI_API_KEY=your_gemini_api_key
SUPABASE_URL=your_supabase_url
SUPABASE_KEY=your_server_side_supabase_key
LLM_AUTOPILOT_API_KEY=your_client_api_key
```

Keep this file private. Do not expose server-side Supabase credentials or provider keys in a browser/client application.

### 3. Install dependencies

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
pip install google-genai
```

The current `requirements.txt` does not yet list `google-genai`, although the Gemini provider imports it. Install it explicitly as shown above. The embedding-classifier experiment additionally requires `sentence-transformers`.

The current `pyproject.toml` describes the client package dependencies rather than the complete backend service environment; `uv sync` alone is therefore not the backend installation path.

### 4. Apply the database migration

Apply `migrations/001_hard_budget_enforcement.sql` to the existing Supabase database after the baseline tables have been created. The migration adds budget and per-call accounting columns; it does not create the original tables.

### 5. Start the API

```bash
uvicorn main:app --reload --port 8000
```

Open [http://localhost:8000/docs](http://localhost:8000/docs) for the interactive API documentation.

### 6. Start the Streamlit applications

In separate terminals:

```bash
streamlit run query_console.py --server.port 8501
streamlit run dashboard.py --server.port 8502
```

## Python SDK

The SDK is available from the repository package. It exposes `LLMCostAutopilotClient` and returns a `RoutingResult` dataclass.

```python
from llm_cost_autopilot_client import LLMCostAutopilotClient

client = LLMCostAutopilotClient(
    api_key="YOUR_API_KEY",
    base_url="http://localhost:8000",
)

result = client.ask(
    "Explain quantum entanglement in simple terms",
    routing_mode="economy",
    max_cost_usd=0.001,
)

print("Answer:", result.answer)
print("Model:", result.model)
print("Routing reason:", result.routing_reason)
print("Accounted cost:", result.total_cost_usd)
print("Actual cost:", result.actual_total_cost_usd)
print("Quality:", result.quality_score)
```

## Tests and experiments

### Mocked regression tests

The focused budget-enforcement suite uses mocked provider responses:

```bash
python -m unittest -v test_budget_enforcement
```

It covers budget preflight, exact-budget eligibility, actual-versus-estimated usage, concurrent reservations, retry/fallback accounting, judge expenditure, escalation limits, cancellation, invalid usage, and SDK compatibility.

**Test-suite note:** Several older files named `test_*.py` are manual smoke scripts that execute provider or Supabase operations at module import time. Do not run unrestricted `unittest discover` until those scripts are moved/renamed and the suite is separated into safe automated tests.

The current GitHub Actions workflow runs a focused Flake8 error check and builds the Docker image; it does not yet execute the mocked unit suite.

### Live-provider experiments

These scripts make real provider calls and may incur API charges:

```bash
python classifier_accuracy.py
python compare_baseline.py
python quality_check.py
```

They use the prompt set in `test_prompts.py`. Results depend on provider availability, model versions, current registry prices, rate limits, and generated token usage. Treat previously observed savings/quality numbers as experiment results, not guaranteed performance or CI-enforced benchmarks. Re-run experiments and retain their outputs before publishing numerical claims.

## Current limitations

The following are not implemented or not yet validated as production capabilities:

- **No circuit breaker:** the dispatcher has a fixed fallback map and a bounded retry path; it does not implement closed/open/half-open circuit-breaker states or provider health probes.
- **Per-request, not global, budgets:** reservations coordinate calls within one request tracker. They do not enforce shared user, API-key, daily, or monthly quotas across requests/processes.
- **Provider billing uncertainty:** estimates cannot guarantee the final invoice, especially for failed requests or incomplete usage metadata.
- **Synchronous Supabase paths:** API-key validation and request telemetry use synchronous database calls. Telemetry inserts are sequential rather than transactional, so partial persistence is possible if a later insert fails.
- **Limited tracing and latency analytics:** there is no end-to-end correlation trace, request-duration percentile dashboard (P50/P95/P99), or full provider health history yet.
- **Dashboard scope:** the dashboard queries a limited recent window (up to 500 model calls); its latency chart is per-call average, not end-to-end request latency.
- **Schema bootstrap:** the repository contains the budget migration, not the complete initial database schema.
- **Packaging cleanup:** backend dependencies should be consolidated into the service dependency manifest; in particular, add `google-genai` to `requirements.txt` before relying on the current Docker image.

## Potential next improvements

1. Add a provider circuit breaker and health-aware failover.
2. Add request/call correlation IDs and structured traces.
3. Add P50/P95/P99 latency and provider reliability analytics.
4. Move Supabase auth and telemetry paths to async clients and make lifecycle persistence resilient/transactional.
5. Separate deterministic unit tests from live-provider smoke scripts and run the safe suite in CI.
6. Add shared per-user/API-key quotas if multi-tenant usage is required.

---

**LLM Cost Autopilot** explores how routing policy, live quality feedback, budget accounting, and evaluation can work together to make multi-provider LLM applications more cost-aware and observable.
