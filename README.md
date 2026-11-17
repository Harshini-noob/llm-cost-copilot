# LLM Cost Autopilot

<<<<<<< HEAD
An adaptive, multi-provider LLM gateway that routes each prompt using estimated **quality, cost, and latency**, verifies generated answers with an LLM judge, and tracks request-level expenditure.

The project combines a FastAPI service, Groq and Google Gemini providers, a Supabase telemetry store, a Streamlit dashboard, and a Python client SDK.
=======
**An adaptive LLM routing service that balances cost, quality, and latency—with request-level budget tracking, quality checks, and multi-provider fallback.**

**Built by Harshini A**

LLM Cost Autopilot is a Python project for choosing an appropriate language model for each request instead of sending every prompt to the same model. It exposes a FastAPI endpoint, records request and per-call telemetry in Supabase, provides a Streamlit dashboard and query console, and includes a small Python client SDK.

The project is designed to make model selection and spending more observable and controllable. Its budget checks use estimates and provider-reported usage; they are not a guarantee of the final amount a provider will bill.
>>>>>>> 94437ea (Final README)

> **Project status:** A feature-rich engineering prototype with budget-aware routing and per-request accounting. It is not an absolute provider-billing cap or a fully production-hardened gateway; see [Current limitations](#current-limitations).

<<<<<<< HEAD
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
=======
## What this project does

- **Routes by request complexity.** A deterministic classifier assigns prompts to simple, medium, or complex tiers.
- **Chooses among configured providers and models.** The current registry includes Groq-hosted GPT OSS models and Google Gemini Flash.
- **Supports four routing priorities:** `economy`, `balanced`, `quality`, and `latency`.
- **Applies request constraints.** Callers may set a maximum estimated request budget and a minimum expected quality.
- **Checks answer quality.** An LLM judge can score a generated answer from 1 to 5; a low score can trigger an affordable escalation to a stronger candidate.
- **Handles provider failures.** Configured retry and fallback paths are accounted for as separate provider attempts.
- **Tracks usage and spend.** It records reported token usage, actual and estimated call costs, quality outcomes, fallback and escalation events, and request-level budget totals.
- **Includes user-facing tools.** A Python SDK, an interactive query console, and a telemetry dashboard make the service easier to exercise and inspect.

## Routing strategy

Each request is classified into a complexity tier. Eligible models are filtered by availability, context-window fit, quality requirements, and (when supplied) estimated cost. The selected routing mode determines how eligible candidates are ranked.

| Mode | Selection priority |
| --- | --- |
| `economy` | Lowest estimated cost among candidates that meet the quality requirement |
| `quality` | Highest expected quality among eligible candidates |
| `latency` | Lowest configured average latency among eligible candidates |
| `balanced` | Weighted score across quality, normalized cost, and normalized latency |

The balanced score currently weights quality at 50%, cost at 35%, and latency at 15%. Candidate prices, context windows, static quality estimates, and average latency are maintained in [`model_registry.py`](model_registry.py). Live quality scores can be derived from recent Supabase evaluations after the configured minimum sample count is available; otherwise the registry's static score is used.

## Request lifecycle

```text
Client
  │  POST /query + X-API-Key
  ▼
FastAPI ── validate routing options
  │
  ├── classify prompt and estimate candidate costs
  ├── filter and rank eligible models
  ├── reserve budget and call selected provider
  ├── account for retries/fallback attempts
  ├── run a budget-checked quality judge
  ├── optionally escalate if quality is low and budget permits
  └── persist request and call telemetry to Supabase
         ├── Streamlit dashboard
         └── Streamlit query console
```

The asynchronous provider dispatchers are in [`providers/`](providers/); request orchestration is in [`main.py`](main.py), routing decisions are in [`routing_policy.py`](routing_policy.py), and call-level budgeting is in [`budget.py`](budget.py) and [`models.py`](models.py).

## Budget tracking

When `max_cost_usd` is provided, the service estimates a model call's worst-case cost using an input-token estimate, the model's configured input/output prices, and that candidate's effective output-token limit. The limit is resolved per model and tier, so an escalation candidate does not inherit the primary model's output limit.

Before dispatch, a `BudgetTracker` reserves the estimate. Reservations are coordinated with an asynchronous lock and must be claimed before they can authorize a provider call; a reservation cannot authorize two concurrent calls. On a successful response with valid usage metadata, the reservation is reconciled to the cost calculated from the reported input and output tokens. Failed calls and responses with missing usage metadata retain the conservative estimate. Each retry, fallback, quality-judge call, and escalation is accounted against the same request tracker.

If a judge call cannot be afforded, quality verification is marked unavailable and the service does not treat the answer as verified. If no model is affordable, the API rejects the request rather than silently selecting a model outside the budget.

### Important budget limitations

Budget enforcement is a best-effort control based on available estimates and provider responses—not an absolute billing cap:

- Input tokens are estimated using the project's prompt-token heuristic; provider tokenization may differ.
- Provider pricing and billing rules can change.
- A provider may charge for a failed or interrupted request, even when usage details are unavailable.
- A provider may report incomplete or inaccurate usage, or actual cost may exceed the preflight estimate. In that case the request can be recorded as over budget after the call has completed.
- The budget is per request and per tracker; it is not a shared account-wide spending limit.
- Requests without `max_cost_usd` do not have a budget cap.

Treat `actual_total_cost_usd` as the cost calculated from reported token usage, and `budget_accounted_cost_usd` / `total_cost_usd` as the request's accounted amount, including conservative estimates for calls without known usage.

## API

The service exposes `POST /query`. The API key is sent in the `X-API-Key` header. The query options are URL parameters:

| Parameter | Required | Description |
| --- | --- | --- |
| `prompt` | Yes | Text to send to the selected model |
| `routing_mode` | No | `balanced` (default), `economy`, `quality`, or `latency` |
| `max_cost_usd` | No | Positive finite USD budget for this request |
| `min_quality` | No | Minimum expected quality from `0.0` to `1.0` |

Example:

```bash
curl -X POST "http://localhost:8000/query?prompt=Explain%20how%20LLM%20routing%20works&routing_mode=balanced&max_cost_usd=0.01" -H "X-API-Key: your_api_key"
```

The response includes the answer and chosen model, routing rationale, rejected-candidate details, quality and escalation status, per-call records, and budget fields such as:

- `estimated_cost_usd`
- `actual_total_cost_usd`
- `budget_accounted_cost_usd`
- `remaining_budget_usd`
- `budget_limit_usd`
- `budget_exceeded`
- `quality_verification_status`
- `escalation_attempted` and `escalation_reason`
- `model_calls`

Invalid constraints return HTTP 400; a request with no eligible affordable model returns HTTP 422. Provider failures are reported as HTTP 502.

## Python client SDK

The repository includes [`llm_cost_autopilot_client`](llm_cost_autopilot_client/). Its `LLMCostAutopilotClient.ask()` method returns a `RoutingResult` dataclass. The SDK keeps older response fields usable and defaults newer budget/quality fields when talking to an older API response.

```python
from llm_cost_autopilot_client import LLMCostAutopilotClient

client = LLMCostAutopilotClient(
    api_key="your_api_key",
    base_url="http://localhost:8000",
)

result = client.ask(
    "Explain how LLM routing works",
    routing_mode="balanced",
    max_cost_usd=0.01,
)

print(result.answer)
print("Model:", result.model)
print("Accounted cost:", result.total_cost_usd)
print("Actual reported-usage cost:", result.actual_total_cost_usd)
print("Remaining budget:", result.remaining_budget_usd)
```

The SDK also defines a `compare_models()` convenience method. The current API response does not populate `candidates_considered`, so this method may return an empty list; use `ask()` for the primary SDK workflow. See [`client.py`](llm_cost_autopilot_client/client.py) and [`models.py`](llm_cost_autopilot_client/models.py) for the supported fields and methods.

## Run locally

### 1. Requirements

- Python 3.9 or newer (the included Docker image uses Python 3.11)
- API credentials for the providers you intend to use
- A Supabase project with the tables expected by the application

### 2. Install dependencies

From the repository root, create and activate a virtual environment, then install the application requirements:

```bash
python -m venv .venv

# Windows PowerShell
.\.venv\Scripts\Activate.ps1

# Install dependencies
python -m pip install -r requirements.txt
```

The project also has a `pyproject.toml` for the Python client package. To install the local package in editable mode:

```bash
python -m pip install -e .
```

### 3. Configure environment variables

Create a local `.env` file in the project root. Do not commit credentials.

```dotenv
GROQ_API_KEY=replace_with_groq_key
GEMINI_API_KEY=replace_with_gemini_key
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_KEY=replace_with_supabase_key
LLM_AUTOPILOT_API_KEY=replace_with_client_api_key
```

The API authenticates client requests against active keys in the Supabase `api_keys` table; setting `LLM_AUTOPILOT_API_KEY` does not create or activate a key in that table. The query console uses that variable to prefill its API-key field. `SUPABASE_KEY` is used by this application for its database operations; keep it private and use an appropriately protected server-side key.

### 4. Prepare Supabase

The application expects existing `api_keys`, `requests`, `model_calls`, and `quality_scores` tables with the columns used by the code. The budget-enforcement migration is additive; it does not create the base schema. Apply [`migrations/001_hard_budget_enforcement.sql`](migrations/001_hard_budget_enforcement.sql) to add request-level budget/status fields and per-model-call accounting fields. Review the migration and back up your database according to your deployment process before applying it.

### 5. Start the API

```bash
uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

### 6. Start the optional Streamlit tools

In separate terminals:

```bash
streamlit run query_console.py --server.port 8501
```

```bash
streamlit run dashboard.py --server.port 8502
```

The query console sends requests to `http://localhost:8000/query`. The dashboard reads request and model-call telemetry from Supabase; the migration must be applied before using the new budget metrics.

## Docker

The included [`Dockerfile`](Dockerfile) builds the API service:

```bash
docker build -t llm-cost-autopilot .
docker run --env-file .env -p 8000:8000 llm-cost-autopilot
```

The current Dockerfile copies the build context into the image. Ensure `.env` and all credential files are excluded from that context before building; do not rely on `.gitignore` to protect Docker builds. Provide the required environment variables and a prepared Supabase schema to the container through your deployment's secret-management mechanism. The Docker image runs the API; the Streamlit dashboard and query console are separate processes.

For example, create a `.dockerignore` file before building:

```text
.env
.env.*
.git
.venv
__pycache__
*.py[cod]
```

## Telemetry and dashboard

The service persists one request record and individual model-call records. Call records include generation, judge, retry, fallback, and escalation attempts, with usage and cost-estimation status where available. The Streamlit dashboard presents actual reported-usage spend, remaining budget, utilization, latency, tier distribution, and fallback activity.

Existing records are preserved by the additive migration. Historical actual totals are backfilled where prior call records contain non-estimated cost values; values that cannot be reconstructed from old telemetry cannot be recovered retroactively.

## Testing and evaluation

The focused budget and SDK regression suite uses mocked provider responses and does not make real provider API calls:
>>>>>>> 94437ea (Final README)

```bash
python -m unittest -v test_budget_enforcement
```

<<<<<<< HEAD
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
=======
It covers budget preflight and accounting, concurrent reservations, retries/fallbacks, judge and escalation spend, invalid budgets and usage, SDK import/construction, and legacy/new API response deserialization.

The repository also contains exploratory integration and evaluation scripts for providers, classification, quality comparison, and cost comparison. Some execute API/database calls at import or run time and may require valid credentials, network access, and configured services; inspect a script before running it. Do not assume `python -m unittest discover` is safe as a fully isolated test suite.

## Engineering decisions and learnings

- Kept a deterministic, rule-based classifier as the production routing default to avoid classifier API latency, cost, and output-format variability.
- Normalized cost and latency before combining them with quality in balanced routing, so small dollar values do not dominate the score merely because of their scale.
- Kept model prices and token limits in shared configuration rather than duplicating prices across routing and accounting logic.
- Used `Decimal` for cost arithmetic and per-request async reservations to reduce rounding errors and avoid concurrent calls spending the same estimated headroom.
- Kept telemetry additive so existing request and model-call records can continue to be read.
- Treats missing usage conservatively rather than recording unknown-cost calls as free.

## Project layout

```text
.
├── main.py                         # FastAPI endpoint and request orchestration
├── routing_policy.py               # Candidate filtering, constraints, and selection
├── router.py                       # Complexity classifier and tier token limits
├── model_registry.py               # Configured models, providers, prices, and metadata
├── budget.py                       # Decimal cost accounting and reservations
├── models.py                       # Provider dispatch, retries, and fallback
├── providers/                      # Groq and Gemini provider adapters
├── dynamic_quality.py              # Supabase-backed live quality lookup
├── logger.py                       # Request and call telemetry persistence
├── dashboard.py                    # Streamlit telemetry dashboard
├── query_console.py                 # Streamlit API client
├── llm_cost_autopilot_client/       # Python SDK and response models
├── migrations/                      # Additive Supabase migrations
└── test_budget_enforcement.py       # Mocked budget and SDK regression tests
```

## Current scope

This is a per-request routing and cost-observability project. It does not provide account-wide budget coordination, provider-billing reconciliation, guaranteed service availability, or a provider-independent absolute spend cap. Production deployments should monitor provider invoices, secure API and database credentials, configure provider limits where available, and validate the database schema and migration in their own Supabase environment.
>>>>>>> 94437ea (Final README)
