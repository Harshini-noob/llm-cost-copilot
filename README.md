# LLM Cost Autopilot

**Built by Harshini A**

An adaptive LLM routing service that selects models by cost, quality, and latency. It combines a FastAPI API, Groq and Google Gemini providers, request-level budget tracking, Supabase telemetry, a Streamlit dashboard, and a Python client SDK.

> This is an engineering prototype, not an absolute provider-billing cap. Cost estimates and provider-reported usage may differ from actual charges.

## Features

- Classifies prompts as simple, medium, or complex with a deterministic rule-based classifier.
- Supports `economy`, `balanced`, `quality`, and `latency` routing modes.
- Filters model candidates using availability, context window, quality requirements, and estimated cost.
- Verifies answers with an LLM judge and can escalate to a stronger model when quality is low and budget permits.
- Accounts for generation, judge, retry, fallback, and escalation calls in the request total.
- Uses `Decimal` cost arithmetic and asynchronous budget reservations to coordinate concurrent calls sharing one request tracker.
- Persists request and per-call telemetry to Supabase; the dashboard shows spending, remaining budget, utilization, and latency.
- Includes a query console and a typed Python SDK.

## Architecture

```text
Python SDK / Query Console
          │  POST /query + API key
          ▼
       FastAPI ── authenticate and validate request
          │
          ├── Classify prompt and load model/quality metadata
          ├── Filter candidates by quality, context, availability, and budget
          ├── Rank candidates using the selected routing mode
          ▼
   Budget Tracker ── reserve and account for each provider attempt
          │
          ▼
   Groq / Google Gemini ── retry or fallback when configured
          │
          ├── LLM quality judge
          └── Optional budget-checked escalation
          │
          ▼
    Response + Supabase telemetry
          ├── Streamlit dashboard
          └── Historical quality feedback
```

## Routing modes

The router first filters models by availability, context-window fit, required quality, and (if provided) estimated request budget. The mode controls how the remaining candidates are ranked:

| Mode | Selection behavior |
| --- | --- |
| `economy` | Lowest estimated cost among eligible candidates |
| `quality` | Highest expected quality among eligible candidates |
| `latency` | Lowest configured average latency among eligible candidates |
| `balanced` (default) | Weighted score: 50% quality, 35% normalized cost, and 15% normalized latency |

If `min_quality` is omitted, the quality floor depends on prompt complexity. A small safety margin is also applied. These modes rank only eligible candidates; they do not override the quality, context, or budget filters.

## How budget tracking works

For each provider attempt, the service estimates input cost using its prompt-token heuristic and output cost using the model's configured maximum output tokens. It reserves this estimate before dispatch. Successful calls with valid usage metadata are reconciled to reported input/output token costs; failed calls and calls with missing usage retain the conservative estimate.

The budget applies per request, not across users or API keys. Actual charges can exceed estimates because tokenization, provider billing, failed-call charges, and reported usage are outside the application's control. When usage exceeds an estimate, the response reports `budget_exceeded`; this is not a guarantee of the final invoice. Requests without `max_cost_usd` are uncapped.

## Run locally

Requires Python 3.11 recommended, provider credentials, and a Supabase project with the application's base tables.

1. Install dependencies:

   ```bash
   python -m venv .venv
   # Windows PowerShell
   .\.venv\Scripts\Activate.ps1
   python -m pip install -r requirements.txt
   python -m pip install google-genai
   ```

2. Create a private `.env` file:

   ```dotenv
   GROQ_API_KEY=your_groq_key
   GEMINI_API_KEY=your_gemini_key
   SUPABASE_URL=https://your-project.supabase.co
   SUPABASE_KEY=your_server_side_supabase_key
   LLM_AUTOPILOT_API_KEY=your_client_key
   ```

   Client API keys must also exist and be active in Supabase's `api_keys` table. `LLM_AUTOPILOT_API_KEY` only pre-fills the query console; it does not provision a database key. Keep all credentials private.

3. Apply [`migrations/001_hard_budget_enforcement.sql`](migrations/001_hard_budget_enforcement.sql) to the existing Supabase schema. This migration adds budget and per-call fields; it does not create the base tables.

4. Start the API and optional UI tools:

   ```bash
   uvicorn main:app --reload --port 8000
   streamlit run query_console.py --server.port 8501
   streamlit run dashboard.py --server.port 8502
   ```

   The API documentation is available at `http://localhost:8000/docs`.

The Dockerfile builds the API service. Exclude `.env` and other secrets from the Docker build context (for example, with `.dockerignore`) before building.

## API

`POST /query` accepts query parameters and an `X-API-Key` header:

| Parameter | Description |
| --- | --- |
| `prompt` | Prompt to route |
| `routing_mode` | `balanced` (default), `economy`, `quality`, or `latency` |
| `max_cost_usd` | Optional positive per-request USD budget |
| `min_quality` | Optional expected-quality floor from 0.0 to 1.0 |

```bash
curl -X POST "http://localhost:8000/query?prompt=Explain%20LLM%20routing&routing_mode=balanced&max_cost_usd=0.01" -H "X-API-Key: YOUR_API_KEY"
```

The response includes the selected model and answer, `candidates_considered`, `rejected_candidates`, routing and quality status, per-call records, `estimated_cost_usd`, `actual_total_cost_usd`, `budget_accounted_cost_usd`, `remaining_budget_usd`, `budget_limit_usd`, and `budget_exceeded`. Invalid constraints return HTTP 400; no eligible affordable model returns HTTP 422.

## Python SDK

```python
from llm_cost_autopilot_client import LLMCostAutopilotClient

client = LLMCostAutopilotClient(
    api_key="YOUR_API_KEY",
    base_url="http://localhost:8000",
)
result = client.ask(
    "Explain LLM routing",
    routing_mode="balanced",
    max_cost_usd=0.01,
)

print(result.answer, result.model, result.total_cost_usd)

for candidate in client.compare_models("Explain LLM routing"):
    print(candidate.model, candidate.estimated_cost, candidate.expected_quality)
```

`ask()` returns a `RoutingResult` dataclass. `compare_models()` returns the eligible candidates supplied by the API. The SDK supports older responses by defaulting fields that newer API versions add.

## Testing

Run the mocked budget and SDK regression tests:

```bash
python -m unittest -v test_budget_enforcement
```

The suite covers affordability checks, cost reconciliation, concurrent reservations, cancellation, retries/fallbacks, judge and escalation accounting, validation, and SDK response compatibility. Some other `test_*.py` files are manual provider/database scripts and may make network calls; unrestricted `unittest discover` is not isolated.

## Project layout

```text
main.py                         FastAPI endpoint and request orchestration
routing_policy.py               Candidate filtering and routing decisions
router.py / model_registry.py   Complexity classification and model configuration
budget.py / models.py            Cost tracking, reservations, and provider dispatch
providers/                       Groq and Gemini adapters
dynamic_quality.py / logger.py   Supabase quality lookup and telemetry
dashboard.py / query_console.py  Streamlit dashboard and query UI
llm_cost_autopilot_client/       Python SDK
migrations/                      Supabase migrations
test_budget_enforcement.py       Mocked regression tests
```

## Current limitations

- Budget limits are per request, not shared quotas; provider charges cannot be guaranteed from estimates.
- API-key validation and telemetry persistence use synchronous Supabase calls.
- The repository's migration is additive but does not include the initial database schema.
- Provider/classifier evaluation scripts may make live calls and incur charges; their results depend on providers and current model pricing.
