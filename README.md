# 🚀 LLM Cost Autopilot

An adaptive LLM inference router that classifies incoming queries, estimates cost/quality/latency across candidate models *before* generating a response, applies user-defined constraints, and picks the best eligible model — with real post-generation quality verification, budget-aware escalation, a relational data layer, live-computed quality scores, API authentication, and containerized deployment with CI.

Built solo as a personal engineering project to go deep into production LLM infrastructure patterns — routing policy, evaluation, resilience, explainability, persistence, and deployability — beyond simple API-calling projects.

---

## The Problem

Companies running LLM-powered products routinely send every query — trivial or complex — to the same expensive, top-tier model. A factual lookup costs the same as a deep multi-step reasoning task, even though a much cheaper model could answer the simple one just as well. At scale, this is real, avoidable spend. This is also a real, funded product category (OpenRouter, Not Diamond, Martian, Portkey, LiteLLM all solve exactly this problem commercially).

**The goal:** build a system that automatically decides, per-query, which model to use — under real cost/quality/latency/budget constraints — prove with real measurements that it saves money without degrading quality, and let the system's own performance assumptions improve automatically as it accumulates real usage data.

---

## Architecture

```
User Request  ──[ X-API-Key header ]──▶  Auth Check (Supabase-backed)
     │
     ▼ (rejected here if invalid — no cost incurred)
Validate Constraints ── reject invalid routing_mode / budget / quality bounds
     │
     ▼
Complexity Classifier ── LLM-based: simple / medium / complex
     │
     ▼
Candidate Models (from Model Registry, quality scores fetched live per request)
     │
     ▼
For each candidate, check:
  • availability          • context window
  • quality ≥ floor + safety margin   (live-computed if enough data, else hand-set fallback)
  • estimated cost ≤ budget (pre-call estimate, no API call)
     │
     ▼
Routing Policy ── economy / balanced / quality / latency
     │
     ▼
Select Best Eligible Model ── with full "rejected candidates + reasons" trail
     │
     ▼
Generate Response
     │
     ▼
Quality Verification ── independent LLM-judge scoring (not self-reported confidence)
     │
     ├── Pass → Return
     │
     └── Fail → Check stronger candidates → Check remaining budget
                  → Escalate if allowed → Re-verify final response → Return
     │
     ▼
Full Telemetry Persisted to Supabase (Postgres)
requests → model_calls → quality_scores  (relationally linked)
     │
     ▼
Feeds back into live quality scoring for future requests
```

**Resilience layer:** if a model hits a rate limit mid-request, the system automatically falls back to another available model — with a separate `allow_fallback=False` mode reserved for evaluation scripts, so measurement integrity is never silently compromised by an unplanned model swap.

**Deployability layer:** the entire system is containerized (Docker) and CI-checked (GitHub Actions) — every push automatically verifies the app lints cleanly and the Docker image still builds.

---

## Tech Stack

- **Backend:** FastAPI (Python)
- **LLM Provider:** Groq (free tier) — `llama-3.1-8b-instant`, `openai/gpt-oss-120b`, `llama-3.3-70b-versatile`
- **Database & Auth:** Supabase (hosted PostgreSQL) — relational telemetry + API key store
- **UI:** Two Streamlit apps — historical analytics dashboard + live interactive query console
- **DevOps:** Docker (containerized deployment) + GitHub Actions (CI: lint + build verification on every push)
- **Evaluation:** LLM-as-judge methodology, custom labeled benchmark set

---

## Authentication

Every request to `/query` requires a valid `X-API-Key` header, checked against a `api_keys` table in Supabase *before* any routing, generation, or database write occurs — invalid requests cost nothing. Keys are self-issued (`generate_key.py`) and self-registered; there is no shared master key, and none is distributed with the code. See `.env.example` for what a new user needs to supply to run this themselves (their own Groq key, their own Supabase project, their own generated API key).

This is intentionally API-key auth, not full multi-user login — appropriately scoped for a single-operator backend service rather than a many-user consumer product. (Multi-user JWT-based auth via Supabase Auth is a natural next step if this ever needed to serve multiple independent users with individually tracked budgets — noted as a future direction, not built prematurely.)

---

## Data Layer

Telemetry is stored in three relationally linked Postgres tables (Supabase), not a flat log file:

- **`requests`** — one row per incoming query (prompt, tier, routing mode, constraints)
- **`model_calls`** — one row per actual model invocation (a request can have 2+ rows if escalation occurs), linked via `request_id`
- **`quality_scores`** — one row per quality verification, linked via `model_call_id`
- **`api_keys`** — issued keys, checked on every request

This structure exists because a single request can trigger multiple model attempts (original + escalation), each independently scored — a flat log can't represent that relationship cleanly, while linked tables can.

---

## Dynamic (Live) Quality Scoring

Instead of a permanently hardcoded `quality_tier` per model, `build_model_metadata()` queries Supabase on every routing decision:

- Fewer than 5 real verified quality scores for a model → honestly falls back to its hand-set registry estimate (`source: static_fallback`)
- 5+ scores → quality becomes a **live rolling average of real judge-verified outcomes** (`source: live_data`)

**Observed in testing:** after 7 real queries, `openai/gpt-oss-120b`'s live-computed quality (0.943) had already diverged meaningfully from its hand-set estimate (0.85) — the system discovered, from real usage, that this model was performing better than initially assumed, with no manual retuning.

**Known tradeoff:** querying live scores on every request adds a small amount of latency per request compared to a cached/fixed dict. A production system would likely cache this and refresh on an interval — noted here as a reasonable next optimization, not fixed now.

---

## Model Registry & Routing Policy

All model metadata (provider, pricing, quality tier, context window, availability, latency) lives in `model_registry.py` — adding a new model means adding one entry, no routing logic changes required.

Four selectable routing objectives, all operating over the same constraint-filtered candidate list:

| Mode | Objective |
|---|---|
| `economy` | Cheapest model meeting the quality floor |
| `quality` | Highest-quality model, cost ignored |
| `latency` | Fastest model meeting the quality floor |
| `balanced` | Weighted combination (50% quality / 35% cost / 15% latency), normalized across candidates |

Every routing decision returns **why** it happened — which model was chosen, why alternatives were rejected, and the estimated cost/quality/latency of every candidate considered.

---

## Results

Measured on a balanced, hand-labeled 30-prompt benchmark (10 simple / 10 medium / 10 complex), with a fixed reference model for evaluation integrity.

| Metric | Result |
|---|---|
| Classifier accuracy — LLM-based vs. human-labeled ground truth | **100%** (vs. 73.3% for rule-based) |
| Overall cost savings vs. always-premium baseline | **19.7%** (67.5% simple / 33.9% medium / 0% complex, by design) |
| Quality retention (LLM-as-judge vs. reference model) | **4.50 / 5**, zero scores below 4 |
| Pre-call cost estimate accuracy | within **~5%** of real measured cost |

---

## Real Engineering Problems Found & Fixed

Documented because the debugging process is as important as the final numbers.

1. **Token-count skew inverted expected savings** — uncapped `max_tokens` let a cheaper-per-token model generate up to 4x more output tokens than a pricier one, making the "cheaper" tier cost more overall. Fixed with tier-specific `max_tokens` caps.
2. **Rate-limit fallback silently corrupted evaluation integrity** — automatic fallback initially applied even to fixed reference/baseline models in evaluation scripts, sometimes comparing an answer to itself under a different name. Fixed with an `allow_fallback` flag: live traffic degrades gracefully, evaluation scripts fail loudly instead.
3. **Quality-floor exact-equality bypass** — a model's quality score exactly equaled a tier's quality floor, and the filter used strict `<`, letting it "qualify" by tying rather than clearing the bar. Fixed with honestly-scoped quality values plus a `QUALITY_SAFETY_MARGIN`.
4. **Balanced-mode scoring was silently just quality-mode** — raw cost (tiny numbers) was completely dominated by the quality term regardless of actual cost differences. Fixed by normalizing cost/latency to a 0–1 range relative to the current candidate set before weighting.
5. **Rule-based classification is fragile in ways that only show up empirically** — keyword rules missed complex prompts without trigger words. Measured, not assumed: 73.3% vs. 100% accuracy against the same labeled set.
6. **Bloated, environment-polluted `requirements.txt`** — an initial `pip freeze` pulled in hundreds of unrelated packages from other local projects (including a Windows-only binary that broke the Linux-based Docker build). Fixed by hand-curating the actual 8 packages this project uses.

---

## Features

- Policy-based routing across 4 modes, with full explainability
- Pre-call cost estimation (no wasted API calls to check affordability)
- Real post-generation quality verification (independent LLM-judge)
- Budget-aware escalation — only triggers if a stronger candidate exists AND remaining budget allows it
- Rate-limit-aware fallback with evaluation-safe mode
- Centralized model registry
- Relational data layer (Supabase/Postgres) — requests, model calls, and quality scores properly linked
- Dynamic, live-computed quality scores that override static estimates once enough real data exists
- API key authentication — no request reaches the LLM or database without a valid, self-issued key
- Containerized (Docker) — runs identically anywhere with two commands
- CI (GitHub Actions) — lint + Docker build verified automatically on every push
- Two Streamlit UIs: historical analytics dashboard + live interactive query console

---

## What I'd Build Next

- Cache live quality scores on an interval rather than querying fresh per request
- Multi-user auth (Supabase Auth / JWT) if this ever needed to serve independently tracked users
- Expand the evaluation benchmark set beyond 30 prompts for tighter statistical confidence
- Multi-provider support beyond Groq
- `docker-compose.yml` if additional local services are ever added

---

## Running It

**With your own credentials** (see `.env.example` — you'll need your own free Groq key, Supabase project, and a self-generated API key registered in your `api_keys` table):

```bash
# Local
pip install -r requirements.txt
uvicorn main:app --reload
streamlit run dashboard.py --server.port 8501
streamlit run query_console.py --server.port 8502

# Or containerized
docker build -t llm-cost-autopilot .
docker run -p 8000:8000 --env-file .env llm-cost-autopilot
```

```bash
python classifier_accuracy.py    # classifier comparison
python compare_baseline.py       # cost savings measurement
python quality_check.py          # quality verification (LLM-as-judge)
python test_dynamic_quality.py   # inspect live vs. static quality scores per model
python generate_key.py           # generate a new API key to register in Supabase
```

## API Example

```bash
POST /query
Headers: X-API-Key: <your-generated-key>
{
  "prompt": "Explain how vaccines work",
  "routing_mode": "balanced",
  "max_cost_usd": 0.0005,
  "min_quality": 0.8
}
```

Returns the generated answer plus full routing telemetry: chosen model, routing reason, quality score, escalation status, total cost, and every rejected candidate with its reason.