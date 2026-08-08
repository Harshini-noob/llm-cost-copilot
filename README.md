# 🚀 LLM Cost Autopilot

**An adaptive LLM inference router that automatically picks the cheapest model capable of answering each query well — with live-computed quality scores, budget-aware escalation, full explainability, and a real client SDK.**

Built solo, end-to-end: routing engine, evaluation framework, relational data layer, authentication, containerization, CI, and a pip-installable client package. Every number below is measured, not estimated — and every bug listed was found, diagnosed, and fixed during real development, not hypothetical.

---

## The Problem

Companies running LLM-powered products routinely send *every* query — a one-word factual lookup or a deep multi-step reasoning task — to the same expensive, top-tier model. That's real, avoidable spend at scale. It's also a genuine, funded product category: OpenRouter, Not Diamond, Martian, Portkey, and LiteLLM all exist to solve exactly this problem commercially.

**This project builds that system from scratch:** classify each query's complexity, estimate cost/quality/latency across candidate models *before* generating anything, apply user-defined budget and quality constraints, pick the best eligible model, verify the answer actually met the bar, and escalate if it didn't — all while learning, over time, which models actually perform well from real outcomes instead of hand-set guesses.

---

## Why This Isn't "Called an API and Deployed a Chatbot"

| | |
|---|---|
| **Policy-based routing** | 4 distinct objectives (economy / balanced / quality / latency), not a single fixed lookup table |
| **Live, self-correcting quality scores** | Model quality is computed from real judge-verified outcomes once enough data exists — not permanently hardcoded |
| **Full explainability** | Every routing decision returns *why* — which model was picked, and why every alternative was rejected |
| **Real quality verification** | An independent LLM judge scores every answer post-generation — not self-reported model confidence |
| **Budget-aware escalation** | If quality fails, the system only escalates to a stronger model if one exists *and* the budget allows it |
| **Relational persistence** | Postgres (Supabase), not flat log files — requests, model calls, and quality scores properly linked |
| **Real auth** | API-key gated; no request reaches an LLM or the database without a valid key |
| **A real client SDK** | `pip install`-able package with typed result objects — proven working from a completely separate machine location |
| **Deployable** | Dockerized, with CI (GitHub Actions) verifying lint + build on every push |

---

## Architecture

```
User Request  ──[ X-API-Key ]──▶  Auth Check (Supabase-backed)
     │
     ▼ (rejected here if invalid — zero cost incurred)
Validate Constraints ── reject invalid routing_mode / budget / quality bounds
     │
     ▼
Complexity Classifier ── LLM-based: simple / medium / complex
     │
     ▼
Candidate Models (Model Registry, quality fetched live per request)
     │
     ▼
For each candidate, check:
  • availability   • context window
  • quality ≥ floor + safety margin  (live-computed, or hand-set fallback)
  • estimated cost ≤ budget  (pre-call estimate — no wasted API calls)
     │
     ▼
Routing Policy ── economy / balanced / quality / latency
     │
     ▼
Select Best Eligible Model ── full "rejected candidates + reasons" trail
     │
     ▼
Generate Response
     │
     ▼
Quality Verification ── independent LLM-judge scoring
     │
     ├── Pass ──▶ Return
     │
     └── Fail ──▶ Check stronger candidates ──▶ Check remaining budget
                    ──▶ Escalate if allowed ──▶ Re-verify ──▶ Return
     │
     ▼
Full Telemetry → Supabase (requests → model_calls → quality_scores)
     │
     ▼
Feeds back into live quality scoring for future requests
```

---

## Results

Measured on a balanced, hand-labeled 30-prompt benchmark (10 simple / 10 medium / 10 complex), with a fixed reference model for evaluation integrity.

| Metric | Result |
|---|---|
| Classifier accuracy — LLM-based vs. human-labeled ground truth | **100%** (vs. 73.3% for rule-based keyword matching) |
| Overall cost savings vs. always-premium baseline | **19.7%** (67.5% on simple queries, 33.9% on medium) |
| Quality retention (LLM-as-judge vs. reference model) | **4.50 / 5** — zero scores below 4 |
| Pre-call cost estimate accuracy | within **~5%** of real measured cost |
| Live quality score self-correction (observed) | after 7 real queries, one model's live score (0.943) had already diverged meaningfully from its hand-set estimate (0.85) |

---

## Try It: What a Routed Query Actually Returns

```python
from llm_cost_autopilot_client import LLMCostAutopilotClient

client = LLMCostAutopilotClient(api_key="your_key_here")
result = client.ask("Explain how vaccines work", routing_mode="balanced")

print(result.model)            # openai/gpt-oss-120b
print(result.routing_reason)   # "Best balance of cost, quality, and latency"
print(result.total_cost_usd)   # 0.00025125
print(result.quality_score)    # 5

for c in result.candidates_considered:
    print(c)
# CandidateModel(model='llama-3.1-8b-instant', cost=$0.000005, quality=0.96, latency=0.6s)
# CandidateModel(model='openai/gpt-oss-120b', cost=$0.000037, quality=0.973, latency=1.1s)
# CandidateModel(model='llama-3.3-70b-versatile', cost=$0.000052, quality=1.0, latency=1.4s)
```

No dict-key guessing, no black box — the full decision, typed and readable. Verified working from a completely independent machine location, not just inside this repo.

---

## Real Engineering Problems Found & Fixed

The debugging process here mattered as much as the final numbers. Six real, non-trivial issues surfaced during development:

1. **Token-count skew inverted expected savings** — uncapped `max_tokens` let a cheaper-per-token model generate up to 4x more output tokens than a pricier one for the same prompt, making the "cheaper" tier cost *more* overall. Diagnosed via token-count debug logging; fixed with tier-specific caps.
2. **Rate-limit fallback silently corrupted evaluation integrity** — automatic fallback initially applied even to fixed reference/baseline models in evaluation scripts, occasionally comparing an answer to itself under a different name. Fixed with an `allow_fallback` flag: live traffic degrades gracefully, evaluation scripts fail loudly instead.
3. **Quality-floor exact-equality bypass** — a model's quality score exactly equaled a tier's quality floor, and the filter used strict `<`, letting it "qualify" by tying rather than clearing the bar. Fixed with honestly-scoped quality values plus a safety margin.
4. **Balanced-mode scoring was silently just quality-mode** — raw cost values (tiny numbers) were completely dominated by the quality term regardless of actual cost differences. Fixed by normalizing cost/latency to a 0–1 range relative to the current candidate set before weighting.
5. **Component drift after the Supabase migration** — the dashboard kept reading a now-abandoned local log file, the query console never sent its auth header, and two test scripts used stale function signatures from before a refactor. Found via a structured implementation review, fixed and re-verified end-to-end across all four components.
6. **Environment-polluted `requirements.txt`** — an initial `pip freeze` pulled in hundreds of unrelated packages from other local projects, including a Windows-only binary that broke the Linux-based Docker build. Fixed by hand-curating the actual 8 packages this project uses.

---

## Tech Stack

**Backend:** FastAPI (Python) · **LLM Provider:** Groq — `llama-3.1-8b-instant`, `openai/gpt-oss-120b`, `llama-3.3-70b-versatile` · **Database & Auth:** Supabase (Postgres) · **UI:** Streamlit (analytics dashboard + interactive query console) · **DevOps:** Docker + GitHub Actions CI · **Client:** custom pip-installable SDK

---

## What I'd Build Next

- Cache live quality scores on an interval rather than querying fresh per request
- Multi-user auth (Supabase Auth / JWT) if this needed to serve independently-tracked users
- Expand the evaluation benchmark beyond 30 prompts for tighter statistical confidence
- Multi-provider support beyond Groq

---

## Running It

```bash
# Server
pip install -r requirements.txt
uvicorn main:app --reload
streamlit run dashboard.py --server.port 8501
streamlit run query_console.py --server.port 8502

# Or containerized
docker build -t llm-cost-autopilot .
docker run -p 8000:8000 --env-file .env llm-cost-autopilot

# Client SDK
pip install -e .
```

```bash
python classifier_accuracy.py    # classifier comparison
python compare_baseline.py       # cost savings measurement
python quality_check.py          # LLM-as-judge quality verification
python test_dynamic_quality.py   # live vs. static quality scores per model
```

Requires your own free Groq + Supabase accounts — see `.env.example`. API access is protected by a self-issued key (`generate_key.py`); no credentials ship with the code.

---

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

Returns the answer plus full routing telemetry: chosen model, reasoning, quality score, escalation status, total cost, and every candidate — considered or rejected — with its numbers.