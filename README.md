# 🚀 LLM Cost Autopilot

An adaptive LLM inference router that classifies incoming queries, estimates cost/quality/latency across candidate models *before* generating a response, applies user-defined constraints, and picks the best eligible model — with real post-generation quality verification and budget-aware escalation if the first attempt falls short.

Built solo as a personal engineering project to go deeper into production LLM infrastructure patterns (routing policy, evaluation, resilience, explainability) beyond simple API-calling projects.

---

## The Problem

Companies running LLM-powered products routinely send every query — trivial or complex — to the same expensive, top-tier model. A factual lookup costs the same as a deep multi-step reasoning task, even though a much cheaper model could answer the simple one just as well. At scale, this is real, avoidable spend. This is also a real, funded product category (OpenRouter, Not Diamond, Martian, Portkey, LiteLLM all solve exactly this problem commercially).

**The goal:** build a system that automatically decides, per-query, which model to use — under real cost/quality/latency/budget constraints — and prove, with real measurements, that it saves money *without* degrading answer quality.

---

## Architecture

```
User Request
     │
     ▼
Validate Constraints ── reject invalid routing_mode / budget / quality bounds
     │
     ▼
Complexity Classifier ── LLM-based: simple / medium / complex
     │
     ▼
Candidate Models (from Model Registry)
     │
     ▼
For each candidate, check:
  • availability          • context window
  • quality ≥ floor + safety margin
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
Full Telemetry Logged (JSONL)
```

**Resilience layer:** if a model hits a rate limit mid-request, the system automatically falls back to another available model — with a separate `allow_fallback=False` mode reserved for evaluation scripts, so measurement integrity is never silently compromised by an unplanned model swap.

---

## Tech Stack

- **Backend:** FastAPI (Python)
- **LLM Provider:** Groq (free tier) — `llama-3.1-8b-instant`, `openai/gpt-oss-120b`, `llama-3.3-70b-versatile`
- **UI:** Two Streamlit apps — historical analytics dashboard + live interactive query console
- **Logging:** JSONL (append-only structured telemetry)
- **Evaluation:** LLM-as-judge methodology, custom labeled benchmark set

---

## Model Registry

All model metadata (provider, pricing, quality tier, context window, availability, latency) lives in one place (`model_registry.py`), not scattered across if/else branches. Adding a new model means adding one entry — no routing logic changes required.

```python
MODELS = {
    "small":  {"model": "llama-3.1-8b-instant",     "quality_tier": 0.75, ...},
    "medium": {"model": "openai/gpt-oss-120b",       "quality_tier": 0.85, ...},
    "large":  {"model": "llama-3.3-70b-versatile",   "quality_tier": 1.00, ...},
}
```

Quality scores are **honestly scoped** — each reflects the tier of prompt that model was actually evaluated against in `quality_check.py`, not an assumed universal score (see bugs section below for why this matters).

---

## Routing Policy Engine

Four selectable objectives, all operating over the same constraint-filtered candidate list:

| Mode | Objective |
|---|---|
| `economy` | Cheapest model meeting the quality floor |
| `quality` | Highest-quality model, cost ignored |
| `latency` | Fastest model meeting the quality floor |
| `balanced` | Weighted combination (50% quality / 35% cost / 15% latency), normalized across candidates |

Every routing decision returns **why** it happened — which model was chosen, why alternatives were rejected (unavailable / over budget / below quality floor / context window exceeded), and the estimated cost/quality/latency of every candidate considered.

---

## Results

Measured on a balanced, hand-labeled 30-prompt benchmark (10 simple / 10 medium / 10 complex), with a fixed reference model for evaluation integrity.

### Classifier Accuracy (vs. human-labeled ground truth)

| Classifier | Accuracy |
|---|---|
| Rule-based (keyword matching) | 73.3% |
| LLM-based (Groq judge call) | **100%** |

### Cost Savings (routed vs. always-use-strongest-model baseline)

| Tier | Savings |
|---|---|
| Simple | 67.5% |
| Medium | 33.9% |
| Complex | 0% (by design — already the strongest model) |
| **Overall** | **19.7%** (including classifier call overhead) |

### Quality Retention (LLM-as-judge, routed answer vs. reference model answer)

**4.50 / 5** average across 20 judged queries. Zero scores below 4 — every "4" was scored down for *comprehensiveness*, not correctness; no factual errors introduced by routing to a cheaper tier.

### Cost Estimation Accuracy (pre-call estimate vs. real measured cost)

Word-count-based estimator landed within **~5%** of real measured cost on a complex-tier test prompt ($0.000399 estimated vs. ~$0.00042 actual) — accurate enough to filter candidates by budget without ever calling an API.

---

## Real Engineering Problems Found & Fixed

This project surfaced several genuine bugs — documenting them because the debugging process is as important as the final numbers.

**1. Token-count skew inverted expected savings**
Without a `max_tokens` cap, a cheaper-per-token model generated up to 4x more output tokens than a pricier one for the same prompt, making the "cheaper" tier cost *more* in total. Diagnosed via token-count debug logging; fixed with tier-specific `max_tokens` limits.

**2. Rate-limit fallback silently corrupted evaluation integrity**
Automatic fallback initially applied everywhere, including to the fixed "reference" model used in evaluation scripts — meaning the system sometimes compared an answer to *itself* under a different name, inflating quality scores. Fixed with an `allow_fallback` flag: live traffic degrades gracefully, evaluation scripts fail loudly and skip rather than silently substituting models.

**3. Quality-floor exact-equality bypass**
A model's `expected_quality` was set exactly equal to the `complex`-tier quality floor (0.90), and the filter used strict `<` — so a model that was never actually evaluated on complex tasks still "qualified" by tying the bar rather than clearing it. Fixed two ways: (a) corrected quality scores to honestly reflect only the tier each model was tested against, and (b) added a `QUALITY_SAFETY_MARGIN` so a tie is no longer sufficient.

**4. Balanced-mode scoring was silently just quality-mode**
The `balanced` formula combined quality (0.75–1.0 range) with raw cost (~0.00001–0.0004 range) and raw latency, using fixed multipliers. Because cost values are orders of magnitude smaller than quality values, the quality term dominated regardless of actual cost differences — `balanced` mode always picked the same model as `quality` mode. Fixed by normalizing cost and latency to a 0–1 range *relative to the current candidate set* before applying weights, so all three factors are actually comparable.

**5. Rule-based classification is fragile in ways that only show up empirically**
Keyword-matching rules missed genuinely complex prompts that didn't contain trigger words. Measured, not assumed: 73.3% vs. 100% accuracy against the same labeled set.

---

## Features

- Policy-based routing across 4 modes, with full explainability (chosen model + rejected candidates + reasons)
- Pre-call cost estimation (no wasted API calls to check affordability)
- Real post-generation quality verification (independent LLM-judge, not self-reported confidence)
- Budget-aware escalation — only triggers if a stronger candidate exists AND remaining budget allows it
- Rate-limit-aware fallback with evaluation-safe mode
- Centralized model registry — adding models requires no routing code changes
- Full request-level telemetry (cost, quality, latency, escalation reason, rejected candidates)
- Two Streamlit UIs: historical analytics dashboard + live interactive query console showing cost comparisons across all candidate models per request

---

## What I'd Build Next

- Expand the evaluation benchmark set beyond 30 prompts for tighter statistical confidence
- Log real production traffic over time; only revisit a trained/learned classifier once genuine usage data exists (consciously deferred — 30-100 log entries is not enough to train on without overfitting)
- Add per-user budget caps and multi-provider support (not just Groq)
- Explore bandit-style online learning for tier selection based on accumulated outcome data

---

## Running It

```bash
pip install fastapi uvicorn groq python-dotenv streamlit pandas requests
# add GROQ_API_KEY to .env

uvicorn main:app --reload              # live API at localhost:8000/docs
streamlit run dashboard.py --server.port 8501       # historical analytics dashboard
streamlit run query_console.py --server.port 8502   # live interactive query console

python classifier_accuracy.py    # classifier comparison
python compare_baseline.py       # cost savings measurement
python quality_check.py          # quality verification (LLM-as-judge)
```

## API Example

```bash
POST /query
{
  "prompt": "Explain how vaccines work",
  "routing_mode": "balanced",
  "max_cost_usd": 0.0005,
  "min_quality": 0.8
}
```

Returns the generated answer plus full routing telemetry: chosen model, routing reason, quality score, escalation status, total cost, and every rejected candidate with its reason.