# 🧠 LLM Cost Autopilot

An intelligent request router that classifies incoming LLM queries by complexity and routes each one to the cheapest model tier capable of answering it well — cutting cost without sacrificing answer quality.

Built solo as a personal engineering project to go deeper into production LLM infrastructure patterns (routing, evaluation, resilience) beyond simple API-calling projects.

---

## The Problem

Companies running LLM-powered products routinely send every query — trivial or complex — to the same expensive, top-tier model. A factual lookup ("what's the capital of Japan?") costs the same as a deep multi-step reasoning task, even though a much cheaper model could have answered the simple one just as well. At scale, this is real, avoidable spend. This is also a real, funded product category (OpenRouter, Not Diamond, Martian, Portkey, LiteLLM all exist to solve exactly this).

**The goal:** build a system that automatically decides, per-query, which model tier to use — and prove, with real measurements, that it saves money *without* degrading answer quality.

---

## Architecture

```
User Query
    │
    ▼
LLM Classifier ── rates complexity: simple / medium / complex
    │
    ▼
Router ── maps tier → model (Groq: llama-3.1-8b-instant / gpt-oss-120b / llama-3.3-70b-versatile)
    │
    ▼
Model Call ── generates answer, tracks tokens/cost/latency
    │
    ▼
Confidence Check ── model self-rates its own answer (1-10)
    │
    ├── confident (≥6) → return answer
    └── unsure (<6)    → escalate to next tier, re-answer, return upgraded answer
    │
    ▼
Logger ── every request logged (tier, model, cost, latency, confidence, escalated?)
    │
    ▼
Dashboard ── Streamlit UI: cost trends, tier distribution, latency, fallback events
```

**Resilience layer:** if a model hits a rate limit mid-request, the system automatically falls back to a cheaper available model rather than failing the request — with a separate `allow_fallback=False` mode reserved for evaluation scripts, so measurement integrity is never silently compromised by an unplanned model swap.

---

## Tech Stack

- **Backend:** FastAPI (Python)
- **LLM Provider:** Groq (free tier) — `llama-3.1-8b-instant`, `openai/gpt-oss-120b`, `llama-3.3-70b-versatile`
- **Dashboard:** Streamlit + Pandas
- **Logging:** JSONL (append-only structured logs)
- **Evaluation:** LLM-as-judge methodology, custom labeled benchmark set

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

**4.50 / 5** average, across 20 judged queries (simple + medium tiers). Zero scores below 4 — every "4" was scored down for *comprehensiveness*, not correctness; no factual errors were introduced by routing to a cheaper tier.

---

## Real Engineering Problems Found & Fixed Along the Way

This project surfaced several genuine bugs and design flaws — documenting them here because the debugging process is as important as the final numbers.

**1. Token-count skew inverted expected savings**
Without a `max_tokens` cap, a cheaper-per-token model (`gpt-oss-120b`) generated up to 4x more output tokens than the pricier baseline model for the same prompt, making the "cheaper" tier cost *more* in total. Diagnosed via token-count debug logging, fixed with tier-specific `max_tokens` limits — savings flipped from -88.7% to a real, positive number.

**2. Rate-limit fallback silently corrupted evaluation integrity**
Adding automatic fallback (for resilience) initially applied everywhere, including to the fixed "reference" and "baseline" models used in evaluation scripts. This meant the system was sometimes comparing an answer to *itself* under a different name, artificially inflating quality scores. Fixed by adding an `allow_fallback` flag — live traffic keeps graceful degradation, evaluation scripts fail loudly and skip instead of silently substituting models.

**3. Rule-based classification is fragile in ways that only show up empirically**
Keyword-matching rules missed genuinely complex prompts that didn't happen to contain trigger words (e.g., "write a function to check X" wasn't caught by a "write code" rule). Measured, not assumed: 73.3% vs. 100% accuracy against the same labeled set.

---

## Features

- 3-tier LLM routing with two interchangeable classifiers (rule-based, LLM-based)
- Confidence-based escalation — model self-rates its answer; low-confidence responses automatically get a second attempt from a stronger tier
- Rate-limit-aware fallback with evaluation-safe mode
- Full cost/latency/token logging per request
- Live Streamlit dashboard with tier/cost/latency breakdowns and fallback tracking

---

## What I'd Build Next

- Expand the evaluation benchmark set beyond 30 prompts for tighter statistical confidence
- Log real production traffic over time and revisit whether a genuinely trained classifier (vs. rule-based/LLM-based) outperforms once real usage data exists
- Add per-user budget caps and multi-provider support (not just Groq)
- Explore bandit-style online learning for tier selection based on accumulated outcome data

---

## Running It

```bash
pip install fastapi uvicorn groq python-dotenv streamlit pandas
# add GROQ_API_KEY to .env
uvicorn main:app --reload        # live API at localhost:8000/docs
streamlit run dashboard.py       # dashboard at localhost:8501
python classifier_accuracy.py    # classifier comparison
python compare_baseline.py       # cost savings measurement
python quality_check.py          # quality verification (LLM-as-judge)
```