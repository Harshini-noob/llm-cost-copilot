# 🚀 LLM Cost Autopilot (Copilot)

> **An Adaptive, High-Throughput, Multi-Provider LLM Router & Quality Copilot**
> *Classifies query complexity, estimates cost/quality/latency pre-call, routes across multi-provider LLMs (Groq & Gemini) with automatic failover, and enforces post-generation LLM-judge quality verification & budget-aware escalation.*

[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![AsyncIO](https://img.shields.io/badge/AsyncIO-Concurrency-orange.svg)](https://docs.python.org/3/library/asyncio.html)
[![Supabase](https://img.shields.io/badge/Supabase-PostgreSQL-3ECF8E.svg)](https://supabase.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B.svg)](https://streamlit.io/)

---

## 📌 Executive Summary & Motivation

In enterprise production environments, sending **every query** (from a 5-word factual lookup to a multi-step system design prompt) to a top-tier \$15/M token model incurs massive unnecessary expense. 

**LLM Cost Autopilot** solves this \$100B+ over-provisioning problem by dynamically routing prompts to the cheapest, fastest model capable of satisfying user-defined quality thresholds.

### Key Benchmark Results (Empirically Verified)

| Metric | Measured Value | Benefit / Impact |
| :--- | :--- | :--- |
| **Simple Query Cost Savings** | **80.3% Savings** | Cuts costs by ~5x for routine, high-volume lookups |
| **Overall Cost Savings** | **51.1% Savings** | Cut baseline spend in half across mixed workloads |
| **Quality Retention** | **4.91 / 5.0 Quality Score** | Zero noticeable drop in response quality vs. reference model |
| **Failover Success Rate** | **100% Resilient (8/8 Calls)** | Zero crashes during cross-provider rate/quota exhaustion |
| **Classification Accuracy** | **73.3% Accuracy** | 0ms latency, \$0 cost deterministic classifier |

---

## 💡 Key Architectural Innovations

### 1. 🔀 Multi-Criteria Dynamic Optimization Engine
Unlike naive routers that only look at model tier, **LLM Cost Autopilot** performs pre-call constraint validation and mathematical normalization across candidate models:
$$\text{Score} = (W_{\text{quality}} \cdot Q_{\text{exp}}) + (W_{\text{cost}} \cdot (1 - C_{\text{norm}})) + (W_{\text{latency}} \cdot (1 - L_{\text{norm}}))$$
All candidate metrics (cost, latency, quality) are normalized to $[0, 1]$ before applying routing mode weights (`balanced`, `economy`, `quality`, `latency`).

### 2. 🛡️ Cross-Provider Auto-Failover Circuit Breaker
Provides high availability across decoupled provider APIs (Groq + Google Gemini). If a model encounters rate-limits, 429 quota exhaustion, or API errors:
- Catches provider-specific exceptions (`RateLimitError`, `GeminiAPIError`).
- Intercepts error before response failure and seamlessly re-routes to a fallback model (e.g., Gemini $\rightarrow$ Groq `120b`).
- Logs `fell_back: True` in telemetry for observability.

### 3. ⚖️ Post-Generation Quality Verification & Budget-Aware Escalation
- After answer generation, an independent **LLM-as-a-Judge** scores response quality ($1–5$).
- If quality falls below threshold, the router calculates **remaining user budget** ($B_{\text{rem}} = B_{\text{max}} - C_{\text{spent}}$).
- **Escalates only if budget permits**, selecting a stronger candidate and re-evaluating.

### 4. 📈 Self-Correcting Live Quality Feedback Loop
- Quality tiers are not static dictionary values; they are continuously re-computed live from historical Supabase telemetry (`model_calls` $\rightarrow$ `quality_scores`).
- If a model's live quality degrades, the router automatically downgrades its expected quality score for future routing decisions.

### 5. ⚡ Fully Asynchronous Concurrency Pipeline
- Implemented from bottom-up using `AsyncGroq`, `google.genai.Client.aio`, and Supabase `acreate_client`.
- Non-blocking I/O ensures high throughput under heavy concurrency.

---

## 🏗️ System Architecture

```
User Query ──[ X-API-Key ]──▶ FastAPI Gateway (main.py)
                                  │
                                  ▼
                     Validate Constraints (max_cost_usd, min_quality)
                                  │
                                  ▼
                     Complexity Classifier (rule-based / LLM / embedding)
                                  │
                                  ▼
                     Fetch Candidate Models & Live Quality Scores (Supabase)
                                  │
                                  ▼
                     Filter Candidates (Context Window, Quality Floor, Budget)
                                  │
                                  ▼
                     Score & Select Best Model (balanced / economy / quality / latency)
                                  │
                                  ▼
                     Async Provider Dispatcher (Groq / Gemini)
                                  │
                       ┌──────────┴──────────┐
                    [ Success ]          [ 429 / Rate Limit ]
                       │                     │
                       ▼                     ▼
             Response Generated     Cross-Provider Failover
                       │                     │
                       └──────────┬──────────┘
                                  ▼
                    Post-Gen Quality Verification (LLM Judge)
                                  │
                      ┌───────────┴───────────┐
                   [ Pass ]               [ Fail ]
                      │                      │
                      │           Check Remaining Budget & Escalate
                      │                      │
                      └───────────┬──────────┘
                                  ▼
                    Telemetry Persistence (Supabase)
                                  │
                      ┌───────────┴───────────┐
                      ▼                       ▼
            Streamlit Dashboard     Interactive Query Console
```

---

## 🔬 Empirical Classification Benchmark: Why Rule-Based Beats LLM Classifier

During benchmarking, we discovered a crucial production insight:
- **LLM-Based Classifier**: Scores dropped from $100\%$ to $56\text{--}67\%$ when models were updated to reasoning-oriented architectures. Reason: **invisible reasoning-token consumption** consumed variable output token budgets before emitting the classification word.
- **Rule-Based Classifier**: Deterministic, 0ms latency, \$0 execution cost, and stable **73.3% accuracy** against ground truth.

**Engineering Decision**: Rule-based is set as production default; LLM and embedding (KNN) classifiers are preserved as benchmarked alternatives.

---

## 🛠️ Real-World Engineering Problems Solved

1. **Reasoning-Token Exhaustion Fix**: Diagnosed empty LLM responses by inspecting raw message objects (`repr()`); solved by configuring token budget headroom (`max_tokens`: 250 simple / 600 medium / 1200 complex).
2. **Windows Console Encoding Resilience**: Replaced Unicode symbols (e.g. `⚠️`) in logging with ASCII fallbacks (`[WARN]`) to prevent `UnicodeEncodeError` on Windows `cp1252` terminals.
3. **Evaluation Integrity Protection**: Added `allow_fallback=False` flag during baseline evaluations to prevent reference models from silently evaluating against themselves during rate limits.
4. **Normalized Weighted Scoring**: Solved cost-weighting imbalance where raw sub-cent costs were dwarfed by quality metrics by scaling all parameters to $[0, 1]$ prior to weighting.

---

## 📦 Tech Stack

- **Language & Runtime**: Python 3.9+, AsyncIO
- **API Framework**: FastAPI, Pydantic, Uvicorn
- **LLM Providers**: Groq API (`openai/gpt-oss-20b`, `120b`), Google GenAI SDK (`gemini-2.5-flash`)
- **Database & Telemetry**: Supabase (PostgreSQL), `postgrest-py`
- **Embeddings & ML**: `sentence-transformers` (MiniLM-L6-v2)
- **Frontend / Visualization**: Streamlit (Dashboard & Query Console)
- **Deployment**: Docker, `uv` package manager

---

## ⚙️ Quickstart & Local Setup

### 1. Environment Setup
Create a `.env` file in the project root (see `.env.example`):
```env
GROQ_API_KEY=your_groq_api_key
GEMINI_API_KEY=your_gemini_api_key
SUPABASE_URL=your_supabase_url
SUPABASE_KEY=your_supabase_service_key
LLM_AUTOPILOT_API_KEY=your_secret_api_key
```

### 2. Install Dependencies
```bash
# Using standard pip
pip install -r requirements.txt

# Or using uv (recommended)
uv sync
```

### 3. Run FastAPI Backend Server
```bash
uvicorn main:app --reload --port 8000
```

### 4. Run Interactive Dashboards
```bash
# Launch Live Query Console (Interactive routing tester)
streamlit run query_console.py --server.port 8501

# Launch Telemetry Dashboard (Real-time analytics)
streamlit run dashboard.py --server.port 8502
```

### 5. Run Verification Benchmarks
```bash
python classifier_accuracy.py    # Test classification accuracy
python compare_baseline.py       # Benchmark cost savings vs always-premium
python quality_check.py          # Run LLM-as-a-Judge quality verification
```

---

## 🐳 Docker Deployment

```bash
# Build Docker image
docker build -t llm-cost-autopilot .

# Run container
docker run -p 8000:8000 --env-file .env llm-cost-autopilot
```

---

## 📄 Client SDK Usage

```python
from llm_cost_autopilot_client import CostAutopilotClient

client = CostAutopilotClient(api_key="your_secret_api_key")

response = client.query(
    prompt="Explain quantum entanglement in simple terms",
    routing_mode="economy",
    max_cost_usd=0.001
)

print("Chosen Model:", response.model)
print("Cost:", response.total_cost_usd)
print("Answer:", response.answer)
```