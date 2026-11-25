# LLM Cost Autopilot

**Built by Harshini A**

An adaptive LLM routing service that selects models by cost, quality, and latency. It combines a FastAPI API, Groq and Google Gemini providers, request-level budget tracking, Supabase telemetry, a Streamlit dashboard, and a Python client SDK.

> This is an engineering prototype, not an absolute provider-billing cap. Cost estimates and provider-reported usage may differ from actual charges.

## Benchmark Results (M4 — xRouteBench Offline Evaluation)

Evaluated on **xRouteBench** (`ulab-ai/xRouteBench`, **3,729** held-out test prompts, **18** candidate LLMs, **13** tasks, zero live API calls; CC BY-NC-SA 4.0). Routers are swept over cost-penalty $\lambda \in [0.00, 5.00]$ with **95% prompt-cluster bootstrap confidence intervals**; baselines are single operating points and the Oracle (`λ=0.00`) marks the theoretical upper bound.

![Cost vs quality — routers as λ-sweep curves, baselines as points](reports/m4/hero_cost_vs_quality.svg)

### Multi-Model Routing Summary (18 Candidates, $n = 3,729$ Test Prompts)

| Strategy | Role | Macro Quality [95% CI] | Micro Quality [95% CI] | Cost / 1k Queries | Historical LLM Latency* |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Always-Cheapest** (`gpt-oss-20b`) | Deployable baseline | 0.5806 [0.5627, 0.5965] | 0.6976 [0.6810, 0.7113] | \$0.014 | 3.21s |
| **Random-Uniform** (18 models) | Deployable baseline | 0.5109 [0.4897, 0.5292] | 0.6173 [0.6004, 0.6343] | \$0.181 | 2.76s |
| **Static-Task-Table** | Deployable (needs `task_name`) | 0.7321 [0.7041, 0.7570] | 0.8137 [0.8020, 0.8258] | \$0.315 | 3.15s |
| **kNN ($\lambda = 0.10$)** | Deployable learned router | 0.6529 [0.6251, 0.6739] | 0.7115 [0.6956, 0.7246] | \$0.072 | 2.95s |
| **Learned LR ($\lambda = 0.75$)** | Deployable (quality parity) | **0.6533** [0.6305, 0.6761] | 0.7380 [0.7234, 0.7518] | **\$0.036** | 3.22s |
| **Learned LR ($\lambda = 0.00$)** | Deployable (max quality) | **0.7208** [0.6994, 0.7478] | 0.7994 [0.7848, 0.8121] | \$0.283 | 3.44s |
| **Best Single Model** (`qwen3-next-80b-a3b-instruct`) | Post-hoc test reference | 0.6536 [0.6368, 0.6650] | 0.8054 [0.7931, 0.8188] | \$0.458 | 2.52s |
| **Oracle ($\lambda = 0.00$)** | Theoretical upper bound | 0.8533 [0.8325, 0.8760] | 0.9293 [0.9218, 0.9377] | \$0.032 | 2.64s |

\* *Historical LLM Latency reflects downstream provider response times recorded in xRouteBench, not local router CPU decision overhead (sub-millisecond).*

- **Calibration (M4.3):** Raw Logistic Regression achieves **0.0389 ECE** / **0.1734 Brier** (improved to **0.0283 ECE** via out-of-fold Platt scaling); kNN ($k=5$) improves from **0.1123** to **0.0490 ECE** via isotonic regression.
- **Out-of-Distribution (M4.4):** Leave-one-task-out across 6 unseen tasks yields a 3–3 split (`LEARNED_LR` wins `gsm8k`, `mmlu`, `aime_2020_2024`; `kNN` wins `mbpp`, `human_eval`, `squad`), with calibration degrading sharply on novel domains.
- **Two-Model APGR (M4.5):** On `cogito-v2-1-671b` vs. `llama-3-8b-instruct-lite`, RouteLLM Hybrid Ensemble achieves **0.5906** Micro APGR `[0.5761, 0.6058]` (**0.5904** Macro APGR, null-centered bootstrap $p = 1/1001 \approx 0.000999$, rounded to `0.0010` vs. 0.50 random boundary; overlapping CIs with kNN `0.5818` and LR `0.5810` do not establish between-router superiority).
- **Full Research Report:** See [`reports/m4/M4_FINAL_RESEARCH_REPORT.md`](reports/m4/M4_FINAL_RESEARCH_REPORT.md) for complete methodology, audits, reliability diagrams, and reproducibility commands.

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

## V2 model benchmark (M1)

The V2 benchmark uses the LiteLLM Python SDK as a unified interface to Groq,
Google Gemini, and OpenRouter. The project still owns model selection, registry
pricing, token/cost/latency telemetry, and benchmark evaluation; LiteLLM does
not provide a universal API key or choose fallback models. Each provider
requires its own credential, and benchmark calls explicitly disable retries.

The enabled benchmark pool in [`config/models.yaml`](config/models.yaml) is:

| Registry ID | Provider model ID | Input / output per 1M tokens (USD) |
| --- | --- | --- |
| `gpt_oss_20b` | Groq `openai/gpt-oss-20b` | $0.075 / $0.30 |
| `gpt_oss_120b` | Groq `openai/gpt-oss-120b` | $0.15 / $0.60 |
| `gemini_3_5_flash_lite` | Google `gemini-3.5-flash-lite` | $0.30 / $2.50 |
| `qwen3_30b_a3b_openrouter` | OpenRouter `qwen/qwen3-30b-a3b-instruct-2507` | $0.10 / $0.30 |

Pricing was checked on 2026-11-19 against the provider documentation/catalog
linked from the YAML. Gemini figures are Google Standard tier prices; OpenRouter
pricing is provider-catalog pricing and may vary by routed endpoint. The old
`gemini_3_8_flash` entry remains disabled as historical configuration. Prices
are per-token estimates for benchmark calculations and may differ from invoices.

Set the relevant variable in a private `.env` file or the process environment:

```dotenv
GROQ_API_KEY=your_groq_api_key
GEMINI_API_KEY=your_gemini_api_key
OPENROUTER_API_KEY=your_openrouter_api_key
```

`GOOGLE_API_KEY` is also accepted as an alternative to `GEMINI_API_KEY`.
Never commit `.env`; `.env.example` contains variable names and placeholders
only. Install the project dependencies before running the V2 tests or runner.
From PowerShell:

```powershell
python -m pip install -e .
python -m pytest tests_v2
python -m eval.runner --dry-run --limit 2 --runs 1
python -m eval.runner --limit 2 --runs 1 --models gpt_oss_20b
```

The dry run prints the planned call count and a `benchmark_config_id` without
calling a provider. The ID fingerprints the selected enabled model definitions
and their configured prices, so different model pools/configurations can be
distinguished. Results append to `data/processed/matrix.parquet`; rows from an
older file without an ID are retained and explicitly marked
`legacy-unversioned`. A failed provider call is recorded with
`execution_success=False` and no correctness score. Successful outputs are
evaluated using the configured deterministic methods described below.

### M1.3 output evaluation

The runner evaluates successful outputs and stores `evaluation_status`,
`score`, `evaluation_error`, and `evaluation_details` alongside the existing
`correct` and `evaluation_method` columns. Execution failures bypass evaluation
and always have `correct=null`.

- **Math (`numeric_exact`)** parses the final non-empty answer line and compares
  it exactly using decimal arithmetic. Explanatory work and intermediate
  numbers are ignored; variable assignments are accepted only when the dataset
  configures `answer_variable` (or the reference itself is an assignment).
  Currency prefixes, whitespace, grouped-number commas, and harmless answer
  formatting are normalized; ambiguous or incorrect answers do not pass.
- **Knowledge (`normalized_exact`)** compares case- and whitespace-normalized
  text, ignoring surrounding punctuation. Dataset records may add an
  `acceptable_answers` list for explicitly approved alternatives.
- **Multiple choice (`multiple_choice_exact`)** accepts a configured answer
  label only when the output contains one unambiguous standalone label or
  explicit answer marker. Ambiguous explanatory output remains unscored.
- **Extraction (`json_exact`)** parses generated JSON and checks every expected
  field/value in the reference object. Extra generated fields are allowed;
  malformed JSON or missing/wrong required values fail.
- **Code (`unit_test`)** is marked `unsupported`: this application does not
  provide a process isolation sandbox, so generated code is never executed.
- **Summarization and chit-chat (`llm_judge`)** are unscored unless both a
  reference label and a separate `EVAL_JUDGE_MODEL` are configured. The judge
  must use a supported LiteLLM prefix, for example
  `gemini/gemini-3.5-flash-lite`, and its provider credential must be set. A
  fixed rubric requests JSON containing a 0–1 score and rationale. The result
  is stored as `needs_review`; `correct` remains null because the judge is not
  validated ground truth. The current dataset has no reference labels for
  these tasks, so they remain unscored and do not trigger judge calls.

Deterministic results use `evaluation_status=scored` with `score` 0 or 1.
Missing labels/configuration produce `unscored`; unavailable safe execution
produces `unsupported`. New evaluation fields are appended while prior
benchmark rows are retained unchanged; historical rows have null values for
fields that did not exist when they were recorded.

### M1.4 benchmark integrity analysis

The read-only integrity analyzer reads `data/processed/matrix.parquet` and
writes a uniquely named JSON report beside it. It does not update benchmark
records, historical correctness labels, or earlier reports:

```powershell
python -m eval.integrity
```

Use `--input <path>` and `--output-dir <path>` to analyze another Parquet file
or choose a report directory. Each report records its analysis version, source
row count, and source Parquet SHA-256 for reproducibility. Results are grouped
by `benchmark_config_id`; legacy rows without a fingerprint are isolated as
`legacy-unversioned`. Their duplicate keys are reported as candidates, not
verified duplicates, and they are excluded from repeated-run consistency
calculations.

The report separates verified duplicate groups from legacy candidates without
deleting rows. Execution success rate is successful executions divided by all
records; evaluation coverage is successful executions with a recorded
evaluation status divided by successful executions; accuracy is correct
scored labels divided by all scored labels. Repeated-run label agreement is
the majority-label share within each prompt/model/configuration group, and
summary agreement is weighted across groups with at least two scored runs.
Consistency classifications are `consistent_correct`,
`consistent_incorrect`, `inconsistent`, or `insufficient_sample`. Unsupported,
unscored, failed, and unlabeled records do not become correctness labels.
Each metric includes its denominator/sample sizes and insufficient-sample
flags. This analysis does not rank models or implement routing.

### M1.5 external benchmark pipeline

The original-dataset adapters are separate from the custom benchmark. They do
not change the V2 executor or registry, and their result files never append to
`data/processed/matrix.parquet`. Dataset downloads are not implemented or
performed; local import accepts only publisher-provided Parquet or JSONL files
after a separate data-download approval. Pickle files are rejected.

| Dataset config | Pinned revision | Dataset-card license | Adapter/evaluation |
| --- | --- | --- | --- |
| [GSM8K `main`](https://huggingface.co/datasets/openai/gsm8k) | `740312add88f781978c0658806c59bc2815b9866` | MIT | `question` plus exact numeric answer after `####`; `numeric_exact` |
| [MMLU `all`](https://huggingface.co/datasets/cais/mmlu) | `c30699e8356da336a370243923dbaf21066bb9fe` | MIT | Subject/question/four choices preserved; answer class mapped to A-D; `multiple_choice_exact` |
| [MBPP `sanitized`](https://huggingface.co/datasets/google-research-datasets/mbpp) | `4bb6404fdc6cacfda99d4ac4205087b89d32030c` | CC BY 4.0 | `task_id`, prompt, reference code, imports and tests preserved; `unit_test` |

MIT rows require retaining the copyright/license notice. CC BY 4.0 permits
sharing/adaptation, including commercial use, with attribution, a license link,
and notices of changes; it does not imply endorsement. The adapters record
dataset citation and license metadata per record. These are dataset-card
declarations, not assumptions based on repository code licenses.

The adapters are deterministic and versioned. GSM8K's pinned `train` rows are
assigned to train/validation by a SHA-256 rule (one-in-ten to validation); its
official `test` remains held out. MMLU `dev` maps to the train/example split,
with official validation and test held separate; `auxiliary_train` and
subject-specific duplicate configs are not included. MBPP uses only the
`sanitized` config, with its `prompt` split kept separately as `few_shot` and
official train/validation/test kept distinct. Normalization preserves each
original source row as JSON, plus its source ID or stable source-row locator,
question, original reference answer, choices/tests, source revision, and
attribution.

#### Dataset Provenance & Trusted Approval Registry

Dataset authenticity is enforced via a centralized trusted approval registry
in [`eval/dataset_registry.py`](eval/dataset_registry.py):
- Every source manifest must reference an approved registry entry ID.
- Source bytes are hashed with SHA-256 and verified independently against the
  trusted registry, preventing self-supplied manifest hash spoofing.
- Normalized records preserve verified registry entry IDs, publisher metadata,
  source URLs, and license approval references.
- `load_normalized_dataset()` independently re-authenticates every record and
  Parquet file against the trusted registry upon loading before execution.
- Registry trust overrides are restricted to test-only execution using
  `assert_test_mode()` guards to prevent production bypass.
- Frozen final-test manifests capture complete execution environment fingerprints,
  including Python implementation/version, `pyproject.toml` SHA-256, and `uv.lock` SHA-256.

The external runner validates the complete normalized split set and writes
validation and final-test results to separate directories. Final-test
execution requires both `--authorize-final-evaluation` and a frozen,
create-once evaluation manifest. The manifest fingerprints model IDs and
configuration, prompt/split settings, dataset revision and content/source
hashes, run count, and evaluator configuration; the runner rejects execution
if any setting changes. Test results are for final reporting, not routing
development, prompt tuning, or model selection. This is a workflow safeguard,
not a technical barrier to downstream manual use of test results. No automatic
routing or ranking is included.

The external runner is dry-run by default. Use `--split validation` for
development runs and `--split test` only for the frozen final evaluation.
`--models` limits the enabled registry models and `--runs` controls repeated
executions. Validation results append to an existing configuration file;
final-test results are write-once and cannot overwrite an existing result or
the custom benchmark matrix. Each result row includes execution telemetry,
evaluation fields, source provenance, source-file hashes, and the evaluation
manifest ID when applicable.

Local-only usage after data files have been separately approved and placed on
disk. Create one manifest JSON per source split with the actual file hash, for
example:

```json
{
  "dataset_name": "openai/gsm8k",
  "approved_revision": "740312add88f781978c0658806c59bc2815b9866",
  "source_file_sha256": "<sha256 of the exact local file bytes>",
  "split": "train",
  "format": "parquet",
  "attribution": "Cobbe et al., Training Verifiers to Solve Math Word Problems (2021), https://arxiv.org/abs/2110.14168"
}
```

```powershell
# Normalize local files only after manifest verification; --write refuses overwrite.
python -m eval.external_datasets --dataset gsm8k `
  --split train=.\data\external\gsm8k\train.parquet `
  --split test=.\data\external\gsm8k\test.parquet `
  --manifest train=.\data\external\gsm8k\train.manifest.json `
  --manifest test=.\data\external\gsm8k\test.manifest.json --write

# Validation dry-run; no provider calls.
python -m eval.external_runner --dataset gsm8k

# Freeze final-test settings first; the file is create-once.
python -m eval.external_runner --dataset gsm8k --split test `
  --models gpt_oss_20b --runs 1 `
  --freeze-manifest .\data\external\gsm8k-final-evaluation.json

# Final execution requires the matching frozen manifest and explicit authorization.
python -m eval.external_runner --dataset gsm8k --split test `
  --models gpt_oss_20b --runs 1 `
  --manifest .\data\external\gsm8k-final-evaluation.json `
  --authorize-final-evaluation --execute

# Validation execution remains separately stored from test results.
python -m eval.external_runner --dataset gsm8k --split validation `
  --models gpt_oss_20b --runs 1 --execute

# Synthetic adapter, evaluator, split-integrity and runner tests.
python -m pytest tests_v2\test_external_datasets.py `
  tests_v2\test_external_runner.py tests_v2\test_metrics.py -v
```

MMLU normalization takes `dev`, `validation`, and `test` local files. MBPP
normalization takes `train`, `prompt`, `validation`, and `test` local files.
The benchmark runner stores results under
`data/processed/external_benchmarks/<dataset>/`, separate from the custom
matrix, and preserves earlier external rows by appending.

**MBPP safety limitation:** this project does not currently have an approved
isolated code-execution sandbox. MBPP outputs therefore remain
`evaluation_status=unsupported`, with `correct=null`; neither model-generated
code nor source tests are executed. Actual MBPP correctness scoring is gated
on a separately reviewed sandbox implementation.

### M1.5.1 RouterBench — BLOCKED (optional scaffolding only)

Actual RouterBench ingestion is **BLOCKED** because dataset licensing and
rights for its component records, responses, and costs remain unresolved. The
optional normalizer below is not an ingestion or download path.

The [RouterBench Hub dataset](https://huggingface.co/datasets/withmartian/routerbench)
currently exposes `routerbench_0shot.pkl`, `routerbench_5shot.pkl`, and
`routerbench_raw.pkl`. The two precomputed variants are pandas pickles with
wide-form per-model performance, `|model_response`, and `|total_cost` columns.
The Hub Dataset Viewer does not support these files. The code-only normalizer
in `eval/routerbench_import.py` accepts a trusted in-memory DataFrame and
converts the 0-shot or 5-shot wide form to one row per sample/model. It does
not download or deserialize pickle files. Dataset files belong under
`data/external/routerbench/raw/`; normalized output paths are reserved under
`data/external/routerbench/normalized/`, separate from
`data/processed/matrix.parquet`.

The normalizer preserves the source sample ID, evaluation name, prompt,
model identifier, response, performance value, cost value, and oracle model
identifier. It does not map source model identifiers to the local registry,
coerce performance scores to Boolean correctness, recalculate costs, or infer
a universal task taxonomy. Optional category/metric maps must be supplied
explicitly. RouterBench's reference conversion code estimates costs from
token counts and its own historical model-cost table; those values are not
provider billing telemetry.

The Hub card does not declare a dataset license. The separate reference
repository's MIT license is not evidence that the data itself is MIT-licensed.
The repository README cites Hu et al., *RouterBench: A Benchmark for Multi-LLM
Routing System* ([arXiv:2403.12031](https://arxiv.org/abs/2403.12031)); retain
this attribution and the Hub dataset DOI `10.57967/hf/1996`, and verify
data-use rights before downloading, redistributing, or committing the data.
The Hub revision and file SHA-256 should be recorded and verified during any
future ingestion. The pickle files must not be loaded with unrestricted
`pickle.load`; use a publisher-provided tabular export or an explicitly
reviewed isolated conversion process.

To add a model, verify its provider model ID, context, and current pricing from
the provider’s documentation, then add an enabled entry using the existing YAML
schema. For a new provider, add its credential environment-variable mapping and
LiteLLM model prefix to `executor_v2.py`; adding a model for an already
supported provider requires only a registry entry.
The executor maps the provider and model ID to LiteLLM’s provider-qualified
model string (for example, `groq/openai/gpt-oss-20b`) and calls the unified
async completion API. It reads reported token usage, computes cost from the
registry prices, and measures wall-clock latency locally. It does not route,
retry, or fall back to another model.

## Baseline evaluation (M2)

Phase M2 introduces a reproducible baseline evaluation framework in [`eval/baselines.py`](eval/baselines.py) to benchmark models and routing policies against standard deployable, post-hoc, and theoretical reference baselines.

### Five Baseline Strategies

1. **Always-Cheapest (`deployable_policy`)**:
   - Identifies the candidate model with the lowest mean cost on the training partition ($m^* = \text{argmin}_m \bar{C}_{\text{train}}$).
   - Routes all test prompts to $m^*$.
   - Represents the cost-oriented lower bound.
2. **Always-Best (`post_hoc_reference`)**:
   - Evaluates test-set quality across candidate models and selects the single model achieving highest quality ($m^* = \text{argmax}_m Q_{\text{test}}$), breaking ties by lowest test cost.
   - Strictly declared as a post-hoc reference ceiling, not a deployable train-time selection policy.
3. **Random at Matched Cost (`deployable_policy`)**:
   - Solves a convex combination over candidate models to match a configurable target expected cost ($\mathbb{E}[C] = \text{target\_cost}$).
   - Evaluates across repeated sampling trials ($N=20$) with reproducible seeds to capture empirical sampling variability.
   - Reports target cost, expected cost, realized mean cost, standard deviation, and relative cost deviation ($38.04\%$).
4. **Static Task-Type Table (`deployable_policy`)**:
   - Learns a mapping from prompt `task_type` to best-value model strictly on the training partition.
   - Default objective: $\text{Quality} / \max(\text{Cost}, 1e-9)$ (alternative objectives: `quality_only`, `cost_only`).
   - Supports configurable classifier inference cost (defaults to $0.00 when task metadata labels are supplied).
   - Falls back gracefully to the overall training best-value model when an unseen test category is encountered.
5. **Oracle (`theoretical_reference`)**:
   - Per-prompt post-hoc optimal selection: picks the cheapest model that scored correct ($Q \ge 1.0$).
   - If no model answered correctly, falls back to the cheapest candidate and explicitly records the unanswerable prompt.
   - Represents the theoretical upper bound.

### Data Partitioning & Leakage Prevention

- Prompts are partitioned deterministically using a fixed random seed (`--seed 42`) and stratified by `task_type`.
- Partition metadata is saved to `reports/train_test_split.json`.
- Strict zero-leakage enforcement: deployable policies receive only training prompt records for policy derivation.

### Failure & Evaluation Accounting

- Execution failures (`execution_success == False`) and unscored rows (`evaluation_status != 'scored'`) are preserved and reported separately.
- Incurred costs are preserved across all candidate executions.
- Missing evaluations are never coerced to $0.0$ / incorrect. Quality averages reflect scored evaluations, with evaluation coverage tracked per baseline.

### Dual-Engine Visualization (`reports/cost_vs_quality.png`)

- Implements dual rendering engines: uses Matplotlib when available, with an automatic fallback to Pillow (`pillow>=12.0.0`) when running in environments without Matplotlib.
- **Concentric Markers**: Overlapping / coincident baselines are rendered with concentric multi-shape markers (Oracle star, Always-Best square, Static Table diamond, Always-Cheapest circle) at exact data coordinates without artificial jitter.
- **Intelligent Callout Cards**: Coincident baselines are grouped into a structured callout card with angled leader lines, eliminating text collisions.
- **Microdollar Axis Formatting**: X-axis tick marks are formatted to 6 decimal places (e.g. `$0.000010`, `$0.000021`), ensuring microdollar differences are clearly visible.
- **Exploratory Status Warning**: Prominently displays the red disclaimer banner whenever data coverage is sparse.

### Real-Data Coverage Status (`INSUFFICIENT_DATA`)

The active benchmark configuration (`m1-efa7ec3afd4f`) in `data/processed/matrix.parquet` currently contains runs for only **two prompts** (`math_001` and `math_002`), covering only **1 of 6 task categories** (`math`). A 50/50 partition yields 1 training prompt and 1 test prompt, leaving 5 categories unrepresented. Results on the active matrix are automatically flagged as `INSUFFICIENT_DATA` / `EXPLORATORY`. No statistical claims or 6-category evaluations are asserted until full pilot benchmark runs are authorized.

### CLI Usage & Reproduction

```powershell
# Run baseline evaluation on active benchmark data
python -m eval.baselines --config-id m1-efa7ec3afd4f --split-ratio 0.5 --seed 42 --random-repetitions 20 --output-dir reports

# Run dedicated M2 baseline test suite (synthetic fixtures, zero API calls)
python -m pytest tests_v2/test_baselines.py -v

# Run entire test suite
python -m pytest tests_v2 -v
```

Generated artifacts:
- `reports/train_test_split.json`: Partition metadata and sufficiency status.
- `reports/baseline_comparison.csv`: Tabular comparison metrics.
- `reports/baseline_comparison.parquet`: Serialized baseline results.
- `reports/cost_vs_quality.png`: Visual scatter plot.

## Intelligent routing & empirical decision engine (M3)

Phase M3 develops an offline-trained, empirical ML routing system that dynamically predicts model correctness and selects optimal candidate models under configurable cost-quality preferences. It introduces two machine-learning routers, a mathematical utility decision engine, comparative Pareto evaluation against all five M2 baselines, and comprehensive operational robustness stress testing.

### Architecture Overview

```text
Input Prompt ───────────────────────────────────────────────┐
  │                                                         │
  ▼                                                         ▼
[Embedding Encoder] (all-MiniLM-L6-v2)            [Feature Extraction]
  │  Dense 384-d semantic vector                    │  Prompt length, code indicator,
  │                                                 │  one-hot task category
  ├─────────────────────────────────────────────────┘
  │
  ├──► kNN Router (k=3, cosine similarity, distance weighting)
  │      └── P(correct | prompt, m) for each candidate model
  │
  └──► Learned Router (L2 Logistic Regression per model)
         └── P(correct | prompt, m) with Beta-smoothed empirical fallback
  │
  ▼
[Lambda Utility Decision Engine]
  │  Expected Cost Estimand: attempted-call expenditures per model
  │  Cost Normalizer: Candidate-set-relative min-max, max-scale, or raw USD
  │  Utility Rule: U(m, x) = P(correct | m, x) - lambda * C_norm(m)
  ▼
Optimal Candidate Model Selection (argmax_m U(m, x))
```

### 1. Embedding & Feature Architecture (Step 1)

Implemented in [`eval/routers/encoders.py`](eval/routers/encoders.py):
- **`BaseEncoder`**: Extensible abstract base class enforcing `encode_queries(texts)` and `encode_prompts(prompts)`.
- **`SentenceTransformerEncoder`**: Local offline encoder using `all-MiniLM-L6-v2` with strict local cache validation (`local_files_only=True`). Fails gracefully if weights are missing, preventing silent network calls.
- **`DeterministicTestEncoder`**: Lightweight, reproducible offline encoder projecting text via SHA-256 into arbitrary-dimension unit vectors for testing.

### 2. Dataset Preparation & Repeated-Run Handling (Step 2)

Implemented in [`eval/routers/data_prep.py`](eval/routers/data_prep.py):
- **`prepare_routing_dataset()`**: Extracts structured `PromptRecord` and `ModelOutcomeRecord` objects from the benchmark matrix.
- **Empirical Probability Formulation**: Preserves individual run-level observations and estimates empirical quality as:
  $$q(\text{prompt}, m) = \frac{\text{correct\_runs}}{\text{evaluated\_runs}}$$
  Prevents majority-voting collapse or arbitrary binary thresholding.
- **Prompt-Level Isolation**: Prompts are split deterministically across train/val/test partitions. All candidate runs for a given prompt remain strictly in the same split.
- **Status Preservation**: Failed (`execution_success == False`) and unscored (`evaluation_status != 'scored'`) records are preserved as unavailable labels (`label=None`), never coerced to incorrect or zero.
- **Data Sufficiency Assessment**: Identifies small matrices (< 10 prompts or < 2 categories) and tags them as `INSUFFICIENT_DATA`.

### 3. kNN Correctness Predictor (Step 3)

Implemented in [`eval/routers/knn_router.py`](eval/routers/knn_router.py):
- Non-parametric correctness predictor estimating $P(\text{correct} \mid \text{prompt}, m)$ from the $k$ nearest historical training prompts in embedding space.
- Supports `cosine` or `euclidean` distance metrics, with configurable weighting (`distance` or `uniform`).
- Distance weighting handles exact matches ($d \to 0$) safely with numerical epsilon offsets.
- Fallback mechanism defaults to global empirical model accuracy when historical coverage is limited.

### 4. Learned Router with Beta-Smoothed Fallback (Step 4)

Implemented in [`eval/routers/learned_router.py`](eval/routers/learned_router.py):
- Feature representation concatenates prompt embeddings, normalized prompt length, code-presence indicator, and one-hot task categories.
- Fits independent, L2-regularized Logistic Regression classifiers per candidate model using `scikit-learn`.
- `StandardScaler` transformations are fit strictly on training data.
- Handles fractional empirical quality $q(\text{prompt}, m) \in [0, 1]$ via sample-weighted binary cross-entropy.
- **Beta-Smoothed Empirical Fallback**:
  When training data for a model contains single-class outcomes or insufficient diversity, applies a Bayesian Beta-smoothed fallback:
  $$P(\text{correct}) = \frac{\text{successes} + \alpha}{\text{evaluated\_observations} + \alpha + \beta} \quad (\alpha=1.0, \beta=1.0)$$
  Eliminates unjustified certainty from small samples (e.g. 1 success / 0 failures yields 0.667 rather than 1.0; 0 successes / 1 failure yields 0.333 rather than 0.0).

### 5. Lambda Utility Decision Engine (Step 5)

Implemented in [`eval/routers/decision.py`](eval/routers/decision.py):
- Combines predicted correctness with normalized expected model costs via the utility rule:
  $$U(m, x) = \hat{Q}(m, x) - \lambda \cdot c_{\text{norm}}(m)$$
  $$\text{selected\_model} = \arg\max_{m \in \mathcal{M}} U(m, x)$$
- **Provisional Presets**: `auto:quality` ($\lambda=0.0$), `auto:balanced` ($\lambda=0.5$), and `auto:cheap` ($\lambda=2.0$).
- **Expected-Cost Estimand**: Models expected expenditure per attempted call. Includes valid observed costs across all attempted calls (including execution failures and unscored runs that incur provider charges).
- **Equal Prompt Weighting**: Computes average cost per unique prompt-model pair first, then averages across unique prompts, preventing heavily repeated prompts from distorting cost estimates.
- **Cost Normalization Strategies**:
  - `min_max`: $(c - \min c) / (\max c - \min c)$ (strictly candidate-set-relative).
  - `max_scale`: $c / \max c$.
  - `raw`: Preserves unscaled USD values (effective cost penalty $\lambda \cdot c$).

### 6. Comparative Evaluation & Pareto Sweeps (Step 6)

Implemented in [`eval/router_evaluation.py`](eval/router_evaluation.py):
- Rigorous head-to-head comparison across all seven strategies on the exact same held-out test prompts:
  1. *Always-Cheapest* (M2 Deployable Baseline)
  2. *Static Task-Type Table* (M2 Deployable Baseline)
  3. *Random at Matched Cost* (M2 Deployable Baseline)
  4. *kNN Router* (M3 Deployable Router)
  5. *Learned Logistic Regression Router* (M3 Deployable Router)
  6. *Always-Best* (M2 Post-Hoc Reference)
  7. *Oracle* (M2 Theoretical Upper Bound)
- **Lambda Sweeps**: Sweeps $\lambda \in [0.0, 10.0]$ across 21 points to map the empirical Pareto efficiency frontier (Cost vs. Quality).
- **Evaluation Integrity**: Quality denominators reflect actual evaluated/scored attempts; missing outcomes and execution failures are tracked via coverage metrics without imputation.

### 7. Router Robustness & Generalization Evaluation (Step 7)

Implemented in [`eval/router_robustness.py`](eval/router_robustness.py):
- Reproducible offline stress-testing framework investigating operational edge cases:
  - **Model Availability Stress**: Simulates provider outages, partial availability pools, single-candidate fallback, and graceful handling of empty candidate sets.
  - **Cost Sensitivity**: Evaluates decision-time cost multipliers ($0.5\times$ to $3.0\times$) and non-uniform single-candidate cost shocks. Mathematically proves and tests uniform cost invariance under `min_max` and `max_scale`.
  - **Lambda Sensitivity**: Measures transition stability and adjacent-lambda selection shifts.
  - **Distribution Shift Methodology**: Static dataset audit on benchmark data transparently reports `INSUFFICIENT_DATA` / `EXPLORATORY`. Offline controlled simulation (`simulate_controlled_distribution_shift`) tests synthetic prompt length shifts and unseen task categories.
  - **Granular Accounting**: Distinctly tallies `deliberately_unavailable`, `missing_prediction`, `execution_failure`, `unscored`, and `valid_prediction` events.
- **Dual-Panel Visualization**: Generates `reports/router_robustness.png` illustrating cost sensitivity and availability resilience.

### Verification & Reproduction

```powershell
# Run M3 router unit and integration tests
python -m pytest tests_v2/test_encoders.py tests_v2/test_data_prep.py `
  tests_v2/test_knn_router.py tests_v2/test_learned_router.py `
  tests_v2/test_router_decision.py tests_v2/test_router_evaluation.py `
  tests_v2/test_router_robustness.py -v

# Run comparative Pareto sweep evaluation
python -m eval.router_evaluation --config-id m1-efa7ec3afd4f

# Run operational robustness evaluation
python -m eval.router_robustness --config-id m1-efa7ec3afd4f

# Run full test suite (226 tests, 100% offline, zero network calls)
python -m pytest tests_v2 -v
```

Generated artifacts:
- `reports/router_comparison.csv` & `reports/router_comparison.parquet`: Multi-strategy comparative evaluation data.
- `reports/router_evaluation_summary.json`: Strategy summary, Pareto status, and data sufficiency audit.
- `reports/router_pareto_curve.png`: High-resolution dual-engine Pareto frontier curve.
- `reports/router_robustness.csv` & `reports/router_robustness.parquet`: Operational stress-test scenario logs (68 scenarios).
- `reports/router_robustness_summary.json`: Detailed robustness audit and stability metrics.
- `reports/router_robustness.png`: Two-panel cost sensitivity and availability resilience plot.

## Project layout

```text
main.py                          FastAPI endpoint and request orchestration
routing_policy.py                Candidate filtering and routing decisions
router.py / model_registry.py    Complexity classification and model configuration
budget.py / models.py             Cost tracking, reservations, and provider dispatch
providers/                        Groq and Gemini adapters
dynamic_quality.py / logger.py    Supabase quality lookup and telemetry
dashboard.py / query_console.py   Streamlit dashboard and query UI
llm_cost_autopilot_client/        Python SDK
migrations/                       Supabase migrations
test_budget_enforcement.py        Mocked regression tests
config/models.yaml                V2 model registry and benchmark pricing
model_registry_v2.py              Enabled V2 model definitions
executor_v2.py                    LiteLLM execution and cost/latency telemetry
eval/dataset_registry.py          Trusted dataset approval registry & provenance
eval/runner.py                    Custom benchmark execution
eval/metrics.py                   Deterministic output evaluation
eval/integrity.py                 Read-only benchmark integrity reports
eval/external_datasets.py         Approved local external-dataset adapters
eval/external_runner.py           Separate validation/final-test runner
eval/routerbench_import.py        Blocked RouterBench normalizer scaffolding
eval/baselines.py                 M2 baseline evaluation engine & visualization
eval/routers/encoders.py          M3 BaseEncoder, SentenceTransformer & deterministic encoders
eval/routers/data_prep.py         M3 dataset preparation and repeated-run handling
eval/routers/knn_router.py        M3 kNN correctness predictor
eval/routers/learned_router.py    M3 Learned Logistic Regression router with Beta fallback
eval/routers/decision.py          M3 Lambda utility decision engine & cost normalizers
eval/router_evaluation.py         M3 7-strategy comparative evaluation & Pareto sweeps
eval/router_robustness.py         M3 router robustness & operational stress testing
reports/                          M2 and M3 comparison reports, Pareto curves, and plots
tests_v2/                         Full test suite (226 tests, offline and zero provider spend)
```

## Current limitations

- Budget limits are per request, not shared quotas; provider charges cannot be guaranteed from estimates.
- API-key validation and telemetry persistence use synchronous Supabase calls.
- The repository's migration is additive but does not include the initial database schema.
- Provider/classifier evaluation scripts may make live calls and incur charges; their results depend on providers and current model pricing.
- Benchmark matrix coverage in `data/processed/matrix.parquet` currently contains only two math prompts under the active configuration, leaving 5 of 6 task categories pending authorized live execution. Real-data M2 baseline and M3 router evaluations are therefore automatically flagged as `INSUFFICIENT_DATA` / `EXPLORATORY`. Statistical claims on out-of-domain generalization require broader benchmark collection.
