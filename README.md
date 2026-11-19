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
attribution. Every source split requires a separate JSON manifest naming the
dataset, approved revision, source-file SHA-256, source split, format, and
attribution. The source bytes are hashed and compared before parsing; the hash
and verified manifest are retained in normalized records and run metadata.
Arbitrary local files cannot be labeled with an approved revision without a
matching manifest and byte hash. MBPP normalization appends a versioned code-only response
instruction to the prompt; it does not rewrite the problem, reference code,
imports, or test cases. Exact duplicate prompts and same-source-ID
cross-split matches are hard failures. Token-set near duplicates at Jaccard
similarity 0.8 or higher produce review warnings containing both source IDs
and the similarity; records are not removed or modified. Short and entity-
substitution questions can trigger false-positive warnings, so inspect them
before deciding whether they represent leakage. This is a lexical check, not
semantic paraphrase detection.

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
