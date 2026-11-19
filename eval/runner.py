
import argparse
import asyncio
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from eval.dataset import load_dataset
from eval.metrics import EvaluationResult, evaluate_output
from executor_v2 import execute_model
from model_registry_v2 import MODELS


PROJECT_ROOT = Path(__file__).resolve().parents[1]

OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "matrix.parquet"

RESULT_COLUMNS = [
    "prompt_id",
    "task_type",
    "model_id",
    "provider",
    "run",
    "correct",
    "execution_success",
    "input_tokens",
    "output_tokens",
    "cost",
    "latency_ms",
    "output",
    "error",
    "evaluation_method",
    "timestamp",
    "benchmark_config_id",
    "evaluation_status",
    "score",
    "evaluation_error",
    "evaluation_details",
]


def _benchmark_config_id(models):
    fields = (
        "id",
        "provider",
        "model_name",
        "input_price_per_1m",
        "output_price_per_1m",
        "context_window",
    )
    configuration = [
        {field: model[field] for field in fields}
        for model in sorted(models.values(), key=lambda item: item["id"])
    ]
    serialized = json.dumps(configuration, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:12]
    return f"m1-{digest}"


async def run_benchmark(
    prompt_limit=None,
    model_ids=None,
    runs=2,
    dry_run=False,
):
    dataset = load_dataset()

    if prompt_limit is not None:
        dataset = dataset[:prompt_limit]

    selected_models = {
        model_id: config
        for model_id, config in MODELS.items()
        if model_ids is None or model_id in model_ids
    }

    if not selected_models:
        raise ValueError("No matching enabled models found.")

    benchmark_config_id = _benchmark_config_id(selected_models)

    if dry_run:
        expected = len(dataset) * len(selected_models) * runs
        print("DRY RUN — no API calls will be made.")
        print(f"Prompts: {len(dataset)}")
        print(f"Models: {len(selected_models)}")
        print(f"Runs per prompt/model: {runs}")
        print(f"Expected executions: {expected}")
        print(f"Benchmark configuration: {benchmark_config_id}")
        return

    results = []

    total = len(dataset) * len(selected_models) * runs
    completed = 0

    for prompt in dataset:
        for model_id, model_config in selected_models.items():
            for run_number in range(1, runs + 1):
                completed += 1

                print(
                    f"[{completed}/{total}] "
                    f"{prompt['prompt_id']} -> {model_id} "
                    f"(run {run_number})"
                )

                result = await execute_model(
                    model_id=model_id,
                    prompt=prompt["prompt"],
                )

                if result["success"]:
                    evaluation = await evaluate_output(
                        prompt,
                        result["output"],
                        model_id=model_id,
                    )
                else:
                    evaluation = EvaluationResult(
                        evaluation_status="execution_failed",
                        score=None,
                        correct=None,
                        evaluation_method=prompt["evaluation_method"],
                        evaluation_error="Model execution did not succeed.",
                    )

                results.append({
                    "prompt_id": prompt["prompt_id"],
                    "task_type": prompt["task_type"],
                    "model_id": model_id,
                    "provider": model_config["provider"],
                    "run": run_number,
                    "correct": evaluation.correct,
                    "execution_success": result["success"],
                    "input_tokens": result["input_tokens"],
                    "output_tokens": result["output_tokens"],
                    "cost": result["cost"],
                    "latency_ms": result["latency_ms"],
                    "output": result["output"],
                    "error": result["error"],
                    "evaluation_method": prompt["evaluation_method"],
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "benchmark_config_id": benchmark_config_id,
                    "evaluation_status": evaluation.evaluation_status,
                    "score": evaluation.score,
                    "evaluation_error": evaluation.evaluation_error,
                    "evaluation_details": evaluation.evaluation_details,
                })

                print(
                    f"  success={result['success']} | "
                    f"cost=${result['cost']:.8f} | "
                    f"latency={result['latency_ms']} ms"
                )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    new_results = pd.DataFrame(results, columns=RESULT_COLUMNS)

    # Preserve existing benchmark results if the file already exists.
    if OUTPUT_PATH.exists():
        old_results = pd.read_parquet(OUTPUT_PATH)
        if "benchmark_config_id" not in old_results.columns:
            old_results["benchmark_config_id"] = "legacy-unversioned"
        else:
            old_results["benchmark_config_id"] = old_results[
                "benchmark_config_id"
            ].fillna("legacy-unversioned")
        new_results = pd.concat(
            [old_results, new_results],
            ignore_index=True,
        )

    new_results.to_parquet(OUTPUT_PATH, index=False)

    print("\nBenchmark completed.")
    print(f"Rows recorded in file: {len(results)}")
    print(f"Saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the LLM Cost Copilot benchmark."
    )

    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--models", nargs="*", default=None)
    parser.add_argument("--runs", type=int, default=2)
    parser.add_argument("--dry-run", action="store_true")

    args = parser.parse_args()

    if args.runs < 1:
        parser.error("--runs must be at least 1")

    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be at least 1")

    asyncio.run(
        run_benchmark(
            prompt_limit=args.limit,
            model_ids=args.models,
            runs=args.runs,
            dry_run=args.dry_run,
        )
    )
