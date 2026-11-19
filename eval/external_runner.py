"""Separate runner for the approved, locally normalized external datasets."""

import argparse
import asyncio
import hashlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import litellm
import pandas as pd

from eval.external_datasets import (
    DATASET_SPECS,
    EXTERNAL_DATA_ROOT,
    NORMALIZED_COLUMNS,
    PROJECT_ROOT,
    validate_split_integrity,
)
from eval.metrics import EvaluationResult, evaluate_output
from executor_v2 import execute_model, _API_KEY_ENV_VARS
from model_registry_v2 import MODELS


EXTERNAL_RESULTS_ROOT = (
    PROJECT_ROOT / "data" / "processed" / "external_benchmarks"
)

_CORE_RESULT_COLUMNS = [
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

_PROVENANCE_COLUMNS = [
    "dataset_key",
    "dataset_id",
    "dataset_revision",
    "dataset_config",
    "license_id",
    "license_url",
    "attribution",
    "citation_url",
    "source_split",
    "source_file_sha256",
    "source_file_format",
    "source_manifest_json",
    "benchmark_split",
    "source_id",
    "source_question",
    "source_reference_answer_json",
    "source_record_json",
    "choices",
    "choice_labels",
    "test_cases",
    "test_imports",
    "test_setup_code",
    "source_file_hashes_json",
    "evaluation_manifest_id",
    "evaluation_manifest_json",
    "adapter_version",
    "split_policy_version",
    "prompt_template_version",
    "external_benchmark_config_id",
]

EVALUATOR_CONFIG_VERSION = "external-evaluator-v1"
EVALUATION_MANIFEST_VERSION = "1"

EXTERNAL_RESULT_COLUMNS = _CORE_RESULT_COLUMNS + _PROVENANCE_COLUMNS


def load_normalized_dataset(
    dataset_key: str,
    *,
    root: str | Path | None = None,
    trusted_registry: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    spec = DATASET_SPECS.get(dataset_key)
    if spec is None:
        raise ValueError(f"Unsupported external dataset: {dataset_key}")
    base = Path(root) if root is not None else EXTERNAL_DATA_ROOT
    directory = (
        base / dataset_key / spec["revision"] / spec["config"] / "normalized"
    )
    records: list[dict[str, Any]] = []
    for split in ("train", "validation", "test", "few_shot"):
        path = directory / f"{split}.parquet"
        if not path.exists():
            if split in {"train", "validation", "test"}:
                raise FileNotFoundError(
                    f"Required normalized split is missing: {path}"
                )
            continue
        frame = pd.read_parquet(path)
        if not set(NORMALIZED_COLUMNS).issubset(frame.columns):
            raise ValueError(f"Normalized split has an invalid schema: {path}")
        records.extend(frame.to_dict(orient="records"))
    validate_split_integrity(records, trusted_registry=trusted_registry)
    return records


def _dataset_fingerprint(records: list[Mapping[str, Any]]) -> str:
    source_rows = [
        {
            "prompt_id": record["prompt_id"],
            "benchmark_split": record["benchmark_split"],
            "source_record_json": record["source_record_json"],
            "source_file_sha256": record["source_file_sha256"],
            "source_manifest_json": record["source_manifest_json"],
            "prompt": record["prompt"],
            "reference_answer": record["reference_answer"],
            "evaluation_method": record["evaluation_method"],
            "choices": record["choices"],
            "choice_labels": record["choice_labels"],
            "test_cases": record["test_cases"],
            "test_imports": record["test_imports"],
            "test_setup_code": record["test_setup_code"],
        }
        for record in sorted(records, key=lambda row: row["prompt_id"])
    ]
    serialized = json.dumps(
        source_rows,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _source_file_hashes(records: list[Mapping[str, Any]]) -> dict[str, str]:
    hashes = {}
    for record in records:
        source_split = record["source_split"]
        source_hash = record["source_file_sha256"]
        prior_hash = hashes.get(source_split)
        if prior_hash is not None and prior_hash != source_hash:
            raise ValueError(
                f"Source split {source_split!r} has inconsistent file hashes."
            )
        hashes[source_split] = source_hash
    return dict(sorted(hashes.items()))


def _external_config_id(
    records: list[Mapping[str, Any]],
    models: Mapping[str, dict[str, Any]],
    *,
    split: str = "test",
) -> str:
    spec = DATASET_SPECS[records[0]["dataset_key"]]
    model_fields = (
        "id",
        "provider",
        "model_name",
        "input_price_per_1m",
        "output_price_per_1m",
        "context_window",
    )
    configuration = {
        "dataset_id": spec["dataset_id"],
        "dataset_revision": spec["revision"],
        "dataset_config": spec["config"],
        "evaluation_split": split,
        "dataset_fingerprint": _dataset_fingerprint(records),
        "adapter_version": records[0]["adapter_version"],
        "split_policy_version": records[0]["split_policy_version"],
        "prompt_template_version": records[0]["prompt_template_version"],
        "models": [
            {field: model[field] for field in model_fields}
            for model in sorted(models.values(), key=lambda item: item["id"])
        ],
    }
    serialized = json.dumps(configuration, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:16]
    return f"external-{records[0]['dataset_key']}-{digest}"


def create_frozen_evaluation_manifest(
    records: list[Mapping[str, Any]],
    models: Mapping[str, dict[str, Any]],
    *,
    runs: int,
    split: str = "test",
) -> dict[str, Any]:
    if isinstance(runs, bool) or runs < 1:
        raise ValueError("runs must be at least 1.")
    if not records:
        raise ValueError("Cannot freeze evaluation settings for an empty dataset.")
    dataset_key = records[0]["dataset_key"]
    spec = DATASET_SPECS[dataset_key]
    model_fields = (
        "id",
        "provider",
        "model_name",
        "input_price_per_1m",
        "output_price_per_1m",
        "context_window",
    )
    executor_path = PROJECT_ROOT / "executor_v2.py"
    executor_sha = (
        hashlib.sha256(executor_path.read_bytes()).hexdigest()
        if executor_path.exists()
        else "unknown"
    )
    pyproject_path = PROJECT_ROOT / "pyproject.toml"
    pyproject_sha = (
        hashlib.sha256(pyproject_path.read_bytes()).hexdigest()
        if pyproject_path.exists()
        else None
    )
    uv_lock_path = PROJECT_ROOT / "uv.lock"
    uv_lock_sha = (
        hashlib.sha256(uv_lock_path.read_bytes()).hexdigest()
        if uv_lock_path.exists()
        else None
    )
    model_config_list = [
        {field: model[field] for field in model_fields}
        for model in sorted(models.values(), key=lambda item: item["id"])
    ]
    model_registry_fingerprint = hashlib.sha256(
        json.dumps(model_config_list, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()

    manifest = {
        "manifest_version": EVALUATION_MANIFEST_VERSION,
        "dataset_key": dataset_key,
        "dataset_id": spec["dataset_id"],
        "dataset_revision": spec["revision"],
        "dataset_config": spec["config"],
        "evaluation_split": split,
        "dataset_fingerprint": _dataset_fingerprint(records),
        "source_file_sha256s": _source_file_hashes(records),
        "model_ids": sorted(models),
        "model_configuration": model_config_list,
        "execution_configuration": {
            "executor_version": "executor_v2",
            "executor_source_sha256": executor_sha,
            "litellm_version": getattr(litellm, "__version__", "unknown"),
            "python_implementation": platform.python_implementation(),
            "python_version": platform.python_version(),
            "pyproject_toml_sha256": pyproject_sha,
            "uv_lock_sha256": uv_lock_sha,
            "providers": sorted({model["provider"] for model in models.values()}),
            "provider_configurations": [
                {
                    "provider": provider,
                    "api_key_env_vars": list(_API_KEY_ENV_VARS.get(provider, ())),
                }
                for provider in sorted({model["provider"] for model in models.values()})
            ],
            "model_registry_fingerprint": model_registry_fingerprint,
        },
        "prompt_settings": {
            "template_versions": sorted({
                record["prompt_template_version"] for record in records
            }),
            "split_policy_version": records[0]["split_policy_version"],
        },
        "run_count": runs,
        "evaluator_configuration": {
            "version": EVALUATOR_CONFIG_VERSION,
            "implementation_sha256": hashlib.sha256(
                Path(__file__).with_name("metrics.py").read_bytes()
            ).hexdigest(),
            "methods": sorted({
                record["evaluation_method"] for record in records
            }),
            "code_execution": "unsupported_without_isolation_sandbox",
        },
    }
    serialized = json.dumps(
        manifest,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    manifest["manifest_id"] = hashlib.sha256(
        serialized.encode("utf-8")
    ).hexdigest()
    return manifest


def write_frozen_evaluation_manifest(
    path: str | Path,
    manifest: Mapping[str, Any],
) -> Path:
    manifest_path = Path(path)
    if manifest_path.exists():
        raise FileExistsError(
            f"Refusing to overwrite frozen evaluation manifest: {manifest_path}"
        )
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest_path


def _validate_frozen_manifest(
    supplied: Mapping[str, Any] | None,
    expected: Mapping[str, Any],
) -> None:
    if not isinstance(supplied, Mapping):
        raise ValueError("A frozen evaluation manifest is required for test execution.")
    if dict(supplied) != dict(expected):
        raise ValueError(
            "Frozen evaluation manifest does not match the dataset, source hashes, "
            "model configuration, prompt settings, run count, or evaluator."
        )


def _select_models(model_ids: list[str] | None) -> dict[str, dict[str, Any]]:
    if model_ids is None:
        selected = dict(MODELS)
    else:
        unknown = set(model_ids).difference(MODELS)
        if unknown:
            raise ValueError(f"Unknown or disabled model IDs: {sorted(unknown)}")
        selected = {model_id: MODELS[model_id] for model_id in model_ids}
    if not selected:
        raise ValueError("No enabled models were selected.")
    return selected


def _result_row(
    prompt: Mapping[str, Any],
    model_id: str,
    model_config: Mapping[str, Any],
    run_number: int,
    execution: Mapping[str, Any] | None,
    evaluation: EvaluationResult,
    config_id: str,
    evaluation_manifest: Mapping[str, Any] | None,
    source_file_hashes: Mapping[str, str],
) -> dict[str, Any]:
    result = execution or {
        "output": None,
        "provider": model_config["provider"],
        "input_tokens": 0,
        "output_tokens": 0,
        "cost": 0.0,
        "latency_ms": 0.0,
        "success": False,
        "error": "Model execution did not succeed.",
    }
    row = {
        "prompt_id": prompt["prompt_id"],
        "task_type": prompt["task_type"],
        "model_id": model_id,
        "provider": model_config["provider"],
        "run": run_number,
        "correct": evaluation.correct,
        "execution_success": bool(result["success"]),
        "input_tokens": result["input_tokens"],
        "output_tokens": result["output_tokens"],
        "cost": result["cost"],
        "latency_ms": result["latency_ms"],
        "output": result["output"],
        "error": result["error"],
        "evaluation_method": prompt["evaluation_method"],
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "benchmark_config_id": config_id,
        "evaluation_status": evaluation.evaluation_status,
        "score": evaluation.score,
        "evaluation_error": evaluation.evaluation_error,
        "evaluation_details": evaluation.evaluation_details,
    }
    row.update({
        column: prompt[column]
        for column in _PROVENANCE_COLUMNS
        if column in prompt
    })
    row["external_benchmark_config_id"] = config_id
    row["source_file_hashes_json"] = json.dumps(
        source_file_hashes,
        sort_keys=True,
        separators=(",", ":"),
    )
    row["evaluation_manifest_id"] = (
        evaluation_manifest["manifest_id"]
        if evaluation_manifest is not None
        else None
    )
    row["evaluation_manifest_json"] = (
        json.dumps(
            evaluation_manifest,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        if evaluation_manifest is not None
        else None
    )
    return row


async def run_external_benchmark(
    records: list[Mapping[str, Any]],
    *,
    model_ids: list[str] | None = None,
    runs: int = 1,
    split: str = "validation",
    execute: bool = False,
    authorize_final_evaluation: bool = False,
    evaluation_manifest_path: str | Path | None = None,
    output_root: str | Path | None = None,
) -> dict[str, Any]:
    """Dry-run by default; test execution requires authorization and a frozen manifest."""
    if isinstance(runs, bool) or runs < 1:
        raise ValueError("runs must be at least 1.")
    if split not in {"validation", "test"}:
        raise ValueError("split must be 'validation' or 'test'.")
    validate_split_integrity(records)
    selected_models = _select_models(model_ids)
    split_records = [
        record for record in records if record["benchmark_split"] == split
    ]
    if not split_records:
        raise ValueError(f"No {split} records were supplied.")

    dataset_key = records[0]["dataset_key"]
    source_file_hashes = _source_file_hashes(records)
    config_id = _external_config_id(records, selected_models, split=split)
    frozen_manifest = None
    if split == "test":
        frozen_manifest = create_frozen_evaluation_manifest(
            records,
            selected_models,
            runs=runs,
            split=split,
        )
        if execute:
            if not authorize_final_evaluation:
                raise ValueError(
                    "Final test execution requires explicit authorization."
                )
            if evaluation_manifest_path is None:
                raise ValueError(
                    "Final test execution requires a persisted frozen manifest."
                )
            try:
                supplied_manifest = json.loads(
                    Path(evaluation_manifest_path).read_text(encoding="utf-8")
                )
            except (OSError, json.JSONDecodeError) as exc:
                raise ValueError(
                    f"Cannot read frozen evaluation manifest: {exc}"
                ) from exc
            _validate_frozen_manifest(supplied_manifest, frozen_manifest)
    elif authorize_final_evaluation:
        raise ValueError("Final-evaluation authorization is only valid for test.")

    calls = len(split_records) * len(selected_models) * runs
    result_root = Path(output_root) if output_root is not None else EXTERNAL_RESULTS_ROOT
    output_path = result_root / dataset_key / split / f"{config_id}.parquet"
    if output_path.resolve() == (
        PROJECT_ROOT / "data" / "processed" / "matrix.parquet"
    ).resolve():
        raise ValueError("External results cannot overwrite the custom benchmark.")
    if split == "test" and execute and output_path.exists():
        raise FileExistsError(
            "Final test results already exist for this configuration; refusing "
            "to append or rerun under the same frozen settings."
        )

    summary = {
        "dataset_key": dataset_key,
        "split": split,
        "benchmark_config_id": config_id,
        "records": len(split_records),
        "model_count": len(selected_models),
        "runs": runs,
        "expected_executions": calls,
        "output_path": str(output_path),
        "executed": execute,
        "dataset_revision": DATASET_SPECS[dataset_key]["revision"],
        "source_file_sha256s": source_file_hashes,
        "evaluation_manifest_id": (
            frozen_manifest["manifest_id"]
            if frozen_manifest is not None
            else None
        ),
    }
    if not execute:
        return summary

    results = []
    for prompt in split_records:
        for model_id, model_config in selected_models.items():
            for run_number in range(1, runs + 1):
                execution = await execute_model(
                    model_id=model_id,
                    prompt=prompt["prompt"],
                )
                if execution["success"]:
                    evaluation_record = dict(prompt)
                    evaluation = await evaluate_output(
                        evaluation_record,
                        execution["output"],
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
                results.append(
                    _result_row(
                        prompt,
                        model_id,
                        model_config,
                        run_number,
                        execution,
                        evaluation,
                        config_id,
                        frozen_manifest,
                        source_file_hashes,
                    )
                )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    new_results = pd.DataFrame(results, columns=EXTERNAL_RESULT_COLUMNS)
    if output_path.exists():
        if split == "test":
            raise FileExistsError(
                "Final test results appeared before the result could be saved."
            )
        old_results = pd.read_parquet(output_path)
        new_results = pd.concat([old_results, new_results], ignore_index=True)
    new_results.to_parquet(output_path, index=False)
    summary["rows_written"] = len(results)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a separate external benchmark; defaults to dry-run."
    )
    parser.add_argument("--dataset", choices=sorted(DATASET_SPECS), required=True)
    parser.add_argument("--models", nargs="*", default=None)
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Make provider calls for the selected validation or final test split.",
    )
    parser.add_argument(
        "--authorize-final-evaluation",
        action="store_true",
        help="Explicitly authorize the frozen final test evaluation.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        help="Frozen evaluation-manifest JSON required for final test execution.",
    )
    parser.add_argument(
        "--freeze-manifest",
        type=Path,
        help="Create a new immutable test-evaluation manifest and exit.",
    )
    args = parser.parse_args()
    if args.runs < 1:
        parser.error("--runs must be at least 1")
    records = load_normalized_dataset(args.dataset)
    try:
        selected_models = _select_models(args.models)
        if args.freeze_manifest is not None:
            if args.split != "test":
                parser.error("--freeze-manifest requires --split test.")
            if args.execute:
                parser.error("Freezing a manifest must not execute providers.")
            manifest = create_frozen_evaluation_manifest(
                records,
                selected_models,
                runs=args.runs,
                split=args.split,
            )
            path = write_frozen_evaluation_manifest(args.freeze_manifest, manifest)
            print(f"Frozen final-evaluation manifest: {path}")
            print(f"Manifest ID: {manifest['manifest_id']}")
            return

        if args.split == "test" and args.execute:
            if args.manifest is None:
                parser.error("--split test --execute requires --manifest.")
            if not args.authorize_final_evaluation:
                parser.error(
                    "--split test --execute requires "
                    "--authorize-final-evaluation."
                )
        summary = asyncio.run(
            run_external_benchmark(
                records,
                model_ids=args.models,
                runs=args.runs,
                split=args.split,
                execute=args.execute,
                authorize_final_evaluation=args.authorize_final_evaluation,
                evaluation_manifest_path=args.manifest,
            )
        )
    except (OSError, json.JSONDecodeError, ValueError, FileExistsError) as exc:
        parser.error(str(exc))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
