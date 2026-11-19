import asyncio
import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

import eval.external_runner as external_runner
from eval.dataset_registry import TrustedDatasetEntry, register_trusted_entry
from eval.external_datasets import DATASET_SPECS, normalize_source_splits


def _records(tmp_path):
    train = [
        {
            "question": f"Compute marker_{index}: {index} plus 1.",
            "answer": f"#### {index + 1}",
        }
        for index in range(100)
    ]
    rows_by_split = {
        "train": train,
        "test": [{
            "question": "Compute the unique external test result.",
            "answer": "Work.\n#### 42",
        }],
    }
    spec = DATASET_SPECS["gsm8k"]
    source_paths = {}
    manifests = {}
    for split, rows in rows_by_split.items():
        path = tmp_path / f"{split}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        content = "".join(
            json.dumps(row, separators=(",", ":")) + "\n"
            for row in rows
        ).encode("utf-8")
        path.write_bytes(content)
        source_paths[split] = path
        content_hash = hashlib.sha256(content).hexdigest()
        entry_id = f"{spec['dataset_id']}:{spec['revision']}:{split}:jsonl"
        register_trusted_entry(
            TrustedDatasetEntry(
                entry_id=entry_id,
                dataset_name=spec["dataset_id"],
                publisher="OpenAI",
                approved_revision=spec["revision"],
                split=split,
                source_url="https://github.com/openai/grade-school-math",
                sha256=content_hash,
                file_format="jsonl",
                attribution=spec["attribution"],
                license_approval_reference="TEST-MIT-APPROVED",
                approval_status="approved",
            )
        )
        manifests[split] = {
            "registry_entry_id": entry_id,
            "dataset_name": spec["dataset_id"],
            "approved_revision": spec["revision"],
            "source_file_sha256": content_hash,
            "split": split,
            "format": "jsonl",
            "attribution": spec["attribution"],
        }
    return normalize_source_splits(
        "gsm8k",
        source_paths,
        manifests,
    )


def _model_config():
    return {
        "gpt_oss_20b": {
            "id": "gpt_oss_20b",
            "provider": "groq",
            "model_name": "openai/gpt-oss-20b",
            "input_price_per_1m": 0.075,
            "output_price_per_1m": 0.30,
            "context_window": 131072,
        },
    }


def _freeze_manifest(records, models, path, *, runs=1):
    manifest = external_runner.create_frozen_evaluation_manifest(
        records,
        models,
        runs=runs,
    )
    manifest_path = external_runner.write_frozen_evaluation_manifest(
        path,
        manifest,
    )
    return manifest_path, manifest


def test_external_runner_defaults_to_validation_dry_run(monkeypatch, tmp_path):
    records = _records(tmp_path / "source")
    async def unexpected_execution(**kwargs):
        raise AssertionError(f"unexpected provider execution: {kwargs}")

    monkeypatch.setattr(external_runner, "MODELS", _model_config())
    monkeypatch.setattr(external_runner, "execute_model", unexpected_execution)

    summary = asyncio.run(
        external_runner.run_external_benchmark(
            records,
            output_root=tmp_path,
        )
    )

    assert summary["executed"] is False
    assert summary["split"] == "validation"
    assert summary["records"] > 0
    assert summary["expected_executions"] == summary["records"]
    assert summary["benchmark_config_id"].startswith("external-gsm8k-")
    assert not (tmp_path / "gsm8k").exists()


def test_external_runner_executes_only_test_and_keeps_results_separate(
    monkeypatch,
    tmp_path,
):
    records = _records(tmp_path / "source")
    calls = []

    async def fake_execution(*, model_id, prompt):
        calls.append((model_id, prompt))
        return {
            "output": "42",
            "model_id": model_id,
            "provider": "groq",
            "input_tokens": 3,
            "output_tokens": 1,
            "cost": 0.000001,
            "latency_ms": 5.0,
            "success": True,
            "error": None,
        }

    monkeypatch.setattr(external_runner, "MODELS", _model_config())
    monkeypatch.setattr(external_runner, "execute_model", fake_execution)

    manifest_path, manifest = _freeze_manifest(
        records,
        _model_config(),
        tmp_path / "final-evaluation.json",
    )
    summary = asyncio.run(
        external_runner.run_external_benchmark(
            records,
            execute=True,
            split="test",
            authorize_final_evaluation=True,
            evaluation_manifest_path=manifest_path,
            output_root=tmp_path,
        )
    )

    assert len(calls) == 1
    assert calls[0][1] == "Compute the unique external test result."
    output_path = pd.read_parquet(summary["output_path"])
    assert len(output_path) == 1
    assert output_path.iloc[0]["correct"]
    assert output_path.iloc[0]["dataset_key"] == "gsm8k"
    assert output_path.iloc[0]["source_reference_answer_json"] == '"Work.\\n#### 42"'
    assert output_path.iloc[0]["benchmark_split"] == "test"
    assert Path(summary["output_path"]).parent.name == "test"
    assert Path(summary["output_path"]).name.endswith(".parquet")
    assert summary["evaluation_manifest_id"] == manifest["manifest_id"]
    assert output_path.iloc[0]["evaluation_manifest_id"] == manifest["manifest_id"]
    assert set(json.loads(output_path.iloc[0]["source_file_hashes_json"])) == {
        "test",
        "train",
    }


def test_final_test_requires_authorization_and_matching_frozen_manifest(
    monkeypatch,
    tmp_path,
):
    records = _records(tmp_path / "source")
    monkeypatch.setattr(external_runner, "MODELS", _model_config())

    async def unexpected_execution(**kwargs):
        raise AssertionError(f"execution must not start: {kwargs}")

    monkeypatch.setattr(external_runner, "execute_model", unexpected_execution)
    manifest_path, _ = _freeze_manifest(
        records,
        _model_config(),
        tmp_path / "authorization.json",
    )

    with pytest.raises(ValueError, match="explicit authorization"):
        asyncio.run(
            external_runner.run_external_benchmark(
                records,
                split="test",
                execute=True,
                evaluation_manifest_path=manifest_path,
                output_root=tmp_path / "results",
            )
        )

    with pytest.raises(ValueError, match="does not match"):
        asyncio.run(
            external_runner.run_external_benchmark(
                records,
                split="test",
                runs=2,
                execute=True,
                authorize_final_evaluation=True,
                evaluation_manifest_path=manifest_path,
                output_root=tmp_path / "results",
            )
        )


def test_final_test_rejects_changed_model_prompt_or_evaluator_settings(
    monkeypatch,
    tmp_path,
):
    records = _records(tmp_path / "source")
    models = _model_config()
    monkeypatch.setattr(external_runner, "MODELS", models)

    async def unexpected_execution(**kwargs):
        raise AssertionError(f"execution must not start: {kwargs}")

    monkeypatch.setattr(external_runner, "execute_model", unexpected_execution)
    frozen = external_runner.create_frozen_evaluation_manifest(
        records,
        models,
        runs=1,
    )
    changed = json.loads(json.dumps(frozen))
    changed["model_configuration"][0]["model_name"] = "different/model"
    changed_model_path = external_runner.write_frozen_evaluation_manifest(
        tmp_path / "changed-model.json",
        changed,
    )

    with pytest.raises(ValueError, match="does not match"):
        asyncio.run(
            external_runner.run_external_benchmark(
                records,
                split="test",
                execute=True,
                authorize_final_evaluation=True,
                evaluation_manifest_path=changed_model_path,
                output_root=tmp_path / "results",
            )
        )

    changed = json.loads(json.dumps(frozen))
    changed["prompt_settings"]["template_versions"] = ["different-template"]
    changed_prompt_path = external_runner.write_frozen_evaluation_manifest(
        tmp_path / "changed-prompt.json",
        changed,
    )
    with pytest.raises(ValueError, match="does not match"):
        asyncio.run(
            external_runner.run_external_benchmark(
                records,
                split="test",
                execute=True,
                authorize_final_evaluation=True,
                evaluation_manifest_path=changed_prompt_path,
                output_root=tmp_path / "results",
            )
        )


def test_frozen_evaluation_manifest_is_create_once(tmp_path):
    records = _records(tmp_path / "source")
    manifest = external_runner.create_frozen_evaluation_manifest(
        records,
        _model_config(),
        runs=2,
    )
    manifest_path = external_runner.write_frozen_evaluation_manifest(
        tmp_path / "frozen.json",
        manifest,
    )

    assert manifest_path.exists()
    assert json.loads(manifest_path.read_text(encoding="utf-8")) == manifest
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        external_runner.write_frozen_evaluation_manifest(manifest_path, manifest)


def test_validation_execution_writes_only_validation_rows(monkeypatch, tmp_path):
    records = _records(tmp_path / "source")

    async def fake_execution(*, model_id, prompt):
        return {
            "output": "42",
            "model_id": model_id,
            "provider": "groq",
            "input_tokens": 3,
            "output_tokens": 1,
            "cost": 0.000001,
            "latency_ms": 5.0,
            "success": True,
            "error": None,
        }

    monkeypatch.setattr(external_runner, "MODELS", _model_config())
    monkeypatch.setattr(external_runner, "execute_model", fake_execution)
    summary = asyncio.run(
        external_runner.run_external_benchmark(
            records,
            split="validation",
            execute=True,
            output_root=tmp_path / "results",
        )
    )
    saved = pd.read_parquet(summary["output_path"])

    assert summary["split"] == "validation"
    assert set(saved["benchmark_split"]) == {"validation"}
    assert Path(summary["output_path"]).parent.name == "validation"
    assert saved["evaluation_manifest_id"].isna().all()


def test_external_runner_records_failed_execution_as_unscored(
    monkeypatch,
    tmp_path,
):
    records = _records(tmp_path / "source")
    async def failed_execution(*, model_id, prompt):
        return {
            "output": None,
            "model_id": model_id,
            "provider": "groq",
            "input_tokens": 0,
            "output_tokens": 0,
            "cost": 0.0,
            "latency_ms": 1.0,
            "success": False,
            "error": "Provider unavailable.",
        }

    monkeypatch.setattr(external_runner, "MODELS", _model_config())
    monkeypatch.setattr(external_runner, "execute_model", failed_execution)

    manifest_path, _ = _freeze_manifest(
        records,
        _model_config(),
        tmp_path / "failed-run.json",
    )
    summary = asyncio.run(
        external_runner.run_external_benchmark(
            records,
            execute=True,
            split="test",
            authorize_final_evaluation=True,
            evaluation_manifest_path=manifest_path,
            output_root=tmp_path,
        )
    )
    result = pd.read_parquet(summary["output_path"]).iloc[0]

    assert not bool(result["execution_success"])
    assert pd.isna(result["correct"])
    assert pd.isna(result["score"])
    assert result["evaluation_status"] == "execution_failed"


def test_external_runner_rejects_test_only_input(tmp_path):
    records = [
        record
        for record in _records(tmp_path / "source")
        if record["benchmark_split"] == "test"
    ]
    with pytest.raises(ValueError, match="missing benchmark splits"):
        asyncio.run(external_runner.run_external_benchmark(records))


def test_external_config_fingerprint_changes_with_model_pool(tmp_path):
    records = _records(tmp_path / "source")
    one_model = _model_config()
    two_models = dict(one_model)
    two_models["gpt_oss_120b"] = {
        **one_model["gpt_oss_20b"],
        "id": "gpt_oss_120b",
        "model_name": "openai/gpt-oss-120b",
    }

    assert external_runner._external_config_id(records, one_model) != (
        external_runner._external_config_id(records, two_models)
    )


def test_external_runner_refuses_to_repeat_frozen_final_test(
    monkeypatch,
    tmp_path,
):
    records = _records(tmp_path / "source")
    async def fake_execution(*, model_id, prompt):
        return {
            "output": "42",
            "model_id": model_id,
            "provider": "groq",
            "input_tokens": 3,
            "output_tokens": 1,
            "cost": 0.000001,
            "latency_ms": 5.0,
            "success": True,
            "error": None,
        }

    monkeypatch.setattr(external_runner, "MODELS", _model_config())
    monkeypatch.setattr(external_runner, "execute_model", fake_execution)

    manifest_path, _ = _freeze_manifest(
        records,
        _model_config(),
        tmp_path / "append-run.json",
    )
    first = asyncio.run(
        external_runner.run_external_benchmark(
            records,
            execute=True,
            split="test",
            authorize_final_evaluation=True,
            evaluation_manifest_path=manifest_path,
            output_root=tmp_path,
        )
    )
    with pytest.raises(FileExistsError, match="already exist"):
        asyncio.run(
            external_runner.run_external_benchmark(
                records,
                execute=True,
                split="test",
                authorize_final_evaluation=True,
                evaluation_manifest_path=manifest_path,
                output_root=tmp_path,
            )
        )

    assert len(pd.read_parquet(first["output_path"])) == 1


def test_validation_and_test_configuration_ids_are_distinct(tmp_path):
    """Validation and test runs must generate distinct configuration identities."""
    records = _records(tmp_path / "source")
    models = _model_config()

    val_config_id = external_runner._external_config_id(records, models, split="validation")
    test_config_id = external_runner._external_config_id(records, models, split="test")

    assert val_config_id != test_config_id
    assert "external-gsm8k-" in val_config_id
    assert "external-gsm8k-" in test_config_id


def test_frozen_manifest_captures_execution_and_dependency_fingerprints(tmp_path):
    """Frozen manifest captures executor source SHA, LiteLLM version, provider configs, and model digest."""
    records = _records(tmp_path / "source")
    models = _model_config()

    manifest = external_runner.create_frozen_evaluation_manifest(records, models, runs=1)

    assert "execution_configuration" in manifest
    exec_config = manifest["execution_configuration"]
    assert exec_config["executor_version"] == "executor_v2"
    assert len(exec_config["executor_source_sha256"]) == 64
    assert "litellm_version" in exec_config
    assert exec_config["providers"] == ["groq"]
    assert exec_config["provider_configurations"] == [
        {"provider": "groq", "api_key_env_vars": ["GROQ_API_KEY"]}
    ]
    assert len(exec_config["model_registry_fingerprint"]) == 64
    assert manifest["evaluation_split"] == "test"


def test_frozen_manifest_rejects_executor_tampering(monkeypatch, tmp_path):
    """Tampering with executor source SHA invalidates the final test evaluation."""
    records = _records(tmp_path / "source")
    models = _model_config()
    monkeypatch.setattr(external_runner, "MODELS", models)

    async def unexpected_execution(**kwargs):
        raise AssertionError("execution must not start")

    monkeypatch.setattr(external_runner, "execute_model", unexpected_execution)

    frozen = external_runner.create_frozen_evaluation_manifest(records, models, runs=1)
    tampered = json.loads(json.dumps(frozen))
    tampered["execution_configuration"]["executor_source_sha256"] = "0" * 64
    tampered_path = external_runner.write_frozen_evaluation_manifest(
        tmp_path / "tampered-executor.json",
        tampered,
    )

    with pytest.raises(ValueError, match="does not match"):
        asyncio.run(
            external_runner.run_external_benchmark(
                records,
                split="test",
                execute=True,
                authorize_final_evaluation=True,
                evaluation_manifest_path=tampered_path,
                output_root=tmp_path / "results",
            )
        )


def test_frozen_manifest_rejects_litellm_version_change(monkeypatch, tmp_path):
    """Tampering with LiteLLM version invalidates the final test evaluation."""
    records = _records(tmp_path / "source")
    models = _model_config()
    monkeypatch.setattr(external_runner, "MODELS", models)

    async def unexpected_execution(**kwargs):
        raise AssertionError("execution must not start")

    monkeypatch.setattr(external_runner, "execute_model", unexpected_execution)

    frozen = external_runner.create_frozen_evaluation_manifest(records, models, runs=1)
    tampered = json.loads(json.dumps(frozen))
    tampered["execution_configuration"]["litellm_version"] = "99.99.99"
    tampered_path = external_runner.write_frozen_evaluation_manifest(
        tmp_path / "tampered-litellm.json",
        tampered,
    )

    with pytest.raises(ValueError, match="does not match"):
        asyncio.run(
            external_runner.run_external_benchmark(
                records,
                split="test",
                execute=True,
                authorize_final_evaluation=True,
                evaluation_manifest_path=tampered_path,
                output_root=tmp_path / "results",
            )
        )


def test_frozen_manifest_rejects_model_registry_fingerprint_change(monkeypatch, tmp_path):
    """Tampering with model registry fingerprint invalidates the final test evaluation."""
    records = _records(tmp_path / "source")
    models = _model_config()
    monkeypatch.setattr(external_runner, "MODELS", models)

    async def unexpected_execution(**kwargs):
        raise AssertionError("execution must not start")

    monkeypatch.setattr(external_runner, "execute_model", unexpected_execution)

    frozen = external_runner.create_frozen_evaluation_manifest(records, models, runs=1)
    tampered = json.loads(json.dumps(frozen))
    tampered["execution_configuration"]["model_registry_fingerprint"] = "f" * 64
    tampered_path = external_runner.write_frozen_evaluation_manifest(
        tmp_path / "tampered-registry.json",
        tampered,
    )

    with pytest.raises(ValueError, match="does not match"):
        asyncio.run(
            external_runner.run_external_benchmark(
                records,
                split="test",
                execute=True,
                authorize_final_evaluation=True,
                evaluation_manifest_path=tampered_path,
                output_root=tmp_path / "results",
            )
        )


def test_frozen_manifest_never_contains_secrets_or_api_keys(tmp_path):
    """The frozen evaluation manifest must never leak API keys, credentials, or secrets."""
    records = _records(tmp_path / "source")
    models = _model_config()

    manifest = external_runner.create_frozen_evaluation_manifest(records, models, runs=1)
    serialized = json.dumps(manifest)

    # Asserts that standard secret indicators or fake credentials are not present
    for secret_token in ("sk-", "AIza", "Bearer ", "password", "secret_key"):
        assert secret_token not in serialized


def test_frozen_manifest_captures_complete_environment_fingerprint(tmp_path):
    """The frozen evaluation manifest captures complete python and lockfile environment fingerprints."""
    import platform
    records = _records(tmp_path / "source")
    models = _model_config()

    manifest = external_runner.create_frozen_evaluation_manifest(records, models, runs=1)
    exec_config = manifest["execution_configuration"]

    assert exec_config["python_implementation"] == platform.python_implementation()
    assert exec_config["python_version"] == platform.python_version()
    assert "pyproject_toml_sha256" in exec_config
    assert "uv_lock_sha256" in exec_config
    if (external_runner.PROJECT_ROOT / "pyproject.toml").exists():
        assert len(exec_config["pyproject_toml_sha256"]) == 64
    if (external_runner.PROJECT_ROOT / "uv.lock").exists():
        assert len(exec_config["uv_lock_sha256"]) == 64


def test_frozen_manifest_rejects_python_version_change(monkeypatch, tmp_path):
    records = _records(tmp_path / "source")
    models = _model_config()
    monkeypatch.setattr(external_runner, "MODELS", models)

    async def unexpected_execution(**kwargs):
        raise AssertionError("execution must not start")

    monkeypatch.setattr(external_runner, "execute_model", unexpected_execution)

    frozen = external_runner.create_frozen_evaluation_manifest(records, models, runs=1)
    tampered = json.loads(json.dumps(frozen))
    tampered["execution_configuration"]["python_version"] = "2.7.18"
    tampered_path = external_runner.write_frozen_evaluation_manifest(
        tmp_path / "tampered-python.json",
        tampered,
    )

    with pytest.raises(ValueError, match="does not match"):
        asyncio.run(
            external_runner.run_external_benchmark(
                records,
                split="test",
                execute=True,
                authorize_final_evaluation=True,
                evaluation_manifest_path=tampered_path,
                output_root=tmp_path / "results",
            )
        )


def test_frozen_manifest_rejects_pyproject_toml_change(monkeypatch, tmp_path):
    records = _records(tmp_path / "source")
    models = _model_config()
    monkeypatch.setattr(external_runner, "MODELS", models)

    async def unexpected_execution(**kwargs):
        raise AssertionError("execution must not start")

    monkeypatch.setattr(external_runner, "execute_model", unexpected_execution)

    frozen = external_runner.create_frozen_evaluation_manifest(records, models, runs=1)
    tampered = json.loads(json.dumps(frozen))
    tampered["execution_configuration"]["pyproject_toml_sha256"] = "0" * 64
    tampered_path = external_runner.write_frozen_evaluation_manifest(
        tmp_path / "tampered-pyproject.json",
        tampered,
    )

    with pytest.raises(ValueError, match="does not match"):
        asyncio.run(
            external_runner.run_external_benchmark(
                records,
                split="test",
                execute=True,
                authorize_final_evaluation=True,
                evaluation_manifest_path=tampered_path,
                output_root=tmp_path / "results",
            )
        )


def test_frozen_manifest_rejects_uv_lock_change(monkeypatch, tmp_path):
    records = _records(tmp_path / "source")
    models = _model_config()
    monkeypatch.setattr(external_runner, "MODELS", models)

    async def unexpected_execution(**kwargs):
        raise AssertionError("execution must not start")

    monkeypatch.setattr(external_runner, "execute_model", unexpected_execution)

    frozen = external_runner.create_frozen_evaluation_manifest(records, models, runs=1)
    tampered = json.loads(json.dumps(frozen))
    tampered["execution_configuration"]["uv_lock_sha256"] = "1" * 64
    tampered_path = external_runner.write_frozen_evaluation_manifest(
        tmp_path / "tampered-uvlock.json",
        tampered,
    )

    with pytest.raises(ValueError, match="does not match"):
        asyncio.run(
            external_runner.run_external_benchmark(
                records,
                split="test",
                execute=True,
                authorize_final_evaluation=True,
                evaluation_manifest_path=tampered_path,
                output_root=tmp_path / "results",
            )
        )


