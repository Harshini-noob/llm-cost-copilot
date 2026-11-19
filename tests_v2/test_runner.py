import asyncio

import pandas as pd

import eval.runner as runner
import executor_v2


def _execution_result(success=True):
    return {
        "output": "mock output" if success else None,
        "model_id": "test-model",
        "provider": "groq",
        "input_tokens": 2 if success else 0,
        "output_tokens": 3 if success else 0,
        "cost": 0.000001 if success else 0.0,
        "latency_ms": 4.0,
        "success": success,
        "error": None if success else "Execution failed (RuntimeError).",
    }


def test_runner_appends_and_records_benchmark_configuration(monkeypatch, tmp_path):
    models = {
        "gpt_oss_20b": {
            "id": "gpt_oss_20b",
            "provider": "groq",
            "model_name": "openai/gpt-oss-20b",
            "input_price_per_1m": 0.075,
            "output_price_per_1m": 0.30,
            "context_window": 131072,
        },
        "gpt_oss_120b": {
            "id": "gpt_oss_120b",
            "provider": "groq",
            "model_name": "openai/gpt-oss-120b",
            "input_price_per_1m": 0.15,
            "output_price_per_1m": 0.60,
            "context_window": 131072,
        },
    }
    prompt = {
        "prompt_id": "math_001",
        "task_type": "math",
        "prompt": "2 + 2",
        "evaluation_method": "numeric_exact",
    }
    output_path = tmp_path / "matrix.parquet"
    old_row = {column: None for column in runner.RESULT_COLUMNS if column != "benchmark_config_id"}
    old_row.update(
        prompt_id="old_prompt",
        model_id="old_model",
        output="preserve me",
    )
    pd.DataFrame([old_row]).to_parquet(output_path, index=False)
    calls = []

    async def fake_execute_model(model_id, prompt):
        assert model_id in models
        assert prompt == "2 + 2"
        calls.append((model_id, prompt))
        return _execution_result()

    monkeypatch.setattr(runner, "MODELS", models)
    monkeypatch.setattr(runner, "OUTPUT_PATH", output_path)
    monkeypatch.setattr(runner, "load_dataset", lambda: [prompt])
    monkeypatch.setattr(runner, "execute_model", fake_execute_model)

    asyncio.run(runner.run_benchmark(runs=2))

    saved = pd.read_parquet(output_path)
    assert len(calls) == 4
    assert len(saved) == 5
    assert saved.iloc[0]["output"] == "preserve me"
    assert saved.iloc[0]["benchmark_config_id"] == "legacy-unversioned"
    new_rows = saved.iloc[1:]
    assert new_rows["benchmark_config_id"].nunique() == 1
    assert new_rows["benchmark_config_id"].iloc[0].startswith("m1-")
    assert set(new_rows["model_id"]) == set(models)
    assert set(new_rows["run"]) == {1, 2}
    assert "correct" in saved.columns


def test_runner_records_failed_execution_as_unscored(monkeypatch, tmp_path):
    prompt = {
        "prompt_id": "math_002",
        "task_type": "math",
        "prompt": "2 + 2",
        "evaluation_method": "numeric_exact",
    }
    output_path = tmp_path / "matrix.parquet"

    async def fake_execute_model(model_id, prompt):
        return _execution_result(success=False)

    async def unexpected_evaluation(*args, **kwargs):
        raise AssertionError("failed execution must not be evaluated")

    monkeypatch.setattr(runner, "MODELS", {
        "gpt_oss_20b": {
            "id": "gpt_oss_20b",
            "provider": "groq",
            "model_name": "openai/gpt-oss-20b",
            "input_price_per_1m": 0.075,
            "output_price_per_1m": 0.30,
            "context_window": 131072,
        },
    })
    monkeypatch.setattr(runner, "OUTPUT_PATH", output_path)
    monkeypatch.setattr(runner, "load_dataset", lambda: [prompt])
    monkeypatch.setattr(runner, "execute_model", fake_execute_model)
    monkeypatch.setattr(runner, "evaluate_output", unexpected_evaluation)

    asyncio.run(runner.run_benchmark(runs=1))

    result = pd.read_parquet(output_path).iloc[0]
    assert not bool(result["execution_success"])
    assert pd.isna(result["correct"])
    assert result["error"] == "Execution failed (RuntimeError)."
    assert result["evaluation_status"] == "execution_failed"
    assert pd.isna(result["score"])


def test_runner_persists_deterministic_evaluation(monkeypatch, tmp_path):
    prompt = {
        "prompt_id": "math_004",
        "task_type": "math",
        "prompt": "What is 2 + 2?",
        "reference_answer": "4",
        "evaluation_method": "numeric_exact",
    }
    output_path = tmp_path / "matrix.parquet"

    async def fake_execute_model(model_id, prompt):
        return {
            **_execution_result(),
            "model_id": model_id,
            "output": "Answer: 4.",
        }

    monkeypatch.setattr(runner, "MODELS", {
        "gpt_oss_20b": {
            "id": "gpt_oss_20b",
            "provider": "groq",
            "model_name": "openai/gpt-oss-20b",
            "input_price_per_1m": 0.075,
            "output_price_per_1m": 0.30,
            "context_window": 131072,
        },
    })
    monkeypatch.setattr(runner, "OUTPUT_PATH", output_path)
    monkeypatch.setattr(runner, "load_dataset", lambda: [prompt])
    monkeypatch.setattr(runner, "execute_model", fake_execute_model)

    asyncio.run(runner.run_benchmark(runs=1))

    result = pd.read_parquet(output_path).iloc[0]
    assert bool(result["execution_success"])
    assert bool(result["correct"])
    assert result["score"] == 1.0
    assert result["evaluation_status"] == "scored"
    assert result["evaluation_method"] == "numeric_exact"


def test_dry_run_does_not_execute_models(monkeypatch, capsys):
    prompt = {
        "prompt_id": "math_003",
        "task_type": "math",
        "prompt": "2 + 2",
        "evaluation_method": "numeric_exact",
    }

    async def unexpected_call(*args, **kwargs):
        raise AssertionError(f"unexpected dry-run call: {args}, {kwargs}")

    monkeypatch.setattr(runner, "MODELS", {
        "gpt_oss_20b": {
            "id": "gpt_oss_20b",
            "provider": "groq",
            "model_name": "openai/gpt-oss-20b",
            "input_price_per_1m": 0.075,
            "output_price_per_1m": 0.30,
            "context_window": 131072,
        },
    })
    monkeypatch.setattr(runner, "load_dataset", lambda: [prompt])
    monkeypatch.setattr(runner, "execute_model", unexpected_call)

    asyncio.run(runner.run_benchmark(runs=1, dry_run=True))

    assert "Benchmark configuration: m1-" in capsys.readouterr().out


def test_runner_uses_litellm_without_legacy_adapters(monkeypatch, tmp_path):
    prompt = {
        "prompt_id": "math_litellm",
        "task_type": "math",
        "prompt": "2 + 2",
        "evaluation_method": "numeric_exact",
    }
    output_path = tmp_path / "matrix.parquet"
    completion_calls = []
    execution_calls = []

    async def fake_acompletion(**kwargs):
        completion_calls.append(kwargs)
        return {
            "choices": [{"message": {"content": "4"}}],
            "usage": {"prompt_tokens": 3, "completion_tokens": 1},
        }

    real_execute_model = runner.execute_model

    async def checked_execute_model(**kwargs):
        execution_calls.append(kwargs)
        assert "adapters" not in kwargs
        return await real_execute_model(**kwargs)

    monkeypatch.setenv("GROQ_API_KEY", "benchmark-test-key")
    monkeypatch.setattr(executor_v2.litellm, "acompletion", fake_acompletion)
    monkeypatch.setattr(runner, "execute_model", checked_execute_model)
    monkeypatch.setattr(runner, "MODELS", {
        "gpt_oss_20b": runner.MODELS["gpt_oss_20b"],
    })
    monkeypatch.setattr(runner, "OUTPUT_PATH", output_path)
    monkeypatch.setattr(runner, "load_dataset", lambda: [prompt])

    asyncio.run(runner.run_benchmark(runs=1))

    result = pd.read_parquet(output_path).iloc[0]
    assert len(execution_calls) == 1
    assert len(completion_calls) == 1
    assert completion_calls[0]["model"] == "groq/openai/gpt-oss-20b"
    assert result["execution_success"]
    assert result["output"] == "4"
