import asyncio
import hashlib
import json

import pandas as pd
import pytest

from eval.dataset_registry import (
    TrustedDatasetEntry,
    register_trusted_entry,
    reset_trusted_registry,
)
from eval.external_datasets import (
    DATASET_SPECS,
    NORMALIZED_COLUMNS,
    normalize_source_splits,
    read_safe_source,
    validate_normalized_records,
    validate_split_integrity,
    write_normalized_splits,
)
from eval.metrics import evaluate_output


def _gsm_row(index, question=None):
    question = question or f"Compute marker_{index}: {index} plus 1."
    return {
        "question": question,
        "answer": f"Work for item {index}.\n#### {index + 1}",
    }


def _gsm_splits(count=100):
    return {
        "train": [_gsm_row(index) for index in range(count)],
        "test": [_gsm_row(10001, "Compute the unique final split marker.")],
    }


def _write_source_set(tmp_path, dataset_key, rows_by_split):
    spec = DATASET_SPECS[dataset_key]
    paths = {}
    manifests = {}
    for split, rows in rows_by_split.items():
        path = tmp_path / f"{split}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        content = "".join(
            json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
            for row in rows
        ).encode("utf-8")
        path.write_bytes(content)
        paths[split] = path
        content_hash = hashlib.sha256(content).hexdigest()
        entry_id = f"{spec['dataset_id']}:{spec['revision']}:{split}:jsonl"
        register_trusted_entry(
            TrustedDatasetEntry(
                entry_id=entry_id,
                dataset_name=spec["dataset_id"],
                publisher="Test Publisher",
                approved_revision=spec["revision"],
                split=split,
                source_url=spec.get("citation_url", "https://example.com"),
                sha256=content_hash,
                file_format="jsonl",
                attribution=spec["attribution"],
                license_approval_reference="TEST-LICENSE-APPROVED",
                approval_status="approved",
            )
        )
        manifests[split] = {
            "dataset_name": spec["dataset_id"],
            "approved_revision": spec["revision"],
            "source_file_sha256": content_hash,
            "split": split,
            "format": "jsonl",
            "attribution": spec["attribution"],
            "registry_entry_id": entry_id,
        }
    return paths, manifests


def _normalize(tmp_path, dataset_key, rows_by_split, manifests=None):
    paths, generated_manifests = _write_source_set(
        tmp_path,
        dataset_key,
        rows_by_split,
    )
    return normalize_source_splits(
        dataset_key,
        paths,
        manifests if manifests is not None else generated_manifests,
    ), paths, generated_manifests


def test_gsm8k_adapter_preserves_answer_and_creates_fixed_splits(tmp_path):
    # Separate directories prove identical ordered source files produce identical splits.
    first, _, _ = _normalize(tmp_path / "first", "gsm8k", _gsm_splits())
    second, _, _ = _normalize(tmp_path / "second", "gsm8k", _gsm_splits())

    assert [
        (row["prompt_id"], row["benchmark_split"], row["source_file_sha256"])
        for row in first
    ] == [
        (row["prompt_id"], row["benchmark_split"], row["source_file_sha256"])
        for row in second
    ]
    assert {row["benchmark_split"] for row in first} == {
        "train",
        "validation",
        "test",
    }
    test_record = next(row for row in first if row["benchmark_split"] == "test")
    assert test_record["source_question"] == "Compute the unique final split marker."
    assert test_record["reference_answer"] == "10002"
    assert test_record["source_reference_answer_json"] == json.dumps(
        "Work for item 10001.\n#### 10002",
        ensure_ascii=False,
        separators=(",", ":"),
    )
    assert test_record["dataset_revision"] == DATASET_SPECS["gsm8k"]["revision"]
    assert test_record["license_id"] == "MIT"
    assert test_record["source_file_sha256"]
    assert json.loads(test_record["source_manifest_json"])["format"] == "jsonl"


def test_gsm8k_numeric_evaluator_scores_normalized_reference(tmp_path):
    records, _, _ = _normalize(tmp_path, "gsm8k", _gsm_splits())
    record = next(row for row in records if row["benchmark_split"] == "test")

    result = asyncio.run(evaluate_output(record, "Answer: 10002"))
    assert result.evaluation_status == "scored"
    assert result.correct is True


def test_mmlu_adapter_preserves_choices_and_maps_source_label(tmp_path):
    row = {
        "question": "Which option is correct?",
        "subject": "test_subject",
        "choices": ["first", "second", "third", "fourth"],
        "answer": 2,
    }
    records, _, _ = _normalize(
        tmp_path,
        "mmlu",
        {
            "dev": [{**row, "question": "A unique development question?"}],
            "validation": [{
                **row,
                "question": "A unique validation question?",
            }],
            "test": [row],
        },
    )
    record = next(row for row in records if row["source_split"] == "test")

    assert record["source_question"] == row["question"]
    assert record["choices"] == row["choices"]
    assert record["choice_labels"] == ["A", "B", "C", "D"]
    assert record["reference_answer"] == "C"
    assert json.loads(record["source_record_json"]) == row
    assert "C. third" in record["prompt"]
    result = asyncio.run(evaluate_output(record, "Answer: C."))
    assert result.evaluation_status == "scored"
    assert result.correct is True


def test_mbpp_sanitized_adapter_preserves_solution_and_tests_verbatim(tmp_path):
    row = {
        "task_id": 41,
        "prompt": "Return the square of a number.",
        "code": "def square(n):\n    return n * n\n",
        "test_imports": ["import math"],
        "test_list": ["assert square(3) == 9", "assert square(0) == 0"],
    }
    records, _, _ = _normalize(
        tmp_path,
        "mbpp",
        {
            "train": [{**row, "task_id": 1, "prompt": "Unique train task."}],
            "prompt": [{**row, "task_id": 2, "prompt": "Unique demo task."}],
            "validation": [{
                **row,
                "task_id": 3,
                "prompt": "Unique validation task.",
            }],
            "test": [row],
        },
    )
    record = next(row for row in records if row["source_split"] == "test")

    assert record["dataset_config"] == "sanitized"
    assert record["source_id"] == "41"
    assert record["source_question"] == row["prompt"]
    assert record["reference_answer"] == row["code"]
    assert record["test_cases"] == row["test_list"]
    assert record["test_imports"] == row["test_imports"]
    assert json.loads(record["source_record_json"]) == row
    assert record["evaluation_method"] == "unit_test"


def test_mbpp_evaluation_stays_unsupported_without_sandbox(tmp_path):
    records, _, _ = _normalize(
        tmp_path,
        "mbpp",
        {
            "train": [{
                "task_id": 1,
                "prompt": "Unique train prompt.",
                "code": "def f(): return 1",
                "test_imports": [],
                "test_list": ["assert f() == 1"],
            }],
            "prompt": [{
                "task_id": 2,
                "prompt": "Unique demo prompt.",
                "code": "def f(): return 1",
                "test_imports": [],
                "test_list": ["assert f() == 1"],
            }],
            "validation": [{
                "task_id": 3,
                "prompt": "Unique validation prompt.",
                "code": "def f(): return 1",
                "test_imports": [],
                "test_list": ["assert f() == 1"],
            }],
            "test": [{
                "task_id": 4,
                "prompt": "Return 1.",
                "code": "def f(): return 1",
                "test_imports": [],
                "test_list": ["assert f() == 1"],
            }],
        },
    )
    record = next(row for row in records if row["source_split"] == "test")

    result = asyncio.run(evaluate_output(record, "def f(): return 1"))
    assert result.evaluation_status == "unsupported"
    assert result.correct is None


def test_normalized_schema_rejects_missing_fields(tmp_path):
    records, _, _ = _normalize(
        tmp_path,
        "gsm8k",
        _gsm_splits(),
    )
    record = next(row for row in records if row["benchmark_split"] == "test")
    del record["source_record_json"]

    with pytest.raises(ValueError, match="missing columns"):
        validate_normalized_records([record])


def test_normalized_schema_rejects_changed_mbpp_test_case(tmp_path):
    records, _, _ = _normalize(
        tmp_path,
        "mbpp",
        {
            "train": [{
                "task_id": 1,
                "prompt": "Unique train task.",
                "code": "def f(): return 1",
                "test_imports": [],
                "test_list": ["assert f() == 1"],
            }],
            "prompt": [{
                "task_id": 3,
                "prompt": "Unique few-shot task.",
                "code": "def f(): return 1",
                "test_imports": [],
                "test_list": ["assert f() == 1"],
            }],
            "validation": [{
                "task_id": 4,
                "prompt": "Unique validation task.",
                "code": "def f(): return 1",
                "test_imports": [],
                "test_list": ["assert f() == 1"],
            }],
            "test": [{
                "task_id": 2,
                "prompt": "Return 1.",
                "code": "def f(): return 1",
                "test_imports": [],
                "test_list": ["assert f() == 1"],
            }],
        },
    )
    record = next(row for row in records if row["source_split"] == "test")
    record["test_cases"] = ["assert f() == 0"]

    with pytest.raises(ValueError, match="source code, prompt, or tests"):
        validate_normalized_records([record])


def test_split_integrity_hard_fails_exact_duplicate_prompts(tmp_path):
    splits = _gsm_splits(count=100)
    splits["test"] = [_gsm_row(20001, splits["train"][0]["question"])]

    with pytest.raises(ValueError, match="Exact duplicate"):
        _normalize(tmp_path, "gsm8k", splits)


@pytest.mark.parametrize(
    ("left_question", "right_question"),
    [
        (
            "What is the capital city of France in Europe today?",
            "What is the capital city of Spain in Europe today?",
        ),
        (
            "A worker completed 12 tasks before the final review on Friday.",
            "A worker completed 13 tasks before the final review on Friday.",
        ),
        (
            "Which candidate won the award after the final public vote?",
            "Which candidate lost the award after the final public vote?",
        ),
    ],
)
def test_near_duplicate_matches_warn_without_changing_records(
    tmp_path,
    left_question,
    right_question,
):
    splits = _gsm_splits(count=100)
    splits["train"][0] = _gsm_row(0, left_question)
    splits["test"] = [_gsm_row(10001, right_question)]

    with pytest.warns(UserWarning, match=r"source_ids=.*token_jaccard=") as caught:
        records, _, _ = _normalize(tmp_path, "gsm8k", splits)

    assert len(records) == 101
    warning_text = str(caught[0].message)
    assert "train:0000000" in warning_text
    assert "test:0000000" in warning_text
    assert "token_jaccard=" in warning_text
    assert next(
        row for row in records if row["source_split"] == "train"
    )["source_question"] == left_question


def test_split_integrity_rejects_tampered_fixed_split_assignment(tmp_path):
    records, _, _ = _normalize(tmp_path, "gsm8k", _gsm_splits())
    duplicate = dict(next(
        row for row in records if row["benchmark_split"] == "train"
    ))
    duplicate["benchmark_split"] = "test"
    duplicate["prompt_id"] += ":duplicate"
    records.append(duplicate)

    with pytest.raises(ValueError, match="fixed split policy"):
        validate_split_integrity(records)


def test_safe_source_reader_accepts_jsonl_and_rejects_pickle(tmp_path):
    jsonl_path = tmp_path / "sample.jsonl"
    jsonl_path.write_text('{"question":"Q","answer":"#### 1"}\n', encoding="utf-8")
    assert read_safe_source(jsonl_path) == [
        {"question": "Q", "answer": "#### 1"}
    ]

    with pytest.raises(ValueError, match="Pickle"):
        read_safe_source(tmp_path / "not_loaded.pkl")


def test_source_normalization_requires_complete_matching_manifest(tmp_path):
    _, paths, manifests = _normalize(
        tmp_path,
        "gsm8k",
        _gsm_splits(),
    )

    with pytest.raises(ValueError, match="manifests mismatch"):
        normalize_source_splits("gsm8k", paths, {})

    wrong_revision = {split: dict(manifest) for split, manifest in manifests.items()}
    wrong_revision["train"]["approved_revision"] = "0" * 40
    with pytest.raises(ValueError, match="revision is not approved"):
        normalize_source_splits("gsm8k", paths, wrong_revision)

    wrong_attribution = {
        split: dict(manifest) for split, manifest in manifests.items()
    }
    wrong_attribution["test"]["attribution"] = "unverified"
    with pytest.raises(ValueError, match="attribution"):
        normalize_source_splits("gsm8k", paths, wrong_attribution)


def test_source_normalization_verifies_source_file_hash_before_parsing(tmp_path):
    _, paths, manifests = _normalize(
        tmp_path,
        "gsm8k",
        _gsm_splits(),
    )
    paths["test"].write_text(
        '{"question":"substituted","answer":"#### 9"}\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        normalize_source_splits("gsm8k", paths, manifests)


def test_source_manifest_format_must_match_local_file(tmp_path):
    _, paths, manifests = _normalize(
        tmp_path,
        "gsm8k",
        _gsm_splits(),
    )
    manifests["test"]["format"] = "parquet"

    with pytest.raises(ValueError, match="format"):
        normalize_source_splits("gsm8k", paths, manifests)


def test_same_mbpp_task_id_in_different_source_splits_is_rejected(tmp_path):
    row = {
        "task_id": 41,
        "prompt": "One task prompt.",
        "code": "def f(): return 1",
        "test_imports": [],
        "test_list": ["assert f() == 1"],
    }
    with pytest.raises(ValueError, match="Duplicate normalized prompt_id"):
        _normalize(
            tmp_path,
            "mbpp",
            {
                "train": [row],
                "prompt": [{
                    **row,
                    "task_id": 42,
                    "prompt": "A unique demonstration task.",
                }],
                "validation": [{
                    **row,
                    "task_id": 43,
                    "prompt": "A unique validation task.",
                }],
                "test": [row],
            },
        )


def test_normalized_writer_refuses_to_overwrite(tmp_path):
    records, _, _ = _normalize(tmp_path / "source", "gsm8k", _gsm_splits())
    outputs = write_normalized_splits(
        "gsm8k",
        records,
        root=tmp_path,
    )
    assert outputs
    assert all(path.exists() for path in outputs)

    with pytest.raises(FileExistsError, match="overwrite"):
        write_normalized_splits("gsm8k", records, root=tmp_path)


def test_gsm8k_rejects_missing_answer_delimiter(tmp_path):
    splits = _gsm_splits()
    splits["test"] = [{
        "question": "A unique question",
        "answer": "Answer is 2",
    }]
    with pytest.raises(ValueError, match="####"):
        _normalize(tmp_path, "gsm8k", splits)


def test_normalized_schema_has_all_declared_columns(tmp_path):
    records, _, _ = _normalize(tmp_path, "gsm8k", _gsm_splits())
    record = records[0]
    assert set(record) == set(NORMALIZED_COLUMNS)


def test_arbitrary_file_with_self_supplied_hash_is_rejected(tmp_path):
    """An arbitrary local file pairing with its own self-generated manifest must be rejected."""
    spec = DATASET_SPECS["gsm8k"]
    source_path = tmp_path / "arbitrary.jsonl"
    content = b'{"question": "Unvetted question?", "answer": "#### 42"}\n'
    source_path.write_bytes(content)
    arbitrary_hash = hashlib.sha256(content).hexdigest()

    # User creates a manifest with their own calculated hash
    user_manifest = {
        "dataset_name": spec["dataset_id"],
        "approved_revision": spec["revision"],
        "source_file_sha256": arbitrary_hash,
        "split": "test",
        "format": "jsonl",
        "attribution": spec["attribution"],
    }

    # Reset registry to default unapproved state for GSM8K
    reset_trusted_registry()

    # Ingestion must be rejected because the registry does not approve this unvetted file
    with pytest.raises(ValueError, match="(blocked|approval status|No trusted)"):
        normalize_source_splits(
            "gsm8k",
            {"test": source_path, "train": source_path},
            {"test": user_manifest, "train": user_manifest},
        )


def test_approved_hash_matching_actual_source_bytes(tmp_path):
    """When actual source bytes match an approved trusted registry entry, normalization succeeds."""
    spec = DATASET_SPECS["gsm8k"]
    rows = _gsm_splits()
    paths = {}
    manifests = {}
    custom_registry = {}

    for split, split_rows in rows.items():
        path = tmp_path / f"{split}.jsonl"
        content = "".join(
            json.dumps(row, separators=(",", ":")) + "\n"
            for row in split_rows
        ).encode("utf-8")
        path.write_bytes(content)
        paths[split] = path
        content_hash = hashlib.sha256(content).hexdigest()
        entry_id = f"test-approved:{split}"
        custom_registry[entry_id] = TrustedDatasetEntry(
            entry_id=entry_id,
            dataset_name=spec["dataset_id"],
            publisher="Verified Publisher",
            approved_revision=spec["revision"],
            split=split,
            source_url="https://example.com/verified",
            sha256=content_hash,
            file_format="jsonl",
            attribution=spec["attribution"],
            license_approval_reference="LEGAL-REVIEW-APPROVED",
            approval_status="approved",
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

    records = normalize_source_splits(
        "gsm8k",
        paths,
        manifests,
        trusted_registry=custom_registry,
    )
    assert len(records) > 0
    assert records[0]["source_file_sha256"] == manifests[records[0]["source_split"]]["source_file_sha256"]


def test_modified_source_bytes_rejected_against_trusted_registry(tmp_path):
    """If source bytes are altered after approval, verification fails even if manifest is updated."""
    spec = DATASET_SPECS["gsm8k"]
    train_path = tmp_path / "train.jsonl"
    test_path = tmp_path / "test.jsonl"
    train_content = b'{"question":"Q1","answer":"#### 1"}\n' * 10
    test_content = b'{"question":"Q2","answer":"#### 2"}\n'
    train_path.write_bytes(train_content)
    test_path.write_bytes(test_content)

    train_hash = hashlib.sha256(train_content).hexdigest()
    test_hash = hashlib.sha256(test_content).hexdigest()

    custom_registry = {
        "reg:train": TrustedDatasetEntry(
            entry_id="reg:train",
            dataset_name=spec["dataset_id"],
            publisher="OpenAI",
            approved_revision=spec["revision"],
            split="train",
            source_url="https://example.com",
            sha256=train_hash,
            file_format="jsonl",
            attribution=spec["attribution"],
            license_approval_reference="APPROVED",
            approval_status="approved",
        ),
        "reg:test": TrustedDatasetEntry(
            entry_id="reg:test",
            dataset_name=spec["dataset_id"],
            publisher="OpenAI",
            approved_revision=spec["revision"],
            split="test",
            source_url="https://example.com",
            sha256=test_hash,
            file_format="jsonl",
            attribution=spec["attribution"],
            license_approval_reference="APPROVED",
            approval_status="approved",
        ),
    }

    # Tamper with test file content
    tampered_content = b'{"question":"Tampered Q","answer":"#### 999"}\n'
    test_path.write_bytes(tampered_content)
    tampered_hash = hashlib.sha256(tampered_content).hexdigest()

    # User attempts to supply updated manifest matching tampered content
    manifests = {
        "train": {
            "registry_entry_id": "reg:train",
            "dataset_name": spec["dataset_id"],
            "approved_revision": spec["revision"],
            "source_file_sha256": train_hash,
            "split": "train",
            "format": "jsonl",
            "attribution": spec["attribution"],
        },
        "test": {
            "registry_entry_id": "reg:test",
            "dataset_name": spec["dataset_id"],
            "approved_revision": spec["revision"],
            "source_file_sha256": tampered_hash,
            "split": "test",
            "format": "jsonl",
            "attribution": spec["attribution"],
        },
    }

    with pytest.raises(ValueError, match="(Source manifest SHA-256|Source SHA-256 mismatch)"):
        normalize_source_splits(
            "gsm8k",
            {"train": train_path, "test": test_path},
            manifests,
            trusted_registry=custom_registry,
        )


def test_missing_trusted_registry_entry_rejected(tmp_path):
    """If a dataset split is not in the trusted registry, it must be rejected."""
    spec = DATASET_SPECS["gsm8k"]
    empty_registry = {}
    path = tmp_path / "test.jsonl"
    path.write_bytes(b'{"question":"Q","answer":"#### 1"}\n')
    manifest = {
        "dataset_name": spec["dataset_id"],
        "approved_revision": spec["revision"],
        "source_file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "split": "test",
        "format": "jsonl",
        "attribution": spec["attribution"],
    }
    with pytest.raises(ValueError, match="No trusted approval registry entry found"):
        normalize_source_splits(
            "gsm8k",
            {"test": path, "train": path},
            {"test": manifest, "train": manifest},
            trusted_registry=empty_registry,
        )


def test_pending_approval_entry_blocks_ingestion(tmp_path):
    """An entry with 'pending' approval status must block ingestion."""
    spec = DATASET_SPECS["gsm8k"]
    path = tmp_path / "test.jsonl"
    content = b'{"question":"Q","answer":"#### 1"}\n'
    path.write_bytes(content)
    content_hash = hashlib.sha256(content).hexdigest()

    pending_registry = {
        "pending_entry": TrustedDatasetEntry(
            entry_id="pending_entry",
            dataset_name=spec["dataset_id"],
            publisher="OpenAI",
            approved_revision=spec["revision"],
            split="test",
            source_url="https://example.com",
            sha256=content_hash,
            file_format="jsonl",
            attribution=spec["attribution"],
            license_approval_reference="PENDING-LEGAL",
            approval_status="pending",
        ),
    }
    manifest = {
        "registry_entry_id": "pending_entry",
        "dataset_name": spec["dataset_id"],
        "approved_revision": spec["revision"],
        "source_file_sha256": content_hash,
        "split": "test",
        "format": "jsonl",
        "attribution": spec["attribution"],
    }
    with pytest.raises(ValueError, match="approval status is 'pending'; ingestion is blocked"):
        normalize_source_splits(
            "gsm8k",
            {"test": path, "train": path},
            {"test": manifest, "train": manifest},
            trusted_registry=pending_registry,
        )


def test_verified_provenance_preserved_in_source_manifest(tmp_path):
    """Verified provenance from trusted registry is retained in source_manifest_json."""
    records, _, _ = _normalize(tmp_path, "gsm8k", _gsm_splits())
    first_record = records[0]
    manifest = json.loads(first_record["source_manifest_json"])

    assert manifest["approval_status"] == "approved"
    assert manifest["publisher"] == "Test Publisher"
    assert manifest["license_approval_reference"] == "TEST-LICENSE-APPROVED"
    assert "registry_entry_id" in manifest


def test_production_registry_override_is_forbidden(monkeypatch, tmp_path):
    """Attempting registry mutations or overrides outside of a test session must fail."""
    spec = DATASET_SPECS["gsm8k"]
    entry = TrustedDatasetEntry(
        entry_id="prod_test",
        dataset_name=spec["dataset_id"],
        publisher="OpenAI",
        approved_revision=spec["revision"],
        split="test",
        source_url="https://example.com",
        sha256="0" * 64,
        file_format="jsonl",
        attribution=spec["attribution"],
        license_approval_reference="APPROVED",
        approval_status="approved",
    )
    # Simulate production environment without pytest
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    monkeypatch.delenv("COPILOT_TESTING", raising=False)

    with pytest.raises(PermissionError, match="prohibited in production"):
        register_trusted_entry(entry)

    with pytest.raises(PermissionError, match="prohibited in production"):
        normalize_source_splits("gsm8k", {}, {}, trusted_registry={})


def test_normalized_record_missing_registry_entry_id_is_rejected(tmp_path):
    records, _, _ = _normalize(tmp_path, "gsm8k", _gsm_splits())
    record = dict(records[0])
    manifest = json.loads(record["source_manifest_json"])
    del manifest["registry_entry_id"]
    record["source_manifest_json"] = json.dumps(manifest)

    with pytest.raises(ValueError, match="missing a valid registry_entry_id"):
        validate_normalized_records([record])


def test_normalized_record_unknown_registry_entry_id_is_rejected(tmp_path):
    records, _, _ = _normalize(tmp_path, "gsm8k", _gsm_splits())
    record = dict(records[0])
    manifest = json.loads(record["source_manifest_json"])
    manifest["registry_entry_id"] = "unknown:unapproved:entry"
    record["source_manifest_json"] = json.dumps(manifest)

    with pytest.raises(ValueError, match="No trusted approval registry entry found"):
        validate_normalized_records([record])


def test_normalized_record_pending_approval_is_rejected(tmp_path):
    spec = DATASET_SPECS["gsm8k"]
    records, _, _ = _normalize(tmp_path, "gsm8k", _gsm_splits())
    record = dict(records[0])
    pending_id = "test:pending:entry"
    register_trusted_entry(
        TrustedDatasetEntry(
            entry_id=pending_id,
            dataset_name=spec["dataset_id"],
            publisher="Test Publisher",
            approved_revision=spec["revision"],
            split=record["source_split"],
            source_url="https://example.com",
            sha256=record["source_file_sha256"],
            file_format="jsonl",
            attribution=spec["attribution"],
            license_approval_reference="TEST-LICENSE-APPROVED",
            approval_status="pending",
        )
    )
    manifest = json.loads(record["source_manifest_json"])
    manifest["registry_entry_id"] = pending_id
    manifest["approval_status"] = "pending"
    record["source_manifest_json"] = json.dumps(manifest)

    with pytest.raises(ValueError, match="approval status.*must be 'approved'"):
        validate_normalized_records([record])


def test_normalized_record_fake_internally_consistent_source_hash_is_rejected(tmp_path):
    """Internally consistent Parquet metadata with a forged hash must be rejected."""
    records, _, _ = _normalize(tmp_path, "gsm8k", _gsm_splits())
    record = dict(records[0])
    fake_hash = "f" * 64
    record["source_file_sha256"] = fake_hash
    manifest = json.loads(record["source_manifest_json"])
    manifest["source_file_sha256"] = fake_hash
    record["source_manifest_json"] = json.dumps(manifest)

    with pytest.raises(ValueError, match="does not match trusted registry"):
        validate_normalized_records([record])


def test_normalized_record_wrong_publisher_or_source_url_is_rejected(tmp_path):
    records, _, _ = _normalize(tmp_path, "gsm8k", _gsm_splits())
    record = dict(records[0])
    manifest = json.loads(record["source_manifest_json"])
    manifest["publisher"] = "Forged Malicious Publisher"
    record["source_manifest_json"] = json.dumps(manifest)

    with pytest.raises(ValueError, match="publisher does not match"):
        validate_normalized_records([record])

    manifest["publisher"] = "Test Publisher"
    manifest["source_url"] = "https://phishing.example.com"
    record["source_manifest_json"] = json.dumps(manifest)

    with pytest.raises(ValueError, match="source_url does not match"):
        validate_normalized_records([record])


def test_normalized_record_wrong_license_approval_reference_is_rejected(tmp_path):
    records, _, _ = _normalize(tmp_path, "gsm8k", _gsm_splits())
    record = dict(records[0])
    manifest = json.loads(record["source_manifest_json"])
    manifest["license_approval_reference"] = "UNVERIFIED-LICENSE-CLAIM"
    record["source_manifest_json"] = json.dumps(manifest)

    with pytest.raises(ValueError, match="license_approval_reference does not match"):
        validate_normalized_records([record])


def test_parquet_loading_authenticates_provenance_and_rejects_forgery(tmp_path):
    """Loading Parquet files from disk independently re-authenticates every record."""
    from eval.external_runner import load_normalized_dataset
    records, _, _ = _normalize(tmp_path / "raw", "gsm8k", _gsm_splits())
    output_files = write_normalized_splits("gsm8k", records, root=tmp_path / "normalized_root")
    assert len(output_files) == 3

    # Loading authentic records succeeds
    loaded = load_normalized_dataset("gsm8k", root=tmp_path / "normalized_root")
    assert len(loaded) == len(records)

    # Tamper with the test.parquet file on disk: inject forged metadata
    test_parquet = next(p for p in output_files if p.name == "test.parquet")
    frame = pd.read_parquet(test_parquet)
    forged_manifest = json.loads(frame.iloc[0]["source_manifest_json"])
    forged_manifest["publisher"] = "Forged Publisher"
    frame.at[0, "source_manifest_json"] = json.dumps(forged_manifest)
    frame.to_parquet(test_parquet, index=False)

    # Must be rejected at load time before any benchmark execution
    with pytest.raises(ValueError, match="publisher does not match"):
        load_normalized_dataset("gsm8k", root=tmp_path / "normalized_root")


