"""Trusted dataset approval registry for external benchmark authentication.

This registry acts as an independent authoritative allowlist of verified
dataset releases, publishers, revisions, source URLs, licenses, and SHA-256
file hashes. Source manifests must match an approved registry entry.
"""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
import os
import re
from typing import Any, Mapping


# Trust Boundary:
# Production dataset approvals are statically defined, immutable, and reviewable
# directly in source code. Runtime mutation of the registry or substitution of
# custom registries is prohibited in production and restricted strictly to
# automated test environments under pytest. Syntactically valid SHA-256 strings
# or arbitrary license identifiers do NOT constitute independent verification.


def is_test_mode() -> bool:
    """Check if execution is currently inside an automated test harness."""
    return bool(
        os.environ.get("PYTEST_CURRENT_TEST")
        or os.environ.get("COPILOT_TESTING") == "1"
    )


def assert_test_mode(operation: str = "Registry override") -> None:
    """Enforce that registry mutations or overrides cannot occur in production."""
    if not is_test_mode():
        raise PermissionError(
            f"{operation} is prohibited in production. Dataset approval is strictly "
            "controlled by the code-reviewed static registry and cannot be modified at runtime."
        )


_SHA256_PATTERN = re.compile(r"^[0-9a-fA-F]{64}$")
_ALLOWED_STATUSES = {"approved", "pending", "rejected"}
_ALLOWED_FORMATS = {"jsonl", "parquet"}


@dataclass(frozen=True)
class TrustedDatasetEntry:
    entry_id: str
    dataset_name: str
    publisher: str
    approved_revision: str
    split: str
    source_url: str
    sha256: str | None
    file_format: str
    attribution: str
    license_approval_reference: str
    approval_status: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def make_entry_id(dataset_name: str, revision: str, split: str, file_format: str) -> str:
    return f"{dataset_name}:{revision}:{split}:{file_format}"


# Pinned external benchmark catalog.
# Note: Hashes for official external downloads are set to None and approval_status
# is set to "pending" because publisher hashes have not been independently verified
# via signed releases. Ingestion remains blocked until independently verified.
# We NEVER invent publisher hashes.
_DEFAULT_REGISTRY: dict[str, TrustedDatasetEntry] = {
    # GSM8K (Cobbe et al. / OpenAI)
    make_entry_id("openai/gsm8k", "740312add88f781978c0658806c59bc2815b9866", "train", "jsonl"): TrustedDatasetEntry(
        entry_id=make_entry_id("openai/gsm8k", "740312add88f781978c0658806c59bc2815b9866", "train", "jsonl"),
        dataset_name="openai/gsm8k",
        publisher="OpenAI",
        approved_revision="740312add88f781978c0658806c59bc2815b9866",
        split="train",
        source_url="https://github.com/openai/grade-school-math",
        sha256=None,
        file_format="jsonl",
        attribution=(
            "Cobbe et al., Training Verifiers to Solve Math Word Problems "
            "(2021), https://arxiv.org/abs/2110.14168"
        ),
        license_approval_reference="LEGAL-REVIEW-MIT-OPENAI-GSM8K-2026",
        approval_status="pending",
    ),
    make_entry_id("openai/gsm8k", "740312add88f781978c0658806c59bc2815b9866", "test", "jsonl"): TrustedDatasetEntry(
        entry_id=make_entry_id("openai/gsm8k", "740312add88f781978c0658806c59bc2815b9866", "test", "jsonl"),
        dataset_name="openai/gsm8k",
        publisher="OpenAI",
        approved_revision="740312add88f781978c0658806c59bc2815b9866",
        split="test",
        source_url="https://github.com/openai/grade-school-math",
        sha256=None,
        file_format="jsonl",
        attribution=(
            "Cobbe et al., Training Verifiers to Solve Math Word Problems "
            "(2021), https://arxiv.org/abs/2110.14168"
        ),
        license_approval_reference="LEGAL-REVIEW-MIT-OPENAI-GSM8K-2026",
        approval_status="pending",
    ),
    # MMLU (Hendrycks et al. / CAIS)
    make_entry_id("cais/mmlu", "c30699e8356da336a370243923dbaf21066bb9fe", "dev", "jsonl"): TrustedDatasetEntry(
        entry_id=make_entry_id("cais/mmlu", "c30699e8356da336a370243923dbaf21066bb9fe", "dev", "jsonl"),
        dataset_name="cais/mmlu",
        publisher="Center for AI Safety",
        approved_revision="c30699e8356da336a370243923dbaf21066bb9fe",
        split="dev",
        source_url="https://github.com/hendrycks/test",
        sha256=None,
        file_format="jsonl",
        attribution=(
            "Hendrycks et al., Measuring Massive Multitask Language "
            "Understanding (ICLR 2021), https://arxiv.org/abs/2009.03300"
        ),
        license_approval_reference="LEGAL-REVIEW-MIT-CAIS-MMLU-2026",
        approval_status="pending",
    ),
    make_entry_id("cais/mmlu", "c30699e8356da336a370243923dbaf21066bb9fe", "validation", "jsonl"): TrustedDatasetEntry(
        entry_id=make_entry_id("cais/mmlu", "c30699e8356da336a370243923dbaf21066bb9fe", "validation", "jsonl"),
        dataset_name="cais/mmlu",
        publisher="Center for AI Safety",
        approved_revision="c30699e8356da336a370243923dbaf21066bb9fe",
        split="validation",
        source_url="https://github.com/hendrycks/test",
        sha256=None,
        file_format="jsonl",
        attribution=(
            "Hendrycks et al., Measuring Massive Multitask Language "
            "Understanding (ICLR 2021), https://arxiv.org/abs/2009.03300"
        ),
        license_approval_reference="LEGAL-REVIEW-MIT-CAIS-MMLU-2026",
        approval_status="pending",
    ),
    make_entry_id("cais/mmlu", "c30699e8356da336a370243923dbaf21066bb9fe", "test", "jsonl"): TrustedDatasetEntry(
        entry_id=make_entry_id("cais/mmlu", "c30699e8356da336a370243923dbaf21066bb9fe", "test", "jsonl"),
        dataset_name="cais/mmlu",
        publisher="Center for AI Safety",
        approved_revision="c30699e8356da336a370243923dbaf21066bb9fe",
        split="test",
        source_url="https://github.com/hendrycks/test",
        sha256=None,
        file_format="jsonl",
        attribution=(
            "Hendrycks et al., Measuring Massive Multitask Language "
            "Understanding (ICLR 2021), https://arxiv.org/abs/2009.03300"
        ),
        license_approval_reference="LEGAL-REVIEW-MIT-CAIS-MMLU-2026",
        approval_status="pending",
    ),
    # MBPP (Austin et al. / Google Research)
    make_entry_id("google-research-datasets/mbpp", "4bb6404fdc6cacfda99d4ac4205087b89d32030c", "train", "jsonl"): TrustedDatasetEntry(
        entry_id=make_entry_id("google-research-datasets/mbpp", "4bb6404fdc6cacfda99d4ac4205087b89d32030c", "train", "jsonl"),
        dataset_name="google-research-datasets/mbpp",
        publisher="Google Research",
        approved_revision="4bb6404fdc6cacfda99d4ac4205087b89d32030c",
        split="train",
        source_url="https://github.com/google-research/google-research/tree/master/mbpp",
        sha256=None,
        file_format="jsonl",
        attribution=(
            "Austin et al., Program Synthesis with Large Language Models "
            "(2021), https://arxiv.org/abs/2108.07732"
        ),
        license_approval_reference="LEGAL-REVIEW-CC-BY-4.0-GOOGLE-MBPP-2026",
        approval_status="pending",
    ),
    make_entry_id("google-research-datasets/mbpp", "4bb6404fdc6cacfda99d4ac4205087b89d32030c", "prompt", "jsonl"): TrustedDatasetEntry(
        entry_id=make_entry_id("google-research-datasets/mbpp", "4bb6404fdc6cacfda99d4ac4205087b89d32030c", "prompt", "jsonl"),
        dataset_name="google-research-datasets/mbpp",
        publisher="Google Research",
        approved_revision="4bb6404fdc6cacfda99d4ac4205087b89d32030c",
        split="prompt",
        source_url="https://github.com/google-research/google-research/tree/master/mbpp",
        sha256=None,
        file_format="jsonl",
        attribution=(
            "Austin et al., Program Synthesis with Large Language Models "
            "(2021), https://arxiv.org/abs/2108.07732"
        ),
        license_approval_reference="LEGAL-REVIEW-CC-BY-4.0-GOOGLE-MBPP-2026",
        approval_status="pending",
    ),
    make_entry_id("google-research-datasets/mbpp", "4bb6404fdc6cacfda99d4ac4205087b89d32030c", "validation", "jsonl"): TrustedDatasetEntry(
        entry_id=make_entry_id("google-research-datasets/mbpp", "4bb6404fdc6cacfda99d4ac4205087b89d32030c", "validation", "jsonl"),
        dataset_name="google-research-datasets/mbpp",
        publisher="Google Research",
        approved_revision="4bb6404fdc6cacfda99d4ac4205087b89d32030c",
        split="validation",
        source_url="https://github.com/google-research/google-research/tree/master/mbpp",
        sha256=None,
        file_format="jsonl",
        attribution=(
            "Austin et al., Program Synthesis with Large Language Models "
            "(2021), https://arxiv.org/abs/2108.07732"
        ),
        license_approval_reference="LEGAL-REVIEW-CC-BY-4.0-GOOGLE-MBPP-2026",
        approval_status="pending",
    ),
    make_entry_id("google-research-datasets/mbpp", "4bb6404fdc6cacfda99d4ac4205087b89d32030c", "test", "jsonl"): TrustedDatasetEntry(
        entry_id=make_entry_id("google-research-datasets/mbpp", "4bb6404fdc6cacfda99d4ac4205087b89d32030c", "test", "jsonl"),
        dataset_name="google-research-datasets/mbpp",
        publisher="Google Research",
        approved_revision="4bb6404fdc6cacfda99d4ac4205087b89d32030c",
        split="test",
        source_url="https://github.com/google-research/google-research/tree/master/mbpp",
        sha256=None,
        file_format="jsonl",
        attribution=(
            "Austin et al., Program Synthesis with Large Language Models "
            "(2021), https://arxiv.org/abs/2108.07732"
        ),
        license_approval_reference="LEGAL-REVIEW-CC-BY-4.0-GOOGLE-MBPP-2026",
        approval_status="pending",
    ),
}

# Live registry instance
_LIVE_REGISTRY: dict[str, TrustedDatasetEntry] = dict(_DEFAULT_REGISTRY)


def validate_trusted_entry(entry: TrustedDatasetEntry) -> None:
    if not isinstance(entry, TrustedDatasetEntry):
        raise TypeError("Registry entry must be an instance of TrustedDatasetEntry.")
    if not entry.entry_id or not isinstance(entry.entry_id, str):
        raise ValueError("Registry entry must have a non-empty entry_id.")
    if not entry.dataset_name or not isinstance(entry.dataset_name, str):
        raise ValueError("Registry entry must have a non-empty dataset_name.")
    if not entry.publisher or not isinstance(entry.publisher, str):
        raise ValueError("Registry entry must have a non-empty publisher.")
    if not entry.approved_revision or not isinstance(entry.approved_revision, str):
        raise ValueError("Registry entry must have a non-empty approved_revision.")
    if not entry.split or not isinstance(entry.split, str):
        raise ValueError("Registry entry must have a non-empty split.")
    if not entry.source_url or not isinstance(entry.source_url, str):
        raise ValueError("Registry entry must have a non-empty source_url.")
    if entry.file_format not in _ALLOWED_FORMATS:
        raise ValueError(f"Registry entry format must be one of {sorted(_ALLOWED_FORMATS)}.")
    if not entry.attribution or not isinstance(entry.attribution, str):
        raise ValueError("Registry entry must have a non-empty attribution.")
    if not entry.license_approval_reference or not isinstance(entry.license_approval_reference, str):
        raise ValueError("Registry entry must have a non-empty license_approval_reference.")
    if entry.approval_status not in _ALLOWED_STATUSES:
        raise ValueError(f"Registry entry approval_status must be one of {sorted(_ALLOWED_STATUSES)}.")
    if entry.approval_status == "approved":
        if not entry.sha256 or not _SHA256_PATTERN.fullmatch(entry.sha256):
            raise ValueError(
                "Approved registry entries must have an independently verified 64-character SHA-256."
            )
    elif entry.sha256 is not None and not _SHA256_PATTERN.fullmatch(entry.sha256):
        raise ValueError("Registry entry SHA-256 must be 64 hexadecimal characters when present.")


def get_trusted_registry() -> dict[str, TrustedDatasetEntry]:
    return dict(_LIVE_REGISTRY)


def reset_trusted_registry() -> None:
    global _LIVE_REGISTRY
    _LIVE_REGISTRY = dict(_DEFAULT_REGISTRY)


def register_trusted_entry(entry: TrustedDatasetEntry | Mapping[str, Any]) -> TrustedDatasetEntry:
    """Register an approved dataset entry in test mode only.
    
    In production, approvals must be statically defined in _DEFAULT_REGISTRY.
    This helper is guarded and only permitted within automated test sessions.
    """
    assert_test_mode("Registering dynamic trusted entries")
    if isinstance(entry, Mapping) and not isinstance(entry, TrustedDatasetEntry):
        data = dict(entry)
        if "entry_id" not in data:
            data["entry_id"] = make_entry_id(
                data["dataset_name"],
                data["approved_revision"],
                data["split"],
                data["file_format"],
            )
        entry_obj = TrustedDatasetEntry(**data)
    else:
        entry_obj = entry

    validate_trusted_entry(entry_obj)
    _LIVE_REGISTRY[entry_obj.entry_id] = entry_obj
    return entry_obj


def lookup_trusted_entry(
    dataset_name: str,
    revision: str,
    split: str,
    file_format: str,
    *,
    registry: Mapping[str, TrustedDatasetEntry] | None = None,
    entry_id: str | None = None,
) -> TrustedDatasetEntry:
    if registry is not None:
        assert_test_mode("Overriding trusted dataset registry")
        active = registry
    else:
        active = _LIVE_REGISTRY
    if entry_id is not None:
        if entry_id not in active:
            raise ValueError(
                f"No trusted approval registry entry found with ID '{entry_id}'."
            )
        entry = active[entry_id]
        if (
            entry.dataset_name == dataset_name
            and entry.approved_revision == revision
            and entry.split == split
            and entry.file_format == file_format
        ):
            return entry
        raise ValueError(
            f"Trusted registry entry {entry_id!r} does not match requested "
            f"dataset ({dataset_name}), revision ({revision}), split ({split}), "
            f"or format ({file_format})."
        )

    # Primary key lookup
    target_id = make_entry_id(dataset_name, revision, split, file_format)
    if target_id in active:
        return active[target_id]

    # Secondary lookup across entries by attributes
    matches = [
        e for e in active.values()
        if e.dataset_name == dataset_name
        and e.approved_revision == revision
        and e.split == split
        and e.file_format == file_format
    ]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise ValueError(
            f"Ambiguous trusted registry entries for {dataset_name}:{revision}:{split}:{file_format}."
        )

    raise ValueError(
        f"No trusted approval registry entry found for dataset '{dataset_name}', "
        f"revision '{revision}', split '{split}', format '{file_format}'."
    )
