"""
routers/base.py

Unified runtime router interface for LLM Cost Copilot (M5).
Wraps the M3/M4 research routers (eval/routers/) into a consistent serving contract:
    route(prompt, lam) -> RouteDecision

Zero live API calls; loads cached embedding weights locally only.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any, Sequence

from eval.routers.decision import PROVISIONAL_PRESETS, resolve_preset_lambda
from eval.routers.encoders import BaseEncoder, SentenceTransformerEncoder

# Supported API model presets -> lambda cost penalty values.
# Empirical basis from M4.2 xRouteBench Pareto sweep (18 models, 3,729 prompts):
# - auto:quality (0.0): Empirically supported max-quality point (0.7208 Macro, $0.283/1k)
# - auto:balanced / auto (0.5): Empirically supported balanced knee (0.6664 Macro, $0.040/1k);
#   provisional for live provider pool until live calibration data is collected.
# - auto:cheap (2.0): Empirically supported cost-focused point (0.6508 Macro, $0.023/1k);
#   provisional for live provider pool until live calibration data is collected.
LAMBDA_PRESETS: dict[str, float] = {
    "auto": PROVISIONAL_PRESETS["auto:balanced"],
    **PROVISIONAL_PRESETS,
}


def resolve_runtime_lambda(preset_or_lambda: str | float) -> float:
    """Resolve 'auto', 'auto:quality', 'auto:balanced', 'auto:cheap', or numeric lambda."""
    if isinstance(preset_or_lambda, str):
        key = preset_or_lambda.strip().lower()
        if key in LAMBDA_PRESETS:
            return LAMBDA_PRESETS[key]
    return resolve_preset_lambda(preset_or_lambda)


def load_cached_encoder(
    model_name: str = "all-MiniLM-L6-v2",
) -> SentenceTransformerEncoder:
    """Load the locally cached SentenceTransformer encoder without network downloads."""
    return SentenceTransformerEncoder(model_name_or_path=model_name, local_files_only=True)


@dataclass(frozen=True)
class RouteDecision:
    """Standardized runtime routing decision returned by all RuntimeRouter implementations."""

    chosen_model: str
    predicted_scores: dict[str, float | None]
    expected_costs_usd: dict[str, float]
    normalized_costs: dict[str, float]
    utility_scores: dict[str, float | None]
    ranked_candidates: list[str]
    router_version: str
    lambda_value: float
    router_latency_ms: float
    metadata: dict[str, Any] = field(default_factory=dict)


class RuntimeRouter(abc.ABC):
    """Abstract base class for deployable M5 runtime routers."""

    @property
    @abc.abstractmethod
    def router_version(self) -> str:
        """Semantic/versioned identifier of the active router bundle."""
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def router_type(self) -> str:
        """Algorithm family ('knn' or 'learned_lr')."""
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def model_ids(self) -> list[str]:
        """Candidate model IDs known to this router bundle."""
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def encoder_name(self) -> str:
        """Name/identifier of the prompt embedding encoder."""
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def encoder_dim(self) -> int:
        """Expected embedding dimension."""
        raise NotImplementedError

    @abc.abstractmethod
    def route(
        self,
        prompt: str,
        lam: float | str = "auto",
        *,
        eligible_models: Sequence[str] | None = None,
        task_category: str = "unknown",
    ) -> RouteDecision:
        """Route a prompt under lambda preset or numeric value."""
        raise NotImplementedError
