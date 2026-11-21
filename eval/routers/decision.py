"""
eval/routers/decision.py

Lambda-based utility decision engine for intelligent LLM routing.
Combines predicted correctness probabilities from KNNRouter and LearnedRouter
with normalized expected model costs to select the optimal candidate LLM.

Decision Rule:
    U(m, x) = predicted_quality(m, x) - lambda * normalized_expected_cost(m, x)
    Select: argmax_m U(m, x)

Adheres strictly to the ZERO ADDITIONAL SPENDING and OFFLINE EVALUATION policies.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
import math
from typing import Any, Sequence

import numpy as np

logger = logging.getLogger(__name__)

# Configurable provisional presets mapping policy names to lambda values.
# Marked PROVISIONAL until Step 6 completes empirical Pareto frontier sweeps.
PROVISIONAL_PRESETS: dict[str, float] = {
    "auto:quality": 0.0,   # Purely quality-driven (zero cost penalty)
    "auto:balanced": 0.5,  # Provisional: balances accuracy and normalized cost
    "auto:cheap": 2.0,     # Provisional: strongly prioritizes lower cost
}


@dataclass(frozen=True)
class ModelUtilityBreakdown:
    """Detailed utility calculation breakdown for an individual candidate model."""
    model_id: str
    predicted_quality: float | None
    expected_cost_usd: float | None
    normalized_cost: float | None
    utility: float | None
    is_eligible: bool
    disqualification_reason: str | None = None


@dataclass(frozen=True)
class RoutingDecision:
    """Complete routing decision record for an input prompt."""
    selected_model_id: str
    predicted_quality: float
    expected_cost_usd: float
    normalized_cost: float
    lambda_value: float
    utility: float
    model_utilities: dict[str, ModelUtilityBreakdown]
    cost_normalization_strategy: str
    fallback_info: dict[str, Any] = field(default_factory=dict)


def resolve_preset_lambda(preset_or_lambda: str | float) -> float:
    """Resolve a preset name (e.g. 'auto:cheap') or float value into a valid lambda float."""
    if isinstance(preset_or_lambda, str):
        key = preset_or_lambda.strip().lower()
        if key in PROVISIONAL_PRESETS:
            return PROVISIONAL_PRESETS[key]
        try:
            val = float(key)
        except ValueError as err:
            raise ValueError(
                f"Unknown routing preset '{preset_or_lambda}'. "
                f"Valid presets are: {list(PROVISIONAL_PRESETS.keys())} or a non-negative float."
            ) from err
    else:
        val = float(preset_or_lambda)

    if not math.isfinite(val) or val < 0.0:
        raise ValueError(f"lambda_value must be a finite, non-negative float, got {val}")

    return val


class CostNormalizer:
    """Normalizes candidate model costs using training data with documented strategies.

    IMPORTANT NOTE ON CANDIDATE-SET RELATIVITY:
    -------------------------------------------
    Min-max normalization is strictly CANDIDATE-SET-RELATIVE:
        c_norm(m) = (cost(m) - min_cost) / (max_cost - min_cost)
    Its values are bounded in [0.0, 1.0], determined by the current pool of candidate models.
    If models are added or removed from the candidate pool (e.g. introducing an ultra-cheap
    model or removing an expensive frontier model), min_cost and max_cost shift, which will
    rescale the normalized costs across all remaining models.

    Raw USD costs (expected_cost_usd) remain absolute, immutable, and invariant to candidate
    pool modifications, and are always preserved in decision records alongside normalized costs.

    Strategies:
    - 'min_max' (default): (cost - min_cost) / (max_cost - min_cost) in [0.0, 1.0].
      If min_cost == max_cost, maps all costs to 0.0.
    - 'max_scale': cost / max_cost in [0.0, 1.0].
    - 'raw': preserves raw USD costs without scaling.
    """

    def __init__(
        self,
        model_costs: dict[str, float],
        strategy: str = "min_max",
    ) -> None:
        if strategy not in ("min_max", "max_scale", "raw"):
            raise ValueError(f"Unknown cost normalization strategy '{strategy}'")

        self.strategy = strategy
        self._raw_costs = {str(m): float(c) for m, c in model_costs.items()}

        for m, c in self._raw_costs.items():
            if not math.isfinite(c) or c < 0.0:
                raise ValueError(f"Cost for model '{m}' must be a finite non-negative number, got {c}")

        valid_costs = list(self._raw_costs.values())
        if valid_costs:
            self._min_cost = min(valid_costs)
            self._max_cost = max(valid_costs)
        else:
            self._min_cost = 0.0
            self._max_cost = 0.0

    @property
    def raw_costs(self) -> dict[str, float]:
        return dict(self._raw_costs)

    def normalize(self, model_id: str) -> tuple[float, float]:
        """Return (raw_cost_usd, normalized_cost) for a candidate model."""
        if model_id not in self._raw_costs:
            raise KeyError(f"Model '{model_id}' not found in CostNormalizer costs.")

        raw = self._raw_costs[model_id]

        if self.strategy == "raw":
            return raw, raw
        elif self.strategy == "max_scale":
            norm = (raw / self._max_cost) if self._max_cost > 0.0 else 0.0
            return raw, float(norm)
        else:  # min_max
            denom = self._max_cost - self._min_cost
            if denom > 1e-12:
                norm = (raw - self._min_cost) / denom
            else:
                norm = 0.0
            return raw, float(np.clip(norm, 0.0, 1.0))


class LambdaDecisionEngine:
    """Evaluates utility U(m, x) = predicted_quality - lambda * normalized_cost to select models.

    Parameters
    ----------
    cost_normalizer : CostNormalizer
        Fitted cost normalizer with per-model expected costs.
    cost_normalization_strategy : str
        Strategy name for audit metadata.
    """

    def __init__(
        self,
        cost_normalizer: CostNormalizer,
    ) -> None:
        self.cost_normalizer = cost_normalizer
        self.strategy = cost_normalizer.strategy

    @classmethod
    def from_training_dataset(
        cls,
        dataset: Any,
        strategy: str = "min_max",
        fallback_costs: dict[str, float] | None = None,
    ) -> LambdaDecisionEngine:
        """Construct decision engine using expected model costs derived strictly from training data.

        Estimand: Expected actual expenditure per attempted model call.
        --------------------------------------------------------------
        1. Includes valid observed costs for all attempted model calls, regardless of:
           - execution success or failure
           - evaluation status being scored or unscored
           - answer correctness
        2. Includes legitimate observed zero-cost attempts as zero (0.0).
        3. Includes failed or unscored attempts with nonzero observed costs (actual charges incurred).
        4. Missing costs (cost is None) remain missing; never silently replaced with zero.
        5. Negative or non-finite costs are rejected with ValueError.
        6. Preserves equal weighting across unique prompts:
           - First averages valid attempted-call costs within each prompt-model pair.
           - Then averages across unique prompts.
           - Repeated runs do not give any prompt disproportionate influence.
        7. If a candidate model has no valid observed costs, it requires an explicit fallback_costs entry.
        8. Operates exclusively on dataset.train_records; test data is never accessed.
        """
        if not hasattr(dataset, "train_records") or not hasattr(dataset, "candidate_models"):
            raise TypeError("Expected PreparedDataset with train_records and candidate_models")

        train_records = dataset.train_records
        candidate_models = list(dataset.candidate_models)

        model_costs: dict[str, float] = {}

        for m in candidate_models:
            # 1. Filter training records for model m
            m_records = [r for r in train_records if r.model_id == m]

            # 2. Group by prompt_id to aggregate repeated runs per prompt first
            prompt_costs_map: dict[str, list[float]] = {}
            for r in m_records:
                if r.cost is None:
                    # Missing costs remain missing; never replace with zero
                    continue

                cost_val = float(r.cost)
                if not math.isfinite(cost_val) or cost_val < 0.0:
                    raise ValueError(
                        f"Invalid cost {cost_val} observed for model '{m}' on prompt '{r.prompt_id}'. "
                        "Costs must be finite and non-negative."
                    )

                # Include all valid observed costs for attempted calls (zero, nonzero, success, failure)
                prompt_costs_map.setdefault(r.prompt_id, []).append(cost_val)

            # 3. Compute prompt-level average costs
            prompt_avg_costs = [
                float(np.mean(costs)) for costs in prompt_costs_map.values() if len(costs) > 0
            ]

            # 4. Each unique prompt contributes equal weight
            if prompt_avg_costs:
                model_costs[m] = float(np.mean(prompt_avg_costs))
            elif fallback_costs and m in fallback_costs:
                fallback_val = float(fallback_costs[m])
                if not math.isfinite(fallback_val) or fallback_val < 0.0:
                    raise ValueError(f"Invalid fallback cost for model '{m}': {fallback_val}")
                model_costs[m] = fallback_val
            else:
                raise ValueError(
                    f"No valid training costs observed for candidate model '{m}', "
                    "and no valid fallback_costs provided. Cannot estimate expected cost."
                )

        normalizer = CostNormalizer(model_costs=model_costs, strategy=strategy)
        return cls(cost_normalizer=normalizer)

    def select_model(
        self,
        predicted_qualities: dict[str, float | None],
        lambda_value: str | float = 0.0,
        available_models: Sequence[str] | set[str] | None = None,
    ) -> RoutingDecision:
        """Select the highest-utility candidate model for a single query.

        Parameters
        ----------
        predicted_qualities : dict[str, float | None]
            Mapping model_id -> predicted correctness probability in [0, 1].
        lambda_value : str | float
            Cost penalty parameter (e.g. 0.0, 0.5, 2.0) or preset ('auto:cheap').
        available_models : Sequence[str] | set[str] | None
            Optional subset of models currently available/healthy. Models not in this
            set are disqualified with reason 'deliberately_unavailable', distinguishing
            them from 'missing_prediction'.

        Returns
        -------
        RoutingDecision
            Decision containing selected model, predicted quality, expected cost, utility,
            and complete per-model breakdown with deterministic tie-breaking.
        """
        lam = resolve_preset_lambda(lambda_value)
        breakdowns: dict[str, ModelUtilityBreakdown] = {}
        eligible_candidates: list[tuple[float, float, str, ModelUtilityBreakdown]] = []

        all_models = sorted(self.cost_normalizer.raw_costs.keys())
        allowed_set = set(available_models) if available_models is not None else None

        for m in all_models:
            # 1. Check deliberate availability (e.g. service outages / intentional drop)
            if allowed_set is not None and m not in allowed_set:
                breakdowns[m] = ModelUtilityBreakdown(
                    model_id=m,
                    predicted_quality=predicted_qualities.get(m),
                    expected_cost_usd=self.cost_normalizer.raw_costs.get(m),
                    normalized_cost=None,
                    utility=None,
                    is_eligible=False,
                    disqualification_reason="deliberately_unavailable",
                )
                continue

            # 2. Check predicted quality (missing or unpredicted)
            if m not in predicted_qualities or predicted_qualities[m] is None:
                breakdowns[m] = ModelUtilityBreakdown(
                    model_id=m,
                    predicted_quality=None,
                    expected_cost_usd=self.cost_normalizer.raw_costs.get(m),
                    normalized_cost=None,
                    utility=None,
                    is_eligible=False,
                    disqualification_reason="missing_prediction",
                )
                continue

            q = float(predicted_qualities[m])
            if not math.isfinite(q) or q < 0.0 or q > 1.0:
                raise ValueError(f"Predicted quality for model '{m}' must be in [0, 1], got {q}")

            # Check cost
            raw_c, norm_c = self.cost_normalizer.normalize(m)

            # Compute utility U(m) = quality - lambda * normalized_cost
            util = float(q - lam * norm_c)

            item = ModelUtilityBreakdown(
                model_id=m,
                predicted_quality=q,
                expected_cost_usd=raw_c,
                normalized_cost=norm_c,
                utility=util,
                is_eligible=True,
                disqualification_reason=None,
            )
            breakdowns[m] = item
            # Sort tuple: (-utility, raw_c, m) for deterministic tie-breaking
            # 1. Maximize utility (hence -util)
            # 2. Minimize cost in tie-break (hence raw_c)
            # 3. Alphabetical tie-break on model_id
            eligible_candidates.append((-util, raw_c, m, item))

        if not eligible_candidates:
            raise RuntimeError("All candidate models were disqualified; cannot make routing decision.")

        # Deterministic sorting
        eligible_candidates.sort(key=lambda t: (t[0], t[1], t[2]))
        best_neg_u, best_cost, best_model_id, best_item = eligible_candidates[0]

        return RoutingDecision(
            selected_model_id=best_model_id,
            predicted_quality=best_item.predicted_quality,  # type: ignore[arg-type]
            expected_cost_usd=best_item.expected_cost_usd,  # type: ignore[arg-type]
            normalized_cost=best_item.normalized_cost,      # type: ignore[arg-type]
            lambda_value=lam,
            utility=best_item.utility,                      # type: ignore[arg-type]
            model_utilities=breakdowns,
            cost_normalization_strategy=self.strategy,
            fallback_info={},
        )

    def select_models_batch(
        self,
        batch_predictions: dict[str, np.ndarray],
        lambda_value: str | float = 0.0,
        available_models: Sequence[str] | set[str] | None = None,
    ) -> list[RoutingDecision]:
        """Vectorized/batched decision making across multiple queries."""
        lam = resolve_preset_lambda(lambda_value)
        models = sorted(batch_predictions.keys())
        if not models:
            return []

        n_queries = len(batch_predictions[models[0]])
        for m in models:
            if len(batch_predictions[m]) != n_queries:
                raise ValueError("All model prediction arrays in batch must have identical length.")

        decisions: list[RoutingDecision] = []
        for i in range(n_queries):
            query_preds = {m: float(batch_predictions[m][i]) for m in models}
            dec = self.select_model(
                query_preds,
                lambda_value=lam,
                available_models=available_models,
            )
            decisions.append(dec)

        return decisions
