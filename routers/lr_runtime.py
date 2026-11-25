"""
routers/lr_runtime.py

Runtime adapter for M3 LearnedRouter (regularized Logistic Regression)
with versioned, serializable bundles.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import pickle
import time
from typing import Any, Sequence

from eval.routers.decision import CostNormalizer, LambdaDecisionEngine
from eval.routers.encoders import BaseEncoder
from eval.routers.learned_router import LearnedRouter
from routers.base import (
    RouteDecision,
    RuntimeRouter,
    load_cached_encoder,
    resolve_runtime_lambda,
)


class LearnedLRRuntimeRouter(RuntimeRouter):
    """Runtime adapter around eval.routers.learned_router.LearnedRouter."""

    def __init__(
        self,
        learned_router: LearnedRouter,
        model_costs: dict[str, float],
        *,
        router_version: str = "lr-v1",
        encoder_name: str = "all-MiniLM-L6-v2",
        cost_normalization: str = "min_max",
        calibration_status: str = "calibrated",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        if not learned_router.is_fitted:
            raise ValueError("LearnedRouter must be fitted before wrapping in LearnedLRRuntimeRouter")
        if learned_router.encoder is None:
            raise ValueError("LearnedRouter must have a non-None encoder for runtime serving")

        self._lr = learned_router
        self._model_costs = {str(m): float(c) for m, c in model_costs.items()}
        self._router_version = str(router_version)
        self._encoder_name = str(encoder_name)
        self._cost_normalization = str(cost_normalization)
        self._calibration_status = str(calibration_status)
        self._metadata = dict(metadata or {})

    @property
    def router_version(self) -> str:
        return self._router_version

    @property
    def router_type(self) -> str:
        return "learned_lr"

    @property
    def model_ids(self) -> list[str]:
        return list(self._lr.candidate_models)

    @property
    def encoder_name(self) -> str:
        return self._encoder_name

    @property
    def encoder_dim(self) -> int:
        return int(self._lr.encoder.dimension)

    @property
    def model_costs(self) -> dict[str, float]:
        return dict(self._model_costs)

    @property
    def calibration_status(self) -> str:
        return self._calibration_status

    def route(
        self,
        prompt: str,
        lam: float | str = "auto",
        *,
        eligible_models: Sequence[str] | None = None,
        task_category: str = "unknown",
    ) -> RouteDecision:
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt must be a non-empty string")

        t0 = time.perf_counter()
        lam_val = resolve_runtime_lambda(lam)

        raw_preds_dict = self._lr.predict_proba([prompt], task_categories=[task_category])
        raw_preds = {m: float(arr[0]) for m, arr in raw_preds_dict.items()}

        pool = list(eligible_models) if eligible_models is not None else self.model_ids
        preds_for_pool: dict[str, float | None] = {}
        costs_for_pool: dict[str, float] = {}
        for m in pool:
            if m in raw_preds and m in self._model_costs:
                preds_for_pool[m] = float(raw_preds[m])
                costs_for_pool[m] = float(self._model_costs[m])
            else:
                preds_for_pool[m] = None
                if m in self._model_costs:
                    costs_for_pool[m] = float(self._model_costs[m])

        eligible_with_data = [
            m for m in pool if preds_for_pool.get(m) is not None and m in costs_for_pool
        ]
        if not eligible_with_data:
            raise ValueError(
                f"No eligible candidate models with valid predictions and costs in pool={pool}"
            )

        normalizer = CostNormalizer(costs_for_pool, strategy=self._cost_normalization)
        engine = LambdaDecisionEngine(normalizer)
        decision = engine.select_model(
            preds_for_pool,
            lambda_value=lam_val,
            available_models=eligible_with_data,
        )

        ranked = sorted(
            eligible_with_data,
            key=lambda m: (
                -(decision.model_utilities[m].utility or -1e9),
                costs_for_pool[m],
                m,
            ),
        )

        norm_costs = {
            m: decision.model_utilities[m].normalized_cost
            for m in eligible_with_data
            if decision.model_utilities[m].normalized_cost is not None
        }
        utilities = {
            m: decision.model_utilities[m].utility
            for m in pool
            if m in decision.model_utilities
        }

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return RouteDecision(
            chosen_model=decision.selected_model_id,
            predicted_scores=preds_for_pool,
            expected_costs_usd=costs_for_pool,
            normalized_costs=norm_costs,  # type: ignore[arg-type]
            utility_scores=utilities,
            ranked_candidates=ranked,
            router_version=self._router_version,
            lambda_value=lam_val,
            router_latency_ms=elapsed_ms,
            metadata={
                "router_type": self.router_type,
                "calibration_status": self._calibration_status,
                "task_category": task_category,
            },
        )

    def save_bundle(self, bundle_dir: str | Path) -> Path:
        """Serialize LearnedLR runtime bundle (manifest JSON + fitted state)."""
        out_dir = Path(bundle_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        manifest = {
            "router_version": self._router_version,
            "router_type": "learned_lr",
            "C": self._lr.C,
            "max_iter": self._lr.max_iter,
            "random_state": self._lr.random_state,
            "default_prior": self._lr.default_prior,
            "cost_normalization": self._cost_normalization,
            "model_ids": self.model_ids,
            "model_costs": self._model_costs,
            "encoder_name": self._encoder_name,
            "encoder_dim": self.encoder_dim,
            "calibration_status": self._calibration_status,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "metadata": self._metadata,
        }
        (out_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )

        state = {
            "feature_extractor": self._lr._feature_extractor,
            "classifiers": self._lr._classifiers,
            "fit_statuses": self._lr._fit_statuses,
            "candidate_models": self._lr._candidate_models,
        }
        with open(out_dir / "lr_state.pkl", "wb") as f:
            pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)
        return out_dir

    @classmethod
    def load_bundle(
        cls,
        bundle_dir: str | Path,
        *,
        encoder: BaseEncoder | None = None,
    ) -> LearnedLRRuntimeRouter:
        """Load a serialized LearnedLR runtime bundle and validate encoder compatibility."""
        b_dir = Path(bundle_dir)
        manifest_path = b_dir / "manifest.json"
        state_path = b_dir / "lr_state.pkl"
        if not manifest_path.exists() or not state_path.exists():
            raise FileNotFoundError(
                f"Incomplete LearnedLR bundle in '{b_dir}': expected manifest.json and lr_state.pkl"
            )

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("router_type") != "learned_lr":
            raise ValueError(
                f"Expected router_type='learned_lr', got {manifest.get('router_type')!r}"
            )

        enc = encoder if encoder is not None else load_cached_encoder(manifest["encoder_name"])
        if enc.dimension != int(manifest["encoder_dim"]):
            raise ValueError(
                f"Encoder dimension mismatch: bundle requires {manifest['encoder_dim']}, "
                f"got {enc.dimension}"
            )

        lr = LearnedRouter(
            encoder=enc,
            C=float(manifest["C"]),
            max_iter=int(manifest["max_iter"]),
            random_state=int(manifest["random_state"]),
            default_prior=float(manifest["default_prior"]),
        )
        with open(state_path, "rb") as f:
            state = pickle.load(f)

        lr._feature_extractor = state["feature_extractor"]
        lr._classifiers = state["classifiers"]
        lr._fit_statuses = state["fit_statuses"]
        lr._candidate_models = list(state["candidate_models"])
        lr._is_fitted = True

        return cls(
            learned_router=lr,
            model_costs=manifest["model_costs"],
            router_version=manifest["router_version"],
            encoder_name=manifest["encoder_name"],
            cost_normalization=manifest.get("cost_normalization", "min_max"),
            calibration_status=manifest.get("calibration_status", "calibrated"),
            metadata=manifest.get("metadata", {}),
        )
