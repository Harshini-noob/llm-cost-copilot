"""
routers/knn_runtime.py

Runtime adapter for M3 KNNRouter with versioned, serializable bundles.
Supports incremental new-model addition on existing calibration prompt embeddings.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import time
from typing import Any, Sequence

import numpy as np

from eval.routers.decision import CostNormalizer, LambdaDecisionEngine
from eval.routers.encoders import BaseEncoder
from eval.routers.knn_router import KNNRouter
from routers.base import (
    RouteDecision,
    RuntimeRouter,
    load_cached_encoder,
    resolve_runtime_lambda,
)


class KNNRuntimeRouter(RuntimeRouter):
    """Runtime adapter around eval.routers.knn_router.KNNRouter."""

    def __init__(
        self,
        knn_router: KNNRouter,
        model_costs: dict[str, float],
        *,
        router_version: str = "knn-v1",
        encoder_name: str = "all-MiniLM-L6-v2",
        cost_normalization: str = "min_max",
        calibration_status: str = "calibrated",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        if not knn_router.is_fitted:
            raise ValueError("KNNRouter must be fitted before wrapping in KNNRuntimeRouter")
        if knn_router.encoder is None:
            raise ValueError("KNNRouter must have a non-None encoder for runtime serving")

        self._knn = knn_router
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
        return "knn"

    @property
    def model_ids(self) -> list[str]:
        return list(self._knn.candidate_models)

    @property
    def encoder_name(self) -> str:
        return self._encoder_name

    @property
    def encoder_dim(self) -> int:
        return int(self._knn.encoder.dimension)

    @property
    def model_costs(self) -> dict[str, float]:
        return dict(self._model_costs)

    @property
    def calibration_status(self) -> str:
        return self._calibration_status

    def add_model_outcomes(
        self,
        model_id: str,
        outcomes_by_prompt_id: dict[str, float],
        expected_cost_usd: float,
    ) -> None:
        """Incrementally register or update a model using existing calibration prompt IDs.

        Does not require recalibrating or modifying any existing model's outcomes.
        """
        valid_pids = set(self._knn._train_prompt_ids)
        filtered = {
            str(pid): float(val)
            for pid, val in outcomes_by_prompt_id.items()
            if str(pid) in valid_pids
        }
        if model_id not in self._knn._candidate_models:
            self._knn._candidate_models.append(model_id)
        self._knn._model_outcomes[model_id] = filtered
        self._model_costs[model_id] = float(expected_cost_usd)

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

        emb = self._knn.encoder.encode_one(prompt)
        raw_preds_dict = self._knn.predict_proba(emb.reshape(1, -1))
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
        """Serialize kNN runtime bundle (manifest JSON + compressed npz state)."""
        out_dir = Path(bundle_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        manifest = {
            "router_version": self._router_version,
            "router_type": "knn",
            "k": self._knn.k,
            "weights": self._knn.weights,
            "metric": self._knn.metric,
            "default_prior": self._knn.default_prior,
            "cost_normalization": self._cost_normalization,
            "model_ids": self.model_ids,
            "model_costs": self._model_costs,
            "encoder_name": self._encoder_name,
            "encoder_dim": self.encoder_dim,
            "calibration_status": self._calibration_status,
            "n_train_prompts": len(self._knn._train_prompt_ids),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "metadata": self._metadata,
        }
        (out_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )

        np.savez_compressed(
            out_dir / "knn_state.npz",
            train_embeddings=self._knn._train_embeddings.astype(np.float32),
            train_prompt_ids=np.array(self._knn._train_prompt_ids, dtype=str),
            train_categories_json=np.array(
                json.dumps(self._knn._train_categories), dtype=str
            ),
            model_outcomes_json=np.array(
                json.dumps(self._knn._model_outcomes), dtype=str
            ),
        )
        return out_dir

    @classmethod
    def load_bundle(
        cls,
        bundle_dir: str | Path,
        *,
        encoder: BaseEncoder | None = None,
    ) -> KNNRuntimeRouter:
        """Load a serialized kNN runtime bundle and validate encoder compatibility."""
        b_dir = Path(bundle_dir)
        manifest_path = b_dir / "manifest.json"
        state_path = b_dir / "knn_state.npz"
        if not manifest_path.exists() or not state_path.exists():
            raise FileNotFoundError(
                f"Incomplete kNN router bundle in '{b_dir}': expected manifest.json and knn_state.npz"
            )

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("router_type") != "knn":
            raise ValueError(f"Expected router_type='knn', got {manifest.get('router_type')!r}")

        enc = encoder if encoder is not None else load_cached_encoder(manifest["encoder_name"])
        if enc.dimension != int(manifest["encoder_dim"]):
            raise ValueError(
                f"Encoder dimension mismatch: bundle requires {manifest['encoder_dim']}, "
                f"got {enc.dimension}"
            )

        knn = KNNRouter(
            k=int(manifest["k"]),
            encoder=enc,
            weights=str(manifest["weights"]),
            metric=str(manifest["metric"]),
            default_prior=float(manifest["default_prior"]),
        )
        with np.load(state_path, allow_pickle=False) as state:
            knn._train_embeddings = state["train_embeddings"].astype(np.float32)
            knn._train_prompt_ids = [str(x) for x in state["train_prompt_ids"].tolist()]
            knn._train_categories = json.loads(str(state["train_categories_json"].item()))
            knn._model_outcomes = json.loads(str(state["model_outcomes_json"].item()))

        knn._candidate_models = [str(m) for m in manifest["model_ids"]]
        knn._is_fitted = True

        return cls(
            knn_router=knn,
            model_costs=manifest["model_costs"],
            router_version=manifest["router_version"],
            encoder_name=manifest["encoder_name"],
            cost_normalization=manifest.get("cost_normalization", "min_max"),
            calibration_status=manifest.get("calibration_status", "calibrated"),
            metadata=manifest.get("metadata", {}),
        )
