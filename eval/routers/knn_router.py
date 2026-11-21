"""
eval/routers/knn_router.py

k-Nearest Neighbors (kNN) router for predicting LLM correctness probabilities.
Adheres strictly to the ZERO ADDITIONAL SPENDING and OFFLINE EVALUATION policies.

Key features:
1. Embeds query prompts using BaseEncoder or takes pre-computed embeddings.
2. Identifies k nearest training prompts using cosine or euclidean distance.
3. Estimates P(correct | prompt, model) from valid empirical training outcomes.
4. Supports uniform and distance-weighted voting schemes.
5. Multi-model support with per-model neighbor filtering for missing evaluations.
6. Robust edge-case handling:
   - k > N (clamps to available neighbors)
   - Zero-distance / identical embeddings (deterministic prompt_id tie-breaking)
   - Missing model outcomes (filters to evaluated training prompts per model)
   - Zero training examples (graceful prior fallback)
   - Single-class training labels (e.g. all 1s or all 0s)
   - Empty input sequence handling
7. Strict leakage prevention: fits strictly on train records; never accesses test data.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Any, Sequence

import numpy as np

from eval.routers.data_prep import PreparedDataset
from eval.routers.encoders import BaseEncoder

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class NeighborInfo:
    """Diagnostic information for a retrieved neighbor."""
    prompt_id: str
    distance: float
    task_category: str
    outcomes: dict[str, float | None]  # model_id -> empirical quality


class KNNRouter:
    """kNN classifier predicting P(correct | prompt, model) across candidate models.

    Parameters
    ----------
    k : int
        Number of nearest neighbors to consider (default: 3).
    encoder : BaseEncoder | None
        Text encoder for query prompts. If None, predict_proba accepts pre-computed embeddings.
    weights : str
        Weighting scheme: 'distance' (default, inverse distance) or 'uniform'.
    metric : str
        Distance metric: 'cosine' (default) or 'euclidean'.
    default_prior : float
        Fallback probability when no training data is available for a model (default: 0.5).
    """

    def __init__(
        self,
        k: int = 3,
        encoder: BaseEncoder | None = None,
        *,
        weights: str = "distance",
        metric: str = "cosine",
        default_prior: float = 0.5,
    ) -> None:
        if not isinstance(k, int) or k <= 0:
            raise ValueError(f"k must be a positive integer, got {k}")
        if weights not in ("distance", "uniform"):
            raise ValueError(f"weights must be 'distance' or 'uniform', got '{weights}'")
        if metric not in ("cosine", "euclidean"):
            raise ValueError(f"metric must be 'cosine' or 'euclidean', got '{metric}'")
        if not 0.0 <= default_prior <= 1.0:
            raise ValueError(f"default_prior must be in [0, 1], got {default_prior}")

        self.k = k
        self.encoder = encoder
        self.weights = weights
        self.metric = metric
        self.default_prior = default_prior

        # Fitted state
        self._is_fitted = False
        self._candidate_models: list[str] = []
        self._train_prompt_ids: list[str] = []
        self._train_categories: dict[str, str] = {}
        self._train_embeddings: np.ndarray = np.empty((0, 0), dtype=np.float32)
        # Mapping: model_id -> dict of {prompt_id: empirical_quality}
        self._model_outcomes: dict[str, dict[str, float]] = {}

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

    @property
    def candidate_models(self) -> list[str]:
        return list(self._candidate_models)

    def fit(self, dataset: Any) -> KNNRouter:
        """Fit router on training split of PreparedDataset with strict leakage prevention.

        Uses strictly dataset.train_records and dataset.prompts for prompts in the train split.
        Never reads or accesses test split records.
        """
        if not hasattr(dataset, "train_records") or not hasattr(dataset, "prompts"):
            raise TypeError(f"Expected PreparedDataset, got {type(dataset).__name__}")

        train_records = dataset.train_records
        if hasattr(dataset, "train_prompt_ids"):
            train_pids = sorted(dataset.train_prompt_ids)
        else:
            train_pids = sorted({r.prompt_id for r in train_records})

        if hasattr(dataset, "candidate_models"):
            self._candidate_models = list(dataset.candidate_models)
        else:
            self._candidate_models = sorted({r.model_id for r in train_records})

        # Extract embeddings for train prompts only
        embeddings_list = []
        valid_train_pids = []

        for pid in train_pids:
            pr = dataset.prompts.get(pid)
            if pr is None or pr.prompt_embedding is None:
                continue
            embeddings_list.append(pr.prompt_embedding)
            valid_train_pids.append(pid)
            self._train_categories[pid] = pr.task_category

        self._train_prompt_ids = valid_train_pids
        if embeddings_list:
            self._train_embeddings = np.asarray(embeddings_list, dtype=np.float32)
        else:
            self._train_embeddings = np.empty((0, 0), dtype=np.float32)

        # Build per-model empirical quality outcomes strictly from train_records
        self._model_outcomes = {m: {} for m in self._candidate_models}
        for r in train_records:
            if r.label is not None and r.prompt_id in self._train_prompt_ids:
                # Store empirical quality in [0, 1]
                self._model_outcomes[r.model_id][r.prompt_id] = float(r.label)

        self._is_fitted = True
        return self

    def predict_proba(
        self,
        inputs: Sequence[str] | np.ndarray,
    ) -> dict[str, np.ndarray]:
        """Predict P(correct | prompt, model) for each candidate model across input queries.

        Parameters
        ----------
        inputs : Sequence[str] | np.ndarray
            Either a sequence of prompt text strings (encoded via self.encoder)
            or an (N, D) numpy array of pre-computed prompt embeddings.

        Returns
        -------
        dict[str, np.ndarray]
            Mapping model_id -> 1D float32 array of shape (N,) with probabilities in [0.0, 1.0].
        """
        if not self._is_fitted:
            raise RuntimeError("KNNRouter must be fitted before predict_proba can be called.")

        # 1. Resolve embeddings
        if isinstance(inputs, np.ndarray):
            if inputs.ndim == 1:
                query_embs = inputs.reshape(1, -1).astype(np.float32)
            elif inputs.ndim == 2:
                query_embs = inputs.astype(np.float32)
            else:
                raise ValueError(f"Query embeddings array must be 1D or 2D, got shape {inputs.shape}")
        elif isinstance(inputs, (list, tuple)):
            if len(inputs) == 0:
                return {m: np.empty(0, dtype=np.float32) for m in self._candidate_models}
            if self.encoder is None:
                raise ValueError("An encoder must be provided to KNNRouter to encode raw text queries.")
            query_embs = self.encoder.encode(inputs)
        else:
            raise TypeError(f"Unsupported inputs type: {type(inputs).__name__}")

        n_queries = len(query_embs)
        if n_queries == 0:
            return {m: np.empty(0, dtype=np.float32) for m in self._candidate_models}

        # 2. Compute probabilities per candidate model
        results: dict[str, np.ndarray] = {}
        for model_id in self._candidate_models:
            model_labels_map = self._model_outcomes.get(model_id, {})

            # Filter to train prompts with valid evaluated labels for this model
            available_pids = [
                pid for pid in self._train_prompt_ids if pid in model_labels_map
            ]
            n_available = len(available_pids)

            if n_available == 0 or len(self._train_embeddings) == 0:
                # No training examples for this model: return default prior
                results[model_id] = np.full(n_queries, self.default_prior, dtype=np.float32)
                continue

            # Sub-matrix of embeddings for available training prompts
            sub_indices = [self._train_prompt_ids.index(pid) for pid in available_pids]
            sub_embeddings = self._train_embeddings[sub_indices]
            sub_labels = np.asarray([model_labels_map[pid] for pid in available_pids], dtype=np.float32)

            # Compute distances for each query to all available candidate prompts
            probs = np.zeros(n_queries, dtype=np.float32)
            effective_k = min(self.k, n_available)

            for i in range(n_queries):
                q = query_embs[i]
                dists = self._compute_distances(q, sub_embeddings)

                # Deterministic sorting: by distance ascending, then secondary tie-break on prompt_id
                sort_keys = [
                    (float(dists[j]), available_pids[j], j) for j in range(n_available)
                ]
                sort_keys.sort(key=lambda item: (item[0], item[1]))

                top_k_indices = [item[2] for item in sort_keys[:effective_k]]
                top_k_dists = np.asarray([dists[j] for j in top_k_indices], dtype=np.float32)
                top_k_labels = sub_labels[top_k_indices]

                # Weighting
                if self.weights == "uniform":
                    p = float(np.mean(top_k_labels))
                else:  # distance-weighted
                    # Check for exact matches (zero or near-zero distance)
                    zero_mask = top_k_dists < 1e-9
                    if np.any(zero_mask):
                        p = float(np.mean(top_k_labels[zero_mask]))
                    else:
                        w = 1.0 / np.maximum(top_k_dists, 1e-7)
                        w_sum = float(np.sum(w))
                        if w_sum > 0:
                            p = float(np.sum(w * top_k_labels) / w_sum)
                        else:
                            p = float(np.mean(top_k_labels))

                probs[i] = float(np.clip(p, 0.0, 1.0))

            results[model_id] = probs

        return results

    def predict_proba_one(self, prompt_text_or_emb: str | np.ndarray) -> dict[str, float]:
        """Convenience method returning {model_id: prob} for a single prompt."""
        if isinstance(prompt_text_or_emb, str):
            res = self.predict_proba([prompt_text_or_emb])
        elif isinstance(prompt_text_or_emb, np.ndarray):
            res = self.predict_proba(prompt_text_or_emb.reshape(1, -1))
        else:
            raise TypeError(f"Expected str or np.ndarray, got {type(prompt_text_or_emb).__name__}")
        return {m: float(probs[0]) for m, probs in res.items()}

    def get_neighbors(
        self,
        prompt_text_or_emb: str | np.ndarray,
        k: int | None = None,
        model_id: str | None = None,
    ) -> list[NeighborInfo]:
        """Diagnostic helper returning top-k nearest neighbor prompts for an input query."""
        if not self._is_fitted:
            raise RuntimeError("KNNRouter must be fitted before get_neighbors can be called.")

        if isinstance(prompt_text_or_emb, str):
            if self.encoder is None:
                raise ValueError("An encoder must be configured to inspect raw text neighbors.")
            q_emb = self.encoder.encode_one(prompt_text_or_emb)
        elif isinstance(prompt_text_or_emb, np.ndarray):
            q_emb = prompt_text_or_emb.ravel().astype(np.float32)
        else:
            raise TypeError("Expected str or np.ndarray")

        if len(self._train_prompt_ids) == 0:
            return []

        # Filter by model_id if requested
        if model_id is not None:
            available_pids = [
                pid for pid in self._train_prompt_ids if pid in self._model_outcomes.get(model_id, {})
            ]
        else:
            available_pids = self._train_prompt_ids

        if not available_pids:
            return []

        sub_indices = [self._train_prompt_ids.index(pid) for pid in available_pids]
        sub_embs = self._train_embeddings[sub_indices]
        dists = self._compute_distances(q_emb, sub_embs)

        sort_keys = [
            (float(dists[j]), available_pids[j], j) for j in range(len(available_pids))
        ]
        sort_keys.sort(key=lambda item: (item[0], item[1]))

        limit_k = min(k if k is not None else self.k, len(available_pids))
        neighbors: list[NeighborInfo] = []
        for d, pid, _ in sort_keys[:limit_k]:
            outcomes = {
                m: self._model_outcomes[m].get(pid) for m in self._candidate_models
            }
            neighbors.append(
                NeighborInfo(
                    prompt_id=pid,
                    distance=d,
                    task_category=self._train_categories.get(pid, "unknown"),
                    outcomes=outcomes,
                )
            )

        return neighbors

    def _compute_distances(self, query: np.ndarray, train_matrix: np.ndarray) -> np.ndarray:
        """Compute distances from 1D query vector to each row in train_matrix."""
        if self.metric == "cosine":
            # Cosine distance = 1 - cosine_similarity
            q_norm = np.linalg.norm(query)
            m_norms = np.linalg.norm(train_matrix, axis=1)

            denom = np.maximum(q_norm * m_norms, 1e-12)
            dots = np.dot(train_matrix, query)
            cos_sim = dots / denom
            cos_dist = 1.0 - cos_sim
            return np.maximum(cos_dist, 0.0).astype(np.float32)
        else:  # euclidean
            diff = train_matrix - query
            return np.sqrt(np.sum(diff ** 2, axis=1)).astype(np.float32)
