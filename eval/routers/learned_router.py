"""
eval/routers/learned_router.py

Learned LLM selection router using regularized Logistic Regression classifiers.
Adheres strictly to the ZERO ADDITIONAL SPENDING and OFFLINE EVALUATION policies.

Key architectural features:
1. Feature Engineering:
   - Dense prompt embedding vectors from BaseEncoder.
   - Standardized prompt length (z-score normalized on training prompts only).
   - Code-presence indicator heuristic.
   - One-hot task category encoding (fit strictly on categories in training data).
2. Repeated-Run Handling (Weighted Binary Observations):
   - For empirical quality q = correct_runs / evaluated_runs:
     - Adds label 1 with sample weight q (if q > 0).
     - Adds label 0 with sample weight (1 - q) (if q < 1).
   - Total sample weight per unique prompt-model pair strictly equals 1.0.
   - Repeated runs never distort neighbor or training influence.
3. Model Architecture & Multi-Model Support:
   - One regularized Logistic Regression classifier per candidate model (L2 penalty).
   - Configurable C, solver, max_iter, and deterministic random_state.
4. Edge-Case & Insufficient-Data Handling:
   - Handles models with no training observations (fallback to prior).
   - Handles models with single-class outcomes (all 1s or all 0s, fallback with exposed reason).
   - Handles tiny datasets safely.
   - Transparently exposes ModelFitStatus per candidate model.
5. Strict Leakage Prevention:
   - Feature scalers and category vocabularies are fit strictly on training prompts.
   - Test prompts, outcomes, and labels are completely isolated from fitting and calibration.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
import re
from typing import Any, Sequence

import numpy as np
from sklearn.linear_model import LogisticRegression

from eval.routers.data_prep import PreparedDataset, PromptRecord
from eval.routers.encoders import BaseEncoder

logger = logging.getLogger(__name__)

# Heuristic patterns identifying programming code in prompts
CODE_PATTERN = re.compile(
    r"(```|"
    r"\bdef\s+\w+\s*\(|"
    r"\bclass\s+\w+\b|"
    r"\bimport\s+[\w\.]+|"
    r"\bfrom\s+[\w\.]+\s+import|"
    r"\breturn\b|"
    r"\bfor\s+\w+\s+in\b|"
    r"\bfunction\s*\(|"
    r"\bpublic\s+static\s+void|"
    r"#include\s*<|"
    r"\bconsole\.log\(|"
    r"\bprint\()",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class ModelFitStatus:
    """Detailed fitting diagnostic for a specific candidate LLM classifier."""
    model_id: str
    is_fitted: bool
    fallback_reason: str | None  # "no_training_data", "single_class_observed", "insufficient_samples", or None
    train_prompt_count: int
    observed_classes: list[int]
    mean_empirical_quality: float | None
    fallback_prob: float
    success_count: float = 0.0
    evaluated_observation_count: float = 0.0


class PromptFeatureExtractor:
    """Extracts and normalizes features from prompt texts, embeddings, and categories.

    Guarantees strict leakage prevention: parameters (mean, std, category vocabulary)
    are fit strictly on training prompts. Test data is transformed using fitted parameters only.
    """

    def __init__(self, known_categories: list[str] | None = None) -> None:
        self.known_categories = list(known_categories) if known_categories is not None else None
        self._is_fitted = False
        self._len_mean = 0.0
        self._len_std = 1.0
        self._categories_vocab: list[str] = []
        self._feature_dim = 0

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

    @property
    def feature_dimension(self) -> int:
        return self._feature_dim

    def fit(
        self,
        prompts: Sequence[PromptRecord] | Sequence[dict[str, Any]],
        sample_embedding_dim: int,
    ) -> PromptFeatureExtractor:
        """Fit feature scaler and category vocabulary on training prompts."""
        lengths = []
        categories_set = set()

        for p in prompts:
            text = p.prompt_text if isinstance(p, PromptRecord) else p.get("prompt_text", p.get("prompt", ""))
            cat = p.task_category if isinstance(p, PromptRecord) else p.get("task_category", p.get("task_type", "unknown"))

            lengths.append(float(len(text)))
            if cat and cat != "unknown":
                categories_set.add(str(cat))

        if lengths:
            self._len_mean = float(np.mean(lengths))
            self._len_std = float(np.std(lengths))
            if self._len_std < 1e-6:
                self._len_std = 1.0
        else:
            self._len_mean = 0.0
            self._len_std = 1.0

        if self.known_categories is not None:
            self._categories_vocab = sorted(self.known_categories)
        else:
            self._categories_vocab = sorted(categories_set)

        # Total feature dimension = embedding_dim + 1 (scaled length) + 1 (is_code) + len(categories_vocab)
        self._feature_dim = sample_embedding_dim + 2 + len(self._categories_vocab)
        self._is_fitted = True
        return self

    def transform(
        self,
        embeddings: np.ndarray,
        texts: Sequence[str],
        categories: Sequence[str] | None = None,
    ) -> np.ndarray:
        """Transform inputs into a 2D float32 feature matrix using fitted parameters."""
        if not self._is_fitted:
            raise RuntimeError("PromptFeatureExtractor must be fitted on training data before transform.")

        n = len(texts)
        if n == 0 or embeddings.shape[0] == 0:
            return np.empty((0, self._feature_dim), dtype=np.float32)

        emb_dim = embeddings.shape[1]
        out = np.zeros((n, self._feature_dim), dtype=np.float32)

        # 1. Embeddings
        out[:, :emb_dim] = embeddings.astype(np.float32)

        # 2. Scaled length & Code indicator & Category one-hot
        cat_start = emb_dim + 2
        cat_to_idx = {c: i for i, c in enumerate(self._categories_vocab)}

        for i, text in enumerate(texts):
            # Scaled length
            raw_len = float(len(text))
            out[i, emb_dim] = (raw_len - self._len_mean) / self._len_std

            # Code indicator
            is_code = bool(CODE_PATTERN.search(text))
            cat = categories[i] if categories is not None and i < len(categories) else "unknown"
            if cat == "code":
                is_code = True
            out[i, emb_dim + 1] = 1.0 if is_code else 0.0

            # One-hot category
            if cat in cat_to_idx:
                out[i, cat_start + cat_to_idx[cat]] = 1.0

        return out


class LearnedRouter:
    """Learned routing policy predicting P(correct | prompt_features, model_id).

    Trains one regularized Logistic Regression binary classifier per candidate model
    using weighted binary observations for fractional empirical success rates.

    Parameters
    ----------
    C : float
        Inverse regularization parameter for L2 regularization (default: 1.0).
    solver : str
        Scikit-learn optimization solver (default: 'lbfgs').
    max_iter : int
        Maximum iterations for convergence (default: 1000).
    random_state : int
        Deterministic random seed (default: 42).
    encoder : BaseEncoder | None
        Text encoder for query prompts.
    default_prior : float
        Fallback probability when no training data is available (default: 0.5).
    """

    def __init__(
        self,
        C: float = 1.0,
        *,
        solver: str = "lbfgs",
        max_iter: int = 1000,
        random_state: int = 42,
        encoder: BaseEncoder | None = None,
        smoothing_alpha: float = 1.0,
        smoothing_beta: float = 1.0,
        default_prior: float = 0.5,
    ) -> None:
        if C <= 0:
            raise ValueError(f"Regularization parameter C must be positive, got {C}")
        if smoothing_alpha <= 0.0 or smoothing_beta <= 0.0:
            raise ValueError(
                f"smoothing_alpha and smoothing_beta must be positive, got alpha={smoothing_alpha}, beta={smoothing_beta}"
            )
        if not 0.0 <= default_prior <= 1.0:
            raise ValueError(f"default_prior must be in [0, 1], got {default_prior}")

        self.C = float(C)
        self.solver = solver
        self.max_iter = max_iter
        self.random_state = random_state
        self.encoder = encoder
        self.smoothing_alpha = float(smoothing_alpha)
        self.smoothing_beta = float(smoothing_beta)
        self.default_prior = float(default_prior)

        # Fitted state
        self._is_fitted = False
        self._candidate_models: list[str] = []
        self._feature_extractor: PromptFeatureExtractor | None = None
        self._classifiers: dict[str, LogisticRegression] = {}
        self._fit_statuses: dict[str, ModelFitStatus] = {}

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

    @property
    def candidate_models(self) -> list[str]:
        return list(self._candidate_models)

    @property
    def model_fit_statuses(self) -> dict[str, ModelFitStatus]:
        return dict(self._fit_statuses)

    @property
    def feature_dimension(self) -> int:
        if self._feature_extractor is not None:
            return self._feature_extractor.feature_dimension
        return 0

    def compute_smoothed_fallback(self, successes: float, total_evals: float) -> float:
        """Compute Beta-smoothed empirical fallback probability:

        P(correct) = (successes + alpha) / (total_evals + alpha + beta)
        """
        return float((successes + self.smoothing_alpha) / (total_evals + self.smoothing_alpha + self.smoothing_beta))

    def fit(self, dataset: Any) -> LearnedRouter:
        """Fit feature extractor and per-model Logistic Regression classifiers on training data.

        Strict leakage prevention:
        Uses strictly dataset.train_records and training prompt representations.
        Test data is never accessed during fitting.
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

        # 1. Collect training prompt metadata
        train_prompts: list[PromptRecord] = []
        train_embeddings_list: list[np.ndarray] = []
        valid_train_pids: list[str] = []

        for pid in train_pids:
            pr = dataset.prompts.get(pid)
            if pr is None or pr.prompt_embedding is None:
                continue
            train_prompts.append(pr)
            train_embeddings_list.append(pr.prompt_embedding)
            valid_train_pids.append(pid)

        if not train_embeddings_list:
            raise ValueError("No training prompts with valid embeddings available for fitting.")

        sample_emb = train_embeddings_list[0]
        sample_dim = sample_emb.shape[0]

        # 2. Fit feature extractor strictly on training prompts
        self._feature_extractor = PromptFeatureExtractor()
        self._feature_extractor.fit(train_prompts, sample_embedding_dim=sample_dim)

        # 3. Transform training prompts into unified feature matrix
        train_emb_matrix = np.asarray(train_embeddings_list, dtype=np.float32)
        train_texts = [p.prompt_text for p in train_prompts]
        train_cats = [p.task_category for p in train_prompts]

        X_train_prompts = self._feature_extractor.transform(
            train_emb_matrix,
            train_texts,
            train_cats,
        )
        pid_to_feature = {
            valid_train_pids[i]: X_train_prompts[i] for i in range(len(valid_train_pids))
        }

        # 4. Fit one Logistic Regression classifier per candidate model
        self._classifiers = {}
        self._fit_statuses = {}

        for model_id in self._candidate_models:
            # Filter training records for this model with evaluated labels
            model_records = [
                r for r in train_records
                if r.model_id == model_id and r.label is not None and r.prompt_id in pid_to_feature
            ]

            n_samples = len(model_records)
            if n_samples == 0:
                fallback_p = self.compute_smoothed_fallback(0.0, 0.0)
                self._fit_statuses[model_id] = ModelFitStatus(
                    model_id=model_id,
                    is_fitted=False,
                    fallback_reason="no_training_data",
                    train_prompt_count=0,
                    observed_classes=[],
                    mean_empirical_quality=None,
                    fallback_prob=fallback_p,
                    success_count=0.0,
                    evaluated_observation_count=0.0,
                )
                continue

            # Repeated-Run Handling: Weighted Binary Observations
            # For each unique prompt-model pair with empirical quality q:
            # - Add label 1 with sample weight q (if q > 0)
            # - Add label 0 with sample weight (1 - q) (if q < 1)
            X_rows: list[np.ndarray] = []
            y_rows: list[int] = []
            weights: list[float] = []
            q_values: list[float] = []
            total_successes = 0.0
            total_evals = 0.0

            for r in model_records:
                q = float(r.label)
                q_values.append(q)
                x_vec = pid_to_feature[r.prompt_id]

                if hasattr(r, "correct_runs") and hasattr(r, "evaluated_runs") and r.evaluated_runs > 0:
                    total_successes += float(r.correct_runs)
                    total_evals += float(r.evaluated_runs)
                else:
                    total_successes += q
                    total_evals += 1.0

                if q > 0.0:
                    X_rows.append(x_vec)
                    y_rows.append(1)
                    weights.append(q)

                if q < 1.0:
                    X_rows.append(x_vec)
                    y_rows.append(0)
                    weights.append(1.0 - q)

            mean_q = float(np.mean(q_values))
            observed_classes = sorted(list(set(y_rows)))

            # Handle single-class training targets (e.g. all 1s or all 0s)
            if len(observed_classes) < 2:
                fallback_p = self.compute_smoothed_fallback(total_successes, total_evals)
                self._fit_statuses[model_id] = ModelFitStatus(
                    model_id=model_id,
                    is_fitted=False,
                    fallback_reason="single_class_observed",
                    train_prompt_count=n_samples,
                    observed_classes=observed_classes,
                    mean_empirical_quality=mean_q,
                    fallback_prob=fallback_p,
                    success_count=total_successes,
                    evaluated_observation_count=total_evals,
                )
                continue

            # Fit regularized Logistic Regression with sample weights
            X_train_model = np.asarray(X_rows, dtype=np.float32)
            y_train_model = np.asarray(y_rows, dtype=np.int64)
            w_train_model = np.asarray(weights, dtype=np.float32)

            clf = LogisticRegression(
                C=self.C,
                solver=self.solver,
                max_iter=self.max_iter,
                random_state=self.random_state,
            )
            clf.fit(X_train_model, y_train_model, sample_weight=w_train_model)

            self._classifiers[model_id] = clf
            self._fit_statuses[model_id] = ModelFitStatus(
                model_id=model_id,
                is_fitted=True,
                fallback_reason=None,
                train_prompt_count=n_samples,
                observed_classes=observed_classes,
                mean_empirical_quality=mean_q,
                fallback_prob=self.compute_smoothed_fallback(total_successes, total_evals),
                success_count=total_successes,
                evaluated_observation_count=total_evals,
            )

        self._is_fitted = True
        return self

    def predict_proba(
        self,
        inputs: Sequence[str] | Sequence[PromptRecord] | np.ndarray,
        task_categories: Sequence[str] | None = None,
    ) -> dict[str, np.ndarray]:
        """Predict P(correct | prompt_features, model_id) for every candidate model.

        Parameters
        ----------
        inputs : Sequence[str] | Sequence[PromptRecord] | np.ndarray
            Prompt queries as strings, PromptRecord objects, or pre-computed embeddings.
        task_categories : Sequence[str] | None
            Optional task categories corresponding to input queries.

        Returns
        -------
        dict[str, np.ndarray]
            Mapping model_id -> 1D float32 array of shape (N,) with probabilities in [0.0, 1.0].
        """
        if not self._is_fitted or self._feature_extractor is None:
            raise RuntimeError("LearnedRouter must be fitted before predict_proba can be called.")

        # 1. Resolve prompt texts, categories, and embeddings
        texts: list[str] = []
        cats: list[str] = []
        embs_list: list[np.ndarray] = []

        if isinstance(inputs, np.ndarray):
            if inputs.ndim == 1:
                embeddings = inputs.reshape(1, -1).astype(np.float32)
            elif inputs.ndim == 2:
                embeddings = inputs.astype(np.float32)
            else:
                raise ValueError(f"Embeddings array must be 1D or 2D, got {inputs.shape}")
            n = len(embeddings)
            texts = [f"query_{i}" for i in range(n)]
            cats = list(task_categories) if task_categories is not None else ["unknown"] * n
        elif isinstance(inputs, (list, tuple)):
            n = len(inputs)
            if n == 0:
                return {m: np.empty(0, dtype=np.float32) for m in self._candidate_models}

            if isinstance(inputs[0], PromptRecord):
                for p in inputs:
                    texts.append(p.prompt_text)
                    cats.append(p.task_category)
                    if p.prompt_embedding is not None:
                        embs_list.append(p.prompt_embedding)
                if len(embs_list) == n:
                    embeddings = np.asarray(embs_list, dtype=np.float32)
                elif self.encoder is not None:
                    embeddings = self.encoder.encode(texts)
                else:
                    raise ValueError("Embeddings missing from PromptRecord and no encoder provided.")
            elif isinstance(inputs[0], str):
                texts = list(inputs)
                cats = list(task_categories) if task_categories is not None else ["unknown"] * n
                if self.encoder is None:
                    raise ValueError("An encoder must be provided to LearnedRouter to embed raw text queries.")
                embeddings = self.encoder.encode(texts)
            else:
                raise TypeError(f"Unsupported input element type: {type(inputs[0]).__name__}")
        else:
            raise TypeError(f"Unsupported inputs type: {type(inputs).__name__}")

        if len(texts) == 0:
            return {m: np.empty(0, dtype=np.float32) for m in self._candidate_models}

        # 2. Extract unified feature representations
        X_features = self._feature_extractor.transform(embeddings, texts, cats)

        # 3. Predict per model
        results: dict[str, np.ndarray] = {}
        for model_id in self._candidate_models:
            status = self._fit_statuses[model_id]

            if status.is_fitted:
                clf = self._classifiers[model_id]
                # Identify class 1 column index
                classes = list(clf.classes_)
                class_1_idx = classes.index(1) if 1 in classes else 0
                probs = clf.predict_proba(X_features)[:, class_1_idx]
                results[model_id] = np.clip(probs, 0.0, 1.0).astype(np.float32)
            else:
                # Fallback probability
                results[model_id] = np.full(len(texts), status.fallback_prob, dtype=np.float32)

        return results

    def predict_proba_one(
        self,
        prompt_text: str,
        task_category: str | None = None,
    ) -> dict[str, float]:
        """Convenience method returning {model_id: prob} for a single prompt query."""
        res = self.predict_proba([prompt_text], task_categories=[task_category] if task_category else None)
        return {m: float(probs[0]) for m, probs in res.items()}
