"""
eval/routers/encoders.py

Feature encoders for text embeddings in LLM Cost Copilot Phase M3.
Adheres strictly to the ZERO ADDITIONAL SPENDING policy:
1. BaseEncoder: Abstract base interface defining dimension and batch encoding.
2. SentenceTransformerEncoder: Uses locally cached sentence-transformers models
   (default: all-MiniLM-L6-v2) with local_files_only=True to prevent automatic network downloads.
3. DeterministicTestEncoder: Lightweight, hash-based deterministic encoder for offline
   unit testing without neural network weights or network dependencies.
"""

from __future__ import annotations

import abc
import hashlib
from typing import Sequence

import numpy as np


class ModelNotCachedError(FileNotFoundError):
    """Raised when an embedding model is not present in local cache and auto-download is disabled."""
    pass


class BaseEncoder(abc.ABC):
    """Abstract base class for prompt text encoders."""

    @property
    @abc.abstractmethod
    def dimension(self) -> int:
        """Return the embedding vector dimension."""
        raise NotImplementedError

    @abc.abstractmethod
    def encode(self, texts: Sequence[str]) -> np.ndarray:
        """Encode a sequence of texts into an (N, dimension) float32 numpy array.

        Parameters
        ----------
        texts : Sequence[str]
            List or sequence of input strings to encode.

        Returns
        -------
        np.ndarray
            Array of shape (len(texts), dimension) and dtype float32.
            If texts is empty, returns an array of shape (0, dimension).
        """
        raise NotImplementedError

    def encode_one(self, text: str) -> np.ndarray:
        """Convenience method to encode a single text string into a 1D vector."""
        if not isinstance(text, str):
            raise TypeError(f"Expected str, got {type(text).__name__}")
        res = self.encode([text])
        return res[0]


class DeterministicTestEncoder(BaseEncoder):
    """Deterministic, offline hash-based encoder for tests and validation.

    Uses sha256 feature hashing across character n-grams and whitespace tokens
    to construct fixed-dimensional pseudo-embeddings with unit L2 norm.
    Completely zero-dependency, instantaneous, and strictly reproducible.
    """

    def __init__(self, dimension: int = 64, *, normalize: bool = True) -> None:
        if not isinstance(dimension, int) or dimension <= 0:
            raise ValueError(f"Dimension must be a positive integer, got {dimension}")
        self._dim = dimension
        self._normalize = normalize

    @property
    def dimension(self) -> int:
        return self._dim

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        if texts is None:
            raise TypeError("texts cannot be None")

        text_list = list(texts)
        for i, t in enumerate(text_list):
            if not isinstance(t, str):
                raise TypeError(f"All elements in texts must be str, got {type(t).__name__} at index {i}")

        n = len(text_list)
        if n == 0:
            return np.empty((0, self._dim), dtype=np.float32)

        vectors = np.zeros((n, self._dim), dtype=np.float32)
        for i, text in enumerate(text_list):
            if not text:
                continue

            tokens = text.split()
            trigrams = [text[j : j + 3] for j in range(max(0, len(text) - 2))]
            features = tokens + trigrams if trigrams else tokens

            if not features:
                features = [text]

            vec = np.zeros(self._dim, dtype=np.float32)
            for feat in features:
                digest = hashlib.sha256(feat.encode("utf-8")).digest()
                hash_val = int.from_bytes(digest[:8], byteorder="big", signed=False)
                sign = 1.0 if (hash_val & 1) == 0 else -1.0
                idx = (hash_val >> 1) % self._dim
                vec[idx] += sign

            if self._normalize:
                norm = float(np.linalg.norm(vec))
                if norm > 1e-12:
                    vec = vec / norm

            vectors[i] = vec

        return vectors


class SentenceTransformerEncoder(BaseEncoder):
    """SentenceTransformers embedding encoder enforcing local/offline model loading.

    Parameters
    ----------
    model_name_or_path : str
        Name of the model (e.g. 'all-MiniLM-L6-v2') or local path.
    local_files_only : bool
        If True (default), prevents network calls and raises ModelNotCachedError
        if weights are not present in local cache.
    normalize_embeddings : bool
        If True (default), normalizes embeddings to unit length.
    device : str | None
        Torch device to use (default: 'cpu').
    """

    def __init__(
        self,
        model_name_or_path: str = "all-MiniLM-L6-v2",
        *,
        local_files_only: bool = True,
        normalize_embeddings: bool = True,
        device: str | None = "cpu",
    ) -> None:
        self._model_name = model_name_or_path
        self._local_files_only = local_files_only
        self._normalize_embeddings = normalize_embeddings
        self._device = device or "cpu"

        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as err:
            raise ImportError(
                "sentence-transformers is required for SentenceTransformerEncoder. "
                "Install it or use DeterministicTestEncoder."
            ) from err

        try:
            self._model = SentenceTransformer(
                self._model_name,
                device=self._device,
                local_files_only=self._local_files_only,
            )
        except Exception as err:
            msg = str(err)
            if (
                self._local_files_only
                or "local_files_only" in msg
                or "couldn't connect" in msg.lower()
                or "not find" in msg.lower()
                or "offline" in msg.lower()
            ):
                raise ModelNotCachedError(
                    f"Model '{self._model_name}' is not available in local cache. "
                    "Automatic download is prohibited under the strict zero-spending and offline evaluation policy. "
                    "Please ensure model weights are cached locally or use DeterministicTestEncoder."
                ) from err
            raise

        if hasattr(self._model, "get_embedding_dimension"):
            dim = self._model.get_embedding_dimension()
        elif hasattr(self._model, "get_sentence_embedding_dimension"):
            dim = self._model.get_sentence_embedding_dimension()
        else:
            dim = None

        if dim is None:
            dummy = self._model.encode([""], convert_to_numpy=True)
            dim = int(dummy.shape[1])
        self._dim = int(dim)

    @property
    def dimension(self) -> int:
        return self._dim

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        if texts is None:
            raise TypeError("texts cannot be None")

        text_list = list(texts)
        for i, t in enumerate(text_list):
            if not isinstance(t, str):
                raise TypeError(f"All elements in texts must be str, got {type(t).__name__} at index {i}")

        if len(text_list) == 0:
            return np.empty((0, self._dim), dtype=np.float32)

        embeddings = self._model.encode(
            text_list,
            convert_to_numpy=True,
            normalize_embeddings=self._normalize_embeddings,
            show_progress_bar=False,
        )
        return np.asarray(embeddings, dtype=np.float32)
