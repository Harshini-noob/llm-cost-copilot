"""
tests_v2/test_encoders.py

Unit tests for text encoders under Phase M3.
Verifies interface compliance, shape validation, determinism, offline caching,
and strictly zero network activity.
"""

from __future__ import annotations

import socket
from typing import Sequence
import numpy as np
import pytest

from eval.routers.encoders import (
    BaseEncoder,
    DeterministicTestEncoder,
    ModelNotCachedError,
    SentenceTransformerEncoder,
)


def test_base_encoder_interface() -> None:
    """BaseEncoder cannot be instantiated directly without implementing abstract methods."""
    with pytest.raises(TypeError):
        BaseEncoder()  # type: ignore[abstract]

    class IncompleteEncoder(BaseEncoder):
        @property
        def dimension(self) -> int:
            return 10

    with pytest.raises(TypeError):
        IncompleteEncoder()  # type: ignore[abstract]


def test_deterministic_encoder_dimensions_and_shapes() -> None:
    """DeterministicTestEncoder correctly reports dimensions and returns expected array shapes."""
    for dim in (16, 64, 128, 384):
        enc = DeterministicTestEncoder(dimension=dim)
        assert enc.dimension == dim

        # Empty sequence
        res_empty = enc.encode([])
        assert isinstance(res_empty, np.ndarray)
        assert res_empty.shape == (0, dim)
        assert res_empty.dtype == np.float32

        # Single text
        res_single = enc.encode(["Calculate 2 + 2."])
        assert res_single.shape == (1, dim)
        assert res_single.dtype == np.float32

        # encode_one convenience method
        res_one = enc.encode_one("Calculate 2 + 2.")
        assert res_one.shape == (dim,)
        assert np.allclose(res_single[0], res_one)

        # Batch
        texts = ["What is 4 * 5?", "Write a function in python.", "Extract JSON keys."]
        res_batch = enc.encode(texts)
        assert res_batch.shape == (3, dim)
        assert res_batch.dtype == np.float32


def test_deterministic_encoder_determinism() -> None:
    """DeterministicTestEncoder yields identical embeddings across repeated calls and batching."""
    enc = DeterministicTestEncoder(dimension=64)
    prompt = "Find the sum of all prime numbers between 1 and 20."

    v1 = enc.encode_one(prompt)
    v2 = enc.encode_one(prompt)
    assert np.array_equal(v1, v2)

    # In a batch vs standalone
    batch_res = enc.encode(["irrelevant first prompt", prompt, "third prompt"])
    assert np.array_equal(batch_res[1], v1)


def test_deterministic_encoder_normalization() -> None:
    """Normalized embeddings have unit L2 norm, while unnormalized embeddings do not."""
    enc_norm = DeterministicTestEncoder(dimension=64, normalize=True)
    enc_raw = DeterministicTestEncoder(dimension=64, normalize=False)

    text = "The quick brown fox jumps over the lazy dog."
    vec_norm = enc_norm.encode_one(text)
    vec_raw = enc_raw.encode_one(text)

    norm_val = float(np.linalg.norm(vec_norm))
    assert pytest.approx(norm_val, abs=1e-5) == 1.0

    raw_norm = float(np.linalg.norm(vec_raw))
    assert raw_norm > 1.0


def test_deterministic_encoder_validation_errors() -> None:
    """DeterministicTestEncoder rejects invalid initialization and arguments."""
    with pytest.raises(ValueError, match="Dimension must be a positive integer"):
        DeterministicTestEncoder(dimension=0)

    with pytest.raises(ValueError, match="Dimension must be a positive integer"):
        DeterministicTestEncoder(dimension=-5)

    enc = DeterministicTestEncoder(dimension=32)

    with pytest.raises(TypeError, match="texts cannot be None"):
        enc.encode(None)  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="All elements in texts must be str"):
        enc.encode(["valid text", 12345])  # type: ignore[list-item]

    with pytest.raises(TypeError, match="Expected str"):
        enc.encode_one(123)  # type: ignore[arg-type]


def test_sentence_transformer_encoder_local_cache() -> None:
    """SentenceTransformerEncoder loads from local cache and produces valid embeddings."""
    enc = SentenceTransformerEncoder("all-MiniLM-L6-v2", local_files_only=True)
    assert enc.dimension == 384

    # Empty inputs
    empty_res = enc.encode([])
    assert empty_res.shape == (0, 384)
    assert empty_res.dtype == np.float32

    # Real prompt encoding
    prompts = [
        "Compute the integral of x^2 dx from 0 to 3.",
        "Summarize the main differences between TCP and UDP.",
    ]
    vectors = enc.encode(prompts)
    assert vectors.shape == (2, 384)
    assert vectors.dtype == np.float32

    # Normalized unit length
    for vec in vectors:
        assert pytest.approx(float(np.linalg.norm(vec)), abs=1e-4) == 1.0

    # encode_one
    vec_one = enc.encode_one(prompts[0])
    assert vec_one.shape == (384,)
    assert np.allclose(vectors[0], vec_one, atol=1e-5)


def test_sentence_transformer_encoder_fails_gracefully_when_not_cached() -> None:
    """SentenceTransformerEncoder raises ModelNotCachedError when local weights are absent."""
    with pytest.raises(ModelNotCachedError) as exc_info:
        SentenceTransformerEncoder("non-existent-uncached-model-xyz", local_files_only=True)

    assert "is not available in local cache" in str(exc_info.value)
    assert "zero-spending" in str(exc_info.value)


def test_zero_network_calls_during_encoding(monkeypatch: pytest.MonkeyPatch) -> None:
    """Confirms no socket network connections are attempted during model loading and encoding."""
    def guarded_connect(*args, **kwargs):
        raise RuntimeError("Prohibited network connection attempt detected during offline encoding!")

    monkeypatch.setattr(socket.socket, "connect", guarded_connect)

    # Deterministic encoder
    det_enc = DeterministicTestEncoder(dimension=64)
    det_vecs = det_enc.encode(["Test prompt without network."])
    assert det_vecs.shape == (1, 64)

    # Local sentence transformer
    st_enc = SentenceTransformerEncoder("all-MiniLM-L6-v2", local_files_only=True)
    st_vecs = st_enc.encode(["Test sentence transformer without network."])
    assert st_vecs.shape == (1, 384)
