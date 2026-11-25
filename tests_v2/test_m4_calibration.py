"""
tests_v2/test_m4_calibration.py

Unit and integration tests for M4.3: Probability Calibration and Reliability Analysis.
Verifies calibration metric computations, binning mechanics, continuous SQuAD support,
leakage-free calibrator fitting, and SVG/ASCII reliability diagram generation.
"""

from __future__ import annotations

import math
from pathlib import Path
import numpy as np
import pytest

from eval.m4_calibration import (
    CalibrationBin,
    CalibrationMetrics,
    IsotonicCalibrator,
    PlattCalibrator,
    compute_calibration_metrics,
    fast_vectorized_knn_predict,
    generate_ascii_reliability_diagram,
    generate_reliability_diagram_svg,
)


def test_compute_calibration_metrics_empty() -> None:
    """Empty inputs should yield zero metrics and empty bins."""
    metrics = compute_calibration_metrics([], [])
    assert metrics.total_samples == 0
    assert metrics.brier_score == 0.0
    assert metrics.ece == 0.0
    assert metrics.mce == 0.0
    assert len(metrics.bins) == 0


def test_compute_calibration_metrics_length_mismatch() -> None:
    """Mismatched array lengths should raise ValueError."""
    with pytest.raises(ValueError, match="Length mismatch"):
        compute_calibration_metrics([1, 0], [0.5])


def test_compute_calibration_metrics_perfect() -> None:
    """Perfect predictions should yield near-zero Brier score and ECE."""
    y_true = np.array([0.0, 0.0, 1.0, 1.0])
    y_prob = np.array([0.0, 0.0, 1.0, 1.0])
    metrics = compute_calibration_metrics(y_true, y_prob, n_bins=10)

    assert metrics.total_samples == 4
    assert pytest.approx(metrics.brier_score, abs=1e-6) == 0.0
    assert pytest.approx(metrics.ece, abs=1e-6) == 0.0
    assert pytest.approx(metrics.mce, abs=1e-6) == 0.0
    assert len(metrics.bins) == 10


def test_compute_calibration_metrics_known_values() -> None:
    """Verify exact formula calculations on a controlled distribution."""
    # 10 samples in [0.0, 0.1): mean confidence 0.05, 1 positive -> acc 0.1, gap = 0.05
    # 10 samples in [0.9, 1.0]: mean confidence 0.95, 8 positives -> acc 0.8, gap = 0.15
    y_prob = np.array([0.05] * 10 + [0.95] * 10)
    y_true = np.array([0.0] * 9 + [1.0] * 1 + [1.0] * 8 + [0.0] * 2)

    metrics = compute_calibration_metrics(y_true, y_prob, n_bins=10)
    assert metrics.total_samples == 20

    # Brier score: 9 * (0.05)^2 + 1 * (0.95)^2 + 8 * (0.05)^2 + 2 * (0.95)^2
    # = (9 * 0.0025 + 0.9025 + 8 * 0.0025 + 2 * 0.9025) / 20 = (0.0225 + 0.9025 + 0.02 + 1.805) / 20 = 2.75 / 20 = 0.1375
    assert pytest.approx(metrics.brier_score, abs=1e-5) == 0.1375

    # Weighted ECE: (10/20) * |0.1 - 0.05| + (10/20) * |0.8 - 0.95| = 0.5 * 0.05 + 0.5 * 0.15 = 0.025 + 0.075 = 0.10
    assert pytest.approx(metrics.ece, abs=1e-5) == 0.10
    # MCE: max(0.05, 0.15) = 0.15
    assert pytest.approx(metrics.mce, abs=1e-5) == 0.15


def test_compute_calibration_metrics_squad_continuous() -> None:
    """Continuous SQuAD token F1 scores in [0, 1] must be handled properly without crash."""
    y_prob = np.array([0.2, 0.4, 0.6, 0.8])
    y_true = np.array([0.15, 0.45, 0.75, 0.85])  # Continuous F1 partial credit

    metrics = compute_calibration_metrics(y_true, y_prob, n_bins=5)
    assert metrics.total_samples == 4
    expected_brier = np.mean((y_prob - y_true) ** 2)
    assert pytest.approx(metrics.brier_score, abs=1e-6) == expected_brier
    assert 0.0 <= metrics.ece <= 1.0


def test_platt_calibrator() -> None:
    """Platt scaling (sigmoid calibration) must fit and produce valid probabilities."""
    calibrator = PlattCalibrator()
    with pytest.raises(RuntimeError, match="must be fitted before transform"):
        calibrator.transform(np.array([0.5]))

    # Synthetic uncalibrated probabilities and labels
    probs = np.array([0.1, 0.2, 0.3, 0.4, 0.7, 0.8, 0.9, 0.95], dtype=np.float32)
    labels = np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.float32)

    calibrator.fit(probs, labels)
    calibrated = calibrator.transform(probs)

    assert len(calibrated) == len(probs)
    assert np.all(calibrated >= 0.0)
    assert np.all(calibrated <= 1.0)
    # Check monotonicity for 1D logistic regression
    assert np.all(np.diff(calibrated) >= 0.0)


def test_isotonic_calibrator() -> None:
    """Isotonic regression calibrator must fit and produce monotonically non-decreasing output."""
    calibrator = IsotonicCalibrator()
    with pytest.raises(RuntimeError, match="must be fitted before transform"):
        calibrator.transform(np.array([0.5]))

    probs = np.array([0.1, 0.2, 0.4, 0.5, 0.7, 0.85, 0.9], dtype=np.float32)
    labels = np.array([0.0, 0.2, 0.1, 0.6, 0.8, 0.7, 1.0], dtype=np.float32)

    calibrator.fit(probs, labels)
    calibrated = calibrator.transform(probs)

    assert len(calibrated) == len(probs)
    assert np.all(calibrated >= 0.0)
    assert np.all(calibrated <= 1.0)
    # Isotonic transform must be non-decreasing
    assert np.all(np.diff(calibrated) >= -1e-6)


def test_fast_vectorized_knn_predict() -> None:
    """Test fast vectorized kNN prediction on synthetic embeddings."""
    rng = np.random.RandomState(42)
    D = 16
    N_tr = 20
    N_q = 5
    N_models = 3

    # Normalized embeddings
    train_embs = rng.randn(N_tr, D).astype(np.float32)
    train_embs /= np.linalg.norm(train_embs, axis=1, keepdims=True)

    query_embs = rng.randn(N_q, D).astype(np.float32)
    query_embs /= np.linalg.norm(query_embs, axis=1, keepdims=True)

    train_labels = rng.uniform(0.0, 1.0, size=(N_tr, N_models)).astype(np.float32)

    probs = fast_vectorized_knn_predict(query_embs, train_embs, train_labels, k=4)
    assert probs.shape == (N_q, N_models)
    assert np.all(probs >= 0.0)
    assert np.all(probs <= 1.0)


def test_ascii_reliability_diagram() -> None:
    """ASCII diagram helper must return formatted string containing expected headers."""
    metrics = compute_calibration_metrics([0, 1], [0.1, 0.9])
    ascii_table = generate_ascii_reliability_diagram(metrics)
    assert "Bin Range" in ascii_table
    assert "Samples" in ascii_table
    assert "Conf" in ascii_table
    assert "Acc" in ascii_table
    assert "Legend" in ascii_table


def test_svg_reliability_diagram(tmp_path: Path) -> None:
    """SVG generator must write a valid standalone SVG file."""
    metrics = compute_calibration_metrics([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9])
    svg_file = tmp_path / "test_diagram.svg"
    generate_reliability_diagram_svg(metrics, "Test Reliability", svg_file)

    assert svg_file.exists()
    content = svg_file.read_text(encoding="utf-8")
    assert "<svg" in content
    assert "</svg>" in content
    assert "Test Reliability" in content
    assert f"ECE: {metrics.ece:.4f}" in content
    assert "Mean Predicted Confidence" in content
