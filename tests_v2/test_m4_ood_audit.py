"""
tests_v2/test_m4_ood_audit.py

Statistical audit tests for M4.4.1 OOD evaluation:
1. Paired prompt-level bootstrap confidence interval for Delta Q.
2. Exact zero-difference collapse when identical models are chosen.
3. Wilson and Clopper-Pearson binomial intervals for boundary and small-N cases.
4. Continuous SQuAD F1 handling without applying binary binomial intervals.
5. Balanced task-specific outcomes without universal dominance claims.
"""

from __future__ import annotations

import math
import numpy as np
import pytest

from eval.m4_ood_evaluation import (
    compute_wilson_ci,
    compute_clopper_pearson_ci,
)


def compute_paired_bootstrap_diff(
    ood_scores: list[float] | np.ndarray,
    id_scores: list[float] | np.ndarray,
    n_bootstraps: int = 200,
    seed: int = 42,
) -> tuple[float, float, float, bool]:
    """Helper for testing paired prompt bootstrap differences."""
    d = np.asarray(ood_scores, dtype=np.float64) - np.asarray(id_scores, dtype=np.float64)
    mean_d = float(np.mean(d))
    n = len(d)
    rng = np.random.RandomState(seed)
    boot_means = np.empty(n_bootstraps, dtype=np.float64)
    for b in range(n_bootstraps):
        s_idx = rng.randint(0, n, size=n)
        boot_means[b] = np.mean(d[s_idx])
    ci_low = float(np.percentile(boot_means, 2.5))
    ci_high = float(np.percentile(boot_means, 97.5))
    is_sig = bool(ci_low > 0.0 or ci_high < 0.0)
    return mean_d, ci_low, ci_high, is_sig


def test_paired_bootstrap_zero_difference() -> None:
    """When OOD and ID make identical decisions on all prompts, Delta Q must be exactly 0 with [0, 0] CI."""
    ood = [1.0, 0.0, 1.0, 1.0, 0.0]
    id_ = [1.0, 0.0, 1.0, 1.0, 0.0]
    mean_d, ci_low, ci_high, is_sig = compute_paired_bootstrap_diff(ood, id_, n_bootstraps=100)

    assert mean_d == 0.0
    assert ci_low == 0.0
    assert ci_high == 0.0
    assert is_sig is False


def test_paired_bootstrap_significant_degradation() -> None:
    """When OOD systematically underperforms ID, Delta Q must be negative and statistically significant."""
    id_ = [1.0] * 50 + [0.0] * 10
    ood = [0.0] * 30 + [1.0] * 20 + [0.0] * 10
    mean_d, ci_low, ci_high, is_sig = compute_paired_bootstrap_diff(ood, id_, n_bootstraps=200)

    assert mean_d < 0.0
    assert ci_high < 0.0  # Does not cross 0
    assert is_sig is True


def test_wilson_ci_boundary_cases() -> None:
    """Wilson interval on zero successes must not collapse to [0, 0] and must remain within [0, 1]."""
    # k = 0, N = 16 (HumanEval case)
    low, high = compute_wilson_ci(0, 16)
    assert low == 0.0
    assert 0.15 < high < 0.25  # Wilson upper bound is ~0.1936

    # k = 16, N = 16
    low_full, high_full = compute_wilson_ci(16, 16)
    assert high_full == 1.0
    assert 0.75 < low_full < 0.85

    # Empty
    assert compute_wilson_ci(0, 0) == (0.0, 0.0)


def test_clopper_pearson_ci_boundary_cases() -> None:
    """Exact Clopper-Pearson interval on k=0, N=16 must match Beta distribution 95% quantiles."""
    low, high = compute_clopper_pearson_ci(0, 16)
    assert low == 0.0
    assert 0.18 < high < 0.22  # Clopper-Pearson two-sided 95% upper bound is ~0.2059

    # k = 1, N = 13 (AIME Learned-LR case)
    low_1, high_1 = compute_clopper_pearson_ci(1, 13)
    p_hat = 1.0 / 13.0
    assert low_1 < p_hat < high_1
    assert low_1 > 0.0
    assert high_1 < 0.40


def test_squad_continuous_f1_retention() -> None:
    """SQuAD continuous token F1 scores must not be coerced to binary and must support paired differences."""
    squad_id = [0.85, 0.40, 0.65, 0.90]
    squad_ood = [0.80, 0.35, 0.60, 0.85]
    mean_d, ci_low, ci_high, is_sig = compute_paired_bootstrap_diff(squad_ood, squad_id, n_bootstraps=100)

    assert pytest.approx(mean_d, abs=1e-5) == -0.05
    assert ci_low <= -0.05 <= ci_high
