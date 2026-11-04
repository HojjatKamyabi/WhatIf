import numpy as np
import pytest

from whatif.scoring import covered, crps


def test_crps_of_a_point_ensemble_is_the_absolute_error():
    samples = np.full((10, 1), 3.0)
    assert crps(samples, np.array([5.0])) == pytest.approx([2.0])


def test_crps_matches_hand_computation():
    # Ensemble {0, 2}, actual 1: E|X-y| = 1, E|X-X'| = (0+2+2+0)/4 = 1 -> 1 - 0.5 = 0.5
    samples = np.array([[0.0], [2.0]])
    assert crps(samples, np.array([1.0])) == pytest.approx([0.5])


def test_centred_distribution_beats_a_shifted_one():
    rng = np.random.default_rng(0)
    centred = rng.normal(10, 1, size=(500, 1))
    assert crps(centred, np.array([10.0])) < crps(centred + 3, np.array([10.0]))


def test_coverage_counts_values_inside_the_central_interval():
    samples = np.arange(1, 101, dtype=float).reshape(100, 1).repeat(3, axis=1)  # 1..100
    actual = np.array([50.0, 5.0, 95.0])
    assert covered(samples, actual, 0.8).tolist() == [True, False, False]
