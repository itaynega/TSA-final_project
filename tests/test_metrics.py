"""Tests for common.metrics.

Written before the implementation. Every expected value below was computed by
hand from the formulas as `COURSE_NOTATION_2026-07-30.md` §2.1 prints them, not
from any library.

Worked example used throughout:

    y_true = [ 3.0, -0.5,  2.0,  7.0]
    y_pred = [ 2.5,  0.0,  2.0,  8.0]
    e_t = y_t - f(x_t)
        = [ 0.5, -0.5,  0.0, -1.0]
    |e_t|
        = [ 0.5,  0.5,  0.0,  1.0]

    MSE  = (0.25 + 0.25 + 0.0 + 1.0) / 4 = 0.375
    MAE  = (0.5  + 0.5  + 0.0 + 1.0) / 4 = 0.5
    RMSE = sqrt(0.375)                   = 0.6123724356957945
    MdAE = median(0.0, 0.5, 0.5, 1.0)    = 0.5
"""

import numpy as np
import pytest

from common import metrics


Y_TRUE = np.array([3.0, -0.5, 2.0, 7.0])
Y_PRED = np.array([2.5, 0.0, 2.0, 8.0])


# --------------------------------------------------------------------------
# The error term itself. The course fixes the sign; the paper's code does not.
# --------------------------------------------------------------------------


def test_error_term_follows_the_course_sign_convention():
    """e_t = y_t - f(x_t), per the definition quoted on printed sl. 47."""
    e = metrics.errors(Y_TRUE, Y_PRED)
    np.testing.assert_allclose(e, [0.5, -0.5, 0.0, -1.0])


def test_metrics_are_unchanged_if_the_arguments_are_swapped():
    """TQNet's own utils compute `pred - true`, the opposite sign to the course.

    All four of our metrics square or take the modulus, so the divergence is
    harmless. This test is what lets the report say that rather than assume it.
    """
    forward = metrics.all_metrics(Y_TRUE, Y_PRED)
    backward = metrics.all_metrics(Y_PRED, Y_TRUE)
    assert forward == backward


# --------------------------------------------------------------------------
# The four metrics, against hand-computed values.
# --------------------------------------------------------------------------


def test_mse_matches_the_hand_computed_value():
    assert metrics.mse(Y_TRUE, Y_PRED) == pytest.approx(0.375)


def test_mae_matches_the_hand_computed_value():
    assert metrics.mae(Y_TRUE, Y_PRED) == pytest.approx(0.5)


def test_rmse_matches_the_hand_computed_value():
    assert metrics.rmse(Y_TRUE, Y_PRED) == pytest.approx(0.6123724356957945)


def test_rmse_is_the_square_root_of_mse():
    """RMSE = sqrt(MSE) is how the course defines it, not as a separate sum."""
    rng = np.random.default_rng(0)
    a, b = rng.normal(size=500), rng.normal(size=500)
    assert metrics.rmse(a, b) == pytest.approx(np.sqrt(metrics.mse(a, b)))


def test_mdae_matches_the_hand_computed_value():
    assert metrics.mdae(Y_TRUE, Y_PRED) == pytest.approx(0.5)


def test_all_four_metrics_are_zero_for_a_perfect_forecast():
    m = metrics.all_metrics(Y_TRUE, Y_TRUE)
    assert m == {"MSE": 0.0, "MAE": 0.0, "RMSE": 0.0, "MdAE": 0.0}


def test_mdae_is_robust_to_an_outlier_that_moves_mae():
    """The reason MdAE earns its place next to MAE."""
    clean_true = np.zeros(101)
    clean_pred = np.ones(101)
    spiked_pred = clean_pred.copy()
    spiked_pred[0] = 1000.0

    assert metrics.mdae(clean_true, clean_pred) == pytest.approx(
        metrics.mdae(clean_true, spiked_pred)
    )
    assert metrics.mae(clean_true, spiked_pred) > metrics.mae(clean_true, clean_pred)


# --------------------------------------------------------------------------
# Shape handling. LTSF predictions arrive as (n_windows, pred_len, n_features).
# --------------------------------------------------------------------------


def test_three_dimensional_input_is_reduced_over_every_element():
    """The LTSF convention TQNet reports under: one flat mean over everything.

    Not a per-window mean that is then averaged. For MSE and MAE on a
    rectangular array the two agree, so this is pinned on MdAE, where they
    do not: the flat median here is 1.5, the median-of-window-medians is 1.0.
    """
    y_true = np.array([[[0.0], [0.0], [10.0]], [[1.0], [2.0], [3.0]]])
    y_pred = np.zeros_like(y_true)

    assert y_true.shape == (2, 3, 1)
    assert metrics.mdae(y_true, y_pred) == pytest.approx(1.5)


def test_a_scalar_is_returned_not_an_array():
    y_true = np.zeros((2, 3, 4))
    y_pred = np.ones((2, 3, 4))
    value = metrics.mse(y_true, y_pred)
    assert isinstance(value, float)


def test_lists_are_accepted_as_well_as_arrays():
    assert metrics.mae([1.0, 2.0], [1.0, 4.0]) == pytest.approx(1.0)


# --------------------------------------------------------------------------
# Guards. Every one of these is a silent-wrong-number failure if unguarded.
# --------------------------------------------------------------------------


def test_mismatched_shapes_raise():
    with pytest.raises(ValueError, match="shape"):
        metrics.mse(np.zeros(4), np.zeros(5))


def test_shapes_that_would_broadcast_still_raise():
    """(4,) against (4, 1) broadcasts to (4, 4) in numpy and returns a number.

    That number is meaningless. It must not be returned.
    """
    with pytest.raises(ValueError, match="shape"):
        metrics.mse(np.zeros(4), np.zeros((4, 1)))


def test_nan_in_the_truth_raises():
    with pytest.raises(ValueError, match="finite"):
        metrics.mse(np.array([1.0, np.nan]), np.array([1.0, 1.0]))


def test_nan_in_the_prediction_raises():
    with pytest.raises(ValueError, match="finite"):
        metrics.mse(np.array([1.0, 1.0]), np.array([1.0, np.nan]))


def test_infinity_raises():
    with pytest.raises(ValueError, match="finite"):
        metrics.mse(np.array([1.0, 1.0]), np.array([1.0, np.inf]))


def test_empty_input_raises():
    with pytest.raises(ValueError, match="empty"):
        metrics.mse(np.array([]), np.array([]))


# --------------------------------------------------------------------------
# The dictionary the results-writer and the report table consume.
# --------------------------------------------------------------------------


def test_all_metrics_returns_exactly_the_four_agreed_metrics():
    m = metrics.all_metrics(Y_TRUE, Y_PRED)
    assert set(m) == {"MSE", "MAE", "RMSE", "MdAE"}


def test_all_metrics_agrees_with_the_individual_functions():
    m = metrics.all_metrics(Y_TRUE, Y_PRED)
    assert m["MSE"] == pytest.approx(metrics.mse(Y_TRUE, Y_PRED))
    assert m["MAE"] == pytest.approx(metrics.mae(Y_TRUE, Y_PRED))
    assert m["RMSE"] == pytest.approx(metrics.rmse(Y_TRUE, Y_PRED))
    assert m["MdAE"] == pytest.approx(metrics.mdae(Y_TRUE, Y_PRED))


def test_percentage_metrics_are_deliberately_absent():
    """MAPE and SMAPE divide by |y_t|.

    LTSF results are computed on z-scored data, so the series crosses zero and
    both metrics blow up. `COURSE_NOTATION` §2.1 already flags MAPE as undefined
    at y_t = 0. This test exists so that adding them is a deliberate act with a
    conversation attached, not a quiet import.
    """
    assert not hasattr(metrics, "mape")
    assert not hasattr(metrics, "smape")
