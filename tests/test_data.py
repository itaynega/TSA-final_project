"""Tests for common.data and common.split.

The load-bearing test in this file is
`test_windows_match_the_upstream_dataset_exactly`. `common/data.py` claims to be a
faithful numpy reimplementation of TQNet's `Dataset_ETT_hour`, and the baseline and
the leakage audit are only meaningful if that claim holds -- a baseline scored on
almost-the-same windows would produce a plausible number that is not comparable to
the paper's 0.3712. So the claim is checked against the real upstream class, by
importing it, rather than by re-reading the source and agreeing with ourselves.

Tests that need the CSV skip cleanly when it is absent, so the suite still runs on a
fresh clone before `tools/get_data.py` has been called.
"""

import os
import sys

import numpy as np
import pytest

from common import data as data_mod
from common import split as split_mod

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TQNET_DIR = os.path.join(REPO_ROOT, "TQNet")

has_csv = pytest.mark.skipif(
    not os.path.exists(data_mod.DEFAULT_CSV),
    reason="ETTh1.csv not present; run python3 tools/get_data.py",
)


# --------------------------------------------------------------------------
# The split arithmetic, which is transcribed from the loader and must not drift.
# --------------------------------------------------------------------------


def test_borders_are_the_informer_monthly_boundaries():
    """12 / 4 / 4 months of 30 days, with val and test reaching back one window."""
    assert split_mod.borders(96) == {
        "train": (0, 8640),
        "val": (8544, 11520),
        "test": (11424, 14400),
    }


def test_only_14400_rows_are_ever_used():
    """The paper's Table 1 says 14,400; the CSV has 17,420. Both are true."""
    assert split_mod.TOTAL_USED_ROWS == 14400
    assert split_mod.borders(96)["test"][1] == 14400


def test_window_counts_match_the_papers_test_set_size():
    """2,785 test windows is the denominator behind 0.3712."""
    counts = split_mod.n_windows(seq_len=96, pred_len=96)
    assert counts == {"train": 8449, "val": 2785, "test": 2785}


def test_longer_horizons_yield_fewer_windows():
    """A sanity check on the formula rather than on a memorised number."""
    for pred_len in (96, 192, 336, 720):
        expected = 2976 - 96 - pred_len + 1
        assert split_mod.n_windows(96, pred_len)["test"] == expected


def test_split_hash_is_stable_and_depends_on_everything_it_should():
    digest = "0" * 64
    base = split_mod.split_hash(96, 96, digest)

    assert base == split_mod.split_hash(96, 96, digest), "hash must be deterministic"
    assert base != split_mod.split_hash(96, 192, digest), "horizon must change it"
    assert base != split_mod.split_hash(336, 96, digest), "look-back must change it"
    assert base != split_mod.split_hash(96, 96, "1" * 64), "data file must change it"


def test_split_hash_refuses_to_ignore_the_data_file():
    """Hashing the arithmetic alone would call two different CSVs the same split."""
    with pytest.raises(ValueError, match="data_sha256 is required"):
        split_mod.split_hash(96, 96, "")


# --------------------------------------------------------------------------
# Equivalence with upstream. This is the test the rest of the project rests on.
# --------------------------------------------------------------------------


@has_csv
@pytest.mark.parametrize("which", ["train", "val", "test"])
def test_windows_match_the_upstream_dataset_exactly(which):
    """Our numpy windows equal TQNet's `Dataset_ETT_hour`, element for element.

    Checked on the first and last window of the split plus a scattering in between,
    rather than all 8,449, because the upstream `__getitem__` is a Python-level slice
    per call and iterating every window of every split makes the suite slow for no
    extra coverage: an indexing bug that spares the endpoints and the interior
    samples does not exist.
    """
    sys.path.insert(0, TQNET_DIR)
    try:
        from data_provider.data_loader import Dataset_ETT_hour
    finally:
        sys.path.remove(TQNET_DIR)

    upstream = Dataset_ETT_hour(
        root_path=os.path.join(TQNET_DIR, "dataset"),
        data_path="ETTh1.csv",
        flag=which,
        size=[96, 0, 96],
        features="M",
        target="OT",
        timeenc=1,
        freq="h",
        cycle=24,
    )
    ours = data_mod.make_windows(which, seq_len=96, pred_len=96, cycle=24)

    assert len(upstream) == len(ours)

    probes = [0, 1, 7, len(ours) // 3, len(ours) // 2, len(ours) - 2, len(ours) - 1]
    for index in sorted(set(probes)):
        seq_x, seq_y, _, _, cycle_index = upstream[index]
        np.testing.assert_allclose(ours.x[index], seq_x, rtol=0, atol=0)
        np.testing.assert_allclose(ours.y[index], seq_y, rtol=0, atol=0)
        assert int(ours.cycle_index[index]) == int(cycle_index)


@has_csv
def test_scaler_is_fitted_on_training_rows_only():
    """Inspect the fitted object, not the call site.

    The audit is supposed to check what the scaler actually learned, so this
    recomputes the training mean and variance independently and compares.
    """
    frame = data_mod.load_raw()
    scaler = data_mod.fit_scaler(frame)
    train = frame[list(data_mod.CHANNELS)].values[0:8640]

    np.testing.assert_allclose(scaler.mean_, train.mean(axis=0), rtol=1e-12)
    # sklearn uses the population standard deviation, ddof=0.
    np.testing.assert_allclose(scaler.scale_, train.std(axis=0, ddof=0), rtol=1e-12)

    full = frame[list(data_mod.CHANNELS)].values
    assert not np.allclose(scaler.mean_, full.mean(axis=0)), (
        "training mean coincides with the whole-series mean, so this test could not "
        "tell a leaking scaler from a clean one"
    )


@has_csv
def test_training_split_is_standardised_and_later_splits_are_not_forced_to_be():
    """Train is mean 0 / sd 1 by construction; val and test drift, and should."""
    frame = data_mod.load_raw()
    matrix, _ = data_mod.scaled_matrix(frame)

    np.testing.assert_allclose(matrix[0:8640].mean(axis=0), 0.0, atol=1e-10)
    np.testing.assert_allclose(matrix[0:8640].std(axis=0, ddof=0), 1.0, atol=1e-10)

    test_mean = matrix[11424:14400].mean(axis=0)
    assert np.abs(test_mean).max() > 0.05, (
        "the test split is centred like the training split, which would suggest the "
        "scaler saw it"
    )


@has_csv
def test_cycle_index_is_calendar_phase_of_the_first_forecast_step():
    """No lookahead: the phase is arithmetic on the row number, not on any value."""
    ours = data_mod.make_windows("test", seq_len=96, pred_len=96, cycle=24)
    expected = (ours.row_index + 96) % 24
    np.testing.assert_array_equal(ours.cycle_index, expected)


@has_csv
def test_windows_are_chronological_and_contiguous():
    ours = data_mod.make_windows("test", seq_len=96, pred_len=96)
    np.testing.assert_array_equal(np.diff(ours.row_index), 1)
    assert ours.row_index[0] == 11424


@has_csv
def test_train_windows_never_reach_into_validation_or_test():
    """The no-future-information requirement, as an assertion about indices."""
    train = data_mod.make_windows("train", seq_len=96, pred_len=96)
    last_row_touched = int(train.row_index[-1]) + 96 + 96 - 1
    assert last_row_touched == 8639, (
        "training windows end at row {}, but the training split stops at 8639".format(
            last_row_touched
        )
    )


@has_csv
def test_input_and_target_of_a_window_are_adjacent_and_disjoint():
    """y starts exactly where x ends: no gap, no overlap, no label_len offset."""
    frame = data_mod.load_raw()
    matrix, _ = data_mod.scaled_matrix(frame)
    ours = data_mod.make_windows("test", seq_len=96, pred_len=96)

    for index in (0, 100, len(ours) - 1):
        start = int(ours.row_index[index])
        np.testing.assert_array_equal(ours.x[index], matrix[start:start + 96])
        np.testing.assert_array_equal(ours.y[index], matrix[start + 96:start + 192])


# --------------------------------------------------------------------------
# The baseline. Its correctness is entirely about which values get reused.
# --------------------------------------------------------------------------


@has_csv
def test_seasonal_naive_repeats_the_last_observed_day():
    """Step h of the forecast is the input value 24 hours before that step."""
    ours = data_mod.make_windows("test", seq_len=96, pred_len=96)
    pred = data_mod.seasonal_naive(ours, period=24)

    assert pred.shape == ours.y.shape
    np.testing.assert_array_equal(pred[:, 0:24, :], ours.x[:, -24:, :])
    # Horizon steps beyond one period re-use the same day again.
    for block in range(1, 4):
        np.testing.assert_array_equal(
            pred[:, block * 24:(block + 1) * 24, :], ours.x[:, -24:, :]
        )


def test_seasonal_naive_uses_only_the_input_window():
    """Constructed so that any leak from `y` would show up as a wrong value."""
    x = np.arange(2 * 48 * 1, dtype=np.float64).reshape(2, 48, 1)
    y = np.full((2, 48, 1), -999.0)
    window = data_mod.Windows(x=x, y=y, cycle_index=np.zeros(2, int),
                              row_index=np.zeros(2, int), split="test")

    pred = data_mod.seasonal_naive(window, period=24)
    assert not np.any(pred == -999.0), "baseline read from the target array"
    np.testing.assert_array_equal(pred[:, :24, :], x[:, -24:, :])


def test_seasonal_naive_rejects_a_period_longer_than_the_look_back():
    x = np.zeros((1, 12, 1))
    window = data_mod.Windows(x=x, y=np.zeros((1, 12, 1)), cycle_index=np.zeros(1, int),
                              row_index=np.zeros(1, int), split="test")
    with pytest.raises(ValueError, match="exceeds the look-back"):
        data_mod.seasonal_naive(window, period=24)


@has_csv
def test_horizon_not_divisible_by_the_period_is_truncated_not_padded():
    ours = data_mod.make_windows("test", seq_len=96, pred_len=100)
    pred = data_mod.seasonal_naive(ours, period=24)
    assert pred.shape == ours.y.shape
    np.testing.assert_array_equal(pred[:, 96:100, :], ours.x[:, -24:-20, :])
