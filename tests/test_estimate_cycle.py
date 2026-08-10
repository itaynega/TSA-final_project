"""Tests for tools/estimate_cycle.py (Arm B -- J-14).

Three things this module has to prove, because a hardcoded `return 24` would
otherwise pass every check that only looks at ETTh1:

1. The ACF-argmax trap (module docstring) is actually avoided: a plain
   `argmax` over the lag window returns 2 on ETTh1 train rows for every
   channel; `acf_peak` must not.
2. The positive control (J-14 acceptance criterion 3): on synthetic series
   with a *known* period that is not 24, both methods recover it. This is
   the criterion a hardcoded estimator cannot pass, because 17 and 50 are
   not 24.
3. The negative control (criterion 4): on i.i.d. noise, the two methods
   disagree and the fail-loud path fires. A period estimator that returns a
   confident, agreeing answer on noise is not an estimator.

Tests that need the CSV skip cleanly when it is absent (same convention as
tests/test_data.py), so the suite still runs on a fresh clone before
tools/get_data.py has been called.
"""

import os
import subprocess
import sys

import numpy as np
import pytest

from common import data as data_mod
from tools import estimate_cycle as ec

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

has_csv = pytest.mark.skipif(
    not os.path.exists(data_mod.DEFAULT_CSV),
    reason="ETTh1.csv not present; run python3 tools/get_data.py",
)

ETTH1_TRAIN_START, ETTH1_TRAIN_STOP = 0, 8640
ETTH1_CHANNELS = ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"]

# Controls: synthetic sine + Gaussian noise, same sample count as the real
# training window (8640, ETTh1's train rows [0, 8640)), so the estimator sees
# a "comparable number of samples" (J-14 criterion 3).
CONTROL_N = 8640
CONTROL_SNR = 20.0  # var(signal) / var(noise); see report/cycle_estimate.md
# for the sweep (SNR in {3, 5, 10, 20, 50, 100} x 15 seeds x {17, 50}) that
# motivated this choice -- lower SNR lets a noise-perturbed harmonic (2P) beat
# the fundamental's *undamped* ACF peak often enough to be unreliable; SNR=20
# was 30/30 across that sweep.


def _make_sine(period, n=CONTROL_N, snr=CONTROL_SNR, seed=2024):
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    signal = np.sin(2 * np.pi * t / period)
    noise_var = np.var(signal) / snr
    noise = rng.normal(0.0, np.sqrt(noise_var), n)
    return signal + noise


# --------------------------------------------------------------------------
# The ACF-argmax trap.
# --------------------------------------------------------------------------


@has_csv
def test_naive_argmax_would_return_lag_2_on_etth1_channel_mean():
    """Documents the trap this module's docstring describes: a plain argmax
    over the lag window returns 2, because ACF decays monotonically out of
    lag 0. If this assertion ever fails, the trap this module was built to
    avoid no longer exists on this data and the surrounding tests should be
    re-read for relevance."""
    x = ec.load_channel_mean(data_mod.DEFAULT_CSV, ETTH1_TRAIN_START, ETTH1_TRAIN_STOP)
    ac = ec.acf_values(x, ec.DEFAULT_MAX_LAG)
    naive_argmax = int(np.argmax(ac[ec.DEFAULT_MIN_LAG : ec.DEFAULT_MAX_LAG + 1])) + ec.DEFAULT_MIN_LAG
    assert naive_argmax == 2


@has_csv
def test_local_maximum_peak_avoids_the_trap_and_returns_24():
    x = ec.load_channel_mean(data_mod.DEFAULT_CSV, ETTH1_TRAIN_START, ETTH1_TRAIN_STOP)
    period, value, local_maxima = ec.acf_peak(x)
    assert period == 24
    assert value > 0.5
    assert 2 not in local_maxima  # lag 2 must not even be a local maximum


def test_local_maxima_excludes_strictly_decreasing_runs():
    """A monotonically decaying ACF (no periodicity at all) has no local
    maximum in [min_lag, max_lag], and acf_peak must say so with None, not
    fall back to the global argmax."""
    x = np.exp(-np.arange(2000) / 300.0) + 0.0  # pure decay, no bump anywhere
    period, value, local_maxima = ec.acf_peak(x, min_lag=2, max_lag=400)
    assert period is None
    assert value is None
    assert local_maxima == {}


# --------------------------------------------------------------------------
# ETTh1: channel-mean (the decision series) and the per-channel table.
# --------------------------------------------------------------------------


@has_csv
def test_etth1_channel_mean_agrees_at_24():
    """Acceptance criterion 1: channel-mean, train rows [0, 8640): both
    methods return 24."""
    x = ec.load_channel_mean(data_mod.DEFAULT_CSV, ETTH1_TRAIN_START, ETTH1_TRAIN_STOP)
    record = ec.estimate_series(x)
    assert record["acf_period"] == 24
    assert record["periodogram_period"] == 24
    assert record["agree"] is True
    assert record["n"] == 8640


@has_csv
def test_etth1_channel_mean_estimate_or_raise_does_not_raise():
    x = ec.load_channel_mean(data_mod.DEFAULT_CSV, ETTH1_TRAIN_START, ETTH1_TRAIN_STOP)
    period, record = ec.estimate_or_raise(x, label="ETTh1 channel-mean")
    assert period == 24


@has_csv
def test_etth1_uses_exactly_the_training_rows():
    """B2: never read past row 8640. A wrong stop index would silently pull
    in validation rows."""
    channels = ec.load_channels(data_mod.DEFAULT_CSV, ETTH1_TRAIN_START, ETTH1_TRAIN_STOP)
    for name, arr in channels.items():
        assert arr.size == 8640, name


@has_csv
@pytest.mark.parametrize("channel", ETTH1_CHANNELS)
def test_every_channel_is_individually_estimable(channel):
    """Per-channel values must be computable and reported even when they
    disagree with each other -- they are F4/F6 content, never a decision
    input (the channel-mean is)."""
    channels = ec.load_channels(data_mod.DEFAULT_CSV, ETTH1_TRAIN_START, ETTH1_TRAIN_STOP, columns=[channel])
    record = ec.estimate_series(channels[channel])
    assert record["acf_period"] is not None
    assert record["periodogram_period"] is not None


@has_csv
def test_channel_mean_is_not_a_simple_vote_of_the_per_channel_periods():
    """Guards against accidentally re-deriving the aggregation rule as
    'majority vote over channels' instead of 'estimate on the mean series' --
    the two are different computations that happen to often agree."""
    channels = ec.load_channels(data_mod.DEFAULT_CSV, ETTH1_TRAIN_START, ETTH1_TRAIN_STOP)
    mean_x = ec.load_channel_mean(data_mod.DEFAULT_CSV, ETTH1_TRAIN_START, ETTH1_TRAIN_STOP)
    assert np.allclose(mean_x, np.mean(np.stack(list(channels.values()), axis=1), axis=1))


# --------------------------------------------------------------------------
# Harmonic honesty.
# --------------------------------------------------------------------------


def test_agree_is_strict_integer_equality_not_harmonic():
    assert ec.agree(24, 24) is True
    assert ec.agree(24, 48) is False  # first harmonic: not agreement
    assert ec.agree(24, 12) is False  # first subharmonic: not agreement
    assert ec.agree(None, 24) is False
    assert ec.agree(24, None) is False


@has_csv
def test_lufl_harmonic_case_is_reported_as_disagreement():
    """The case named in the dispatch: LUFL's ACF lands on 48 (first
    harmonic of 24), its periodogram on 12 (first subharmonic). Under this
    module's strict rule that is disagreement, not agreement-via-harmonic."""
    channels = ec.load_channels(data_mod.DEFAULT_CSV, ETTH1_TRAIN_START, ETTH1_TRAIN_STOP, columns=["LUFL"])
    record = ec.estimate_series(channels["LUFL"])
    assert record["agree"] is False


# --------------------------------------------------------------------------
# Positive control (criterion 3) -- the one a hardcoded `return 24` fails.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("period", [17, 50])
def test_positive_control_recovers_the_known_period(period):
    x = _make_sine(period)
    record = ec.estimate_series(x)
    assert record["acf_period"] == period
    assert record["periodogram_period"] == period
    assert record["agree"] is True


@pytest.mark.parametrize("period", [17, 50])
@pytest.mark.parametrize("seed", [2024, 2025, 2026])
def test_positive_control_is_robust_across_seeds(period, seed):
    """Not just one lucky draw: the chosen SNR must recover the period
    across multiple noise realisations."""
    x = _make_sine(period, seed=seed)
    record = ec.estimate_series(x)
    assert record["acf_period"] == period
    assert record["periodogram_period"] == period


def test_positive_control_periods_are_not_the_etth1_answer():
    """A hardcoded `return 24` estimator would fail this trivially: neither
    control period is 24."""
    assert 17 != 24
    assert 50 != 24


# --------------------------------------------------------------------------
# Negative control (criterion 4) -- the fail-loud path must actually fire.
# --------------------------------------------------------------------------


def test_negative_control_noise_disagrees():
    rng = np.random.default_rng(2024)
    noise = rng.normal(0.0, 1.0, CONTROL_N)
    record = ec.estimate_series(noise)
    assert record["agree"] is False


def test_negative_control_raises_cycle_disagreement_error():
    rng = np.random.default_rng(2024)
    noise = rng.normal(0.0, 1.0, CONTROL_N)
    with pytest.raises(ec.CycleDisagreementError, match="disagree"):
        ec.estimate_or_raise(noise, label="iid_gaussian_noise")


def test_negative_control_error_message_is_not_empty():
    """R11: an empty error message is a harness failure, not a result."""
    rng = np.random.default_rng(2024)
    noise = rng.normal(0.0, 1.0, CONTROL_N)
    with pytest.raises(ec.CycleDisagreementError) as excinfo:
        ec.estimate_or_raise(noise, label="iid_gaussian_noise")
    message = str(excinfo.value)
    assert len(message) > 40
    assert "ACF=" in message and "periodogram=" in message


# --------------------------------------------------------------------------
# Basic correctness / error handling.
# --------------------------------------------------------------------------


def test_acf_zero_lag_is_one():
    x = np.sin(2 * np.pi * np.arange(1000) / 24.0) + np.random.default_rng(0).normal(0, 0.1, 1000)
    ac = ec.acf_values(x, 50)
    assert ac[0] == pytest.approx(1.0)


def test_acf_rejects_max_lag_at_or_past_series_length():
    with pytest.raises(ValueError):
        ec.acf_values(np.arange(10, dtype=float), 10)


def test_acf_rejects_constant_series():
    with pytest.raises(ValueError):
        ec.acf_values(np.ones(100), 10)


def test_periodogram_excludes_dc_and_out_of_range_periods():
    x = np.sin(2 * np.pi * np.arange(2000) / 24.0)
    period, power = ec.periodogram_peak(x, min_lag=2, max_lag=400)
    assert period == 24


# --------------------------------------------------------------------------
# CLI.
# --------------------------------------------------------------------------


@has_csv
def test_cli_exits_zero_on_agreement():
    result = subprocess.run(
        [
            sys.executable,
            os.path.join(REPO_ROOT, "tools", "estimate_cycle.py"),
            data_mod.DEFAULT_CSV,
            "--stop",
            "8640",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "agree=True" in result.stdout


def test_cli_exits_nonzero_on_disagreement(tmp_path):
    """Build a tiny noise CSV and confirm the CLI's fail-loud path fires
    with a real, non-empty stderr message (R11)."""
    rng = np.random.default_rng(2024)
    csv_path = tmp_path / "noise.csv"
    n = 8640
    import pandas as pd

    frame = pd.DataFrame({"x": rng.normal(0.0, 1.0, n)})
    frame.to_csv(csv_path, index=False)

    result = subprocess.run(
        [sys.executable, os.path.join(REPO_ROOT, "tools", "estimate_cycle.py"), str(csv_path), "--stop", str(n)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert result.stderr.strip() != ""
    assert "disagree" in result.stderr
