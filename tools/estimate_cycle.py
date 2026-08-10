"""Dominant-period estimator (Arm B -- J-14).

`report/prereg-improvement.md` sec 3 "Arm B" specifies "an estimator (ACF peak
and periodogram argmax, computed on training rows [0,8640) only, agreeing or
the run fails loudly) plus a --cycle auto path", and fixes the abandon
condition: **ACF and periodogram disagree on the dominant period.**

It does not say which series the estimator runs on, and J-14's dispatch says
that choice was fixed by the PM in STAGE2_WORKPLAN_2026-08-09.md sec "7j".
**That section does not exist in the workplan as delivered** -- the file's
final numbered section is 7i, followed directly by section 8. This module
implements the rule as it was stated inline in the J-14 dispatch text itself
(reproduced below), and the discrepancy is reported rather than silently
patched over (J-14 return, item 9).

**Aggregation rule** (as given). The estimator runs on the channel-mean of
the seven ETTh1 columns over training rows [0, 8640). Justification, stated
independent of any outcome: `TQNet.models.TQNet.Model.temporalQuery` has
shape `(cycle_len, enc_in)` -- the architecture consumes exactly *one* W for
all seven channels, so an estimator that required per-channel unanimity would
be estimating a quantity the model cannot accept. Per-channel values are
computed and reported (F4/F6 content) but never used to decide.

**The ACF-argmax trap.** A naive "ACF peak" as `argmax(ac[min_lag:max_lag])`
returns whichever lag is closest to lag 0, because autocorrelation on this
kind of data decays monotonically out of lag 0 -- on ETTh1 train rows that is
lag 2, for every channel, independent of any real periodicity. "Peak" here
means the largest **local maximum**: a lag k with `ac[k] > ac[k-1]` and
`ac[k] >= ac[k+1]`, searched over `[min_lag, max_lag]`. If the search range
contains no local maximum, that is reported as a failure (`None`), never
silently coerced to the global argmax.

**Lag / period range: [2, 400].** 2 excludes lags 0 and 1, which are governed
by adjacent-sample correlation and are never a plausible "cycle" on hourly
data (this is exactly the value that the argmax trap above returns). 400
covers everything from sub-daily periods up to just over 16 days -- comfortably
spanning ETTh1's known diurnal (24h) and any plausible weekly (168h)
structure -- while staying far inside the 8640-row training window, so the
periodogram still has many spectral bins per candidate period even near the
top of the range (8640 / 400 = 21.6 bins/period at the coarsest point).

**Harmonic honesty** -- decided here, once, in code and in words together.
ACF and periodogram "agree" under `agree()` **iff they return the exact same
integer period.** A period-24 series' ACF also has a local maximum near lag
48 (its first harmonic), and a non-sinusoidal 24h cycle's periodogram often
carries real power at period 12 (its first subharmonic in frequency). Neither
is treated as agreement with 24: **if one method returns 48 and the other
returns 24 (or 12 and 24), that is disagreement**, full stop, and the abandon
condition in sec 3 fires. This is deliberately the stricter of the two
readings available -- the one that cannot be talked into passing after the
fact. It is applied uniformly, including to ETTh1's own LUFL channel, which
is exactly this case (see `report/cycle_estimate.md`).
"""

import argparse
import json
import sys

import numpy as np
import pandas as pd

__all__ = [
    "DEFAULT_MIN_LAG",
    "DEFAULT_MAX_LAG",
    "CycleDisagreementError",
    "acf_values",
    "acf_peak",
    "periodogram_peak",
    "agree",
    "estimate_series",
    "estimate_or_raise",
    "load_channel_mean",
    "load_channels",
]

DEFAULT_MIN_LAG = 2
DEFAULT_MAX_LAG = 400


class CycleDisagreementError(RuntimeError):
    """ACF and periodogram do not agree on the dominant period.

    `report/prereg-improvement.md` sec 3 "Arm B": "Abandon if: ACF and
    periodogram disagree on the dominant period." Raising this, rather than
    returning a null, is what makes the failure loud (standing order R11: an
    empty error message is a harness failure, not a result -- so the message
    always carries both periods and the row/lag range that produced them).
    """


# ---------------------------------------------------------------------------
# The two methods.
# ---------------------------------------------------------------------------


def acf_values(x, max_lag):
    """Autocorrelation of 1-D array `x` at lags 0..max_lag, via FFT.

    Biased estimator (each lag divided by n, not n - lag -- the conventional
    choice for spectral/ACF work since it guarantees a positive-semidefinite
    sequence), mean-centered, normalised so `ac[0] == 1.0`.
    """
    x = np.asarray(x, dtype=np.float64)
    n = x.size
    if max_lag >= n:
        raise ValueError(
            "max_lag ({}) must be < series length ({})".format(max_lag, n)
        )
    xc = x - x.mean()
    size = 1
    while size < 2 * n:
        size *= 2
    f = np.fft.rfft(xc, n=size)
    acov_full = np.fft.irfft(f * np.conj(f), n=size)[:n]
    acov = acov_full / n
    denom = acov[0]
    if denom == 0:
        raise ValueError("series is constant; autocorrelation is undefined")
    ac = acov / denom
    return ac[: max_lag + 1]


def _local_maxima(ac, min_lag, max_lag):
    """Lags k in [min_lag, max_lag] with ac[k] > ac[k-1] and ac[k] >= ac[k+1]."""
    peaks = []
    upper = min(max_lag, ac.size - 2)
    for k in range(min_lag, upper + 1):
        if ac[k] > ac[k - 1] and ac[k] >= ac[k + 1]:
            peaks.append(k)
    return peaks


def acf_peak(x, min_lag=DEFAULT_MIN_LAG, max_lag=DEFAULT_MAX_LAG):
    """Largest local-maximum lag of the ACF in [min_lag, max_lag].

    "Largest" = highest ac value among the local maxima found (not the
    largest lag). Returns `(period_or_None, ac_value_or_None, all_local_maxima)`
    where `all_local_maxima` maps every local-maximum lag found in range to
    its ac value, for the report's harmonic discussion.
    """
    x = np.asarray(x, dtype=np.float64)
    max_lag = min(max_lag, x.size - 2)
    ac = acf_values(x, max_lag)
    peaks = _local_maxima(ac, min_lag, max_lag)
    if not peaks:
        return None, None, {}
    peak_vals = {int(k): float(ac[k]) for k in peaks}
    best = max(peaks, key=lambda k: ac[k])
    return int(best), float(ac[best]), peak_vals


def periodogram_peak(x, min_lag=DEFAULT_MIN_LAG, max_lag=DEFAULT_MAX_LAG):
    """argmax of the power spectrum, expressed as an integer period, restricted
    to periods in `[min_lag, max_lag]`.

    Single FFT of the mean-centered series, no windowing beyond that (matching
    the ACF side, which also does no windowing/tapering). The DC bin (period
    = infinity) is always excluded. Returns `(period_or_None, power_or_None)`.
    """
    x = np.asarray(x, dtype=np.float64)
    n = x.size
    xc = x - x.mean()
    fft = np.fft.rfft(xc)
    power = np.abs(fft) ** 2
    freqs = np.fft.rfftfreq(n)  # cycles per sample

    best_i, best_period, best_power = None, None, -1.0
    for i in range(1, len(freqs)):
        if freqs[i] <= 0:
            continue
        period = 1.0 / freqs[i]
        if min_lag <= period <= max_lag and power[i] > best_power:
            best_i, best_period, best_power = i, period, power[i]

    if best_i is None:
        return None, None
    return int(round(best_period)), float(best_power)


def agree(acf_period, pg_period):
    """Strict agreement: same integer period, harmonics and subharmonics do
    NOT count (see module docstring, "Harmonic honesty")."""
    if acf_period is None or pg_period is None:
        return False
    return int(acf_period) == int(pg_period)


def estimate_series(x, min_lag=DEFAULT_MIN_LAG, max_lag=DEFAULT_MAX_LAG):
    """Run both methods on `x` and report the full record. Never raises on
    disagreement -- that is `estimate_or_raise`'s job."""
    acf_period, acf_val, acf_local_maxima = acf_peak(x, min_lag, max_lag)
    pg_period, pg_power = periodogram_peak(x, min_lag, max_lag)
    return {
        "n": int(np.asarray(x).size),
        "min_lag": int(min_lag),
        "max_lag": int(max_lag),
        "acf_period": acf_period,
        "acf_peak_value": acf_val,
        "acf_local_maxima": acf_local_maxima,
        "periodogram_period": pg_period,
        "periodogram_power": pg_power,
        "agree": agree(acf_period, pg_period),
    }


def estimate_or_raise(x, min_lag=DEFAULT_MIN_LAG, max_lag=DEFAULT_MAX_LAG, label=None):
    """`estimate_series`, then fail loudly (raise `CycleDisagreementError`) if
    the two methods disagree. On agreement, returns the integer period."""
    record = estimate_series(x, min_lag=min_lag, max_lag=max_lag)
    if not record["agree"]:
        raise CycleDisagreementError(
            "ACF and periodogram disagree on the dominant period{}: "
            "ACF={} (peak ac={}), periodogram={} (power={}), "
            "lag/period range=[{}, {}], n={}. "
            "report/prereg-improvement.md sec 3 'Arm B': abandon.".format(
                "" if label is None else " for {!r}".format(label),
                record["acf_period"],
                record["acf_peak_value"],
                record["periodogram_period"],
                record["periodogram_power"],
                min_lag,
                max_lag,
                record["n"],
            )
        )
    return record["acf_period"], record


# ---------------------------------------------------------------------------
# CSV loading.
# ---------------------------------------------------------------------------


def _select_columns(frame, columns):
    if columns is None:
        return [c for c in frame.columns if c != "date"]
    return list(columns)


def load_channels(csv_path, row_start, row_stop, columns=None):
    """Return `{channel_name: 1-D float64 array}` for `columns` (default: every
    non-`date` column, in file order) over rows `[row_start, row_stop)`."""
    frame = pd.read_csv(csv_path)
    cols = _select_columns(frame, columns)
    sliced = frame[cols].values[row_start:row_stop].astype(np.float64)
    return {c: sliced[:, i] for i, c in enumerate(cols)}


def load_channel_mean(csv_path, row_start, row_stop, columns=None):
    """Row-wise mean across `columns` (default: every non-`date` column) over
    rows `[row_start, row_stop)` -- the aggregation rule this module's
    docstring fixes. Returns a 1-D float64 array."""
    channels = load_channels(csv_path, row_start, row_stop, columns=columns)
    stacked = np.stack(list(channels.values()), axis=1)
    return stacked.mean(axis=1)


# ---------------------------------------------------------------------------
# CLI.
# ---------------------------------------------------------------------------


def _build_parser():
    p = argparse.ArgumentParser(
        description=(
            "Estimate the dominant period of a CSV column (or channel-mean of "
            "several columns) by ACF local-maximum and periodogram argmax. "
            "Fails loudly (non-zero exit) if the two disagree."
        )
    )
    p.add_argument("csv_path", help="path to the CSV file")
    p.add_argument("--start", type=int, default=0, help="row range start (inclusive)")
    p.add_argument("--stop", type=int, required=True, help="row range stop (exclusive)")
    p.add_argument(
        "--columns",
        default=None,
        help="comma-separated column names to include (default: every non-'date' "
        "column). If more than one column is given, they are averaged row-wise "
        "(channel-mean) before estimation.",
    )
    p.add_argument("--min-lag", type=int, default=DEFAULT_MIN_LAG)
    p.add_argument("--max-lag", type=int, default=DEFAULT_MAX_LAG)
    p.add_argument("--json", action="store_true", help="print the full record as JSON")
    return p


def main(argv=None):
    args = _build_parser().parse_args(argv)
    columns = args.columns.split(",") if args.columns else None
    x = load_channel_mean(args.csv_path, args.start, args.stop, columns=columns)

    record = estimate_series(x, min_lag=args.min_lag, max_lag=args.max_lag)

    if args.json:
        print(json.dumps(record, indent=2, sort_keys=True))
    else:
        print(
            "rows=[{}, {}) columns={} lag_range=[{}, {}] "
            "acf={} (peak ac={}) periodogram={} (power={}) agree={}".format(
                args.start,
                args.stop,
                "all" if columns is None else ",".join(columns),
                args.min_lag,
                args.max_lag,
                record["acf_period"],
                record["acf_peak_value"],
                record["periodogram_period"],
                record["periodogram_power"],
                record["agree"],
            )
        )

    if not record["agree"]:
        sys.stderr.write(
            "estimate_cycle: ACF and periodogram disagree "
            "(ACF={}, periodogram={}) -- report/prereg-improvement.md sec 3 "
            "'Arm B': abandon.\n".format(record["acf_period"], record["periodogram_period"])
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
