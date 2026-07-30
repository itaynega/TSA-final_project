"""Forecast error metrics, in the course's notation.

Written to `COURSE_NOTATION_2026-07-30.md` §2.1, which records these formulas as
they are printed on slides 47-48 of `Time-Series Forecasting.pdf` (PDF pp. 45-46).
The course defines the error term first:

    e_t = y_t - f(x_t)

and then, over N points:

    MSE  = (1/N) * sum_{t=1..N} e_t^2
    MAE  = (1/N) * sum_{t=1..N} |e_t|
    RMSE = sqrt(MSE)
    MdAE = median(|e_t|)

Four metrics, and only four. Between them they discharge two requirements:

  * B7 asks for the paper's own metric. TQNet reports MSE and MAE.
  * B8 asks for at least one metric studied in class. All four are in the course,
    which means MAE satisfies B7 and B8 at the same time; RMSE and MdAE are
    carried as well so that B8 does not rest on a single line.

MAPE and SMAPE are deliberately not here. Both divide by |y_t|, and long-horizon
forecasting results — including TQNet's — are computed on z-scored data, where the
series crosses zero. The course's own table already marks MAPE undefined at
y_t = 0. If either is ever wanted, compute it on the original scale, label it as
such, and follow the course in writing it without a x100 factor.

Two conventions worth stating once, because getting either wrong produces a
number that looks plausible and is not comparable to anything:

  1. **Sign.** The course writes e_t = y_t - f(x_t); TQNet's own `utils/metrics.py`
     writes `pred - true`, the opposite. Every metric here squares or takes the
     modulus, so the two agree exactly. `tests/test_metrics.py` pins that.

  2. **Reduction.** Predictions arrive shaped (n_windows, pred_len, n_features).
     The long-horizon literature, TQNet included, reduces over every element at
     once — one flat mean, not a per-window mean that is then averaged. For MSE
     and MAE on a rectangular array the two happen to agree; for MdAE they do
     not. Flat is what is implemented, because flat is what 0.3712 / 0.3928 were
     computed with.

Kept to numpy, and to syntax Python 3.8 accepts, because TQNet's environment
pins 3.8 and this module has to import there unchanged.
"""

from typing import Dict, Sequence, Union

import numpy as np

__all__ = ["errors", "mse", "mae", "rmse", "mdae", "all_metrics", "METRIC_NAMES"]

# The order the report's tables use. MSE and MAE first: those are the paper's.
METRIC_NAMES = ("MSE", "MAE", "RMSE", "MdAE")

ArrayLike = Union[np.ndarray, Sequence[float]]


def _validate(y_true: ArrayLike, y_pred: ArrayLike):
    """Return the two inputs as float arrays, or explain why they are unusable.

    Each check here corresponds to a way of getting a wrong number quietly:

      * empty input makes numpy return nan with a RuntimeWarning that is easy
        to miss in a training log;
      * mismatched shapes that happen to broadcast — (N,) against (N, 1) is the
        common one — return a real number computed over an outer product;
      * a single nan anywhere propagates to the mean and turns the whole run's
        headline into nan, usually noticed long after the run has finished.
    """
    true = np.asarray(y_true, dtype=np.float64)
    pred = np.asarray(y_pred, dtype=np.float64)

    if true.size == 0 or pred.size == 0:
        raise ValueError("cannot compute a metric on empty input")

    if true.shape != pred.shape:
        raise ValueError(
            "y_true and y_pred must have the same shape; got {} and {}. "
            "Note that shapes such as (N,) and (N, 1) would broadcast silently, "
            "so this is rejected rather than reduced.".format(true.shape, pred.shape)
        )

    if not np.isfinite(true).all():
        raise ValueError("y_true contains values that are not finite (nan or inf)")
    if not np.isfinite(pred).all():
        raise ValueError("y_pred contains values that are not finite (nan or inf)")

    return true, pred


def errors(y_true: ArrayLike, y_pred: ArrayLike) -> np.ndarray:
    """The forecast error, e_t = y_t - f(x_t).

    Shape is preserved, so this can be used for residual diagnostics as well as
    for the metrics below.
    """
    true, pred = _validate(y_true, y_pred)
    return true - pred


def mse(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Mean squared error: (1/N) * sum e_t^2.

    Penalises large errors quadratically, so it is dominated by the worst
    windows. This is TQNet's headline metric.
    """
    e = errors(y_true, y_pred)
    return float(np.mean(e ** 2))


def mae(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Mean absolute error: (1/N) * sum |e_t|.

    Same units as the target, and weights every error linearly.
    """
    e = errors(y_true, y_pred)
    return float(np.mean(np.abs(e)))


def rmse(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Root mean squared error: sqrt(MSE).

    Defined by the course as the square root of MSE rather than as its own sum,
    and implemented that way here so the identity holds exactly.
    """
    return float(np.sqrt(mse(y_true, y_pred)))


def mdae(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Median absolute error: median(|e_t|).

    The robust counterpart to MAE. A handful of badly-missed windows move MAE
    and leave MdAE where it was, so a gap between the two is itself a finding.
    """
    e = errors(y_true, y_pred)
    return float(np.median(np.abs(e)))


def all_metrics(y_true: ArrayLike, y_pred: ArrayLike) -> Dict[str, float]:
    """All four metrics at once, keyed by the names the report's tables use.

    This is the function the results-writer calls, so that every number reaching
    the report is produced by one code path rather than by four call sites that
    might disagree.
    """
    true, pred = _validate(y_true, y_pred)
    return {
        "MSE": mse(true, pred),
        "MAE": mae(true, pred),
        "RMSE": rmse(true, pred),
        "MdAE": mdae(true, pred),
    }
