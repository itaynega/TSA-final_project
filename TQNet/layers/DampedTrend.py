"""Damped-trend instance normalisation (Arm A -- J-11).

Implements the mechanism pre-registered in `report/prereg-improvement.md`
section 3, "Arm A -- Damped-trend instance normalisation". That file is
frozen; this module implements it and does not edit it, does not change its
predictions or thresholds, and does not select phi (phi selection is J-12,
on validation MSE at H=96 only).

The registered mechanism, quoted:

    Least-squares line per window per channel (closed form on a fixed 0..95
    index -- no loop), subtract, then normalise the residual as now; on the
    output, de-normalise as now and add the damped trend.

        trend added back at step h  =  slope * sum_{k=1..h} phi^k   0 < phi <= 1

    phi = 1 recovers plain linear extrapolation; phi -> 0 recovers the
    current model.

Two implementation choices are recorded here because section 3's prose does
not pin them down and a reader must be able to tell what was done:

1.  **"Subtract that line" is implemented as subtracting the slope component
    `b * (t - (S-1))`, taken about the window's LAST observation (t = S-1),
    not about the window's centre and not the full fitted line `a + b*t`.**

    This is J-12d's fix. It was previously implemented as `b * (t - t_bar)`,
    `t_bar = (S-1)/2` the window centre, on the rationale that this leaves the
    window's mean in the residual and so lets the existing `seq_mean` /
    `seq_var` instance normalisation carry the level through and restore it
    unchanged. **That rationale is correct on its own terms and wrong for
    what this module actually needs.** `retrend()` adds
    `slope * sum_{k=1..h} phi^k`, which is damped Holt's forecast projected
    forward from the window's LAST point, t = S-1 -- that is what "recovers
    plain linear extrapolation" at phi = 1 means, and it is the only
    convention under which `sum_{k=1..h} phi^k -> h` at phi = 1 reproduces
    `slope * h` continued from t = S-1. A centre-origin residual carries the
    fitted line's value AT THE CENTRE, not at t = S-1, so `retrend` was
    projecting the right slope forward from the wrong level -- the two
    disagreed by exactly `slope * (S-1)/2`, a per-window CONSTANT added at
    every forecast step, for every window, proportional to that window's
    slope. On a perfectly linear window this is exactly measurable: at
    phi = 1, an oracle forecaster returning the (exactly flat) detrended
    level reconstructs the window's own continuation short by `slope *
    (S-1)/2` at every step -- see
    `tests/test_damped_trend.py::test_phi_one_oracle_reconstructs_a_perfectly_linear_window_exactly`,
    which pins this down to 1e-9 and is the test that caught it (a constant
    offset, not a drift, is the signature of a misplaced origin rather than a
    wrong `sum_{k=1..h} phi^k` convention -- that convention is separately
    correct and tested by `test_damped_sum_*` above).

    Undetected by J-11's own two 1e-6 gate tests, because neither exercises
    both a non-zero slope AND phi = 1 with a ground truth to check against:
    the zero-slope test has `b = 0`, so the origin does not matter and the
    test passes at either origin; the trended test runs at phi -> 0, where
    `retrend` adds (almost) nothing regardless of which level `detrend` left
    behind, so it also passes at either origin. This is the same failure
    class as sec 3's own two prior silent failures in this component
    (STAGE2_WORKPLAN_2026-08-09.md sec 7h, sec 7i): a gate that cannot detect
    the failure it exists for.

    Fixed form: subtract `b * (t - (S-1))`. This still leaves the level in
    the residual, so `seq_mean` / `seq_var` and their inverse still work
    verbatim -- moving the origin only changes WHICH level survives (the
    line's value at t = S-1 instead of at t_bar), not whether one does. The
    network's input is provably unchanged by this fix: the two residuals
    (old origin vs new) differ only by the per-window constant `slope *
    (S-1)/2` (added uniformly over `t`, since only the origin subtracted from
    the centred index moved), and instance normalisation subtracts the
    window's own mean and divides by its own standard deviation -- a
    per-window additive constant shifts `seq_mean` by that same constant and
    leaves `seq_var` exactly unchanged, so the normalised tensor the network
    sees is identical bit-for-bit up to floating-point rounding. Only the
    *restored* level differs, which is exactly where the fix needed to act.
    `ols_slope_weights` (used by `slope()`) is untouched and keeps its own
    centre origin `t_bar = (S-1)/2` -- the OLS slope estimate is origin-
    invariant, so this does not need to and must not match `centred_index`.

2.  **The detrending is unconditional -- it is not gated by phi.** This
    follows damped Holt, where the trend is estimated from the data
    regardless of phi and phi damps only the extrapolation. A consequence,
    stated plainly: with a *trended* input, `phi -> 0` does **not** make
    this module numerically identical to the un-modified model, because the
    network is handed a detrended window either way. `phi -> 0` recovers
    the current model in the sense damped Holt means it -- no trend is
    projected forward -- and it is numerically exact whenever the window's
    fitted slope is zero. See `tests/test_damped_trend.py`, which pins both
    statements down.

No learned parameters are added: both tensors below are constants of
`seq_len`, `pred_len` and `phi`, computed once at construction. They are
registered non-persistently so they never enter `state_dict()` and the
model's trainable-parameter count is unchanged.
"""

import torch
import torch.nn as nn

__all__ = ["damped_trend_sum", "ols_slope_weights", "DampedTrendInstanceNorm"]


def damped_trend_sum(pred_len, phi, dtype=torch.float64):
    """`sum_{k=1..h} phi^k` for h = 1..pred_len, as a tensor of shape (pred_len,).

    Closed form used (geometric series, first term phi, ratio phi):

        sum_{k=1..h} phi^k = phi * (1 - phi**h) / (1 - phi)      for phi != 1
        sum_{k=1..h} phi^k = h                                    for phi == 1

    The `phi == 1` branch is the removable singularity of the first
    expression, taken exactly rather than approached. Note `h` starts at
    **1**, not 0: the first forecast step gets `phi`, not `1`. That is the
    off-by-one this function's test exists to catch -- with the wrong
    convention `sum` tends to 1 rather than to 0 as `phi -> 0`, and every
    horizon is then silently offset by one slope unit.
    """
    phi = float(phi)
    if not (0.0 <= phi <= 1.0):
        raise ValueError(
            "damped trend phi must satisfy 0 <= phi <= 1 (the pre-registration "
            "states 0 < phi <= 1; phi = 0 is admitted only as the exact limit "
            "the unit test uses), got {!r}".format(phi)
        )

    h = torch.arange(1, int(pred_len) + 1, dtype=dtype)
    if abs(1.0 - phi) < 1e-12:
        return h.clone()
    phi_t = torch.tensor(phi, dtype=dtype)
    return phi_t * (1.0 - torch.pow(phi_t, h)) / (1.0 - phi_t)


def ols_slope_weights(seq_len, dtype=torch.float64):
    """Constant weights `w` with `slope = sum_t w_t * x_t`, shape (seq_len,).

    The design matrix `X = [1, t]` on the fixed index `t = 0..seq_len-1` is
    the same for every window and every channel, so its pseudo-inverse is a
    constant and is formed once here rather than per window. `w` is the
    slope row of `(X^T X)^-1 X^T`:

        t_bar = (seq_len - 1) / 2
        S_tt  = sum_t (t - t_bar)**2 = seq_len * (seq_len**2 - 1) / 12
        w_t   = (t - t_bar) / S_tt

    Because `sum_t w_t == 0`, the intercept term drops out of `w . x` and the
    contraction gives the ordinary-least-squares slope directly, with no
    loop over windows or channels -- a single tensor contraction over the
    time axis handles the whole batch.
    """
    seq_len = int(seq_len)
    if seq_len < 2:
        raise ValueError("a slope needs at least 2 points, got seq_len={}".format(seq_len))
    t = torch.arange(seq_len, dtype=dtype)
    centred = t - (seq_len - 1) / 2.0
    s_tt = torch.sum(centred * centred)
    return centred / s_tt


class DampedTrendInstanceNorm(nn.Module):
    """Detrend a window before instance norm; re-add the damped trend after.

    Usage inside a forward pass, wrapping the existing instance norm:

        x, slope = self.damped_trend.detrend(x)      # before instance norm
        ...  existing normalise / network / de-normalise, unchanged  ...
        output = self.damped_trend.retrend(output, slope)

    `phi` is fixed at construction (it is a config value, selected once on
    validation by J-12 and then frozen), so both constants are precomputed.
    """

    def __init__(self, seq_len, pred_len, phi):
        super(DampedTrendInstanceNorm, self).__init__()
        self.seq_len = int(seq_len)
        self.pred_len = int(pred_len)
        self.phi = float(phi)

        t = torch.arange(self.seq_len, dtype=torch.float64)
        # Both constants are kept in float64 and cast to the activation dtype
        # at use, so the trend maths does not depend on the model's dtype.
        # Origin is the window's LAST observation (t = seq_len - 1), not its
        # centre -- see docstring point 1 for why. `retrend` projects forward
        # from t = seq_len - 1 (damped Holt's own convention), so the level
        # `detrend` must leave behind is the fitted line's value at that same
        # point, or the two disagree by a constant multiple of the slope.
        self.register_buffer(
            "centred_index", t - (self.seq_len - 1), persistent=False
        )
        self.register_buffer(
            "slope_weights", ols_slope_weights(self.seq_len), persistent=False
        )
        self.register_buffer(
            "damped_sum", damped_trend_sum(self.pred_len, self.phi), persistent=False
        )

    def extra_repr(self):
        return "seq_len={}, pred_len={}, phi={}".format(
            self.seq_len, self.pred_len, self.phi
        )

    def slope(self, x):
        """Per-window, per-channel OLS slope of `x` (b, s, c) -> (b, 1, c)."""
        w = self.slope_weights.to(dtype=x.dtype, device=x.device)
        return torch.einsum("s,bsc->bc", w, x).unsqueeze(1)

    def detrend(self, x):
        """`x` (b, s, c) -> (residual, slope), residual (b, s, c), slope (b, 1, c)."""
        slope = self.slope(x)
        centred = self.centred_index.to(dtype=x.dtype, device=x.device).view(1, -1, 1)
        return x - slope * centred, slope

    def retrend(self, y, slope):
        """Add `slope * sum_{k=1..h} phi^k` to `y` (b, pred_len, c)."""
        damped = self.damped_sum.to(dtype=y.dtype, device=y.device).view(1, -1, 1)
        return y + slope * damped
