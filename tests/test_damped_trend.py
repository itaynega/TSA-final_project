"""Tests for Arm A -- damped-trend instance normalisation (J-11).

The mechanism is pre-registered in `report/prereg-improvement.md` section 3,
"Arm A", and implemented in `TQNet/layers/DampedTrend.py` +
`TQNet/models/TQNet.py`. This file's job is to prove three things:

1.  With the arm **off**, `TQNet.Model.forward` is the published forward pass,
    bit for bit. Demonstrated, not asserted: the pre-change body is copied
    verbatim into `_published_forward` below and both are run on the same
    weights and the same input.

2.  With the arm **on** and `phi -> 0`, no trend is projected forward. This is
    the gate the job exists for. The failure mode it catches is an off-by-one
    or a sign error in `sum_{k=1..h} phi^k`, which produces output that looks
    reasonable, trains fine, and is quietly wrong at every horizon.

3.  With the arm **on** and `phi = 1`, the trend added back is exactly plain
    linear extrapolation, `slope * h`.

**On what "phi -> 0 recovers the current model" can and cannot mean.**
Section 3 says the detrending is unconditional -- fit the line, subtract it,
normalise the residual -- and separately that `phi -> 0 recovers the current
model`. Read as a claim about the whole network those two statements cannot
both hold for an input that has a trend: the network is handed a *detrended*
window whatever phi is, and a network is not the identity, so its output moves.
`phi` controls only what is added back afterwards. So this file pins the claim
down in the two forms in which it is actually well-posed, and both are
required to pass:

* `test_phi_to_zero_matches_arm_off_when_the_window_has_no_trend` -- the
  literal comparison, arm on at `phi -> 0` against arm off, on a fixed input
  whose fitted slope is zero. Exact agreement is expected and required. This
  is what proves the detrend / normalise / de-normalise / re-trend round trip
  is neutral: any sign error on the centring, any mismatch between the
  statistics used to normalise and to de-normalise, and any non-zero constant
  leaking out of `sum_{k=1..h} phi^k` breaks it.

* `test_phi_to_zero_matches_arm_off_on_the_detrended_window` -- the same
  comparison on a strongly trended input, against the arm-off model run on the
  detrended window. Exact agreement is expected and required, and *this* is
  the test with a non-zero slope in play, so it is the one that catches an
  off-by-one in `sum_{k=1..h} phi^k`: with the wrong convention the sum tends
  to 1 rather than 0 and a whole slope unit is added at every horizon.

Everything runs in float64 (`model.double()`) so that the 1e-6 tolerance the
dispatch asks for measures the arithmetic rather than float32 rounding, and in
`eval()` so the two dropout layers (p = 0.5) are off and the forward pass is
deterministic.
"""

import math
import os
import sys

import numpy as np
import pytest
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TQNET_DIR = os.path.join(REPO_ROOT, "TQNet")

TOLERANCE = 1e-6

SEQ_LEN = 96
PRED_LEN = 96
ENC_IN = 7
CYCLE = 24
BATCH = 4


def _import_from_tqnet():
    """Import the model and the trend module the way `run.py` does."""
    sys.path.insert(0, TQNET_DIR)
    try:
        from layers.DampedTrend import (  # noqa: E402
            DampedTrendInstanceNorm,
            damped_trend_sum,
            ols_slope_weights,
        )
        from models.TQNet import Model  # noqa: E402
    finally:
        if TQNET_DIR in sys.path:
            sys.path.remove(TQNET_DIR)
    return Model, DampedTrendInstanceNorm, damped_trend_sum, ols_slope_weights


Model, DampedTrendInstanceNorm, damped_trend_sum, ols_slope_weights = _import_from_tqnet()


class _Configs(object):
    """The attributes `TQNet.Model.__init__` reads, and nothing else."""

    def __init__(self, **overrides):
        self.seq_len = SEQ_LEN
        self.pred_len = PRED_LEN
        self.enc_in = ENC_IN
        self.cycle = CYCLE
        self.model_type = "mlp"
        self.d_model = 64
        self.dropout = 0.5
        self.use_revin = 1
        self.use_tq = 1
        self.channel_aggre = 1
        self.use_damped_trend = 0
        self.damped_phi = 0.9
        for key, value in overrides.items():
            setattr(self, key, value)


def _build(**overrides):
    """A deterministic model in float64, eval mode."""
    torch.manual_seed(20260809)
    model = Model(_Configs(**overrides)).double()
    model.eval()
    return model


def _pair(**overrides):
    """An arm-on model and an arm-off model carrying identical weights.

    `DampedTrendInstanceNorm` registers its two constants non-persistently and
    adds no parameters, so the two `state_dict()`s have identical keys and the
    copy below is exact.
    """
    on = _build(use_damped_trend=1, **overrides)
    off = _build(use_damped_trend=0, **overrides)
    off.load_state_dict(on.state_dict())
    off.eval()
    assert set(on.state_dict()) == set(off.state_dict())
    return on, off


def _fixed_input(trend_per_step=0.0, seed=11):
    """A fixed (b, s, c) window, optionally with a linear trend laid on top."""
    generator = torch.Generator().manual_seed(seed)
    x = torch.randn(BATCH, SEQ_LEN, ENC_IN, generator=generator, dtype=torch.float64)
    # A deterministic seasonal component, so the window is not pure noise.
    t = torch.arange(SEQ_LEN, dtype=torch.float64).view(1, -1, 1)
    x = x + 2.0 * torch.sin(2.0 * math.pi * t / CYCLE)
    if trend_per_step:
        per_channel = torch.arange(1, ENC_IN + 1, dtype=torch.float64).view(1, 1, -1)
        x = x + trend_per_step * t * per_channel
    return x


def _cycle_index():
    return torch.arange(BATCH, dtype=torch.long) % CYCLE


def _detrended(x, module):
    residual, _ = module.detrend(x)
    return residual


def _max_abs_diff(a, b):
    return float(torch.max(torch.abs(a - b)))


# --------------------------------------------------------------------------
# 1. The arm-off path is the published forward pass, unchanged.
# --------------------------------------------------------------------------


def _published_forward(self, x, cycle_index):
    """`TQNet.Model.forward` exactly as it stood before Arm A was added.

    Copied verbatim from `TQNet/models/TQNet.py` at the commit this job
    started from (lines 44-78), so that "the arm-off path is unchanged" is
    demonstrated against the old code rather than asserted in prose.
    """

    # instance norm
    if self.use_revin:
        seq_mean = torch.mean(x, dim=1, keepdim=True)
        seq_var = torch.var(x, dim=1, keepdim=True) + 1e-5
        x = (x - seq_mean) / torch.sqrt(seq_var)

    # b,s,c -> b,c,s
    x_input = x.permute(0, 2, 1)

    if self.use_tq:
        gather_index = (cycle_index.view(-1, 1) + torch.arange(self.seq_len, device=cycle_index.device).view(1, -1)) % self.cycle_len
        query_input = self.temporalQuery[gather_index].permute(0, 2, 1)  # (b, c, s)
        if self.channel_aggre:
            channel_information = self.channelAggregator(query=query_input, key=x_input, value=x_input)[0]
        else:
            channel_information = query_input
    else:
        if self.channel_aggre:
            channel_information = self.channelAggregator(query=x_input, key=x_input, value=x_input)[0]
        else:
            channel_information = 0

    input = self.input_proj(x_input+channel_information)

    hidden = self.model(input)

    output = self.output_proj(hidden+input).permute(0, 2, 1)

    # instance denorm
    if self.use_revin:
        output = output * torch.sqrt(seq_var) + seq_mean

    return output


@pytest.mark.parametrize("use_revin", [1, 0])
@pytest.mark.parametrize("use_tq,channel_aggre", [(1, 1), (0, 0), (1, 0), (0, 1)])
def test_arm_off_is_the_published_forward_pass_bit_for_bit(use_revin, use_tq, channel_aggre):
    """With the arm off, the new forward equals the pre-change one exactly."""
    model = _build(
        use_damped_trend=0,
        use_revin=use_revin,
        use_tq=use_tq,
        channel_aggre=channel_aggre,
    )
    x = _fixed_input(trend_per_step=0.02)
    cycle_index = _cycle_index()

    with torch.no_grad():
        produced = model(x, cycle_index)
        published = _published_forward(model, x, cycle_index)

    assert torch.equal(produced, published), (
        "arm-off forward differs from the published one by "
        "{:.3e}".format(_max_abs_diff(produced, published))
    )


def test_arm_off_model_has_no_damped_trend_state():
    """The flag defaults to off, and off means the submodule is never built."""
    default = _build()
    assert default.use_damped_trend is False
    assert not hasattr(default, "damped_trend")

    on, off = _pair()
    assert hasattr(on, "damped_trend")
    n_on = sum(p.numel() for p in on.parameters())
    n_off = sum(p.numel() for p in off.parameters())
    assert n_on == n_off, "Arm A must not add trainable parameters"


# --------------------------------------------------------------------------
# 2. The trend maths, on its own.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("phi", [0.0, 1e-8, 0.5, 0.8, 0.9, 0.95, 1.0])
@pytest.mark.parametrize("pred_len", [1, 96, 720])
def test_damped_sum_closed_form_matches_a_naive_loop(phi, pred_len):
    """`phi (1 - phi**h) / (1 - phi)` equals `sum_{k=1..h} phi**k`, term by term."""
    closed = damped_trend_sum(pred_len, phi)

    naive = []
    running = 0.0
    for h in range(1, pred_len + 1):
        running += phi ** h
        naive.append(running)
    naive = torch.tensor(naive, dtype=torch.float64)

    assert closed.shape == (pred_len,)
    assert torch.max(torch.abs(closed - naive)) < 1e-12


@pytest.mark.parametrize("phi", [0.0, 1e-8, 0.5, 0.8, 0.9, 0.95, 1.0])
def test_damped_sum_starts_at_phi_not_at_one(phi):
    """h = 1 gets phi. The off-by-one this guards against is `sum_{k=0..h-1}`."""
    assert abs(float(damped_trend_sum(8, phi)[0]) - phi) < 1e-15


def test_damped_sum_at_phi_one_is_plain_linear_extrapolation():
    """phi = 1 gives sum = h, i.e. the undamped line continued."""
    total = damped_trend_sum(720, 1.0)
    expected = torch.arange(1, 721, dtype=torch.float64)
    assert torch.equal(total, expected)


def test_damped_sum_vanishes_as_phi_goes_to_zero():
    assert float(torch.max(torch.abs(damped_trend_sum(720, 0.0)))) == 0.0
    assert float(torch.max(torch.abs(damped_trend_sum(720, 1e-8)))) < 1e-7


@pytest.mark.parametrize("phi", [-0.1, 1.5])
def test_damped_sum_rejects_phi_outside_the_unit_interval(phi):
    with pytest.raises(ValueError):
        damped_trend_sum(96, phi)


def test_slope_weights_are_a_precomputed_constant_of_seq_len():
    """One constant vector, formed once -- no loop over windows or channels."""
    module = DampedTrendInstanceNorm(SEQ_LEN, PRED_LEN, 0.9)
    assert module.slope_weights.shape == (SEQ_LEN,)
    assert torch.equal(module.slope_weights, ols_slope_weights(SEQ_LEN))
    # sum_t w_t == 0 is what makes the intercept drop out of the contraction.
    assert abs(float(torch.sum(module.slope_weights))) < 1e-15
    # And the constants stay out of state_dict, so checkpoints are unaffected.
    assert module.state_dict() == {}


def test_closed_form_slope_equals_a_least_squares_fit():
    """The contraction reproduces `numpy.polyfit` slopes, per window per channel.

    Checked against a genuinely independent estimator rather than against a
    rearrangement of the same formula.
    """
    module = DampedTrendInstanceNorm(SEQ_LEN, PRED_LEN, 0.9)
    x = _fixed_input(trend_per_step=0.037)

    produced = module.slope(x)
    assert produced.shape == (BATCH, 1, ENC_IN)

    t = np.arange(SEQ_LEN, dtype=np.float64)
    values = x.numpy()
    for b in range(BATCH):
        for c in range(ENC_IN):
            reference = np.polyfit(t, values[b, :, c], 1)[0]
            assert abs(float(produced[b, 0, c]) - float(reference)) < 1e-10


def test_detrending_removes_the_slope_it_measured():
    module = DampedTrendInstanceNorm(SEQ_LEN, PRED_LEN, 0.9)
    x = _fixed_input(trend_per_step=0.05)
    residual, slope = module.detrend(x)
    assert float(torch.max(torch.abs(slope))) > 0.05  # there was a trend to remove
    assert float(torch.max(torch.abs(module.slope(residual)))) < 1e-12
    # J-12d: updated reference, not a relaxed tolerance. Under the pre-fix
    # centre origin (t - (S-1)/2), the residual's mean equalled the window's
    # own mean exactly, because that origin is where `ols_slope_weights` also
    # centres, so the two coincided. Under the fixed origin (t - (S-1), the
    # window's LAST point -- see DampedTrend.py docstring point 1), the level
    # that survives detrending is the fitted line's value at t = S-1, not at
    # the window mean, and the two differ by exactly `slope * (S-1)/2`. This
    # is not a side effect to tolerate: it is the fix (see
    # test_phi_one_oracle_reconstructs_a_perfectly_linear_window_exactly),
    # restated here as the residual-level invariant this test checks.
    expected_level_shift = slope.squeeze(1) * (SEQ_LEN - 1) / 2.0
    actual_shift = residual.mean(dim=1) - x.mean(dim=1)
    assert torch.max(torch.abs(actual_shift - expected_level_shift)) < 1e-9


# --------------------------------------------------------------------------
# 3. The acceptance gate: phi -> 0 projects no trend.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("phi", [0.0, 1e-8])
@pytest.mark.parametrize("use_revin", [1, 0])
def test_phi_to_zero_matches_arm_off_when_the_window_has_no_trend(phi, use_revin):
    """The literal comparison: arm on at phi -> 0 == arm off, to 1e-6.

    The fixed input is detrended first, so its fitted slope is zero and the
    arm has no trend to remove or to re-apply. Agreement then has to be
    exact, and any leak -- a sign error on the centred index, a mismatch
    between the statistics used to normalise and to de-normalise, a constant
    surviving in `sum_{k=1..h} phi^k` -- shows up here.
    """
    on, off = _pair(use_revin=use_revin, damped_phi=phi)
    x = _detrended(_fixed_input(trend_per_step=0.05), on.damped_trend)
    cycle_index = _cycle_index()

    with torch.no_grad():
        produced = on(x, cycle_index)
        reference = off(x, cycle_index)

    discrepancy = _max_abs_diff(produced, reference)
    assert discrepancy < TOLERANCE, "max |arm-on - arm-off| = {:.6e}".format(discrepancy)


@pytest.mark.parametrize("phi", [0.0, 1e-8])
@pytest.mark.parametrize("use_revin", [1, 0])
def test_phi_to_zero_matches_arm_off_on_the_detrended_window(phi, use_revin):
    """Same claim, with a real trend in play so the sum is actually exercised.

    At `phi -> 0` the arm must reduce to the published model applied to the
    detrended window and nothing more. An off-by-one in `sum_{k=1..h} phi^k`
    makes the sum tend to 1 instead of 0 and adds a whole slope unit at every
    horizon; a sign error flips it. Either fails here by roughly the size of
    the window's slope, which is order 0.05 -- far above 1e-6.
    """
    on, off = _pair(use_revin=use_revin, damped_phi=phi)
    x = _fixed_input(trend_per_step=0.05)
    cycle_index = _cycle_index()

    with torch.no_grad():
        produced = on(x, cycle_index)
        reference = off(_detrended(x, on.damped_trend), cycle_index)

    discrepancy = _max_abs_diff(produced, reference)
    assert discrepancy < TOLERANCE, "max |arm-on - arm-off| = {:.6e}".format(discrepancy)


@pytest.mark.parametrize("use_revin", [1, 0])
def test_phi_one_adds_exactly_the_extrapolated_line(use_revin):
    """phi = 1: the difference from phi -> 0 is `slope * h`, exactly."""
    on_one, _ = _pair(use_revin=use_revin, damped_phi=1.0)
    on_zero, _ = _pair(use_revin=use_revin, damped_phi=0.0)
    x = _fixed_input(trend_per_step=0.05)
    cycle_index = _cycle_index()

    with torch.no_grad():
        damped = on_one(x, cycle_index)
        undamped = on_zero(x, cycle_index)

    slope = on_one.damped_trend.slope(x)
    h = torch.arange(1, PRED_LEN + 1, dtype=torch.float64).view(1, -1, 1)
    expected = slope * h

    discrepancy = _max_abs_diff(damped - undamped, expected)
    assert discrepancy < TOLERANCE, "max |added trend - slope*h| = {:.6e}".format(discrepancy)


# --------------------------------------------------------------------------
# 4. J-12d: phi = 1 must recover plain linear extrapolation EXACTLY, on a
#    window that is perfectly linear, with an oracle forecaster that returns
#    the (exactly flat) detrended level. Sec 3 Arm A states "phi = 1 recovers
#    plain linear extrapolation" as a specification; this is that
#    specification made testable. See STAGE2_WORKPLAN_2026-08-09.md / the
#    J-12d dispatch for the derivation of why the pre-fix origin fails this
#    by exactly `slope * (seq_len - 1) / 2`.
# --------------------------------------------------------------------------


def test_phi_one_oracle_reconstructs_a_perfectly_linear_window_exactly():
    """PM arithmetic: x(t) = 3.0 + 0.02*(t - 47.5), seq_len = pred_len = 96.

    The window is exactly linear, so the OLS slope recovered is exactly the
    generating slope `b` and `detrend()` leaves an exactly flat residual --
    whatever origin `detrend` uses, because a perfectly linear signal minus
    its own fitted line (about any origin on that same line) is a constant.
    An oracle forecaster is simulated by feeding that flat residual straight
    to `retrend()` (skipping normalisation and the network entirely, since
    both are the identity on an already-constant input in the noiseless
    case). At phi = 1, `retrend` must then reproduce the line's true
    continuation for h = 1..pred_len, to 1e-9.

    This is the test the dispatch requires be seen to fail before the fix:
    with the centre origin `t - (S-1)/2`, the level carried through is the
    line's value at the window CENTRE, but `retrend` projects forward from
    the window END (t = S-1, damped Holt's own convention) -- so every
    reconstructed step is short by exactly `b * (S-1)/2`, a *constant*, not a
    drift (the signature of a misplaced origin, not a wrong Sigma
    convention).
    """
    seq_len = pred_len = 96
    b = 0.02
    a = 3.0  # value at the window centre, t = 47.5 = (seq_len - 1) / 2

    module = DampedTrendInstanceNorm(seq_len, pred_len, phi=1.0)

    t = torch.arange(seq_len, dtype=torch.float64)
    centre = (seq_len - 1) / 2.0
    x = (a + b * (t - centre)).view(1, -1, 1)  # (1, 96, 1), exactly linear

    residual, slope = module.detrend(x)
    assert float(torch.max(torch.abs(slope - b))) < 1e-12, (
        "OLS slope on an exactly linear window must recover b exactly"
    )
    # Flat residual: the "oracle forecaster" just repeats it for every
    # forecast step -- this stands in for a network that is the identity on
    # an already-constant input.
    flat_level = residual[:, :1, :]  # (1, 1, 1)
    oracle_forecast = flat_level.expand(1, pred_len, 1)

    reconstruction = module.retrend(oracle_forecast, slope)

    h = torch.arange(1, pred_len + 1, dtype=torch.float64).view(1, -1, 1)
    true_continuation = a + b * ((seq_len - 1 + h) - centre)

    diff = reconstruction - true_continuation
    max_abs_error = float(torch.max(torch.abs(diff)))
    first_step_error = float(diff[0, 0, 0])
    last_step_error = float(diff[0, -1, 0])

    expected_pre_fix_error = -b * (seq_len - 1) / 2.0  # -0.95 for this b

    print(
        "test_phi_one_oracle_reconstructs_a_perfectly_linear_window_exactly: "
        "max_abs_error={:.9f} first_step_error={:.9f} last_step_error={:.9f} "
        "expected_pre_fix_error(b*(S-1)/2)={:.9f}".format(
            max_abs_error, first_step_error, last_step_error, expected_pre_fix_error
        )
    )

    assert max_abs_error < 1e-9, (
        "phi=1 oracle reconstruction of a perfectly linear window must match "
        "the true continuation to 1e-9; max abs error = {:.9e} "
        "(first_step={:.9e}, last_step={:.9e}). Pre-fix (centre origin) this "
        "is expected to be a CONSTANT error of b*(S-1)/2 = {:.9e} at every "
        "step.".format(max_abs_error, first_step_error, last_step_error,
                        expected_pre_fix_error)
    )


@pytest.mark.parametrize("phi", [0.8, 0.9, 0.95])
def test_intermediate_phi_adds_the_damped_line(phi):
    """For the pre-registered candidates, the added term is slope * sum phi^k."""
    on_phi, _ = _pair(damped_phi=phi)
    on_zero, _ = _pair(damped_phi=0.0)
    x = _fixed_input(trend_per_step=0.05)
    cycle_index = _cycle_index()

    with torch.no_grad():
        damped = on_phi(x, cycle_index)
        undamped = on_zero(x, cycle_index)

    slope = on_phi.damped_trend.slope(x)
    expected = slope * damped_trend_sum(PRED_LEN, phi).view(1, -1, 1)

    discrepancy = _max_abs_diff(damped - undamped, expected)
    assert discrepancy < TOLERANCE, "max |added trend - slope*sum| = {:.6e}".format(discrepancy)
