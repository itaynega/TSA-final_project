"""NumPy mirror of the Arm A forward-pass algebra.

torch cannot be installed in this container (PyPI is refused) so the pytest
gate could not be executed here. This script reproduces the *exact* algebra of
TQNet/models/TQNet.py's forward pass with a deterministic non-linear stand-in
for the network body, and measures the same discrepancies the pytest gate
asserts. The equivalence claims are properties of the normalise / de-normalise
/ detrend / re-trend wrapper and do not depend on what the network is, only on
it being a deterministic function of its input -- which is why a stand-in is
sufficient to check the algebra.
"""

import numpy as np

SEQ_LEN, PRED_LEN, ENC_IN, CYCLE, BATCH = 96, 96, 7, 24, 4


# --- the module under test, transcribed from TQNet/layers/DampedTrend.py ----

def damped_trend_sum(pred_len, phi):
    h = np.arange(1, pred_len + 1, dtype=np.float64)
    if abs(1.0 - phi) < 1e-12:
        return h.copy()
    return phi * (1.0 - phi ** h) / (1.0 - phi)


def ols_slope_weights(seq_len):
    t = np.arange(seq_len, dtype=np.float64)
    centred = t - (seq_len - 1) / 2.0
    return centred / np.sum(centred * centred)


CENTRED = np.arange(SEQ_LEN, dtype=np.float64) - (SEQ_LEN - 1) / 2.0
W = ols_slope_weights(SEQ_LEN)


def slope_of(x):
    return np.einsum("s,bsc->bc", W, x)[:, None, :]


def detrend(x):
    s = slope_of(x)
    return x - s * CENTRED.reshape(1, -1, 1), s


# --- a deterministic stand-in for the network body -------------------------

RNG = np.random.default_rng(20260809)
A = RNG.standard_normal((SEQ_LEN, PRED_LEN)) / np.sqrt(SEQ_LEN)
B = RNG.standard_normal((SEQ_LEN, SEQ_LEN)) / np.sqrt(SEQ_LEN)


def network(x_norm):
    """(b, s, c) -> (b, pred_len, c). Non-linear, deterministic, order-fixed."""
    xi = np.transpose(x_norm, (0, 2, 1))            # b, c, s
    mixed = np.tanh(xi @ B) + xi                    # a channel/time mix
    out = mixed @ A                                 # b, c, pred_len
    return np.transpose(out, (0, 2, 1))             # b, pred_len, c


def forward(x, use_damped_trend, phi, use_revin=True):
    if use_damped_trend:
        x, trend_slope = detrend(x)

    if use_revin:
        seq_mean = np.mean(x, axis=1, keepdims=True)
        seq_var = np.var(x, axis=1, keepdims=True) + 1e-5
        x = (x - seq_mean) / np.sqrt(seq_var)

    output = network(x)

    if use_revin:
        output = output * np.sqrt(seq_var) + seq_mean

    if use_damped_trend:
        output = output + trend_slope * damped_trend_sum(PRED_LEN, phi).reshape(1, -1, 1)

    return output


# --- fixtures, matching tests/test_damped_trend.py -------------------------

def fixed_input(trend_per_step=0.0, seed=11):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((BATCH, SEQ_LEN, ENC_IN))
    t = np.arange(SEQ_LEN, dtype=np.float64).reshape(1, -1, 1)
    x = x + 2.0 * np.sin(2.0 * np.pi * t / CYCLE)
    if trend_per_step:
        per_channel = np.arange(1, ENC_IN + 1, dtype=np.float64).reshape(1, 1, -1)
        x = x + trend_per_step * t * per_channel
    return x


def report(label, value, tol=1e-6):
    print("{:<66s} {:.6e}  {}".format(label, value, "PASS" if value < tol else "FAIL"))
    return value < tol


ok = True

print("=" * 96)
print("sum_{k=1..h} phi^k  --  closed form vs naive loop")
print("=" * 96)
for phi in (0.0, 1e-8, 0.5, 0.8, 0.9, 0.95, 1.0):
    for pl in (1, 96, 720):
        closed = damped_trend_sum(pl, phi)
        naive, running = [], 0.0
        for h in range(1, pl + 1):
            running += phi ** h
            naive.append(running)
        d = float(np.max(np.abs(closed - np.array(naive))))
        ok &= d < 1e-12
        if pl == 720:
            print("  phi={:<8g} h=720  max|closed-naive| = {:.3e}   S(1)={:.6e} (must equal phi)"
                  .format(phi, d, closed[0]))
        assert abs(closed[0] - phi) < 1e-15, "off-by-one at h=1"

print()
print("=" * 96)
print("slope closed form vs numpy.polyfit")
print("=" * 96)
x_tr = fixed_input(trend_per_step=0.037)
t = np.arange(SEQ_LEN, dtype=np.float64)
ref = np.array([[np.polyfit(t, x_tr[b, :, c], 1)[0] for c in range(ENC_IN)] for b in range(BATCH)])
ok &= report("max |closed-form slope - polyfit slope|",
             float(np.max(np.abs(slope_of(x_tr)[:, 0, :] - ref))), 1e-10)
ok &= report("sum_t w_t (must be 0)", abs(float(np.sum(W))), 1e-15)
r, _ = detrend(x_tr)
ok &= report("residual slope after detrending", float(np.max(np.abs(slope_of(r)))), 1e-12)
ok &= report("window mean preserved by detrending",
             float(np.max(np.abs(r.mean(axis=1) - x_tr.mean(axis=1)))), 1e-12)

print()
print("=" * 96)
print("THE GATE: phi -> 0 with the arm on vs the arm off")
print("=" * 96)
for use_revin in (True, False):
    for phi in (0.0, 1e-8):
        x0 = detrend(fixed_input(trend_per_step=0.05))[0]      # zero-slope window
        d = float(np.max(np.abs(forward(x0, True, phi, use_revin)
                                - forward(x0, False, phi, use_revin))))
        ok &= report("literal   revin={:<5} phi={:<8g} zero-trend window".format(str(use_revin), phi), d)

        xt = fixed_input(trend_per_step=0.05)                  # strongly trended
        d = float(np.max(np.abs(forward(xt, True, phi, use_revin)
                                - forward(detrend(xt)[0], False, phi, use_revin))))
        ok &= report("detrended revin={:<5} phi={:<8g} trended window".format(str(use_revin), phi), d)

print()
print("=" * 96)
print("What the gate catches -- the same run with sum_{k=0..h-1} phi^k (off by one)")
print("=" * 96)


def forward_offbyone(x, phi):
    x, s = detrend(x)
    seq_mean = np.mean(x, axis=1, keepdims=True)
    seq_var = np.var(x, axis=1, keepdims=True) + 1e-5
    out = network((x - seq_mean) / np.sqrt(seq_var)) * np.sqrt(seq_var) + seq_mean
    wrong = np.cumsum(phi ** np.arange(0, PRED_LEN))          # sum_{k=0..h-1}
    return out + s * wrong.reshape(1, -1, 1)


xt = fixed_input(trend_per_step=0.05)
d = float(np.max(np.abs(forward_offbyone(xt, 0.0) - forward(detrend(xt)[0], False, 0.0))))
print("  off-by-one variant, phi=0, trended window: discrepancy = {:.6e}  ({})"
      .format(d, "caught" if d > 1e-6 else "NOT CAUGHT"))
ok &= d > 1e-6
d = float(np.max(np.abs(forward_offbyone(detrend(xt)[0], 0.0)
                        - forward(detrend(xt)[0], False, 0.0))))
print("  off-by-one variant, phi=0, zero-trend window: discrepancy = {:.6e}  ({})"
      .format(d, "caught" if d > 1e-6 else "NOT caught -- this is why both tests exist"))

print()
print("=" * 96)
print("phi = 1 is plain linear extrapolation; intermediate phi is the damped line")
print("=" * 96)
xt = fixed_input(trend_per_step=0.05)
s = slope_of(xt)
h = np.arange(1, PRED_LEN + 1, dtype=np.float64).reshape(1, -1, 1)
ok &= report("max |(phi=1) - (phi=0) - slope*h|",
             float(np.max(np.abs(forward(xt, True, 1.0) - forward(xt, True, 0.0) - s * h))))
for phi in (0.8, 0.9, 0.95):
    ok &= report("max |(phi={}) - (phi=0) - slope*sum phi^k|".format(phi),
                 float(np.max(np.abs(forward(xt, True, phi) - forward(xt, True, 0.0)
                                     - s * damped_trend_sum(PRED_LEN, phi).reshape(1, -1, 1)))))

print()
print("=" * 96)
print("For the record: arm on at phi -> 0 vs arm off on a TRENDED window")
print("(section 3's detrending is unconditional, so these differ by construction)")
print("=" * 96)
xt = fixed_input(trend_per_step=0.05)
print("  max |arm-on(phi=0, x) - arm-off(x)| = {:.6e}".format(
    float(np.max(np.abs(forward(xt, True, 0.0) - forward(xt, False, 0.0))))))

print()
print("ALL CHECKS PASS" if ok else "SOME CHECKS FAILED")
