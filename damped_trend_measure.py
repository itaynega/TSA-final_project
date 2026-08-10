"""Print the discrepancies `tests/test_damped_trend.py` asserts on (J-11, Arm A).

The tests assert; they only print a number when they fail. The dispatch asks
for the measured value, so this reruns the same comparisons through the same
helpers and prints what they measured. Run from the repository root:

    python3 damped_trend_measure.py | tee damped_trend_torch_check.log

Reads nothing but a synthetic fixed input. No dataset, no split, no training,
no validation or test number is touched.
"""

import os
import platform
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "tests"))

import torch  # noqa: E402

import test_damped_trend as T  # noqa: E402

TOL = T.TOLERANCE


def line(label, value, tol=TOL):
    verdict = "PASS" if value < tol else "FAIL"
    print("{:<62s} {:.6e}   {}".format(label, value, verdict))
    return value < tol


print("python           :", sys.version.split()[0], sys.executable)
print("platform         :", platform.platform())
print("torch            :", torch.__version__)
print("tolerance        : {:g}".format(TOL))
print("dtype            : float64 (model.double()), eval mode")
print()

ok = True

print("=" * 84)
print("THE GATE -- arm on at phi -> 0 vs arm off")
print("=" * 84)
for use_revin in (1, 0):
    for phi in (0.0, 1e-8):
        on, off = T._pair(use_revin=use_revin, damped_phi=phi)
        ci = T._cycle_index()

        x0 = T._detrended(T._fixed_input(trend_per_step=0.05), on.damped_trend)
        with torch.no_grad():
            d = T._max_abs_diff(on(x0, ci), off(x0, ci))
        ok &= line("literal   revin={} phi={:<8g} zero-trend window".format(use_revin, phi), d)

        xt = T._fixed_input(trend_per_step=0.05)
        with torch.no_grad():
            d = T._max_abs_diff(on(xt, ci), off(T._detrended(xt, on.damped_trend), ci))
        ok &= line("detrended revin={} phi={:<8g} trended window".format(use_revin, phi), d)

print()
print("=" * 84)
print("phi = 1 is plain linear extrapolation; the candidates are the damped line")
print("=" * 84)
xt = T._fixed_input(trend_per_step=0.05)
ci = T._cycle_index()
on_zero, _ = T._pair(damped_phi=0.0)
with torch.no_grad():
    undamped = on_zero(xt, ci)
for phi in (0.8, 0.9, 0.95, 1.0):
    on_phi, _ = T._pair(damped_phi=phi)
    with torch.no_grad():
        damped = on_phi(xt, ci)
    slope = on_phi.damped_trend.slope(xt)
    expected = slope * T.damped_trend_sum(T.PRED_LEN, phi).view(1, -1, 1)
    ok &= line("max |(phi={:<4g}) - (phi=0) - slope*sum phi^k|".format(phi),
               T._max_abs_diff(damped - undamped, expected))

print()
print("=" * 84)
print("Arm off vs the published forward pass (bit-identical, not tolerance)")
print("=" * 84)
xt = T._fixed_input(trend_per_step=0.02)
for use_revin in (1, 0):
    for use_tq, ca in ((1, 1), (0, 0), (1, 0), (0, 1)):
        m = T._build(use_damped_trend=0, use_revin=use_revin, use_tq=use_tq, channel_aggre=ca)
        with torch.no_grad():
            a = m(xt, ci)
            b = T._published_forward(m, xt, ci)
        identical = torch.equal(a, b)
        ok &= identical
        print("revin={} use_tq={} channel_aggre={}  torch.equal = {}   max|diff| = {:.6e}"
              .format(use_revin, use_tq, ca, identical, T._max_abs_diff(a, b)))

on, off = T._pair()
print()
print("trainable parameters  arm on = {}   arm off = {}".format(
    sum(p.numel() for p in on.parameters()),
    sum(p.numel() for p in off.parameters())))
print("damped_trend state_dict entries: {} (must be 0 -- checkpoints unaffected)".format(
    len(on.damped_trend.state_dict())))

print()
print("=" * 84)
print("For the record -- arm on at phi -> 0 vs arm off on a TRENDED window.")
print("Section 3's detrending is unconditional, so these differ by construction;")
print("this is the number behind item 9 of the J-11 return, not a failure.")
print("=" * 84)
on, off = T._pair(damped_phi=0.0)
xt = T._fixed_input(trend_per_step=0.05)
with torch.no_grad():
    print("max |arm-on(phi=0, x) - arm-off(x)| = {:.6e}".format(
        T._max_abs_diff(on(xt, ci), off(xt, ci))))

print()
print("ALL CHECKS PASS" if ok else "SOME CHECKS FAILED")
sys.exit(0 if ok else 1)
