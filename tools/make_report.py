"""Assemble the report's tables and figures from recorded runs.

Nothing here computes a forecast. It reads the run records written by
`tools/collect_results.py` and `tools/run_baseline.py`, reads the window-level arrays
those runs left behind, and renders them. Keeping presentation separate from
computation is what stops a chart from quietly disagreeing with the table above it.

Two things this produces that the report needs and that a printed MSE cannot give:

  * **A verdict, not a comparison.** "Our MSE is 0.3710 and theirs is 0.3712" is not
    a finding until it is measured against something. The yardstick used here is the
    paper's *own* three-seed standard deviation (Table 9, p. 18), which at horizon 96
    is 0.001 MSE. A gap smaller than that is inside the noise the authors themselves
    measured, and calling it a mismatch would be over-reading the data. A gap of ten
    times that is a bug in our setup, not a fact about the paper.

  * **Where the error actually lives.** The headline number is one mean over
    2,785 x 96 x 7 = 1.87M predictions. The figures break that mean down by forecast
    step, by channel, and by window, which is where the interesting structure is --
    and which is what makes the difference between the model and the baseline legible.

Usage, from the repository root:

    python3 tools/make_report.py
    python3 tools/make_report.py --no-figures       # tables only
"""

import argparse
import json
import os
import sys
from collections import OrderedDict

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from common import data as data_mod  # noqa: E402
from common import metrics as metrics_mod  # noqa: E402
from common import results as results_mod  # noqa: E402
from common import split as split_mod  # noqa: E402
from tools import paper_reference  # noqa: E402

RESULTS_DIR = os.path.join(REPO_ROOT, "results")
TQNET_RESULTS = os.path.join(REPO_ROOT, "TQNet", "results")
REPORT_DIR = os.path.join(REPO_ROOT, "report")
FIGURE_DIR = os.path.join(REPORT_DIR, "figures")

# How many multiples of the paper's own seed standard deviation we are willing to call
# "reproduced". 1x = inside the noise they measured; up to 3x = ordinary hardware and
# library drift; beyond that, suspect our setup first.
WITHIN_NOISE = 1.0
ORDINARY_DRIFT = 3.0


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_records():
    """Recorded runs, grouped by arm, newest last."""
    records = results_mod.load_runs(RESULTS_DIR)
    grouped = OrderedDict((arm, []) for arm in results_mod.ARMS)
    for record in records:
        grouped.setdefault(record["arm"], []).append(record)
    return grouped


def variant_of(record):
    """Which architecture variant a record came from. Baselines have none."""
    return (record.get("extra") or {}).get("variant", "published")


def pick(records, pred_len, seed=None, variant="published"):
    """The most recent matching record at this horizon.

    Filtering on `variant` matters more than it looks. Ablation runs are recorded
    under the same arm as the reconstruction, so an unfiltered "most recent" would
    happily return the pure-MLP run and label it as the reconstruction in the headline
    table. Baseline records carry no variant, so `variant=None` disables the filter.
    """
    candidates = [r for r in records if r.get("pred_len") == pred_len]
    if seed is not None:
        candidates = [r for r in candidates if r.get("seed") == seed]
    if variant is not None:
        candidates = [r for r in candidates if variant_of(r) == variant]
    return candidates[-1] if candidates else None


def load_arrays(record):
    """The (true, pred) arrays behind a reconstruction/improved record.

    Returns None when the record has no `setting` (the baseline, which is recomputed
    rather than stored) or when the run directory has been cleaned away.
    """
    setting = (record.get("extra") or {}).get("setting")
    if not setting:
        return None
    directory = os.path.join(TQNET_RESULTS, setting)
    pred_path = os.path.join(directory, "pred.npy")
    true_path = os.path.join(directory, "true.npy")
    if not (os.path.exists(pred_path) and os.path.exists(true_path)):
        return None
    return np.load(true_path), np.load(pred_path)


def baseline_arrays(seq_len, pred_len, period=24):
    """Recompute the baseline's forecast so the figures can show it alongside."""
    windows = data_mod.make_windows("test", seq_len=seq_len, pred_len=pred_len)
    return windows, data_mod.seasonal_naive(windows, period=period)


# ---------------------------------------------------------------------------
# Verdict
# ---------------------------------------------------------------------------


def verdict(ours, published, seed_sigma):
    """Classify a reproduction gap against the paper's own run-to-run spread."""
    delta = ours - published
    if seed_sigma in (None, 0):
        return delta, None, "no seed study published at this horizon; gap uncalibrated"

    multiples = abs(delta) / seed_sigma
    if multiples <= WITHIN_NOISE:
        label = ("reproduced: the gap is {:.2f}x the paper's own seed sigma, i.e. inside "
                 "the run-to-run noise they measured".format(multiples))
    elif multiples <= ORDINARY_DRIFT:
        label = ("close: {:.2f}x seed sigma. Consistent with ordinary hardware and "
                 "library drift; report the deviation rather than the discrepancy"
                 .format(multiples))
    else:
        label = ("NOT reproduced: {:.2f}x seed sigma. Look for the fault in our setup "
                 "before concluding anything about the paper".format(multiples))
    return delta, multiples, label


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------


def fmt(value, places=4):
    return "--" if value is None else "{:.{}f}".format(value, places)


def table_headline(grouped, pred_len, seq_len):
    """The three-way table requirement F5 asks for: paper / reconstruction / improved."""
    # Pinned to seed 2024 so the headline row is the cell the paper reports, not
    # whichever seed happened to be run last.
    reconstruction = pick(grouped.get("reconstruction", []), pred_len, seed=2024)
    improved = pick(grouped.get("improved", []), pred_len, variant=None)
    baseline = pick(grouped.get("baseline", []), pred_len, variant=None)
    published = paper_reference.AUTHORS_RESULT_TXT.get(pred_len)

    lines = []
    add = lines.append

    add("### Target cell: ETTh1, multivariate, L={} -> H={}, seed 2024".format(seq_len, pred_len))
    add("")
    add("All figures are on **z-scored** data, over the **{:,} test windows** the split "
        "defines. MSE and MAE are the paper's metrics; RMSE and MdAE are the course's.".format(
            split_mod.n_windows(seq_len, pred_len)["test"]))
    add("")
    add("| Arm | MSE | MAE | RMSE | MdAE | Source |")
    add("|---|---|---|---|---|---|")

    if published:
        add("| Paper (as printed, Table 5) | {} | {} | -- | -- | TQnet.pdf p. 15 |".format(
            fmt(published["mse"], 3), fmt(published["mae"], 3)))
        add("| Paper (authors' own run) | {} | {} | -- | -- | `result_authors_reference.txt` |".format(
            fmt(published["mse"], 6), fmt(published["mae"], 6)))

    for label, record in (("Seasonal-naive baseline (period 24)", baseline),
                          ("**Our reconstruction**", reconstruction),
                          ("Our improvement", improved)):
        if record is None:
            continue
        m = record["metrics"]
        add("| {} | {} | {} | {} | {} | `{}` |".format(
            label, fmt(m["MSE"], 6), fmt(m["MAE"], 6), fmt(m["RMSE"], 6), fmt(m["MdAE"], 6),
            record["run_id"][:40]))

    add("")

    if reconstruction and published:
        sigma_mse = paper_reference.seed_std(pred_len, "mse")
        sigma_mae = paper_reference.seed_std(pred_len, "mae")
        d_mse, mult_mse, label_mse = verdict(reconstruction["metrics"]["MSE"], published["mse"], sigma_mse)
        d_mae, mult_mae, _ = verdict(reconstruction["metrics"]["MAE"], published["mae"], sigma_mae)

        add("#### Reproduction gap")
        add("")
        add("| Metric | Authors' run | Ours | Difference | Relative | Paper's seed sigma | Gap / sigma |")
        add("|---|---|---|---|---|---|---|")
        add("| MSE | {} | {} | {:+.6f} | {:+.3f}% | {} | {} |".format(
            fmt(published["mse"], 6), fmt(reconstruction["metrics"]["MSE"], 6), d_mse,
            100.0 * d_mse / published["mse"], fmt(sigma_mse, 3),
            "{:.2f}x".format(mult_mse) if mult_mse is not None else "--"))
        add("| MAE | {} | {} | {:+.6f} | {:+.3f}% | {} | {} |".format(
            fmt(published["mae"], 6), fmt(reconstruction["metrics"]["MAE"], 6), d_mae,
            100.0 * d_mae / published["mae"], fmt(sigma_mae, 3),
            "{:.2f}x".format(mult_mae) if mult_mae is not None else "--"))
        add("")
        add("**Verdict on MSE:** {}.".format(label_mse))
        add("")
        if sigma_mae == 0:
            add("The paper reports MAE sigma as 0.000 at this horizon, which is 0.0005 "
                "rounded down rather than literal determinism, so the MAE ratio above is "
                "not meaningful and only the MSE verdict should be read.")
            add("")

    if reconstruction and baseline:
        gain_mse = 1.0 - reconstruction["metrics"]["MSE"] / baseline["metrics"]["MSE"]
        add("#### Against the baseline")
        add("")
        add("TQNet reduces MSE by **{:.1f}%** relative to a seasonal-naive forecast at the "
            "same period (24) on the same windows and the same scale. Both make the same "
            "daily-periodicity assumption, so this gap is what the network adds on top of "
            "knowing that ETTh1 repeats daily.".format(100.0 * gain_mse))
        add("")
        mae_ratio = reconstruction["metrics"]["MAE"] / reconstruction["metrics"]["MdAE"]
        add("Note also that MAE / MdAE = {:.2f} for the reconstruction. MdAE being well "
            "below MAE means the error distribution is right-skewed: most predictions are "
            "much better than the mean suggests, and the headline is carried by a minority "
            "of badly-missed windows.".format(mae_ratio))
        add("")

    return "\n".join(lines)


def table_context(pred_len):
    """The paper's own baseline column at this horizon, as context."""
    row = paper_reference.PAPER_TABLE_5.get(pred_len)
    if not row:
        return ""

    lines = []
    add = lines.append
    add("### The paper's comparison set at H={}".format(pred_len))
    add("")
    add("Transcribed from Table 5, p. 15. The caption states these baseline numbers were "
        "**copied** from TimeXer, iTransformer and CycleNet rather than re-run, so each "
        "inherits its own paper's setup. They are context for our result, not a "
        "like-for-like comparison.")
    add("")
    add("| Model | MSE | MAE |")
    add("|---|---|---|")
    for name, (mse, mae) in sorted(row.items(), key=lambda kv: kv[1][0]):
        mark = " **(the paper's method)**" if name == "TQNet" else ""
        add("| {}{} | {:.3f} | {:.3f} |".format(name, mark, mse, mae))
    add("")

    best_name, best_mse = paper_reference.best_baseline(pred_len, "mse")
    sigma = paper_reference.seed_std(pred_len, "mse")
    margin = row["TQNet"][0] - best_mse
    add("TQNet's margin over the strongest baseline ({}) is **{:+.3f} MSE**, against a "
        "run-to-run sigma of **{:.3f}**.".format(best_name, margin, sigma))
    if sigma and abs(margin) <= 2 * sigma:
        add("")
        add("That margin is only {:.1f}x the seed noise. Any improvement aimed at MSE on "
            "this cell is aiming at a target barely above the measurement error, which is "
            "worth stating before Stage 2 picks a direction.".format(abs(margin) / sigma))
    add("")
    return "\n".join(lines)


def table_seed_spread(grouped, pred_len):
    """Our own seed spread, when more than one seed has been run."""
    records = [
        r for r in grouped.get("reconstruction", [])
        if r.get("pred_len") == pred_len and variant_of(r) == "published"
    ]
    by_seed = OrderedDict()
    for record in records:
        by_seed[record.get("seed")] = record
    if len(by_seed) < 2:
        return ""

    lines = []
    add = lines.append
    add("### Our seed spread at H={}".format(pred_len))
    add("")
    add("| Seed | MSE | MAE |")
    add("|---|---|---|")
    for seed, record in sorted(by_seed.items(), key=lambda kv: (kv[0] is None, kv[0])):
        add("| {} | {:.6f} | {:.6f} |".format(seed, record["metrics"]["MSE"], record["metrics"]["MAE"]))

    mses = np.array([r["metrics"]["MSE"] for r in by_seed.values()])
    maes = np.array([r["metrics"]["MAE"] for r in by_seed.values()])
    add("| **mean** | {:.6f} | {:.6f} |".format(mses.mean(), maes.mean()))
    add("| **sd** | {:.6f} | {:.6f} |".format(mses.std(ddof=1), maes.std(ddof=1)))
    add("")
    published_sigma = paper_reference.seed_std(pred_len, "mse")
    add("The paper reports sigma = {:.3f} MSE at this horizon on their hardware. Ours is "
        "{:.6f}. This is the number any claimed improvement has to beat to be real."
        .format(published_sigma, mses.std(ddof=1)))
    add("")
    return "\n".join(lines)


def table_ablation(grouped, pred_len, seed=2024):
    """The ETTh1 ablation the paper never published, if those runs exist.

    Pinned to a single seed. The ablation runs use seed 2024, and comparing them
    against a different seed's baseline would attribute a seed difference to the
    mechanism being ablated -- which, given that the effect here is smaller than the
    seed spread, would invert the conclusion.
    """
    records = [
        r for r in grouped.get("reconstruction", [])
        if r.get("pred_len") == pred_len and r.get("seed") == seed
    ]
    by_variant = OrderedDict()
    for record in records:
        by_variant[(record.get("extra") or {}).get("variant", "published")] = record
    if len(by_variant) < 2:
        return ""

    published = by_variant.get("published")
    lines = []
    add = lines.append
    add("### Does the Temporal Query help on ETTh1? (unpublished ablation)")
    add("")
    add("The paper ablates TQNet only on Electricity, PEMS03 and PEMS04 -- all with more "
        "than 100 channels. ETTh1 has 7, so the channel attention map is 7x7 and there is "
        "no published evidence that TQ contributes anything at that width. These runs "
        "answer that. All are seed {}, with every other flag identical.".format(seed))
    add("")
    add("| Variant | Params | MSE | MAE | MSE vs published | vs our seed sd |")
    add("|---|---|---|---|---|---|")

    our_sigma = _our_seed_sd(grouped, pred_len)
    for variant, record in by_variant.items():
        params = (record.get("extra") or {}).get("n_params")
        delta_text, ratio_text = "--", "--"
        if published and record is not published:
            delta = record["metrics"]["MSE"] - published["metrics"]["MSE"]
            delta_text = "{:+.6f}".format(delta)
            if our_sigma:
                ratio_text = "{:.2f}x".format(abs(delta) / our_sigma)
        add("| {} | {} | {:.6f} | {:.6f} | {} | {} |".format(
            variant, "{:,}".format(params) if params else "--",
            record["metrics"]["MSE"], record["metrics"]["MAE"], delta_text, ratio_text))
    add("")

    if published and our_sigma:
        deltas = [
            abs(record["metrics"]["MSE"] - published["metrics"]["MSE"])
            for variant, record in by_variant.items() if record is not published
        ]
        if deltas and max(deltas) < our_sigma:
            add("**Every ablation lands within one seed standard deviation of the full "
                "model.** On this dataset, at 7 channels, neither the Temporal Query nor "
                "the channel-attention layer is measurable above run-to-run noise -- the "
                "MLP alone accounts for essentially all of the reported accuracy. That is "
                "not a refutation of the paper, whose claims are made on datasets with "
                "more than 100 channels; it is a limit on what ETTh1 can demonstrate, and "
                "the paper's own decision to ablate elsewhere is consistent with it.")
            add("")
    return "\n".join(lines)


def _our_seed_sd(grouped, pred_len, metric="MSE"):
    """Our own across-seed standard deviation of the published variant, or None."""
    by_seed = {}
    for record in grouped.get("reconstruction", []):
        if record.get("pred_len") != pred_len:
            continue
        if (record.get("extra") or {}).get("variant", "published") != "published":
            continue
        by_seed[record.get("seed")] = record["metrics"][metric]
    if len(by_seed) < 2:
        return None
    return float(np.std(np.array(list(by_seed.values())), ddof=1))


def table_environment():
    lines = []
    add = lines.append
    add("### Environment deviations from the paper")
    add("")
    add("The brief asks that differences between our numbers and the paper's be explained. "
        "These are the candidate explanations, recorded before the run rather than "
        "reached for afterwards.")
    add("")
    add("| Aspect | Paper | Ours |")
    add("|---|---|---|")

    env_path = os.path.join(RESULTS_DIR, "environment.json")
    ours = {}
    if os.path.exists(env_path):
        with open(env_path) as handle:
            ours = json.load(handle)

    add("| Device | RTX 4090, 24 GB | {} |".format(ours.get("accelerator", "see tools/check_env.py")))
    add("| Python | 3.8 (upstream README; now end-of-life) | {} |".format(ours.get("python", "?")))
    packages = ours.get("packages", {})
    add("| PyTorch | unstated | {} |".format(packages.get("torch", "?")))
    add("| numpy | unstated | {} |".format(packages.get("numpy", "?")))
    add("| pandas | unstated (code requires < 2.0) | {} |".format(packages.get("pandas", "?")))
    add("| Seed | 2024 | 2024 |")
    add("")
    add("PyTorch and cuDNN version differences change floating-point reduction order and "
        "kernel selection, which is the ordinary reason two runs of identical code differ "
        "in the fourth decimal. Running on CPU rather than an RTX 4090 changes it further. "
        "Neither changes the model, the data or the protocol.")
    add("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def _style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "figure.dpi": 130,
        "savefig.dpi": 130,
        "font.size": 9,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.autolayout": True,
    })
    return plt


OURS_C, PAPER_C, BASE_C, TRUTH_C = "#2b6cb0", "#718096", "#c05621", "#1a202c"


def fig_reproduction(plt, reconstruction, published, pred_len, path):
    """Ours against the authors' own run, with their seed noise drawn to scale."""
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.1))

    for ax, key, metric_name in ((axes[0], "mse", "MSE"), (axes[1], "mae", "MAE")):
        theirs = published[key]
        mine = reconstruction["metrics"][metric_name]
        sigma = paper_reference.seed_std(pred_len, key) or 0.0

        ax.bar([0], [theirs], width=0.55, color=PAPER_C, label="Authors' run")
        ax.bar([1], [mine], width=0.55, color=OURS_C, label="Ours")

        if sigma:
            ax.axhspan(theirs - sigma, theirs + sigma, color=PAPER_C, alpha=0.18, zorder=0)
            ax.annotate(
                "+/- 1 sigma across\nthe paper's 3 seeds",
                xy=(1.42, theirs), fontsize=7, color="#4a5568",
                va="center", ha="left", annotation_clip=False,
            )

        for x, value in ((0, theirs), (1, mine)):
            ax.text(x, value, "{:.4f}".format(value), ha="center", va="bottom", fontsize=8)

        low = min(theirs, mine)
        high = max(theirs, mine)
        pad = max(sigma * 3.5, (high - low) * 4, high * 0.012)
        ax.set_ylim(low - pad, high + pad)
        ax.set_xlim(-0.6, 2.1)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Paper", "Ours"])
        ax.set_title("{} (z-scored)".format(metric_name))

    axes[0].set_ylabel("error")
    fig.suptitle(
        "Reproduction of TQNet on ETTh1, L=96 -> H={}, seed 2024\n"
        "the shaded band is the paper's own run-to-run spread".format(pred_len),
        fontsize=9.5,
    )
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def fig_error_by_step(plt, trues, preds, base_true, base_pred, path):
    """MSE as a function of how far ahead we are forecasting."""
    steps = np.arange(1, trues.shape[1] + 1)
    model_mse = ((trues - preds) ** 2).mean(axis=(0, 2))
    base_mse = ((base_true - base_pred) ** 2).mean(axis=(0, 2))

    fig, ax = plt.subplots(figsize=(7.4, 3.3))
    ax.plot(steps, base_mse, color=BASE_C, lw=1.4, label="Seasonal-naive (period 24)")
    ax.plot(steps, model_mse, color=OURS_C, lw=1.8, label="TQNet (our reconstruction)")
    ax.axhline(model_mse.mean(), color=OURS_C, ls=":", lw=1,
               label="TQNet mean = {:.4f}".format(model_mse.mean()))

    for day in range(24, len(steps), 24):
        ax.axvline(day, color="#cbd5e0", lw=0.7, zorder=0)

    ax.set_xlabel("forecast step ahead (hours)")
    ax.set_ylabel("MSE at that step")
    ax.set_xlim(1, len(steps))
    ax.set_title("TQNet's error grows smoothly with horizon; the baseline's steps up at "
                 "each\n24-hour boundary (grey lines), because it re-uses the same day "
                 "again", fontsize=9.5)
    ax.legend(fontsize=8, loc="lower right")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def fig_per_channel(plt, trues, preds, base_true, base_pred, path):
    """Which of the seven variables the error comes from."""
    channels = list(data_mod.CHANNELS)
    model_mse = ((trues - preds) ** 2).mean(axis=(0, 1))
    base_mse = ((base_true - base_pred) ** 2).mean(axis=(0, 1))

    order = np.argsort(-model_mse)
    positions = np.arange(len(channels))
    width = 0.38

    fig, ax = plt.subplots(figsize=(7.4, 3.2))
    ax.bar(positions - width / 2, base_mse[order], width, color=BASE_C, label="Seasonal-naive")
    ax.bar(positions + width / 2, model_mse[order], width, color=OURS_C, label="TQNet")

    for pos, value in zip(positions + width / 2, model_mse[order]):
        ax.text(pos, value, "{:.3f}".format(value), ha="center", va="bottom", fontsize=7)

    ax.set_xticks(positions)
    ax.set_xticklabels([channels[i] for i in order])
    ax.set_ylabel("MSE (z-scored)")
    ax.set_title("Per-channel error. The headline MSE is one mean over all seven;\n"
                 "the spread across them is wide", fontsize=9.5)
    ax.legend(fontsize=8)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def fig_examples(plt, windows, trues, preds, base_pred, path, channel=-1, n=3):
    """A handful of actual forecasts, so the numbers have something behind them."""
    channel_name = data_mod.CHANNELS[channel]
    seq_len = windows.x.shape[1]
    pred_len = trues.shape[1]

    # Pick a median, a good and a bad window by this channel's MSE, so the panel is
    # representative rather than flattering.
    per_window = ((trues[:, :, channel] - preds[:, :, channel]) ** 2).mean(axis=1)
    ranked = np.argsort(per_window)
    chosen = [
        (ranked[len(ranked) // 20], "5th percentile (easy)"),
        (ranked[len(ranked) // 2], "median"),
        (ranked[-max(1, len(ranked) // 20)], "95th percentile (hard)"),
    ][:n]

    fig, axes = plt.subplots(len(chosen), 1, figsize=(7.4, 2.25 * len(chosen)), sharex=True)
    axes = np.atleast_1d(axes)

    history = np.arange(-seq_len, 0)
    future = np.arange(0, pred_len)

    for ax, (index, label) in zip(axes, chosen):
        ax.plot(history, windows.x[index, :, channel], color="#a0aec0", lw=1.1, label="input history")
        ax.plot(future, trues[index, :, channel], color=TRUTH_C, lw=1.6, label="actual")
        ax.plot(future, preds[index, :, channel], color=OURS_C, lw=1.5, label="TQNet")
        ax.plot(future, base_pred[index, :, channel], color=BASE_C, lw=1.1, ls="--",
                label="seasonal-naive")
        ax.axvline(0, color="#e53e3e", lw=0.9)
        ax.set_ylabel("{} (z)".format(channel_name))
        ax.set_title("window {} -- {} (MSE {:.3f})".format(index, label, per_window[index]),
                     fontsize=8.5, loc="left")

    axes[-1].set_xlabel("hours relative to the forecast origin (red line)")
    axes[0].legend(fontsize=7.5, ncol=4, loc="upper left")
    fig.suptitle("Individual forecasts on the {} channel, chosen by difficulty".format(channel_name),
                 fontsize=9.5)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def fig_error_distribution(plt, trues, preds, base_true, base_pred, path):
    """Where the model's advantage actually is: the tail, not the body.

    Worth drawing because the two summary numbers disagree about how much better
    TQNet is. It beats the baseline by 9% on MAE but 28% on MSE, and the only way that
    happens is if the two are similar for typical predictions and diverge on the worst
    ones. A plain CDF cannot show this -- both curves sit on top of each other over the
    range that holds 90% of the mass -- so the right panel plots the survival function
    on a log axis, where the tail is legible.
    """
    model_err = np.abs(trues - preds).ravel()
    base_err = np.abs(base_true - base_pred).ravel()

    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.2))

    upper = float(np.percentile(base_err, 99))
    bins = np.linspace(0, upper, 80)
    axes[0].hist(base_err, bins=bins, histtype="step", lw=1.5, color=BASE_C,
                 density=True, label="Seasonal-naive")
    axes[0].hist(model_err, bins=bins, histtype="stepfilled", lw=1.5, color=OURS_C,
                 alpha=0.35, density=True, label="TQNet")
    axes[0].axvline(model_err.mean(), color=OURS_C, ls="--", lw=1.2,
                    label="MAE = {:.3f}".format(model_err.mean()))
    axes[0].axvline(np.median(model_err), color=OURS_C, ls=":", lw=1.2,
                    label="MdAE = {:.3f}".format(np.median(model_err)))
    axes[0].set_xlabel("absolute error |e| (z-scored)")
    axes[0].set_ylabel("density")
    axes[0].set_title("Body of the distribution", fontsize=9.5)
    axes[0].legend(fontsize=7)

    # Survival function P(|e| > t) on a log axis. Subsampling every Nth element of the
    # unsorted array is a systematic sample of an exchangeable population, so it
    # estimates the same curve as all 1.87M points at a fraction of the cost.
    for values, colour, label in ((base_err, BASE_C, "Seasonal-naive"), (model_err, OURS_C, "TQNet")):
        sample = np.sort(values[:: max(1, len(values) // 40000)])
        survival = 1.0 - np.linspace(0, 1, len(sample), endpoint=False)
        axes[1].semilogy(sample, survival, color=colour, lw=1.7, label=label)

    axes[1].set_xlim(0, float(np.percentile(base_err, 99.99)))
    axes[1].set_ylim(1e-4, 1.05)
    axes[1].set_xlabel("threshold t")
    axes[1].set_ylabel("P(|e| > t), log scale")
    axes[1].set_title("Tail, where the two separate", fontsize=9.5)
    axes[1].legend(fontsize=8)

    ratio_mae = base_err.mean() / model_err.mean()
    ratio_mse = (base_err ** 2).mean() / (model_err ** 2).mean()
    fig.suptitle(
        "TQNet beats the baseline by {:.0f}% on MAE but {:.0f}% on MSE: the gain is in "
        "the tail".format(100 * (ratio_mae - 1), 100 * (ratio_mse - 1)),
        fontsize=9.5,
    )
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def fig_horizon_sweep(plt, grouped, path):
    """Ours against the paper across every horizon we have run. Skipped if only one."""
    records = grouped.get("reconstruction", [])
    by_horizon = OrderedDict()
    for record in records:
        if (record.get("extra") or {}).get("variant", "published") == "published":
            by_horizon[record["pred_len"]] = record
    horizons = sorted(h for h in by_horizon if h in paper_reference.AUTHORS_RESULT_TXT)
    if len(horizons) < 2:
        return False

    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.1))
    for ax, key, name in ((axes[0], "mse", "MSE"), (axes[1], "mae", "MAE")):
        theirs = [paper_reference.AUTHORS_RESULT_TXT[h][key] for h in horizons]
        mine = [by_horizon[h]["metrics"][name] for h in horizons]
        sigma = [paper_reference.seed_std(h, key) or 0.0 for h in horizons]

        ax.errorbar(horizons, theirs, yerr=sigma, color=PAPER_C, marker="s", lw=1.4,
                    capsize=3, label="Authors' run (+/- seed sigma)")
        ax.plot(horizons, mine, color=OURS_C, marker="o", lw=1.8, label="Ours")
        ax.set_xscale("log")
        ax.set_xticks(horizons)
        ax.set_xticklabels([str(h) for h in horizons])
        ax.set_xlabel("forecast horizon H")
        ax.set_title(name, fontsize=9.5)
    axes[0].set_ylabel("error (z-scored)")
    axes[0].legend(fontsize=7.5)
    fig.suptitle("Reproduction across horizons", fontsize=9.5)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return True


def build_figures(grouped, pred_len, seq_len):
    plt = _style()
    os.makedirs(FIGURE_DIR, exist_ok=True)

    reconstruction = pick(grouped.get("reconstruction", []), pred_len, seed=2024)
    published = paper_reference.AUTHORS_RESULT_TXT.get(pred_len)
    made = []

    if reconstruction and published:
        path = os.path.join(FIGURE_DIR, "fig1_reproduction.png")
        fig_reproduction(plt, reconstruction, published, pred_len, path)
        made.append(("fig1_reproduction.png",
                     "Our MSE and MAE against the authors' own run, with their "
                     "three-seed spread drawn to scale."))

    arrays = load_arrays(reconstruction) if reconstruction else None
    if arrays is not None:
        trues, preds = arrays
        windows, base_pred = baseline_arrays(seq_len, pred_len)
        base_true = windows.y

        if base_true.shape != trues.shape:
            print("  ! baseline windows are {} but the run is {}; skipping comparison "
                  "figures".format(base_true.shape, trues.shape))
        else:
            for name, description, fn in (
                ("fig2_error_by_horizon_step.png",
                 "MSE against how far ahead the forecast is, for TQNet and the baseline.",
                 lambda p: fig_error_by_step(plt, trues, preds, base_true, base_pred, p)),
                ("fig3_per_channel.png",
                 "Per-channel MSE, showing how uneven the headline mean is.",
                 lambda p: fig_per_channel(plt, trues, preds, base_true, base_pred, p)),
                ("fig4_forecast_examples.png",
                 "Three actual forecasts on the OT channel, at the 5th, 50th and 95th "
                 "percentile of difficulty.",
                 lambda p: fig_examples(plt, windows, trues, preds, base_pred, p)),
                ("fig5_error_distribution.png",
                 "Distribution and CDF of absolute error, explaining the MAE / MdAE gap.",
                 lambda p: fig_error_distribution(plt, trues, preds, base_true, base_pred, p)),
            ):
                fn(os.path.join(FIGURE_DIR, name))
                made.append((name, description))

    path = os.path.join(FIGURE_DIR, "fig6_horizon_sweep.png")
    if fig_horizon_sweep(plt, grouped, path):
        made.append(("fig6_horizon_sweep.png",
                     "Ours against the paper at every horizon that has been run."))

    return made


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------


def build(pred_len=96, seq_len=96, with_figures=True):
    grouped = load_records()
    if not any(grouped.values()):
        raise SystemExit(
            "No run records under {}/runs/.\n\n"
            "Produce some first:\n"
            "  python3 tools/run_baseline.py\n"
            "  bash repro/run_reconstruction.sh && python3 tools/collect_results.py"
            .format(os.path.relpath(RESULTS_DIR, REPO_ROOT))
        )

    figures = []
    if with_figures:
        figures = build_figures(grouped, pred_len, seq_len)

    sections = [
        "# Results -- TQNet reconstruction on ETTh1",
        "",
        "Generated by `tools/make_report.py` from the run records in `results/runs/`. "
        "Do not edit by hand: re-run the script instead, or the numbers here will drift "
        "away from the runs they claim to describe.",
        "",
        "Task type: **multivariate, supervised, deterministic point forecasting**. "
        "Seven channels in, seven channels out. Sampling frequency 1 hour, input window "
        "96 hours, output window {} hours, forecast origin advancing at stride 1 across "
        "the test split with a fixed model (rolling-origin evaluation).".format(pred_len),
        "",
        table_headline(grouped, pred_len, seq_len),
        table_seed_spread(grouped, pred_len),
        table_ablation(grouped, pred_len),
        table_context(pred_len),
        table_environment(),
    ]

    if figures:
        sections.append("### Figures")
        sections.append("")
        for name, description in figures:
            sections.append("**`figures/{}`** -- {}".format(name, description))
            sections.append("")
            sections.append("![{}](figures/{})".format(name, name))
            sections.append("")

    text = "\n".join(section for section in sections if section is not None)

    os.makedirs(REPORT_DIR, exist_ok=True)
    out_path = os.path.join(REPORT_DIR, "results.md")
    with open(out_path, "w") as handle:
        handle.write(text.rstrip() + "\n")

    return out_path, figures, grouped


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--pred-len", type=int, default=96)
    parser.add_argument("--seq-len", type=int, default=96)
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args(argv)

    out_path, figures, grouped = build(
        pred_len=args.pred_len, seq_len=args.seq_len, with_figures=not args.no_figures
    )

    counts = {arm: len(records) for arm, records in grouped.items() if records}
    print("run records : {}".format(counts))
    print("figures     : {}".format(len(figures)))
    for name, _ in figures:
        print("              report/figures/{}".format(name))
    print("tables      : {}".format(os.path.relpath(out_path, REPO_ROOT)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
