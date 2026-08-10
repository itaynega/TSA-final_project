"""Tests for the helper tools.

Two things are worth testing here and neither is arithmetic.

The first is `tools/paper_reference.py`, which is a hand transcription of numbers out of
a PDF. Nothing computes those, so nothing would catch a typo -- and a typo there would
propagate silently into every comparison table and reproduction verdict in the report.
The tests below check the transcription against relationships that must hold if it is
right: the full-precision figures must round to the printed ones, the seed study must
agree with the main table at the shared seed, and the paper's own claims about which
model wins must be true of the numbers as transcribed.

The second is the run-directory name parser in `tools/collect_results.py`. It is the
seam between a shell script's string formatting and our records, so a mismatch there
attributes results to the wrong arm or the wrong variant.
"""

import numpy as np
import pytest

from tools import collect_results, paper_reference

HORIZONS = (96, 192, 336, 720)


# --------------------------------------------------------------------------
# The transcription. Checked against relationships, not against itself.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("pred_len", HORIZONS)
def test_full_precision_figures_round_to_the_printed_ones(pred_len):
    """`result.txt` and Table 5 must describe the same run."""
    exact = paper_reference.AUTHORS_RESULT_TXT[pred_len]
    printed_mse, printed_mae = paper_reference.PAPER_TABLE_5[pred_len]["TQNet"]

    assert round(exact["mse"], 3) == printed_mse
    assert round(exact["mae"], 3) == printed_mae


def test_target_cell_agrees_with_the_horizon_96_entry():
    """The target cell is a convenience copy and must not drift from its source."""
    entry = paper_reference.AUTHORS_RESULT_TXT[96]
    assert paper_reference.TARGET_CELL["mse"] == entry["mse"]
    assert paper_reference.TARGET_CELL["mae"] == entry["mae"]
    assert paper_reference.TARGET_CELL["pred_len"] == 96
    assert paper_reference.TARGET_CELL["seed"] == 2024


@pytest.mark.parametrize("pred_len", HORIZONS)
def test_seed_study_matches_the_main_table_at_seed_2024(pred_len):
    """Table 9's seed-2024 column is the same run as Table 5's TQNet row."""
    from_seed_study = paper_reference.SEED_STUDY[pred_len]["seeds"][2024]
    from_main_table = paper_reference.PAPER_TABLE_5[pred_len]["TQNet"]
    assert from_seed_study == from_main_table


@pytest.mark.parametrize("pred_len", HORIZONS)
def test_seed_study_mean_is_consistent_with_its_own_three_seeds(pred_len):
    """The printed mean must be the mean of the printed seeds, to rounding."""
    entry = paper_reference.SEED_STUDY[pred_len]
    for position, metric in enumerate(("mse", "mae")):
        values = [pair[position] for pair in entry["seeds"].values()]
        assert abs(np.mean(values) - entry["mean"][position]) <= 0.0006, (
            "{} at H={}: seeds average to {:.4f} but the mean is printed as {}".format(
                metric, pred_len, np.mean(values), entry["mean"][position]
            )
        )


@pytest.mark.parametrize("pred_len", HORIZONS)
def test_every_model_appears_at_every_horizon(pred_len):
    """A model missing from one horizon would silently drop out of a table."""
    reference = set(paper_reference.PAPER_TABLE_5[96])
    assert set(paper_reference.PAPER_TABLE_5[pred_len]) == reference


def test_tqnet_wins_at_96_and_192_and_loses_at_336_and_720():
    """The paper's ETTh1 story, as a property of the transcribed numbers.

    Stated in `docs/01`: TQNet wins the two short horizons on ETTh1 and loses the two
    long ones to TimeXer. If the transcription were wrong this would be the first thing
    to break, and it is a claim the report makes in prose.
    """
    for pred_len in (96, 192):
        name, best = paper_reference.best_baseline(pred_len, "mse")
        assert paper_reference.PAPER_TABLE_5[pred_len]["TQNet"][0] < best, (
            "TQNet should win at H={} but {} scores {}".format(pred_len, name, best)
        )
    for pred_len in (336, 720):
        name, best = paper_reference.best_baseline(pred_len, "mse")
        assert paper_reference.PAPER_TABLE_5[pred_len]["TQNet"][0] > best
        assert name == "TimeXer"


def test_margin_at_the_target_cell_is_four_times_the_seed_noise():
    """The number that governs how ambitious Stage 2 can be."""
    _, best = paper_reference.best_baseline(96, "mse")
    margin = best - paper_reference.PAPER_TABLE_5[96]["TQNet"][0]
    sigma = paper_reference.seed_std(96, "mse")

    assert margin == pytest.approx(0.004, abs=1e-9)
    assert sigma == 0.001
    assert margin / sigma == pytest.approx(4.0, rel=1e-6)


def test_dataset_facts_record_the_row_count_disagreement():
    """The paper says 14,400; the file has 17,420. Both numbers must be kept."""
    facts = paper_reference.DATASET_FACTS
    assert facts["timesteps_claimed"] == 14400
    assert facts["timesteps_in_csv"] == 17420
    assert facts["timesteps_in_csv"] - facts["timesteps_claimed"] == 3020


def test_target_reference_refuses_an_unknown_horizon():
    """Silently returning None would put an empty column in a report table."""
    with pytest.raises(KeyError, match="no authors' reference"):
        paper_reference.target_reference(48)


def test_hyperparameters_name_their_source():
    """Every hyperparameter must be attributable, or it cannot be reported as pinned."""
    assert len(paper_reference.HYPERPARAMETERS) >= 15
    for name, _value, source in paper_reference.HYPERPARAMETERS:
        assert source, "hyperparameter {!r} has no source".format(name)


# --------------------------------------------------------------------------
# The run-directory name parser.
# --------------------------------------------------------------------------


def test_parses_the_published_setting():
    parsed = collect_results.parse_setting(
        "ETTh1_96_96_TQNet_ETTh1_ftM_sl96_pl96_cycle24_seed2024"
    )
    assert parsed == {
        "model_id": "ETTh1_96_96",
        "model": "TQNet",
        "data": "ETTh1",
        "features": "M",
        "seq_len": 96,
        "pred_len": 96,
        "cycle": 24,
        "seed": 2024,
        "use_tq": 1,
        "channel_aggre": 1,
        "use_damped_trend": False,
        "damped_phi": None,
    }
    assert collect_results.variant_label(parsed) == "published"


@pytest.mark.parametrize("suffix,use_tq,channel_aggre,label", [
    ("_tq0ca1", 0, 1, "no-TQ (self-attention)"),
    ("_tq0ca0", 0, 0, "pure MLP"),
    ("_tq1ca0", 1, 0, "TQ without channel attention"),
])
def test_parses_ablation_suffixes(suffix, use_tq, channel_aggre, label):
    """The suffix is what keeps an ablation from being mistaken for the reconstruction."""
    parsed = collect_results.parse_setting(
        "ETTh1_96_96_TQNet_ETTh1_ftM_sl96_pl96_cycle24_seed2024" + suffix
    )
    assert parsed["use_tq"] == use_tq
    assert parsed["channel_aggre"] == channel_aggre
    assert collect_results.variant_label(parsed) == label


@pytest.mark.parametrize("suffix,phi,label", [
    ("_dphi0.8", 0.8, "damped trend (phi=0.8)"),
    ("_dphi1", 1.0, "damped trend (phi=1)"),
])
def test_parses_damped_trend_suffix(suffix, phi, label):
    """Arm A leaves use_tq and channel_aggre at 1, so only the tag distinguishes it
    from the published model -- and being mislabelled 'published' would make it
    eligible for the reconstruction column in tools/make_report.py."""
    parsed = collect_results.parse_setting(
        "ETTh1_96_96_TQNet_ETTh1_ftM_sl96_pl96_cycle24_seed2024" + suffix
    )
    assert parsed["use_damped_trend"] is True
    assert parsed["damped_phi"] == phi
    assert parsed["use_tq"] == 1 and parsed["channel_aggre"] == 1
    assert collect_results.variant_label(parsed) == label


def test_published_setting_reports_no_damped_trend():
    parsed = collect_results.parse_setting(
        "ETTh1_96_96_TQNet_ETTh1_ftM_sl96_pl96_cycle24_seed2024"
    )
    assert parsed["use_damped_trend"] is False
    assert parsed["damped_phi"] is None
    assert collect_results.variant_label(parsed) == "published"


def test_parses_other_horizons_and_seeds():
    parsed = collect_results.parse_setting(
        "ETTh1_96_720_TQNet_ETTh1_ftM_sl96_pl720_cycle24_seed2026"
    )
    assert parsed["pred_len"] == 720
    assert parsed["seed"] == 2026


def test_parses_a_longer_cycle_and_other_feature_modes():
    """Electricity uses W=168; MS predicts a single target channel."""
    parsed = collect_results.parse_setting(
        "electricity_96_96_TQNet_custom_ftMS_sl96_pl96_cycle168_seed2024"
    )
    assert parsed["cycle"] == 168
    assert parsed["features"] == "MS"
    assert parsed["data"] == "custom"


@pytest.mark.parametrize("name", [
    "not-a-setting",
    "ETTh1_96_96_TQNet_ETTh1_ftM_sl96_pl96_cycle24",          # no seed
    "ETTh1_96_96_TQNet_ETTh1_ftM_sl96_pl96_seed2024",          # no cycle
    "ETTh1_96_96_TQNet_ETTh1_ftM_sl96_pl96_cycle24_seed2024_x",  # unknown suffix
    "",
])
def test_rejects_names_it_cannot_parse(name):
    """Returning a partial parse would attribute a run to the wrong configuration."""
    assert collect_results.parse_setting(name) is None
