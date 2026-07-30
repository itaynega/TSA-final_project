"""Tests for common.results.

The rule this module exists to enforce: every number that reaches the report is
traceable to a file. A number quoted in a chat message does not get printed.
"""

import json

import numpy as np
import pytest

from common import results


Y_TRUE = np.array([3.0, -0.5, 2.0, 7.0])
Y_PRED = np.array([2.5, 0.0, 2.0, 8.0])


def _record(tmp_path, **overrides):
    kwargs = dict(
        results_dir=tmp_path,
        arm="reconstruction",
        model="TQNet",
        seed=2024,
        seq_len=96,
        pred_len=96,
        split_hash="abc123",
        y_true=Y_TRUE,
        y_pred=Y_PRED,
    )
    kwargs.update(overrides)
    return results.record_run(**kwargs)


# --------------------------------------------------------------------------
# Recording a run.
# --------------------------------------------------------------------------


def test_record_run_writes_one_json_file(tmp_path):
    record = _record(tmp_path)
    written = list((tmp_path / "runs").glob("*.json"))
    assert len(written) == 1
    assert json.loads(written[0].read_text())["run_id"] == record["run_id"]


def test_the_recorded_metrics_are_the_ones_metrics_py_computes(tmp_path):
    from common import metrics

    record = _record(tmp_path)
    assert record["metrics"] == metrics.all_metrics(Y_TRUE, Y_PRED)


def test_the_record_carries_everything_needed_to_reproduce_the_number(tmp_path):
    record = _record(tmp_path)
    for field in (
        "run_id",
        "arm",
        "model",
        "seed",
        "seq_len",
        "pred_len",
        "split_hash",
        "n_windows",
        "timestamp",
        "metrics",
    ):
        assert field in record, "missing provenance field: {}".format(field)


def test_the_number_of_windows_is_recorded_from_the_data_not_asserted(tmp_path):
    record = _record(tmp_path, y_true=np.zeros((17, 96, 7)), y_pred=np.zeros((17, 96, 7)))
    assert record["n_windows"] == 17


def test_two_runs_do_not_overwrite_each_other(tmp_path):
    _record(tmp_path, seed=2024)
    _record(tmp_path, seed=2025)
    assert len(list((tmp_path / "runs").glob("*.json"))) == 2


def test_an_unknown_arm_is_rejected(tmp_path):
    """The three-way table has fixed columns; a typo must not create a fourth."""
    with pytest.raises(ValueError, match="arm"):
        _record(tmp_path, arm="reconstrution")


def test_a_missing_split_hash_is_rejected(tmp_path):
    """C2 is pass/fail. A run that cannot name its split cannot be compared."""
    with pytest.raises(ValueError, match="split_hash"):
        _record(tmp_path, split_hash="")


# --------------------------------------------------------------------------
# The C2 check itself.
# --------------------------------------------------------------------------


def test_assert_split_hash_passes_when_the_hashes_agree():
    results.assert_split_hash("abc123", "abc123")


def test_assert_split_hash_raises_when_they_differ():
    with pytest.raises(AssertionError, match="split"):
        results.assert_split_hash("abc123", "def456")


def test_assert_split_hash_raises_on_an_empty_hash():
    """An unset hash comparing equal to another unset hash is the failure mode."""
    with pytest.raises(AssertionError, match="split"):
        results.assert_split_hash("", "")


# --------------------------------------------------------------------------
# Reading them back, which is what builds the report table.
# --------------------------------------------------------------------------


def test_load_runs_returns_every_recorded_run(tmp_path):
    _record(tmp_path, seed=2024)
    _record(tmp_path, seed=2025)
    loaded = results.load_runs(tmp_path)
    assert sorted(r["seed"] for r in loaded) == [2024, 2025]


def test_load_runs_on_an_empty_directory_returns_nothing(tmp_path):
    assert results.load_runs(tmp_path) == []


def test_loaded_runs_are_ordered_by_when_they_were_recorded(tmp_path):
    first = _record(tmp_path, seed=1)
    second = _record(tmp_path, seed=2)
    loaded = results.load_runs(tmp_path)
    assert [r["run_id"] for r in loaded] == [first["run_id"], second["run_id"]]
