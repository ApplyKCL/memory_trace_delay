"""Unit tests for result metadata and multi-seed aggregation."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

module_path = Path(__file__).parents[1] / "scripts" / "analyze_delay_results.py"
spec = importlib.util.spec_from_file_location("analyze_delay_results", module_path)
assert spec is not None and spec.loader is not None
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_training_seed_prefers_explicit_metadata() -> None:
    result = {
        "training_seed": 7,
        "checkpoint": "/logs/run_seed42/model.pt",
    }
    assert module.infer_training_seed(result) == 7


def test_training_seed_falls_back_to_checkpoint_name() -> None:
    result = {"checkpoint": "/logs/run_seed42/model.pt"}
    assert module.infer_training_seed(result) == 42


def test_make_row_rejects_filename_metadata_disagreement() -> None:
    result = {
        "test_delay_steps": 3,
        "evaluation_seed": 123,
    }
    with pytest.raises(ValueError, match="Filename test delay"):
        module.make_row(result, "trace08", filename_test_delay=2, filename_eval_seed=123)


def test_new_multi_seed_filename_is_parsed() -> None:
    parsed = module.parse_trace_filename(Path("trace08_train0to4_trainseed43_test6_evalseed124.json"))
    assert parsed == ("trace08", 6, 124, 43)


def test_fixed_delay_stack_filenames_are_parsed() -> None:
    assert module.parse_stack_filename(Path("stack5_train0_trainseed42_test2_evalseed123.json")) == (
        "stack5_train0",
        2,
        123,
        42,
    )
    assert module.parse_stack_filename(Path("stack5_train2_trainseed44_test6_evalseed123.json")) == (
        "stack5_train2",
        6,
        123,
        44,
    )


def test_fixed_delay_trace_filenames_are_parsed() -> None:
    assert module.parse_trace_filename(Path("trace08_train0_trainseed42_test2_evalseed123.json")) == (
        "trace08_train0",
        2,
        123,
        42,
    )
    assert module.parse_trace_filename(Path("trace08_train2_trainseed44_test6_evalseed123.json")) == (
        "trace08_train2",
        6,
        123,
        44,
    )


def test_legacy_filename_remains_supported() -> None:
    assert module.parse_mlp_filename(Path("train0_test2_seed123.json")) == (
        "train0",
        2,
        123,
        None,
    )


def test_aggregation_averages_evaluation_seeds_before_training_seeds() -> None:
    rows = [
        {"method_key": "trace08", "test_delay": 2, "training_seed": 1, "mean_return": 10.0},
        {"method_key": "trace08", "test_delay": 2, "training_seed": 1, "mean_return": 14.0},
        {"method_key": "trace08", "test_delay": 2, "training_seed": 2, "mean_return": 20.0},
        {"method_key": "trace08", "test_delay": 2, "training_seed": 2, "mean_return": 24.0},
    ]

    aggregated = module.aggregate_metric(rows, "mean_return")

    assert len(aggregated) == 1
    assert aggregated[0]["mean"] == 17.0
    assert aggregated[0]["std_across_training_seeds"] == pytest.approx(5.0 * 2**0.5)
    assert aggregated[0]["num_training_seeds"] == 2
