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
