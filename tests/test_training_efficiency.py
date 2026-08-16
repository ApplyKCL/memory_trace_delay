"""Unit tests for TensorBoard training-efficiency aggregation."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

module_path = Path(__file__).parents[1] / "scripts" / "analyze_training_efficiency.py"
spec = importlib.util.spec_from_file_location("analyze_training_efficiency", module_path)
assert spec is not None and spec.loader is not None
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_scalar_summary_filters_fps_warmup_and_sums_wall_time() -> None:
    scalars = {
        "fps": [(0, 100.0), (1, 200.0), (2, 400.0)],
        "collection_time": [(0, 1.0), (1, 1.5), (2, 2.0)],
        "learning_time": [(0, 0.5), (1, 0.5), (2, 1.0)],
        "reward": [(0, 2.0), (1, 5.0), (2, 4.0)],
        "validation_time": [(1, 3.0), (2, 4.0)],
    }

    summary = module.summarize_scalar_series(scalars, warmup_iterations=1)

    assert summary["first_iteration"] == 1
    assert summary["last_iteration"] == 2
    assert summary["iterations_measured"] == 2
    assert summary["mean_fps"] == pytest.approx(300.0)
    assert summary["mean_iteration_time_seconds"] == pytest.approx(13.0 / 6.0)
    assert summary["total_training_time_seconds"] == pytest.approx(6.5)
    assert summary["total_validation_time_seconds"] == pytest.approx(7.0)
    assert summary["total_end_to_end_time_seconds"] == pytest.approx(13.5)
    assert summary["final_mean_reward"] == pytest.approx(4.0)
    assert summary["best_mean_reward"] == pytest.approx(5.0)


def test_scalar_summary_rejects_empty_post_warmup_series() -> None:
    with pytest.raises(ValueError, match="No Perf/total_fps"):
        module.summarize_scalar_series({"fps": [(0, 100.0)]}, warmup_iterations=1)


def test_latest_runs_use_union_of_training_and_evaluation_seeds(tmp_path: Path) -> None:
    config = {
        "version": 1,
        "common": {"training": {"seeds": [42]}, "evaluation": {"training_seeds": [42, 43]}},
        "methods": [
            {
                "key": "trace08",
                "experiment_name": "trace_experiment",
                "training": {"run_name": "trace_seed{seed}"},
            }
        ],
    }
    experiment_root = tmp_path / "trace_experiment"
    old_run = experiment_root / "2026-01-01_00-00-00_trace_seed42"
    new_run = experiment_root / "2026-01-02_00-00-00_trace_seed42"
    seed43_run = experiment_root / "2026-01-02_00-00-00_trace_seed43"
    for run in (old_run, new_run, seed43_run):
        run.mkdir(parents=True)

    discovered = module.discover_latest_runs(config, tmp_path, ["trace08"])

    assert [(item["training_seed"], item["run_dir"]) for item in discovered] == [
        (42, new_run.resolve()),
        (43, seed43_run.resolve()),
    ]


def test_method_metric_reports_between_seed_variation() -> None:
    rows = [
        {"method_key": "stack5", "mean_fps": 100.0},
        {"method_key": "stack5", "mean_fps": 120.0},
    ]

    aggregated = module.aggregate_method_metric(rows, "mean_fps")

    assert aggregated[0]["mean"] == pytest.approx(110.0)
    assert aggregated[0]["std"] == pytest.approx(10.0 * 2**0.5)
    assert aggregated[0]["num_training_seeds"] == 2
