"""Unit tests for delay-robust periodic-checkpoint selection."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

module_path = Path(__file__).parents[1] / "scripts" / "select_best_checkpoint.py"
spec = importlib.util.spec_from_file_location("select_best_checkpoint", module_path)
assert spec is not None and spec.loader is not None
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_checkpoint_discovery_uses_newest_run_and_iteration_filter(tmp_path: Path) -> None:
    old_run = tmp_path / "2026-01-01_00-00-00_trace_seed42"
    new_run = tmp_path / "2026-01-02_00-00-00_trace_seed42"
    old_run.mkdir()
    new_run.mkdir()
    (old_run / "model_100.pt").touch()
    (new_run / "model_100.pt").touch()
    selected = new_run / "model_200.pt"
    selected.touch()
    pattern = tmp_path / "*_trace_seed42" / "model_*.pt"

    checkpoints = module.discover_periodic_checkpoints(pattern, iterations=[200])

    assert checkpoints == [selected.resolve()]


def test_checkpoint_discovery_can_keep_only_recent_checkpoints(tmp_path: Path) -> None:
    run = tmp_path / "2026-01-02_00-00-00_trace_seed42"
    run.mkdir()
    checkpoints = []
    for iteration in (100, 200, 300):
        checkpoint = run / f"model_{iteration}.pt"
        checkpoint.touch()
        checkpoints.append(checkpoint.resolve())

    discovered = module.discover_periodic_checkpoints(tmp_path / "*_trace_seed42" / "model_*.pt", last_n=2)

    assert discovered == checkpoints[-2:]


def test_checkpoint_summary_averages_validation_seeds_per_delay(tmp_path: Path) -> None:
    checkpoint = tmp_path / "model_100.pt"
    checkpoint.touch()
    results = [
        {"test_delay_steps": 0, "mean_return": 10.0, "failure_rate": 0.0, "evaluation_seed": 901},
        {"test_delay_steps": 0, "mean_return": 14.0, "failure_rate": 0.2, "evaluation_seed": 902},
        {"test_delay_steps": 2, "mean_return": 4.0, "failure_rate": 0.4, "evaluation_seed": 901},
        {"test_delay_steps": 2, "mean_return": 8.0, "failure_rate": 0.6, "evaluation_seed": 902},
    ]

    summary = module.summarize_checkpoint_results("trace08", 42, checkpoint, results)

    assert summary["iteration"] == 100
    assert summary["delay_auc"] == pytest.approx(9.0)
    assert summary["worst_return"] == pytest.approx(6.0)
    assert summary["mean_failure_rate"] == pytest.approx(0.3)
    assert summary["num_validation_seeds"] == 2


def test_checkpoint_selection_uses_documented_tie_breakers() -> None:
    summaries = [
        {"delay_auc": 8.0, "worst_return": 4.0, "mean_failure_rate": 0.1, "iteration": 100},
        {"delay_auc": 8.0, "worst_return": 5.0, "mean_failure_rate": 0.2, "iteration": 200},
        {"delay_auc": 7.0, "worst_return": 9.0, "mean_failure_rate": 0.0, "iteration": 300},
    ]

    assert module.select_best_checkpoint(summaries) is summaries[1]


def test_normalized_auc_uses_delay_spacing() -> None:
    assert module.normalized_auc([(0, 10.0), (2, 6.0), (6, 2.0)]) == pytest.approx(16.0 / 3.0)
