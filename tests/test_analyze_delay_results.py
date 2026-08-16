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


@pytest.mark.parametrize(
    ("filename", "expected"),
    [
        ("stack2_train0to4_trainseed42_test1_evalseed125.json", ("stack2", 1, 125, 42)),
        ("stack8_train0to4_trainseed44_test5_evalseed124.json", ("stack8", 5, 124, 44)),
    ],
)
def test_stack_ablation_filenames_are_parsed(filename: str, expected: tuple) -> None:
    assert module.parse_stack_filename(Path(filename)) == expected


@pytest.mark.parametrize(
    ("filename", "expected"),
    [
        ("trace05_train0to4_trainseed42_test1_evalseed125.json", ("trace05", 1, 125, 42)),
        ("trace0875_train0to4_trainseed44_test5_evalseed124.json", ("trace0875", 5, 124, 44)),
    ],
)
def test_trace_ablation_filenames_are_parsed(filename: str, expected: tuple) -> None:
    assert module.parse_trace_filename(Path(filename)) == expected


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


def test_policy_history_length_uses_metadata_then_method_name() -> None:
    assert module.infer_policy_history_length({"policy_history_length": 2}, "stack8") == 2
    assert module.infer_policy_history_length({}, "stack8") == 8
    assert module.infer_policy_history_length({}, "trace08") == 1


def test_normalized_delay_auc_uses_trapezoidal_rule() -> None:
    assert module._normalized_auc([(0, 10.0), (2, 6.0)]) == pytest.approx(8.0)


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


def test_return_retention_is_normalized_within_each_training_seed() -> None:
    rows = [
        {"method_key": "train0to4", "test_delay": 0, "training_seed": 1, "mean_return": 10.0},
        {"method_key": "train0to4", "test_delay": 0, "training_seed": 1, "mean_return": 14.0},
        {"method_key": "train0to4", "test_delay": 2, "training_seed": 1, "mean_return": 6.0},
        {"method_key": "train0to4", "test_delay": 2, "training_seed": 1, "mean_return": 10.0},
        {"method_key": "train0to4", "test_delay": 0, "training_seed": 2, "mean_return": 20.0},
        {"method_key": "train0to4", "test_delay": 2, "training_seed": 2, "mean_return": 12.0},
    ]

    aggregated = module.aggregate_return_retention(rows, ["train0to4"])
    delay_zero = next(point for point in aggregated if point["test_delay"] == 0)
    delay_two = next(point for point in aggregated if point["test_delay"] == 2)

    assert delay_zero["mean"] == pytest.approx(100.0)
    assert delay_two["mean"] == pytest.approx((100.0 * 8.0 / 12.0 + 60.0) / 2.0)
    assert delay_two["num_training_seeds"] == 2


def test_return_improvement_is_paired_by_training_seed() -> None:
    rows = [
        {"method_key": "train0to4", "test_delay": 2, "training_seed": 1, "mean_return": 10.0},
        {"method_key": "train0to4", "test_delay": 2, "training_seed": 1, "mean_return": 14.0},
        {"method_key": "stack5", "test_delay": 2, "training_seed": 1, "mean_return": 20.0},
        {"method_key": "stack5", "test_delay": 2, "training_seed": 1, "mean_return": 22.0},
        {"method_key": "train0to4", "test_delay": 2, "training_seed": 2, "mean_return": 20.0},
        {"method_key": "stack5", "test_delay": 2, "training_seed": 2, "mean_return": 25.0},
        {"method_key": "stack5", "test_delay": 2, "training_seed": 3, "mean_return": 99.0},
    ]

    aggregated = module.aggregate_return_improvement(rows, ["stack5"], baseline_method="train0to4")

    assert len(aggregated) == 1
    assert aggregated[0]["mean"] == pytest.approx(7.0)
    assert aggregated[0]["std_across_training_seeds"] == pytest.approx(2.0 * 2**0.5)
    assert aggregated[0]["num_training_seeds"] == 2


def test_engineering_summary_joins_robustness_and_benchmark_cost() -> None:
    evaluation_rows = [
        {
            "method_key": "trace08",
            "test_delay": 0,
            "training_seed": 42,
            "mean_return": 10.0,
            "failure_rate": 0.0,
        },
        {
            "method_key": "trace08",
            "test_delay": 2,
            "training_seed": 42,
            "mean_return": 6.0,
            "failure_rate": 0.2,
        },
    ]
    benchmark_records = [
        {
            "method_key": "trace08",
            "backend": "torchscript",
            "device": "cpu",
            "parameter_count": 200,
            "model_size_bytes": 1000,
            "end_to_end_latency": {"mean_ms": 0.2, "p95_ms": 0.3},
        }
    ]

    summaries = module.aggregate_engineering_summary(evaluation_rows, benchmark_records, ["trace08"])

    assert len(summaries) == 1
    assert summaries[0]["delay_auc"] == pytest.approx(8.0)
    assert summaries[0]["parameter_count"] == pytest.approx(200.0)
    assert summaries[0]["end_to_end_p95_ms"] == pytest.approx(0.3)


def test_pareto_frontier_removes_higher_cost_lower_performance_points() -> None:
    rows = [
        {"method_key": "mlp", "cost": 1.0, "score": 5.0},
        {"method_key": "dominated", "cost": 2.0, "score": 4.0},
        {"method_key": "trace", "cost": 3.0, "score": 8.0},
    ]

    frontier = module.pareto_frontier(rows, cost_key="cost", performance_key="score")

    assert [row["method_key"] for row in frontier] == ["mlp", "trace"]


def test_benchmark_loader_filters_backend_and_device(tmp_path: Path) -> None:
    import json

    valid = {
        "method_key": "trace08",
        "backend": "torchscript",
        "device": "cpu",
        "parameter_count": 200,
        "model_size_bytes": 1000,
        "end_to_end_latency": {"mean_ms": 0.2, "p95_ms": 0.3},
    }
    (tmp_path / "valid.json").write_text(json.dumps(valid), encoding="utf-8")
    (tmp_path / "ignored.json").write_text(json.dumps({**valid, "device": "cuda:0"}), encoding="utf-8")

    records = module.load_benchmark_results(
        tmp_path,
        backend="torchscript",
        device="cpu",
        method_keys=["trace08"],
    )

    assert len(records) == 1
    assert records[0]["method_key"] == "trace08"
