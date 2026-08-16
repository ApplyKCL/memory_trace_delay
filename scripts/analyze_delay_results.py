"""Aggregate MLP, Stack-PPO, and Trace-PPO delay-evaluation results."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from statistics import fmean, stdev

METHOD_LABELS = {
    "train0": "Fixed delay 0 MLP",
    "train2": "Fixed delay 2 MLP",
    "train0to4": "Random delay 0-4 MLP",
    "stack5_train0": "Fixed delay 0 Stack-5",
    "stack5_train2": "Fixed delay 2 Stack-5",
    "stack5": "Random delay 0-4 Stack-5",
    "trace08_train0": "Fixed delay 0 Trace (decay 0.8)",
    "trace08_train2": "Fixed delay 2 Trace (decay 0.8)",
    "trace08": "Random delay 0-4 Trace (decay 0.8)",
    "stack2": "Random delay 0-4 Stack-2",
    "stack8": "Random delay 0-4 Stack-8",
    "trace05": "Random delay 0-4 Trace (decay 0.5)",
    "trace0875": "Random delay 0-4 Trace (decay 0.875)",
}

CORE_RANDOM_METHODS = ["train0to4", "stack5", "trace08"]
CORE_METHOD_COLORS = {
    "train0to4": "tab:blue",
    "stack5": "tab:orange",
    "trace08": "tab:green",
}
TRAINING_CONDITION_GROUPS = [
    ("Train fixed d=0", ["train0", "stack5_train0", "trace08_train0"]),
    ("Train fixed d=2", ["train2", "stack5_train2", "trace08_train2"]),
    ("Train random d=0-4", CORE_RANDOM_METHODS),
]
ARCHITECTURE_GROUPS = [
    ("MLP", ["train0", "train2", "train0to4"]),
    ("Stack-5", ["stack5_train0", "stack5_train2", "stack5"]),
    ("Trace decay 0.8", ["trace08_train0", "trace08_train2", "trace08"]),
]
ABLATION_GROUPS = [
    ("Stack window", ["stack2", "stack5", "stack8"]),
    ("Trace decay", ["trace05", "trace08", "trace0875"]),
]

ARCHITECTURE_LABELS = {
    "train0": "MLP",
    "train2": "MLP",
    "train0to4": "MLP",
    "stack5_train0": "Stack-5",
    "stack5_train2": "Stack-5",
    "stack5": "Stack-5",
    "trace08_train0": "Trace",
    "trace08_train2": "Trace",
    "trace08": "Trace",
}
TRAINING_LABELS = {
    "train0": "Train d=0",
    "stack5_train0": "Train d=0",
    "trace08_train0": "Train d=0",
    "train2": "Train d=2",
    "stack5_train2": "Train d=2",
    "trace08_train2": "Train d=2",
    "train0to4": "Train random 0-4",
    "stack5": "Train random 0-4",
    "trace08": "Train random 0-4",
}
ABLATION_LABELS = {
    "stack2": "K=2",
    "stack5": "K=5",
    "stack8": "K=8",
    "trace05": "decay=0.5 (H≈2)",
    "trace08": "decay=0.8 (H≈5)",
    "trace0875": "decay=0.875 (H≈8)",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Aggregate action-delay evaluation results.")
    parser.add_argument("--mlp_dir", default="results/mlp", help="MLP evaluation JSON directory.")
    parser.add_argument("--stack_dir", default="results/stack", help="Stack-PPO evaluation JSON directory.")
    parser.add_argument("--trace_dir", default="results/trace", help="Trace-PPO evaluation JSON directory.")
    parser.add_argument("--output_csv", default="results/combined_summary.csv", help="Output summary CSV.")
    parser.add_argument("--figure_dir", default="results/figures", help="Output figure directory.")
    parser.add_argument("--benchmark_dir", default="results/benchmarks", help="Policy benchmark JSON directory.")
    parser.add_argument("--benchmark_backend", default="torchscript", help="Benchmark backend to compare.")
    parser.add_argument("--benchmark_device", default="cpu", help="Benchmark device to compare.")
    parser.add_argument("--engineering_csv", default="results/engineering_summary.csv")
    return parser


def parse_mlp_filename(path: Path) -> tuple[str, int, int, int | None]:
    match = re.match(
        r"^(train0to4|train0|train2)(?:_trainseed(\d+))?_test(\d+)_(?:seed|evalseed)(\d+)\.json$",
        path.name,
    )
    if match is None:
        raise ValueError(f"Unexpected MLP filename: {path.name}")
    training_seed = int(match.group(2)) if match.group(2) is not None else None
    return match.group(1), int(match.group(3)), int(match.group(4)), training_seed


def parse_stack_filename(path: Path) -> tuple[str, int, int, int | None]:
    match = re.match(
        r"^(stack(?:2|5|8)_train0to4|stack5_train0|stack5_train2)"
        r"(?:_trainseed(\d+))?_test(\d+)_(?:seed|evalseed)(\d+)\.json$",
        path.name,
    )
    if match is None:
        raise ValueError(f"Unexpected Stack filename: {path.name}")
    method_key = match.group(1).removesuffix("_train0to4")
    training_seed = int(match.group(2)) if match.group(2) is not None else None
    return method_key, int(match.group(3)), int(match.group(4)), training_seed


def parse_trace_filename(path: Path) -> tuple[str, int, int, int | None]:
    match = re.match(
        r"^((?:trace|trace05|trace08|trace0875)_train0to4|trace08_train0|trace08_train2)"
        r"(?:_trainseed(\d+))?_test(\d+)_(?:seed|evalseed)(\d+)\.json$",
        path.name,
    )
    if match is None:
        raise ValueError(f"Unexpected Trace filename: {path.name}")
    method_key = match.group(1).removesuffix("_train0to4")
    if method_key == "trace":
        method_key = "trace08"
    training_seed = int(match.group(2)) if match.group(2) is not None else None
    return method_key, int(match.group(3)), int(match.group(4)), training_seed


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def infer_training_seed(result: dict) -> int:
    training_seed = result.get("training_seed")
    if training_seed is not None:
        return int(training_seed)

    # Backward compatibility for results whose checkpoint run name contains
    # e.g. "seed42" but whose JSON lacks an explicit training_seed field.
    checkpoint = str(result.get("checkpoint", ""))
    match = re.search(r"seed(\d+)", checkpoint)
    return int(match.group(1)) if match is not None else -1


def infer_policy_history_length(result: dict, method_key: str) -> int:
    configured = result.get("policy_history_length")
    if configured is not None:
        return int(configured)
    match = re.match(r"stack(\d+)", method_key)
    return int(match.group(1)) if match is not None else 1


def make_row(
    result: dict,
    method_key: str,
    filename_test_delay: int,
    filename_eval_seed: int,
    filename_training_seed: int | None = None,
) -> dict:
    test_delay = int(result.get("test_delay_steps", filename_test_delay))
    evaluation_seed = int(result.get("evaluation_seed", result.get("seed", filename_eval_seed)))

    if test_delay != filename_test_delay:
        raise ValueError(f"Filename test delay {filename_test_delay} disagrees with JSON test delay {test_delay}.")
    if evaluation_seed != filename_eval_seed:
        raise ValueError(f"Filename evaluation seed {filename_eval_seed} disagrees with JSON seed {evaluation_seed}.")

    training_seed = infer_training_seed(result)
    if filename_training_seed is not None and training_seed != filename_training_seed:
        raise ValueError(
            f"Filename training seed {filename_training_seed} disagrees with JSON/checkpoint seed {training_seed}."
        )

    return {
        "method_key": method_key,
        "method": METHOD_LABELS[method_key],
        "test_delay": test_delay,
        "training_seed": training_seed,
        "evaluation_seed": evaluation_seed,
        "mean_return": result["mean_return"],
        "std_return_within_episodes": result["std_return"],
        "mean_episode_length": result["mean_episode_length"],
        "mean_xy_error": result["mean_pre_action_xy_velocity_error"],
        "mean_yaw_error": result["mean_pre_action_yaw_velocity_error"],
        "failure_rate": result["failure_rate"],
        "timeout_rate": result["timeout_rate"],
        "num_episodes": result["num_episodes"],
        "train_delay_min": result.get("train_delay_min"),
        "train_delay_max": result.get("train_delay_max"),
        "trace_enabled": bool(result.get("trace_enabled", method_key.startswith("trace"))),
        "trace_decay": result.get("trace_decay"),
        "policy_history_length": infer_policy_history_length(result, method_key),
        "checkpoint": result["checkpoint"],
    }


def load_results(directory: Path, filename_parser) -> list[dict]:
    rows = []
    if not directory.exists():
        return rows

    for path in sorted(directory.glob("*.json")):
        try:
            method_key, test_delay, evaluation_seed, training_seed = filename_parser(path)
        except ValueError:
            print(f"[SKIP] {path}")
            continue
        rows.append(make_row(read_json(path), method_key, test_delay, evaluation_seed, training_seed))
    return rows


def save_csv(rows: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "method_key",
        "method",
        "test_delay",
        "training_seed",
        "evaluation_seed",
        "mean_return",
        "std_return_within_episodes",
        "mean_episode_length",
        "mean_xy_error",
        "mean_yaw_error",
        "failure_rate",
        "timeout_rate",
        "num_episodes",
        "train_delay_min",
        "train_delay_max",
        "trace_enabled",
        "trace_decay",
        "policy_history_length",
        "checkpoint",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows({key: row[key] for key in fieldnames} for row in rows)


def aggregate_metric(rows: list[dict], metric: str) -> list[dict]:
    """Average evaluation seeds within a model, then aggregate training seeds."""
    per_model: dict[tuple[str, int, int], list[float]] = defaultdict(list)
    for row in rows:
        key = (row["method_key"], row["test_delay"], row["training_seed"])
        per_model[key].append(float(row[metric]))

    per_condition: dict[tuple[str, int], list[float]] = defaultdict(list)
    for (method_key, test_delay, _), values in per_model.items():
        per_condition[(method_key, test_delay)].append(fmean(values))

    aggregated = []
    for (method_key, test_delay), values in sorted(per_condition.items()):
        aggregated.append(
            {
                "method_key": method_key,
                "test_delay": test_delay,
                "mean": fmean(values),
                "std_across_training_seeds": stdev(values) if len(values) > 1 else 0.0,
                "num_training_seeds": len(values),
            }
        )
    return aggregated


def per_model_metric(rows: list[dict], metric: str) -> list[dict]:
    """Average evaluation seeds while retaining each trained model."""
    grouped: dict[tuple[str, int, int], list[float]] = defaultdict(list)
    for row in rows:
        key = (row["method_key"], row["test_delay"], row["training_seed"])
        grouped[key].append(float(row[metric]))
    return [
        {
            "method_key": method_key,
            "test_delay": test_delay,
            "training_seed": training_seed,
            "mean": fmean(values),
        }
        for (method_key, test_delay, training_seed), values in sorted(grouped.items())
    ]


def _aggregate_seed_values(values_by_condition: dict[tuple[str, int], list[float]]) -> list[dict]:
    aggregated = []
    for (method_key, test_delay), values in sorted(values_by_condition.items()):
        aggregated.append(
            {
                "method_key": method_key,
                "test_delay": test_delay,
                "mean": fmean(values),
                "std_across_training_seeds": stdev(values) if len(values) > 1 else 0.0,
                "num_training_seeds": len(values),
            }
        )
    return aggregated


def aggregate_return_retention(rows: list[dict], method_keys: list[str]) -> list[dict]:
    """Return percentage retained from delay zero, normalized within each trained model."""
    per_model = per_model_metric(rows, "mean_return")
    lookup = {
        (row["method_key"], row["training_seed"], row["test_delay"]): row["mean"]
        for row in per_model
        if row["method_key"] in method_keys
    }
    values_by_condition: dict[tuple[str, int], list[float]] = defaultdict(list)
    for (method_key, training_seed, test_delay), value in lookup.items():
        delay_zero_return = lookup.get((method_key, training_seed, 0))
        if delay_zero_return is None:
            continue
        if abs(delay_zero_return) < 1.0e-12:
            raise ValueError(
                f"Cannot compute return retention for {method_key} seed {training_seed}: d=0 return is zero."
            )
        values_by_condition[(method_key, test_delay)].append(100.0 * value / delay_zero_return)
    return _aggregate_seed_values(values_by_condition)


def aggregate_return_improvement(
    rows: list[dict],
    method_keys: list[str],
    baseline_method: str,
) -> list[dict]:
    """Return improvement over a baseline, paired by training seed and test delay."""
    per_model = per_model_metric(rows, "mean_return")
    baseline = {
        (row["training_seed"], row["test_delay"]): row["mean"]
        for row in per_model
        if row["method_key"] == baseline_method
    }
    values_by_condition: dict[tuple[str, int], list[float]] = defaultdict(list)
    for row in per_model:
        if row["method_key"] not in method_keys:
            continue
        paired_baseline = baseline.get((row["training_seed"], row["test_delay"]))
        if paired_baseline is None:
            continue
        values_by_condition[(row["method_key"], row["test_delay"])].append(row["mean"] - paired_baseline)
    return _aggregate_seed_values(values_by_condition)


def _points_for_method(aggregated: list[dict], method_key: str) -> list[dict]:
    return sorted(
        (point for point in aggregated if point["method_key"] == method_key),
        key=lambda point: point["test_delay"],
    )


def _plot_method(axis, points: list[dict], label: str, color: str | None = None) -> None:
    if not points:
        return
    axis.errorbar(
        [point["test_delay"] for point in points],
        [point["mean"] for point in points],
        yerr=[point["std_across_training_seeds"] for point in points],
        marker="o",
        linewidth=2,
        capsize=3,
        label=label,
        color=color,
    )


def plot_metric(rows: list[dict], metric: str, ylabel: str, output_path: Path) -> None:
    import matplotlib.pyplot as plt

    aggregated = aggregate_metric(rows, metric)
    figure, axis = plt.subplots(figsize=(13, 7))

    for method_key, label in METHOD_LABELS.items():
        _plot_method(axis, _points_for_method(aggregated, method_key), label)

    delays = sorted({row["test_delay"] for row in rows})
    axis.set_xlabel("Test action delay (steps)")
    axis.set_ylabel(ylabel)
    axis.set_xticks(delays)
    axis.grid(True, alpha=0.3)
    axis.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=8)
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def plot_grouped_metric(
    rows: list[dict],
    metric: str,
    ylabel: str,
    groups: list[tuple[str, list[str]]],
    labels: dict[str, str],
    output_path: Path,
) -> None:
    import matplotlib.pyplot as plt

    aggregated = aggregate_metric(rows, metric)
    delays = sorted({row["test_delay"] for row in rows})
    figure, axes = plt.subplots(1, len(groups), figsize=(15, 4.8), sharex=True, sharey=True)
    for axis, (title, method_keys) in zip(axes, groups, strict=True):
        for method_key in method_keys:
            _plot_method(axis, _points_for_method(aggregated, method_key), labels[method_key])
        axis.set_title(title)
        axis.set_xlabel("Test delay (steps)")
        axis.set_xticks(delays)
        axis.grid(True, alpha=0.3)
        axis.legend(fontsize=8)
    axes[0].set_ylabel(ylabel)
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def plot_random_training_metrics(rows: list[dict], output_path: Path) -> None:
    import matplotlib.pyplot as plt

    metrics = [
        ("mean_return", "Mean episode return"),
        ("mean_xy_error", "Mean XY error (m/s)"),
        ("failure_rate", "Failure rate"),
    ]
    delays = sorted({row["test_delay"] for row in rows})
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharex=True)
    for axis, (metric, ylabel) in zip(axes, metrics, strict=True):
        aggregated = aggregate_metric(rows, metric)
        for method_key in CORE_RANDOM_METHODS:
            _plot_method(
                axis,
                _points_for_method(aggregated, method_key),
                ARCHITECTURE_LABELS[method_key],
                CORE_METHOD_COLORS[method_key],
            )
        axis.set_xlabel("Test delay (steps)")
        axis.set_ylabel(ylabel)
        axis.set_xticks(delays)
        axis.grid(True, alpha=0.3)
        axis.legend(fontsize=8)
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def plot_return_retention(rows: list[dict], output_path: Path) -> None:
    import matplotlib.pyplot as plt

    aggregated = aggregate_return_retention(rows, CORE_RANDOM_METHODS)
    delays = sorted({point["test_delay"] for point in aggregated})
    figure, axis = plt.subplots(figsize=(8, 5))
    for method_key in CORE_RANDOM_METHODS:
        _plot_method(
            axis,
            _points_for_method(aggregated, method_key),
            ARCHITECTURE_LABELS[method_key],
            CORE_METHOD_COLORS[method_key],
        )
    axis.axhline(100.0, color="black", linestyle="--", linewidth=1, alpha=0.6)
    axis.set_xlabel("Test delay (steps)")
    axis.set_ylabel("Return retention (% of return at d=0)")
    axis.set_title("Robustness relative to each model's delay-zero performance")
    axis.set_xticks(delays)
    axis.grid(True, alpha=0.3)
    axis.legend()
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def plot_return_improvement(rows: list[dict], output_path: Path) -> None:
    import matplotlib.pyplot as plt

    compared_methods = ["stack5", "trace08"]
    aggregated = aggregate_return_improvement(rows, compared_methods, baseline_method="train0to4")
    delays = sorted({point["test_delay"] for point in aggregated})
    figure, axis = plt.subplots(figsize=(8, 5))
    for method_key in compared_methods:
        _plot_method(
            axis,
            _points_for_method(aggregated, method_key),
            f"{ARCHITECTURE_LABELS[method_key]} minus MLP",
            CORE_METHOD_COLORS[method_key],
        )
    axis.axhline(0.0, color="black", linestyle="--", linewidth=1, alpha=0.6)
    axis.set_xlabel("Test delay (steps)")
    axis.set_ylabel("Return improvement over MLP")
    axis.set_title("Benefit of memory under matched random-delay training")
    axis.set_xticks(delays)
    axis.grid(True, alpha=0.3)
    axis.legend()
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def plot_memory_ablation(rows: list[dict], output_path: Path) -> None:
    import matplotlib.pyplot as plt

    aggregated = aggregate_metric(rows, "mean_return")
    delays = sorted({row["test_delay"] for row in rows})
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.8), sharex=True, sharey=True)
    for axis, (title, method_keys) in zip(axes, ABLATION_GROUPS, strict=True):
        for method_key in method_keys:
            _plot_method(axis, _points_for_method(aggregated, method_key), ABLATION_LABELS[method_key])
        axis.set_title(title)
        axis.set_xlabel("Test delay (steps)")
        axis.set_xticks(delays)
        axis.grid(True, alpha=0.3)
        axis.legend(fontsize=8)
    axes[0].set_ylabel("Mean episode return")
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def _normalized_auc(points: list[tuple[int, float]]) -> float:
    points = sorted(points)
    if len(points) < 2 or points[-1][0] == points[0][0]:
        return points[0][1]
    area = sum(
        (right_delay - left_delay) * (left_value + right_value) / 2
        for (left_delay, left_value), (right_delay, right_value) in zip(points[:-1], points[1:], strict=True)
    )
    return area / (points[-1][0] - points[0][0])


def summarize_robustness(rows: list[dict], method_keys: list[str]) -> list[dict]:
    returns = per_model_metric(rows, "mean_return")
    failures = per_model_metric(rows, "failure_rate")
    summaries = []
    for method_key in method_keys:
        seeds = sorted({row["training_seed"] for row in returns if row["method_key"] == method_key})
        seed_values: dict[str, list[float]] = {"delay_auc": [], "worst_return": [], "mean_failure": []}
        for seed in seeds:
            return_points = [
                (row["test_delay"], row["mean"])
                for row in returns
                if row["method_key"] == method_key and row["training_seed"] == seed
            ]
            failure_values = [
                row["mean"] for row in failures if row["method_key"] == method_key and row["training_seed"] == seed
            ]
            if not return_points or not failure_values:
                continue
            seed_values["delay_auc"].append(_normalized_auc(return_points))
            seed_values["worst_return"].append(min(value for _, value in return_points))
            seed_values["mean_failure"].append(fmean(failure_values))

        summary = {"method_key": method_key, "num_training_seeds": len(seed_values["delay_auc"])}
        for metric, values in seed_values.items():
            summary[metric] = fmean(values) if values else 0.0
            summary[f"{metric}_std"] = stdev(values) if len(values) > 1 else 0.0
        summaries.append(summary)
    return summaries


def load_benchmark_results(
    directory: Path,
    *,
    backend: str,
    device: str,
    method_keys: list[str],
) -> list[dict]:
    """Load comparable benchmark records and ignore unrelated JSON files."""
    if not directory.exists():
        return []
    records = []
    for path in sorted(directory.rglob("*.json")):
        try:
            record = read_json(path)
            end_to_end = record["end_to_end_latency"]
            required = (
                float(record["parameter_count"]),
                float(record["model_size_bytes"]),
                float(end_to_end["mean_ms"]),
                float(end_to_end["p95_ms"]),
            )
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            print(f"[SKIP] Invalid benchmark JSON: {path}")
            continue
        if (
            record.get("method_key") not in method_keys
            or record.get("backend") != backend
            or record.get("device") != device
        ):
            continue
        if any(value < 0.0 for value in required):
            print(f"[SKIP] Negative benchmark measurement: {path}")
            continue
        records.append({**record, "benchmark_path": str(path)})
    return records


def aggregate_engineering_summary(
    evaluation_rows: list[dict],
    benchmark_records: list[dict],
    method_keys: list[str],
) -> list[dict]:
    """Join delay robustness with policy cost using one backend/device."""
    robustness = {
        summary["method_key"]: summary
        for summary in summarize_robustness(evaluation_rows, method_keys)
        if summary["num_training_seeds"]
    }
    grouped: dict[str, list[dict]] = defaultdict(list)
    for record in benchmark_records:
        grouped[record["method_key"]].append(record)

    summaries = []
    for method_key in method_keys:
        method_benchmarks = grouped.get(method_key, [])
        if method_key not in robustness or not method_benchmarks:
            continue
        robust = robustness[method_key]
        summaries.append(
            {
                "method_key": method_key,
                "method": METHOD_LABELS[method_key],
                "delay_auc": robust["delay_auc"],
                "delay_auc_std": robust["delay_auc_std"],
                "num_training_seeds": robust["num_training_seeds"],
                "parameter_count": fmean(float(record["parameter_count"]) for record in method_benchmarks),
                "model_size_bytes": fmean(float(record["model_size_bytes"]) for record in method_benchmarks),
                "end_to_end_mean_ms": fmean(
                    float(record["end_to_end_latency"]["mean_ms"]) for record in method_benchmarks
                ),
                "end_to_end_p95_ms": fmean(
                    float(record["end_to_end_latency"]["p95_ms"]) for record in method_benchmarks
                ),
                "num_benchmarks": len(method_benchmarks),
                "backend": method_benchmarks[0]["backend"],
                "device": method_benchmarks[0]["device"],
            }
        )
    return summaries


def pareto_frontier(rows: list[dict], *, cost_key: str, performance_key: str) -> list[dict]:
    """Return points not dominated by a lower-cost, equal-or-better policy."""
    frontier = []
    best_performance = float("-inf")
    for row in sorted(rows, key=lambda item: (float(item[cost_key]), -float(item[performance_key]))):
        performance = float(row[performance_key])
        if performance > best_performance:
            frontier.append(row)
            best_performance = performance
    return frontier


def save_engineering_csv(rows: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "method_key",
        "method",
        "delay_auc",
        "delay_auc_std",
        "num_training_seeds",
        "parameter_count",
        "model_size_bytes",
        "end_to_end_mean_ms",
        "end_to_end_p95_ms",
        "num_benchmarks",
        "backend",
        "device",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows({field: row[field] for field in fieldnames} for row in rows)


def plot_performance_cost(rows: list[dict], *, cost_key: str, xlabel: str, output_path: Path) -> None:
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(8, 5.5))
    for row in rows:
        method_key = row["method_key"]
        axis.errorbar(
            row[cost_key],
            row["delay_auc"],
            yerr=row["delay_auc_std"],
            marker="o",
            markersize=8,
            capsize=3,
            color=CORE_METHOD_COLORS.get(method_key),
        )
        axis.annotate(
            ARCHITECTURE_LABELS.get(method_key, method_key),
            (row[cost_key], row["delay_auc"]),
            xytext=(6, 5),
            textcoords="offset points",
        )
    frontier = pareto_frontier(rows, cost_key=cost_key, performance_key="delay_auc")
    if len(frontier) > 1:
        axis.plot(
            [row[cost_key] for row in frontier],
            [row["delay_auc"] for row in frontier],
            color="black",
            linestyle="--",
            linewidth=1,
            alpha=0.6,
            label="Pareto frontier",
        )
        axis.legend()
    axis.set_xlabel(xlabel)
    axis.set_ylabel("Delay-AUC (higher is better)")
    axis.grid(True, alpha=0.3)
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def plot_robustness_summary(rows: list[dict], output_path: Path) -> None:
    import matplotlib.pyplot as plt

    method_keys = ["train0to4", "stack2", "stack5", "stack8", "trace05", "trace08", "trace0875"]
    summaries = [summary for summary in summarize_robustness(rows, method_keys) if summary["num_training_seeds"]]
    labels = [METHOD_LABELS[summary["method_key"]].replace("Random delay 0-4 ", "") for summary in summaries]
    figure, axes = plt.subplots(1, 3, figsize=(16, 5))
    metrics = [
        ("delay_auc", "Delay-AUC (normalized)"),
        ("worst_return", "Worst-case return"),
        ("mean_failure", "Mean failure rate"),
    ]
    positions = list(range(len(summaries)))
    for axis, (metric, title) in zip(axes, metrics, strict=True):
        axis.bar(
            positions,
            [summary[metric] for summary in summaries],
            yerr=[summary[f"{metric}_std"] for summary in summaries],
            capsize=3,
        )
        axis.set_title(title)
        axis.set_xticks(positions, labels, rotation=30, ha="right")
        axis.grid(True, axis="y", alpha=0.3)
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def plot_seed_stability(rows: list[dict], output_path: Path) -> None:
    import matplotlib.pyplot as plt

    per_model = per_model_metric(rows, "mean_return")
    max_delay = max(row["test_delay"] for row in per_model)
    figure, axis = plt.subplots(figsize=(8, 5))
    for position, method_key in enumerate(CORE_RANDOM_METHODS):
        values = [
            row["mean"] for row in per_model if row["method_key"] == method_key and row["test_delay"] == max_delay
        ]
        offsets = (
            [0.0] if len(values) == 1 else [0.24 * (index / (len(values) - 1) - 0.5) for index in range(len(values))]
        )
        axis.scatter([position + offset for offset in offsets], values, s=42, alpha=0.8)
        if values:
            axis.errorbar(
                position,
                fmean(values),
                yerr=stdev(values) if len(values) > 1 else 0.0,
                fmt="D",
                color="black",
                capsize=5,
            )
    axis.set_xticks(range(len(CORE_RANDOM_METHODS)), [ARCHITECTURE_LABELS[key] for key in CORE_RANDOM_METHODS])
    axis.set_ylabel(f"Return at test delay {max_delay}")
    axis.set_title("Training-seed stability under the largest test delay")
    axis.grid(True, axis="y", alpha=0.3)
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def print_comparison(rows: list[dict]) -> None:
    print()
    print("Aggregated comparison across training seeds")
    print("=" * 78)

    return_rows = aggregate_metric(rows, "mean_return")
    xy_rows = aggregate_metric(rows, "mean_xy_error")
    failure_rows = aggregate_metric(rows, "failure_rate")

    for method_key, label in METHOD_LABELS.items():
        method_returns = [row for row in return_rows if row["method_key"] == method_key]
        if not method_returns:
            continue
        print(f"\n{label}")
        for return_row in method_returns:
            delay = return_row["test_delay"]
            xy_row = next(row for row in xy_rows if row["method_key"] == method_key and row["test_delay"] == delay)
            failure_row = next(
                row for row in failure_rows if row["method_key"] == method_key and row["test_delay"] == delay
            )
            print(
                f"  test={delay}: return={return_row['mean']:.3f} "
                f"xy={xy_row['mean']:.3f} failure={failure_row['mean']:.3f} "
                f"train_seeds={return_row['num_training_seeds']}"
            )


def main() -> None:
    args = build_parser().parse_args()
    rows = []
    rows.extend(load_results(Path(args.mlp_dir), parse_mlp_filename))
    rows.extend(load_results(Path(args.stack_dir), parse_stack_filename))
    rows.extend(load_results(Path(args.trace_dir), parse_trace_filename))

    if not rows:
        raise RuntimeError("No evaluation results were found.")

    method_order = list(METHOD_LABELS)
    rows.sort(
        key=lambda row: (
            method_order.index(row["method_key"]),
            row["test_delay"],
            row["training_seed"],
            row["evaluation_seed"],
        )
    )

    print(f"[INFO] Loaded {len(rows)} evaluation results.")
    save_csv(rows, Path(args.output_csv))

    figure_dir = Path(args.figure_dir)
    plot_metric(rows, "mean_return", "Mean episode return", figure_dir / "return_vs_delay_all_methods.png")
    plot_metric(
        rows,
        "mean_xy_error",
        "Mean XY velocity tracking error (m/s)",
        figure_dir / "xy_error_vs_delay_all_methods.png",
    )
    plot_metric(rows, "failure_rate", "Failure rate", figure_dir / "failure_rate_vs_delay_all_methods.png")
    plot_grouped_metric(
        rows,
        "mean_return",
        "Mean episode return",
        TRAINING_CONDITION_GROUPS,
        ARCHITECTURE_LABELS,
        figure_dir / "return_by_training_condition.png",
    )
    plot_grouped_metric(
        rows,
        "mean_xy_error",
        "Mean XY velocity tracking error (m/s)",
        TRAINING_CONDITION_GROUPS,
        ARCHITECTURE_LABELS,
        figure_dir / "xy_error_by_training_condition.png",
    )
    plot_grouped_metric(
        rows,
        "failure_rate",
        "Failure rate",
        TRAINING_CONDITION_GROUPS,
        ARCHITECTURE_LABELS,
        figure_dir / "failure_rate_by_training_condition.png",
    )
    plot_grouped_metric(
        rows,
        "mean_return",
        "Mean episode return",
        ARCHITECTURE_GROUPS,
        TRAINING_LABELS,
        figure_dir / "return_by_architecture.png",
    )
    plot_random_training_metrics(rows, figure_dir / "random_training_core_metrics.png")
    plot_return_retention(rows, figure_dir / "return_retention_vs_delay.png")
    plot_return_improvement(rows, figure_dir / "return_improvement_vs_mlp.png")
    plot_memory_ablation(rows, figure_dir / "memory_horizon_ablation.png")
    plot_robustness_summary(rows, figure_dir / "robustness_summary.png")
    plot_seed_stability(rows, figure_dir / "seed_stability_random_methods.png")
    benchmark_records = load_benchmark_results(
        Path(args.benchmark_dir),
        backend=args.benchmark_backend,
        device=args.benchmark_device,
        method_keys=CORE_RANDOM_METHODS,
    )
    engineering_rows = aggregate_engineering_summary(rows, benchmark_records, CORE_RANDOM_METHODS)
    if engineering_rows:
        save_engineering_csv(engineering_rows, Path(args.engineering_csv))
        plot_performance_cost(
            engineering_rows,
            cost_key="parameter_count",
            xlabel="Policy parameters",
            output_path=figure_dir / "performance_vs_parameters.png",
        )
        plot_performance_cost(
            engineering_rows,
            cost_key="end_to_end_p95_ms",
            xlabel=f"End-to-end p95 inference latency (ms, {args.benchmark_device})",
            output_path=figure_dir / "performance_vs_latency.png",
        )
        print(f"[INFO] Saved engineering summary to: {args.engineering_csv}")
    else:
        print(
            f"[INFO] No comparable {args.benchmark_backend}/{args.benchmark_device} benchmarks found; "
            "skipped engineering Pareto plots."
        )
    print_comparison(rows)

    print()
    print(f"[INFO] Saved CSV to: {args.output_csv}")
    print(f"[INFO] Saved figures to: {args.figure_dir}")


if __name__ == "__main__":
    main()
