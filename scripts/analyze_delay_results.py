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
    "stack5": "Random delay 0-4 Stack-5",
    "trace08": "Random delay 0-4 Trace (decay 0.8)",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Aggregate action-delay evaluation results.")
    parser.add_argument("--mlp_dir", default="results/mlp", help="MLP evaluation JSON directory.")
    parser.add_argument("--stack_dir", default="results/stack", help="Stack-PPO evaluation JSON directory.")
    parser.add_argument("--trace_dir", default="results/trace", help="Trace-PPO evaluation JSON directory.")
    parser.add_argument("--output_csv", default="results/combined_summary.csv", help="Output summary CSV.")
    parser.add_argument("--figure_dir", default="results/figures", help="Output figure directory.")
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
        r"^stack5_train0to4(?:_trainseed(\d+))?_test(\d+)_(?:seed|evalseed)(\d+)\.json$",
        path.name,
    )
    if match is None:
        raise ValueError(f"Unexpected Stack filename: {path.name}")
    training_seed = int(match.group(1)) if match.group(1) is not None else None
    return "stack5", int(match.group(2)), int(match.group(3)), training_seed


def parse_trace_filename(path: Path) -> tuple[str, int, int, int | None]:
    match = re.match(
        r"^(?:trace|trace08)_train0to4(?:_trainseed(\d+))?_test(\d+)_(?:seed|evalseed)(\d+)\.json$",
        path.name,
    )
    if match is None:
        raise ValueError(f"Unexpected Trace filename: {path.name}")
    training_seed = int(match.group(1)) if match.group(1) is not None else None
    return "trace08", int(match.group(2)), int(match.group(3)), training_seed


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
        raise ValueError(
            f"Filename test delay {filename_test_delay} disagrees with JSON test delay {test_delay}."
        )
    if evaluation_seed != filename_eval_seed:
        raise ValueError(
            f"Filename evaluation seed {filename_eval_seed} disagrees with JSON seed {evaluation_seed}."
        )

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
        "trace_decay": result.get("trace_decay"),
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
        "trace_decay",
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


def plot_metric(rows: list[dict], metric: str, ylabel: str, output_path: Path) -> None:
    import matplotlib.pyplot as plt

    aggregated = aggregate_metric(rows, metric)
    plt.figure(figsize=(8, 5.5))

    for method_key, label in METHOD_LABELS.items():
        points = [point for point in aggregated if point["method_key"] == method_key]
        if not points:
            continue
        points.sort(key=lambda point: point["test_delay"])
        plt.errorbar(
            [point["test_delay"] for point in points],
            [point["mean"] for point in points],
            yerr=[point["std_across_training_seeds"] for point in points],
            marker="o",
            capsize=4,
            label=label,
        )

    delays = sorted({row["test_delay"] for row in rows})
    plt.xlabel("Test action delay (steps)")
    plt.ylabel(ylabel)
    plt.xticks(delays)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()


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
            xy_row = next(
                row for row in xy_rows if row["method_key"] == method_key and row["test_delay"] == delay
            )
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
    print_comparison(rows)

    print()
    print(f"[INFO] Saved CSV to: {args.output_csv}")
    print(f"[INFO] Saved figures to: {args.figure_dir}")


if __name__ == "__main__":
    main()
