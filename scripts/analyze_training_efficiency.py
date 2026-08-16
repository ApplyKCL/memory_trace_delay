"""Summarize RSL-RL TensorBoard training speed and wall-clock metrics."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import fmean, median, stdev
from typing import Any

DEFAULT_METHODS = ["train0to4", "stack5", "trace08"]
SCALAR_TAGS = {
    "fps": "Perf/total_fps",
    "collection_time": "Perf/collection time",
    "learning_time": "Perf/learning_time",
    "reward": "Train/mean_reward",
    "validation_time": "Validation/wall_time_seconds",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize TensorBoard training efficiency across methods and seeds.")
    parser.add_argument("--config", default="configs/experiments/all_algorithms.json")
    parser.add_argument("--log_root", default="logs/rsl_rl")
    parser.add_argument("--method", action="append", default=[], help="Method key; repeat to select multiple methods.")
    parser.add_argument("--warmup_iterations", type=int, default=10)
    parser.add_argument("--output_csv", default="results/training_efficiency.csv")
    parser.add_argument("--figure_dir", default="results/figures")
    parser.add_argument("--skip_plots", action="store_true")
    return parser


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        config = json.load(file)
    if config.get("version") != 1 or not isinstance(config.get("methods"), list):
        raise ValueError("Experiment config must contain version=1 and a methods list.")
    return config


def method_training_seeds(config: dict[str, Any], method: dict[str, Any]) -> list[int]:
    common_training = config.get("common", {}).get("training", {})
    common_evaluation = config.get("common", {}).get("evaluation", {})
    training = {**common_training, **method.get("training", {})}
    evaluation = {**common_evaluation, **method.get("evaluation", {})}
    seeds = set(training.get("seeds", []))
    seeds.update(evaluation.get("training_seeds", training.get("seeds", [])))
    if not seeds or not all(isinstance(seed, int) for seed in seeds):
        raise ValueError(f"{method['key']}: no valid training seeds are configured.")
    return sorted(seeds)


def discover_latest_runs(
    config: dict[str, Any],
    log_root: Path,
    selected_methods: list[str],
) -> list[dict[str, Any]]:
    methods = {method["key"]: method for method in config["methods"]}
    unknown = sorted(set(selected_methods) - set(methods))
    if unknown:
        raise ValueError(f"Unknown method keys: {', '.join(unknown)}")

    discovered = []
    common_training = config.get("common", {}).get("training", {})
    for method_key in selected_methods:
        method = methods[method_key]
        training = {**common_training, **method.get("training", {})}
        run_name_template = training.get("run_name")
        if not isinstance(run_name_template, str) or not run_name_template:
            raise ValueError(f"{method_key}: training.run_name is required.")
        experiment_root = log_root / method["experiment_name"]
        for training_seed in method_training_seeds(config, method):
            run_name = run_name_template.format(seed=training_seed, training_seed=training_seed)
            candidates = sorted(path for path in experiment_root.glob(f"*_{run_name}") if path.is_dir())
            if not candidates:
                print(f"[SKIP] No run directory for {method_key} seed {training_seed}: *_{run_name}")
                continue
            discovered.append(
                {
                    "method_key": method_key,
                    "training_seed": training_seed,
                    "run_dir": candidates[-1].resolve(),
                }
            )
    return discovered


def read_tensorboard_scalars(run_dir: Path) -> dict[str, list[tuple[int, float]]]:
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except ImportError as error:
        raise RuntimeError("TensorBoard is required. Install requirements-dev.txt first.") from error

    accumulator = EventAccumulator(str(run_dir), size_guidance={"scalars": 0})
    accumulator.Reload()
    available = set(accumulator.Tags().get("scalars", []))
    scalars: dict[str, list[tuple[int, float]]] = {}
    for key, tag in SCALAR_TAGS.items():
        if tag not in available:
            scalars[key] = []
            continue
        scalars[key] = [(int(event.step), float(event.value)) for event in accumulator.Scalars(tag)]
    return scalars


def deduplicate_scalar_steps(values: list[tuple[int, float]]) -> list[tuple[int, float]]:
    """Keep the newest value when a resumed run writes a step more than once."""
    return sorted(dict(values).items())


def summarize_scalar_series(
    scalars: dict[str, list[tuple[int, float]]],
    *,
    warmup_iterations: int,
) -> dict[str, float | int]:
    if warmup_iterations < 0:
        raise ValueError("warmup_iterations must be non-negative.")
    fps = [
        (step, value) for step, value in deduplicate_scalar_steps(scalars.get("fps", [])) if step >= warmup_iterations
    ]
    if not fps:
        raise ValueError("No Perf/total_fps samples remain after the warm-up filter.")

    collection = dict(deduplicate_scalar_steps(scalars.get("collection_time", [])))
    learning = dict(deduplicate_scalar_steps(scalars.get("learning_time", [])))
    all_timed_steps = sorted(set(collection) & set(learning))
    iteration_times = [collection[step] + learning[step] for step in all_timed_steps]
    rewards = deduplicate_scalar_steps(scalars.get("reward", []))
    validation_times = [value for _, value in deduplicate_scalar_steps(scalars.get("validation_time", []))]
    fps_values = [value for _, value in fps]
    total_training_time = sum(iteration_times)
    total_validation_time = sum(validation_times)
    return {
        "first_iteration": min(step for step, _ in fps),
        "last_iteration": max(step for step, _ in fps),
        "iterations_measured": len(fps_values),
        "mean_fps": fmean(fps_values),
        "median_fps": median(fps_values),
        "mean_collection_time_seconds": fmean(collection.values()) if collection else 0.0,
        "mean_learning_time_seconds": fmean(learning.values()) if learning else 0.0,
        "mean_iteration_time_seconds": fmean(iteration_times) if iteration_times else 0.0,
        "total_training_time_seconds": total_training_time,
        "total_validation_time_seconds": total_validation_time,
        "total_end_to_end_time_seconds": total_training_time + total_validation_time,
        "final_mean_reward": rewards[-1][1] if rewards else 0.0,
        "best_mean_reward": max((value for _, value in rewards), default=0.0),
    }


def save_csv(rows: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "method_key",
        "training_seed",
        "run_dir",
        "first_iteration",
        "last_iteration",
        "iterations_measured",
        "mean_fps",
        "median_fps",
        "mean_collection_time_seconds",
        "mean_learning_time_seconds",
        "mean_iteration_time_seconds",
        "total_training_time_seconds",
        "total_validation_time_seconds",
        "total_end_to_end_time_seconds",
        "final_mean_reward",
        "best_mean_reward",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows({field: row[field] for field in fieldnames} for row in rows)


def aggregate_method_metric(rows: list[dict[str, Any]], metric: str) -> list[dict[str, Any]]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        grouped[row["method_key"]].append(float(row[metric]))
    return [
        {
            "method_key": method_key,
            "mean": fmean(values),
            "std": stdev(values) if len(values) > 1 else 0.0,
            "num_training_seeds": len(values),
        }
        for method_key, values in sorted(grouped.items())
    ]


def plot_summary(rows: list[dict[str, Any]], figure_dir: Path) -> None:
    import matplotlib.pyplot as plt

    plots = [
        ("mean_fps", "Training throughput (steps/s)", "training_fps_comparison.png", 1.0),
        (
            "total_end_to_end_time_seconds",
            "Training + validation time (hours)",
            "training_wall_time_comparison.png",
            1.0 / 3600.0,
        ),
    ]
    figure_dir.mkdir(parents=True, exist_ok=True)
    for metric, ylabel, filename, scale in plots:
        aggregated = aggregate_method_metric(rows, metric)
        labels = [item["method_key"] for item in aggregated]
        means = [item["mean"] * scale for item in aggregated]
        errors = [item["std"] * scale for item in aggregated]
        figure, axis = plt.subplots(figsize=(8, 5))
        axis.bar(labels, means, yerr=errors, capsize=4)
        axis.set_ylabel(ylabel)
        axis.grid(True, axis="y", alpha=0.3)
        figure.tight_layout()
        figure.savefig(figure_dir / filename, dpi=200)
        plt.close(figure)


def main() -> None:
    args = build_parser().parse_args()
    if args.warmup_iterations < 0:
        raise ValueError("warmup_iterations must be non-negative.")

    repo_root = Path(__file__).resolve().parents[1]
    config_path = Path(args.config).expanduser()
    config_path = config_path if config_path.is_absolute() else repo_root / config_path
    log_root = Path(args.log_root).expanduser()
    log_root = log_root if log_root.is_absolute() else repo_root / log_root
    selected_methods = args.method or DEFAULT_METHODS
    runs = discover_latest_runs(load_config(config_path), log_root, selected_methods)

    rows = []
    for run in runs:
        try:
            summary = summarize_scalar_series(
                read_tensorboard_scalars(run["run_dir"]),
                warmup_iterations=args.warmup_iterations,
            )
        except (OSError, ValueError, RuntimeError) as error:
            print(f"[SKIP] {run['run_dir']}: {error}")
            continue
        rows.append({**run, **summary, "run_dir": str(run["run_dir"])})

    if not rows:
        raise RuntimeError("No usable TensorBoard training runs were found.")
    output_csv = Path(args.output_csv).expanduser()
    output_csv = output_csv if output_csv.is_absolute() else repo_root / output_csv
    save_csv(rows, output_csv)
    if not args.skip_plots:
        figure_dir = Path(args.figure_dir).expanduser()
        figure_dir = figure_dir if figure_dir.is_absolute() else repo_root / figure_dir
        plot_summary(rows, figure_dir)
    print(f"[INFO] Summarized {len(rows)} training runs: {output_csv}")


if __name__ == "__main__":
    main()
