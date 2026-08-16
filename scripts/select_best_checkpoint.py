"""Validate periodic checkpoints and select the most delay-robust model."""

from __future__ import annotations

import argparse
import csv
import glob
import json
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from statistics import fmean
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import run_experiments as experiment_runner  # noqa: E402

DEFAULT_METHODS = ["train0to4", "stack5", "trace08"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Select best periodic checkpoints using held-out delay validation.")
    parser.add_argument("--config", default="configs/experiments/all_algorithms.json")
    parser.add_argument("--only", action="append", default=[], help="Method key; repeat to select multiple methods.")
    parser.add_argument("--training_seed", action="append", type=int, default=[])
    parser.add_argument("--validation_seed", action="append", type=int, default=[])
    parser.add_argument("--test_delays", nargs="+", type=int, default=[0, 2, 4, 6])
    parser.add_argument("--iterations", nargs="+", type=int, default=None, help="Only validate these iterations.")
    parser.add_argument("--last_n", type=int, default=None, help="Validate only the newest N checkpoints per run.")
    parser.add_argument("--num_envs", type=int, default=256)
    parser.add_argument("--num_episodes", type=int, default=256)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--gpus", nargs="*", type=int, default=None)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--output_dir", default="results/checkpoint_validation")
    parser.add_argument("--summary_csv", default="results/checkpoint_validation/checkpoint_summary.csv")
    parser.add_argument("--best_json", default="results/checkpoint_validation/best_checkpoints.json")
    parser.add_argument("--python", default=None)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--continue_on_error", action="store_true")
    parser.add_argument("--copy_best", action="store_true", help="Copy each winner beside its run as best_model.pt.")
    parser.add_argument("--overwrite_best", action="store_true")
    return parser


def configured_training_seeds(config: dict[str, Any], method: dict[str, Any]) -> list[int]:
    common_training = config.get("common", {}).get("training", {})
    common_evaluation = config.get("common", {}).get("evaluation", {})
    training = experiment_runner.merge_stage_config(common_training, method.get("training", {}))
    evaluation = experiment_runner.merge_stage_config(common_evaluation, method.get("evaluation", {}))
    seeds = evaluation.get("training_seeds", training.get("seeds", []))
    if not isinstance(seeds, list) or not seeds or not all(isinstance(seed, int) for seed in seeds):
        raise ValueError(f"{method['key']}: no valid evaluation training seeds are configured.")
    return list(seeds)


def configured_checkpoint_pattern(repo_root: Path, training: dict[str, Any], training_seed: int) -> Path:
    template = training.get("checkpoint_glob")
    if not isinstance(template, str) or not template:
        raise ValueError("training.checkpoint_glob is required for periodic checkpoint selection.")
    pattern = Path(template.format(seed=training_seed, training_seed=training_seed)).expanduser()
    return pattern if pattern.is_absolute() else repo_root / pattern


def discover_periodic_checkpoints(
    pattern: Path,
    iterations: list[int] | None = None,
    last_n: int | None = None,
) -> list[Path]:
    if iterations is not None and last_n is not None:
        raise ValueError("Use either iterations or last_n, not both.")
    if last_n is not None and last_n <= 0:
        raise ValueError("last_n must be positive.")
    candidates = [
        Path(path).resolve()
        for path in glob.glob(str(pattern))
        if experiment_runner.checkpoint_iteration(Path(path)) >= 0
    ]
    if not candidates:
        raise FileNotFoundError(f"No periodic checkpoints match: {pattern}")
    newest_run = max(path.parent.name for path in candidates)
    candidates = [path for path in candidates if path.parent.name == newest_run]
    if iterations is not None:
        requested = set(iterations)
        candidates = [path for path in candidates if experiment_runner.checkpoint_iteration(path) in requested]
    if not candidates:
        raise FileNotFoundError(f"No requested checkpoint iterations match: {pattern}")
    candidates = sorted(candidates, key=experiment_runner.checkpoint_iteration)
    return candidates[-last_n:] if last_n is not None else candidates


def normalized_auc(points: list[tuple[int, float]]) -> float:
    points = sorted(points)
    if not points:
        raise ValueError("Cannot compute Delay-AUC without return points.")
    if len(points) == 1 or points[-1][0] == points[0][0]:
        return points[0][1]
    area = sum(
        (right_delay - left_delay) * (left_value + right_value) / 2.0
        for (left_delay, left_value), (right_delay, right_value) in zip(points[:-1], points[1:], strict=True)
    )
    return area / (points[-1][0] - points[0][0])


def summarize_checkpoint_results(
    method_key: str,
    training_seed: int,
    checkpoint: Path,
    results: list[dict[str, Any]],
) -> dict[str, Any]:
    if not results:
        raise ValueError(f"No validation results for checkpoint: {checkpoint}")
    returns_by_delay: dict[int, list[float]] = defaultdict(list)
    failures_by_delay: dict[int, list[float]] = defaultdict(list)
    validation_seeds = set()
    for result in results:
        delay = int(result["test_delay_steps"])
        returns_by_delay[delay].append(float(result["mean_return"]))
        failures_by_delay[delay].append(float(result["failure_rate"]))
        evaluation_seed = result.get("evaluation_seed")
        if evaluation_seed is None:
            evaluation_seed = result["seed"]
        validation_seeds.add(int(evaluation_seed))
    return_points = [(delay, fmean(values)) for delay, values in sorted(returns_by_delay.items())]
    return {
        "method_key": method_key,
        "training_seed": training_seed,
        "checkpoint": str(checkpoint.resolve()),
        "iteration": experiment_runner.checkpoint_iteration(checkpoint),
        "delay_auc": normalized_auc(return_points),
        "worst_return": min(value for _, value in return_points),
        "mean_failure_rate": fmean(fmean(values) for values in failures_by_delay.values()),
        "num_test_delays": len(returns_by_delay),
        "num_validation_seeds": len(validation_seeds),
        "num_result_files": len(results),
    }


def select_best_checkpoint(summaries: list[dict[str, Any]]) -> dict[str, Any]:
    if not summaries:
        raise ValueError("No checkpoint summaries were supplied.")
    return max(
        summaries,
        key=lambda item: (
            float(item["delay_auc"]),
            float(item["worst_return"]),
            -float(item["mean_failure_rate"]),
            int(item["iteration"]),
        ),
    )


def load_result(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def save_summary_csv(rows: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "method_key",
        "training_seed",
        "checkpoint",
        "iteration",
        "delay_auc",
        "worst_return",
        "mean_failure_rate",
        "num_test_delays",
        "num_validation_seeds",
        "num_result_files",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows({field: row[field] for field in fieldnames} for row in rows)


def copy_best_checkpoint(summary: dict[str, Any], *, overwrite: bool) -> Path:
    source = Path(summary["checkpoint"])
    target = source.parent / "best_model.pt"
    if target.exists() and not overwrite:
        raise FileExistsError(f"Best checkpoint already exists: {target}. Use --overwrite_best to replace it.")
    shutil.copy2(source, target)
    return target


def main() -> int:
    args = build_parser().parse_args()
    if args.num_envs <= 0 or args.num_episodes <= 0:
        raise ValueError("num_envs and num_episodes must be positive.")
    if any(delay < 0 for delay in args.test_delays):
        raise ValueError("test_delays must be non-negative.")
    if args.iterations is not None and any(iteration < 0 for iteration in args.iterations):
        raise ValueError("iterations must be non-negative.")
    if args.iterations is not None and args.last_n is not None:
        raise ValueError("Use either --iterations or --last_n, not both.")
    if args.last_n is not None and args.last_n <= 0:
        raise ValueError("last_n must be positive.")
    validation_seeds = args.validation_seed or [901]
    if any(seed < 0 for seed in validation_seeds):
        raise ValueError("validation seeds must be non-negative.")

    repo_root = Path(__file__).resolve().parents[1]
    config_path = Path(args.config).expanduser()
    config_path = config_path if config_path.is_absolute() else repo_root / config_path
    config = experiment_runner.load_config(config_path.resolve())
    selected = experiment_runner.select_methods(config["methods"], args.only or DEFAULT_METHODS)
    common_training = config.get("common", {}).get("training", {})
    common_evaluation = config.get("common", {}).get("evaluation", {})
    python_executable = args.python or sys.executable
    output_dir = Path(args.output_dir).expanduser()
    output_dir = output_dir if output_dir.is_absolute() else repo_root / output_dir

    execution = experiment_runner.load_execution_settings(config)
    gpus = tuple(args.gpus) if args.gpus is not None else execution.gpus
    if len(set(gpus)) != len(gpus) or any(gpu < 0 for gpu in gpus):
        raise ValueError("gpus must contain unique non-negative integers.")
    workers = args.workers or (min(execution.evaluation_workers, len(gpus)) if gpus else 1)
    if workers <= 0 or (gpus and workers > len(gpus)):
        raise ValueError("workers must be positive and cannot exceed the GPU count.")

    jobs: list[experiment_runner.CommandJob] = []
    result_groups: dict[tuple[str, int, str], list[Path]] = defaultdict(list)
    skipped = 0
    for method in selected:
        training = experiment_runner.merge_stage_config(common_training, method.get("training", {}))
        evaluation = experiment_runner.merge_stage_config(common_evaluation, method.get("evaluation", {}))
        evaluation.update(
            {
                "num_envs": args.num_envs,
                "num_episodes": args.num_episodes,
                "device": args.device,
                "headless": True,
            }
        )
        available_seeds = configured_training_seeds(config, method)
        training_seeds = args.training_seed or available_seeds
        unknown_seeds = sorted(set(training_seeds) - set(available_seeds))
        if unknown_seeds:
            raise ValueError(f"{method['key']}: unconfigured training seeds requested: {unknown_seeds}")

        for training_seed in training_seeds:
            pattern = configured_checkpoint_pattern(repo_root, training, training_seed)
            checkpoints = discover_periodic_checkpoints(pattern, args.iterations, args.last_n)
            for checkpoint in checkpoints:
                iteration = experiment_runner.checkpoint_iteration(checkpoint)
                group_key = (method["key"], training_seed, str(checkpoint))
                for test_delay in args.test_delays:
                    for validation_seed in validation_seeds:
                        output_path = output_dir / (
                            f"{method['key']}_trainseed{training_seed}_iter{iteration}_"
                            f"test{test_delay}_valseed{validation_seed}.json"
                        )
                        result_groups[group_key].append(output_path)
                        if output_path.exists() and not args.overwrite:
                            skipped += 1
                            continue
                        command = experiment_runner.build_evaluation_command(
                            repo_root,
                            python_executable,
                            method,
                            training,
                            evaluation,
                            checkpoint=str(checkpoint),
                            training_seed=training_seed,
                            evaluation_seed=validation_seed,
                            test_delay=test_delay,
                            output_path=output_path,
                        )
                        label = (
                            f"validate:{method['key']}:train-{training_seed}:iter-{iteration}:"
                            f"delay-{test_delay}:seed-{validation_seed}"
                        )
                        jobs.append(experiment_runner.CommandJob(label, tuple(command)))

    print(f"[INFO] Validation jobs={len(jobs)} skipped={skipped} GPUs={gpus or 'configured device'}")
    results = experiment_runner.run_job_batch(
        jobs,
        repo_root,
        gpus=gpus,
        workers=workers,
        dry_run=args.dry_run,
        continue_on_error=args.continue_on_error,
    )
    failures = [result for result in results if result.error is not None]
    if failures and not args.continue_on_error:
        raise RuntimeError(f"Checkpoint validation failed: {failures[0].job.label}: {failures[0].error}")
    if args.dry_run:
        print(f"[SUMMARY] commands={len(jobs)} skipped={skipped} failures={len(failures)}")
        return 1 if failures else 0

    summaries = []
    for (method_key, training_seed, checkpoint_text), paths in result_groups.items():
        missing = [path for path in paths if not path.is_file()]
        if missing:
            print(f"[SKIP] Incomplete checkpoint validation ({len(missing)} missing): {checkpoint_text}")
            continue
        summaries.append(
            summarize_checkpoint_results(
                method_key,
                training_seed,
                Path(checkpoint_text),
                [load_result(path) for path in paths],
            )
        )
    if not summaries:
        raise RuntimeError("No complete checkpoint validation summaries were produced.")

    summary_csv = Path(args.summary_csv).expanduser()
    summary_csv = summary_csv if summary_csv.is_absolute() else repo_root / summary_csv
    save_summary_csv(summaries, summary_csv)
    grouped_summaries: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for summary in summaries:
        grouped_summaries[(summary["method_key"], summary["training_seed"])].append(summary)
    winners = [select_best_checkpoint(group) for _, group in sorted(grouped_summaries.items())]
    if args.copy_best:
        for winner in winners:
            winner["best_model_copy"] = str(copy_best_checkpoint(winner, overwrite=args.overwrite_best))

    best_json = Path(args.best_json).expanduser()
    best_json = best_json if best_json.is_absolute() else repo_root / best_json
    best_json.parent.mkdir(parents=True, exist_ok=True)
    with best_json.open("w", encoding="utf-8") as file:
        json.dump(
            {
                "selection_rule": "max delay_auc, max worst_return, min failure, latest iteration",
                "winners": winners,
            },
            file,
            indent=2,
        )
    print(f"[INFO] Checkpoint summary: {summary_csv}")
    print(f"[INFO] Best checkpoints: {best_json}")
    print(f"[SUMMARY] commands={len(jobs)} skipped={skipped} failures={len(failures)}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
