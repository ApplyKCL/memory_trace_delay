"""Run configured training, delay evaluation, and result analysis jobs."""

from __future__ import annotations

import argparse
import glob
import json
import re
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

DEFAULT_CONFIG = "configs/experiments/train.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the configured Memory Trace experiment suite.")
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="JSON experiment configuration path.")
    parser.add_argument(
        "--only",
        action="append",
        default=[],
        metavar="METHOD",
        help="Run only this method key; repeat the option to select multiple methods.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing them.")
    parser.add_argument("--skip-train", action="store_true", help="Skip every training stage.")
    parser.add_argument("--skip-eval", action="store_true", help="Skip every evaluation stage.")
    parser.add_argument("--skip-analysis", action="store_true", help="Skip the final analysis stage.")
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue with later jobs after a subprocess fails.",
    )
    parser.add_argument("--python", default=None, help="Python executable used for child scripts.")
    parser.add_argument("--list-methods", action="store_true", help="List configured method keys and exit.")
    return parser


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        config = json.load(file)

    if config.get("version") != 1:
        raise ValueError("Experiment config must contain version=1.")
    methods = config.get("methods")
    if not isinstance(methods, list) or not methods:
        raise ValueError("Experiment config must contain a non-empty methods list.")

    seen_keys: set[str] = set()
    for method in methods:
        for field in ("key", "task", "experiment_name"):
            if not method.get(field):
                raise ValueError(f"Each method requires a non-empty {field!r} field.")
        if method["key"] in seen_keys:
            raise ValueError(f"Duplicate method key: {method['key']}")
        seen_keys.add(method["key"])

    return config


def merge_stage_config(common: dict[str, Any], specific: dict[str, Any]) -> dict[str, Any]:
    """Merge one common/stage config while concatenating extra arguments."""
    merged = dict(common)
    merged.update(specific)
    common_extra = _string_list(common.get("extra_args", []), "common extra_args")
    specific_extra = _string_list(specific.get("extra_args", []), "method extra_args")
    merged["extra_args"] = common_extra + specific_extra
    return merged


def _string_list(value: Any, field: str) -> list[str]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"{field} must be a list of strings.")
    return list(value)


def _integer_list(value: Any, field: str, *, non_negative: bool = False) -> list[int]:
    if not isinstance(value, list) or not value or not all(isinstance(item, int) for item in value):
        raise ValueError(f"{field} must be a non-empty list of integers.")
    if non_negative and any(item < 0 for item in value):
        raise ValueError(f"{field} must contain only non-negative integers.")
    return list(value)


def _positive_integer(config: dict[str, Any], field: str) -> int:
    value = config.get(field)
    if not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field} must be a positive integer.")
    return value


def _append_option(command: list[str], name: str, value: Any) -> None:
    if value is not None:
        command.extend((name, str(value)))


def build_train_command(
    repo_root: Path,
    python_executable: str,
    method: dict[str, Any],
    train_cfg: dict[str, Any],
    training_seed: int,
) -> tuple[list[str], str]:
    delay_min = train_cfg.get("delay_min")
    delay_max = train_cfg.get("delay_max")
    if not isinstance(delay_min, int) or delay_min < 0:
        raise ValueError(f"{method['key']}: training delay_min must be a non-negative integer.")
    if not isinstance(delay_max, int) or delay_max < delay_min:
        raise ValueError(f"{method['key']}: training delay_max must be >= delay_min.")

    run_name_template = train_cfg.get("run_name")
    if not isinstance(run_name_template, str) or not run_name_template:
        raise ValueError(f"{method['key']}: training run_name is required.")
    run_name = run_name_template.format(seed=training_seed, training_seed=training_seed)

    command = [
        python_executable,
        str(repo_root / "scripts" / "rsl_rl" / "train.py"),
        "--task",
        method["task"],
        "--seed",
        str(training_seed),
        "--num_envs",
        str(_positive_integer(train_cfg, "num_envs")),
        "--max_iterations",
        str(_positive_integer(train_cfg, "max_iterations")),
        "--train_delay_min",
        str(delay_min),
        "--train_delay_max",
        str(delay_max),
        "--run_name",
        run_name,
    ]
    _append_option(command, "--device", train_cfg.get("device"))
    if train_cfg.get("headless", True):
        command.append("--headless")
    command.extend(_string_list(train_cfg.get("extra_args", []), "training extra_args"))
    return command, run_name


def build_evaluation_command(
    repo_root: Path,
    python_executable: str,
    method: dict[str, Any],
    train_cfg: dict[str, Any],
    evaluation_cfg: dict[str, Any],
    *,
    checkpoint: str,
    training_seed: int,
    evaluation_seed: int,
    test_delay: int,
    output_path: Path,
) -> list[str]:
    command = [
        python_executable,
        str(repo_root / "scripts" / "rsl_rl" / "evaluate_delay.py"),
        "--task",
        method["task"],
        "--checkpoint",
        checkpoint,
        "--test_delay",
        str(test_delay),
        "--seed",
        str(evaluation_seed),
        "--method",
        method["key"],
        "--training_seed",
        str(training_seed),
        "--train_delay_min",
        str(train_cfg["delay_min"]),
        "--train_delay_max",
        str(train_cfg["delay_max"]),
        "--num_envs",
        str(_positive_integer(evaluation_cfg, "num_envs")),
        "--num_episodes",
        str(_positive_integer(evaluation_cfg, "num_episodes")),
        "--output",
        str(output_path),
    ]
    _append_option(command, "--device", evaluation_cfg.get("device"))
    if evaluation_cfg.get("headless", True):
        command.append("--headless")
    command.extend(_string_list(evaluation_cfg.get("extra_args", []), "evaluation extra_args"))
    return command


def checkpoint_iteration(path: Path) -> int:
    match = re.search(r"model_(\d+)\.pt$", path.name)
    return int(match.group(1)) if match is not None else -1


def select_latest_checkpoint(paths: list[Path]) -> Path:
    if not paths:
        raise FileNotFoundError("No checkpoint candidates were found.")
    return max(paths, key=lambda path: (path.parent.name, checkpoint_iteration(path), path.name))


def find_trained_checkpoint(repo_root: Path, experiment_name: str, run_name: str) -> Path:
    log_root = repo_root / "logs" / "rsl_rl" / experiment_name
    candidates = list(log_root.glob(f"*_{run_name}/model_*.pt"))
    return select_latest_checkpoint(candidates).resolve()


def resolve_configured_checkpoint(
    repo_root: Path,
    method: dict[str, Any],
    train_cfg: dict[str, Any],
    training_seed: int,
    *,
    dry_run: bool,
) -> str:
    checkpoints = train_cfg.get("checkpoints", {})
    if checkpoints:
        if not isinstance(checkpoints, dict):
            raise ValueError(f"{method['key']}: checkpoints must be an object keyed by training seed.")
        configured = checkpoints.get(str(training_seed))
        if configured is not None:
            path = Path(str(configured)).expanduser()
            path = path if path.is_absolute() else repo_root / path
            if not path.is_file() and not dry_run:
                raise FileNotFoundError(f"Configured checkpoint does not exist: {path}")
            return str(path.resolve()) if path.exists() else str(path)

    pattern_template = train_cfg.get("checkpoint_glob")
    if not isinstance(pattern_template, str) or not pattern_template:
        if dry_run:
            return f"<checkpoint:{method['key']}:train-seed-{training_seed}>"
        raise ValueError(
            f"{method['key']}: evaluation without a new training run requires checkpoints or checkpoint_glob."
        )

    pattern = pattern_template.format(seed=training_seed, training_seed=training_seed)
    pattern_path = Path(pattern).expanduser()
    if not pattern_path.is_absolute():
        pattern_path = repo_root / pattern_path
    candidates = [Path(path) for path in glob.glob(str(pattern_path))]
    if not candidates:
        if dry_run:
            return f"<latest:{pattern}>"
        raise FileNotFoundError(f"{method['key']}: no checkpoint matches {pattern_path}")
    return str(select_latest_checkpoint(candidates).resolve())


def render_output_path(
    repo_root: Path,
    method: dict[str, Any],
    evaluation_cfg: dict[str, Any],
    *,
    training_seed: int,
    evaluation_seed: int,
    test_delay: int,
) -> Path:
    output_dir = evaluation_cfg.get("output_dir")
    filename_template = evaluation_cfg.get("result_filename")
    if not isinstance(output_dir, str) or not output_dir:
        raise ValueError(f"{method['key']}: evaluation output_dir is required.")
    if not isinstance(filename_template, str) or not filename_template:
        raise ValueError(f"{method['key']}: evaluation result_filename is required.")

    filename = filename_template.format(
        method=method["key"],
        seed=evaluation_seed,
        training_seed=training_seed,
        evaluation_seed=evaluation_seed,
        test_delay=test_delay,
    )
    directory = Path(output_dir).expanduser()
    directory = directory if directory.is_absolute() else repo_root / directory
    return (directory / filename).resolve()


def format_command(command: list[str]) -> str:
    return shlex.join(command)


def run_command(command: list[str], repo_root: Path, *, dry_run: bool) -> None:
    print(f"[COMMAND] {format_command(command)}", flush=True)
    if not dry_run:
        subprocess.run(command, cwd=repo_root, check=True)


def select_methods(methods: list[dict[str, Any]], only: list[str]) -> list[dict[str, Any]]:
    known = {method["key"] for method in methods}
    unknown = set(only) - known
    if unknown:
        raise ValueError(f"Unknown method key(s): {', '.join(sorted(unknown))}")
    selected = [method for method in methods if method.get("enabled", True)]
    if only:
        selected = [method for method in selected if method["key"] in only]
    return selected


def build_analysis_command(
    repo_root: Path,
    python_executable: str,
    analysis_cfg: dict[str, Any],
) -> list[str]:
    command = [python_executable, str(repo_root / "scripts" / "analyze_delay_results.py")]
    option_names = {
        "mlp_dir": "--mlp_dir",
        "stack_dir": "--stack_dir",
        "trace_dir": "--trace_dir",
        "output_csv": "--output_csv",
        "figure_dir": "--figure_dir",
    }
    for field, option in option_names.items():
        _append_option(command, option, analysis_cfg.get(field))
    command.extend(_string_list(analysis_cfg.get("extra_args", []), "analysis extra_args"))
    return command


def main() -> int:
    args = build_parser().parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    config_path = Path(args.config).expanduser()
    config_path = config_path if config_path.is_absolute() else repo_root / config_path
    config = load_config(config_path.resolve())

    if args.list_methods:
        for method in config["methods"]:
            state = "enabled" if method.get("enabled", True) else "disabled"
            print(f"{method['key']}: {method.get('name', method['task'])} [{state}]")
        return 0

    selected = select_methods(config["methods"], args.only)
    if not selected:
        raise ValueError("No enabled experiment methods were selected.")

    common = config.get("common", {})
    common_train = common.get("training", {})
    common_evaluation = common.get("evaluation", {})
    python_executable = args.python or config.get("python") or sys.executable
    continue_on_error = args.continue_on_error or bool(config.get("continue_on_error", False))
    generated_checkpoints: dict[tuple[str, int], str] = {}
    failures: list[str] = []
    commands_run = 0
    commands_skipped = 0

    print(f"[INFO] Config: {config_path.resolve()}")
    print(f"[INFO] Methods: {', '.join(method['key'] for method in selected)}")
    print(f"[INFO] Mode: {'dry-run' if args.dry_run else 'execute'}")

    for method in selected:
        train_cfg = merge_stage_config(common_train, method.get("training", {}))
        evaluation_cfg = merge_stage_config(common_evaluation, method.get("evaluation", {}))
        training_seeds = _integer_list(train_cfg.get("seeds"), f"{method['key']} training seeds")
        train_enabled = bool(train_cfg.get("enabled", True)) and not args.skip_train
        evaluation_enabled = bool(evaluation_cfg.get("enabled", True)) and not args.skip_eval

        print(f"\n[METHOD] {method['key']} - {method.get('name', method['task'])}")

        if train_enabled:
            for training_seed in training_seeds:
                command, run_name = build_train_command(
                    repo_root,
                    python_executable,
                    method,
                    train_cfg,
                    training_seed,
                )
                try:
                    run_command(command, repo_root, dry_run=args.dry_run)
                    commands_run += 1
                    if args.dry_run:
                        generated_checkpoints[(method["key"], training_seed)] = (
                            f"<checkpoint-generated:{method['key']}:train-seed-{training_seed}>"
                        )
                    else:
                        checkpoint = find_trained_checkpoint(repo_root, method["experiment_name"], run_name)
                        generated_checkpoints[(method["key"], training_seed)] = str(checkpoint)
                        print(f"[CHECKPOINT] {checkpoint}")
                except (OSError, subprocess.CalledProcessError, ValueError) as error:
                    failure = f"training {method['key']} seed={training_seed}: {error}"
                    failures.append(failure)
                    print(f"[ERROR] {failure}")
                    if not continue_on_error:
                        raise

        if not evaluation_enabled:
            continue

        test_delays = _integer_list(
            evaluation_cfg.get("test_delays"),
            f"{method['key']} evaluation test_delays",
            non_negative=True,
        )
        evaluation_seeds = _integer_list(
            evaluation_cfg.get("seeds"),
            f"{method['key']} evaluation seeds",
        )

        for training_seed in training_seeds:
            checkpoint = generated_checkpoints.get((method["key"], training_seed))
            if checkpoint is None:
                try:
                    checkpoint = resolve_configured_checkpoint(
                        repo_root,
                        method,
                        train_cfg,
                        training_seed,
                        dry_run=args.dry_run,
                    )
                except (OSError, ValueError) as error:
                    failure = f"checkpoint {method['key']} seed={training_seed}: {error}"
                    failures.append(failure)
                    print(f"[ERROR] {failure}")
                    if not continue_on_error:
                        raise
                    continue

            for test_delay in test_delays:
                for evaluation_seed in evaluation_seeds:
                    output_path = render_output_path(
                        repo_root,
                        method,
                        evaluation_cfg,
                        training_seed=training_seed,
                        evaluation_seed=evaluation_seed,
                        test_delay=test_delay,
                    )
                    if output_path.exists() and not evaluation_cfg.get("overwrite", False):
                        print(f"[SKIP] Result already exists: {output_path}")
                        commands_skipped += 1
                        continue
                    command = build_evaluation_command(
                        repo_root,
                        python_executable,
                        method,
                        train_cfg,
                        evaluation_cfg,
                        checkpoint=checkpoint,
                        training_seed=training_seed,
                        evaluation_seed=evaluation_seed,
                        test_delay=test_delay,
                        output_path=output_path,
                    )
                    try:
                        run_command(command, repo_root, dry_run=args.dry_run)
                        commands_run += 1
                    except (OSError, subprocess.CalledProcessError) as error:
                        failure = (
                            f"evaluation {method['key']} train-seed={training_seed} "
                            f"test-delay={test_delay} eval-seed={evaluation_seed}: {error}"
                        )
                        failures.append(failure)
                        print(f"[ERROR] {failure}")
                        if not continue_on_error:
                            raise

    analysis_cfg = config.get("analysis", {})
    if analysis_cfg.get("enabled", True) and not args.skip_analysis:
        command = build_analysis_command(repo_root, python_executable, analysis_cfg)
        try:
            run_command(command, repo_root, dry_run=args.dry_run)
            commands_run += 1
        except (OSError, subprocess.CalledProcessError) as error:
            failure = f"analysis: {error}"
            failures.append(failure)
            print(f"[ERROR] {failure}")
            if not continue_on_error:
                raise

    print(f"\n[SUMMARY] commands={commands_run} skipped={commands_skipped} failures={len(failures)}")
    for failure in failures:
        print(f"  - {failure}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
