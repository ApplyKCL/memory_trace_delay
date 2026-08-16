"""Training-time validation and best-checkpoint tracking for RSL-RL runs."""

from __future__ import annotations

import csv
import json
import os
import shutil
import subprocess
import time
from collections import defaultdict
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean
from typing import Any

SELECTION_RULE = "max delay_auc, max worst_return, min failure, latest iteration"


@dataclass(frozen=True)
class ValidationSettings:
    """Configuration shared by every candidate checkpoint in one training run."""

    method_key: str
    task: str
    training_seed: int
    train_delay_min: int
    train_delay_max: int
    evaluation_seeds: tuple[int, ...]
    test_delays: tuple[int, ...]
    interval: int
    start_iteration: int
    num_envs: int
    num_episodes: int
    device: str
    lock_path: Path
    extra_args: tuple[str, ...] = ()

    def validate(self) -> None:
        if not self.method_key or not self.task:
            raise ValueError("method_key and task are required for training-time validation.")
        if self.training_seed < 0:
            raise ValueError("training_seed must be non-negative.")
        if self.train_delay_min < 0 or self.train_delay_max < self.train_delay_min:
            raise ValueError("The training delay range is invalid.")
        if not self.evaluation_seeds or any(seed < 0 for seed in self.evaluation_seeds):
            raise ValueError("evaluation_seeds must contain non-negative integers.")
        if len(set(self.evaluation_seeds)) != len(self.evaluation_seeds):
            raise ValueError("evaluation_seeds must not contain duplicates.")
        if not self.test_delays or any(delay < 0 for delay in self.test_delays):
            raise ValueError("test_delays must contain non-negative integers.")
        if len(set(self.test_delays)) != len(self.test_delays):
            raise ValueError("test_delays must not contain duplicates.")
        if self.interval <= 0 or self.start_iteration < 0:
            raise ValueError("interval must be positive and start_iteration must be non-negative.")
        if self.num_envs <= 0 or self.num_episodes <= 0:
            raise ValueError("num_envs and num_episodes must be positive.")
        if not self.device:
            raise ValueError("A validation device is required.")


def should_validate(iteration: int, *, final_iteration: int, interval: int, start_iteration: int) -> bool:
    """Return whether a saved checkpoint belongs to the predeclared validation grid."""
    if iteration < 0 or final_iteration < 0:
        raise ValueError("iteration values must be non-negative.")
    if interval <= 0 or start_iteration < 0:
        raise ValueError("interval must be positive and start_iteration must be non-negative.")
    if iteration == final_iteration:
        return True
    return iteration >= start_iteration and iteration % interval == 0


def normalized_auc(points: list[tuple[int, float]]) -> float:
    """Compute trapezoidal delay-AUC normalized by the evaluated delay range."""
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


def summarize_results(
    settings: ValidationSettings,
    checkpoint: Path,
    iteration: int,
    results: list[dict[str, Any]],
) -> dict[str, Any]:
    """Average evaluation seeds within each delay and score one checkpoint."""
    if not results:
        raise ValueError(f"No validation results were produced for {checkpoint}.")
    returns_by_delay: dict[int, list[float]] = defaultdict(list)
    failures_by_delay: dict[int, list[float]] = defaultdict(list)
    seen_conditions = set()
    for result in results:
        delay = int(result["test_delay_steps"])
        evaluation_seed = result.get("evaluation_seed")
        if evaluation_seed is None:
            evaluation_seed = result["seed"]
        condition = (delay, int(evaluation_seed))
        if condition in seen_conditions:
            raise ValueError(f"Duplicate validation condition: delay={delay}, seed={evaluation_seed}.")
        seen_conditions.add(condition)
        returns_by_delay[delay].append(float(result["mean_return"]))
        failures_by_delay[delay].append(float(result["failure_rate"]))

    expected_conditions = {(delay, seed) for delay in settings.test_delays for seed in settings.evaluation_seeds}
    if seen_conditions != expected_conditions:
        missing = sorted(expected_conditions - seen_conditions)
        extra = sorted(seen_conditions - expected_conditions)
        raise ValueError(f"Incomplete validation grid. Missing={missing}, extra={extra}.")

    return_points = [(delay, fmean(values)) for delay, values in sorted(returns_by_delay.items())]
    return {
        "method_key": settings.method_key,
        "training_seed": settings.training_seed,
        "checkpoint": str(checkpoint.resolve()),
        "iteration": iteration,
        "delay_auc": normalized_auc(return_points),
        "worst_return": min(value for _, value in return_points),
        "mean_failure_rate": fmean(fmean(values) for values in failures_by_delay.values()),
        "validation_seeds": list(settings.evaluation_seeds),
        "test_delays": list(settings.test_delays),
        "num_envs": settings.num_envs,
        "num_episodes": settings.num_episodes,
        "selection_rule": SELECTION_RULE,
    }


def selection_key(summary: dict[str, Any]) -> tuple[float, float, float, int]:
    return (
        float(summary["delay_auc"]),
        float(summary["worst_return"]),
        -float(summary["mean_failure_rate"]),
        int(summary["iteration"]),
    )


def _atomic_json(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)
    os.replace(temporary, path)


def _atomic_copy(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp-{os.getpid()}")
    shutil.copy2(source, temporary)
    os.replace(temporary, target)


@contextmanager
def exclusive_file_lock(path: Path) -> Iterator[None]:
    """Serialize validation processes that share one GPU."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as lock_file:
        if os.name == "nt":
            import msvcrt

            if lock_file.tell() == 0:
                lock_file.write(b"\0")
                lock_file.flush()
            lock_file.seek(0)
            msvcrt.locking(lock_file.fileno(), msvcrt.LK_LOCK, 1)
        else:
            import fcntl

            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            if os.name == "nt":
                lock_file.seek(0)
                msvcrt.locking(lock_file.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


class TrainingValidator:
    """Evaluate saved checkpoints during learning and maintain ``best_model.pt``."""

    def __init__(
        self,
        settings: ValidationSettings,
        *,
        run_dir: Path,
        python_executable: str,
        evaluator_script: Path,
        command_runner: Callable[[list[str]], None] | None = None,
    ) -> None:
        settings.validate()
        self.settings = settings
        self.run_dir = run_dir.resolve()
        self.python_executable = python_executable
        self.evaluator_script = evaluator_script.resolve()
        self.repo_root = self.evaluator_script.parents[2]
        self.command_runner = command_runner or self._run_subprocess
        self.validation_dir = self.run_dir / "validation"
        self.manifest_path = self.run_dir / "best_checkpoint.json"
        self.best_model_path = self.run_dir / "best_model.pt"
        self._validated_iterations: set[int] = set()

    def _run_subprocess(self, command: list[str]) -> None:
        subprocess.run(command, cwd=self.repo_root, check=True)

    def _result_path(self, iteration: int, delay: int, evaluation_seed: int) -> Path:
        return self.validation_dir / f"iter_{iteration:04d}" / (f"delay_{delay}_valseed_{evaluation_seed}.json")

    def _build_command(
        self,
        checkpoint: Path,
        evaluation_seed: int,
        output_template: Path,
    ) -> list[str]:
        settings = self.settings
        command = [
            self.python_executable,
            str(self.evaluator_script),
            "--task",
            settings.task,
            "--checkpoint",
            str(checkpoint),
            "--seed",
            str(evaluation_seed),
            "--method",
            settings.method_key,
            "--training_seed",
            str(settings.training_seed),
            "--train_delay_min",
            str(settings.train_delay_min),
            "--train_delay_max",
            str(settings.train_delay_max),
            "--num_envs",
            str(settings.num_envs),
            "--num_episodes",
            str(settings.num_episodes),
            "--device",
            settings.device,
            "--output",
            str(output_template),
            "--headless",
            *settings.extra_args,
        ]
        for delay in settings.test_delays:
            command.extend(("--test_delay", str(delay)))
        return command

    @staticmethod
    def _read_json(path: Path) -> dict[str, Any]:
        with path.open("r", encoding="utf-8") as file:
            return json.load(file)

    def _current_best(self) -> dict[str, Any] | None:
        return self._read_json(self.manifest_path) if self.manifest_path.is_file() else None

    def _write_summary_csv(self, current_best_iteration: int) -> None:
        rows = []
        for path in sorted(self.validation_dir.glob("iter_*/summary.json")):
            summary = self._read_json(path)
            summary["is_current_best"] = int(summary["iteration"]) == current_best_iteration
            rows.append(summary)
        fieldnames = [
            "method_key",
            "training_seed",
            "checkpoint",
            "iteration",
            "delay_auc",
            "worst_return",
            "mean_failure_rate",
            "validation_wall_time_seconds",
            "is_current_best",
            "validation_seeds",
            "test_delays",
            "num_envs",
            "num_episodes",
            "selection_rule",
        ]
        output_path = self.validation_dir / "checkpoint_summary.csv"
        with output_path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows({field: row[field] for field in fieldnames} for row in rows)

    def validate_if_due(
        self,
        checkpoint: Path,
        *,
        iteration: int,
        final_iteration: int,
    ) -> dict[str, Any] | None:
        """Validate one due checkpoint, update the winner, and return its summary."""
        if iteration in self._validated_iterations or not should_validate(
            iteration,
            final_iteration=final_iteration,
            interval=self.settings.interval,
            start_iteration=self.settings.start_iteration,
        ):
            return None
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Saved checkpoint does not exist: {checkpoint}")

        validation_started = time.perf_counter()
        with exclusive_file_lock(self.settings.lock_path):
            results = []
            for evaluation_seed in self.settings.evaluation_seeds:
                output_paths = [
                    self._result_path(iteration, delay, evaluation_seed)
                    for delay in self.settings.test_delays
                ]
                if any(not output_path.is_file() for output_path in output_paths):
                    output_template = self._result_path(iteration, -1, evaluation_seed)
                    output_template = Path(str(output_template).replace("delay_-1", "delay_{test_delay}"))
                    output_template.parent.mkdir(parents=True, exist_ok=True)
                    command = self._build_command(checkpoint, evaluation_seed, output_template)
                    print(
                        f"[VALIDATION] iteration={iteration} delays={self.settings.test_delays} "
                        f"seed={evaluation_seed}",
                        flush=True,
                    )
                    self.command_runner(command)
                for output_path in output_paths:
                    results.append(self._read_json(output_path))

            summary = summarize_results(self.settings, checkpoint, iteration, results)
            summary["validation_wall_time_seconds"] = time.perf_counter() - validation_started
            current_best = self._current_best()
            became_best = current_best is None or selection_key(summary) > selection_key(current_best)
            summary["became_best"] = became_best
            _atomic_json(summary, self.validation_dir / f"iter_{iteration:04d}" / "summary.json")

            if became_best:
                manifest = {
                    **summary,
                    "source_checkpoint": summary["checkpoint"],
                    "best_model": str(self.best_model_path),
                }
                _atomic_copy(checkpoint, self.best_model_path)
                _atomic_json(manifest, self.manifest_path)
                current_best = manifest
                print(
                    f"[VALIDATION] New best checkpoint: iteration={iteration} Delay-AUC={summary['delay_auc']:.4f}",
                    flush=True,
                )

            assert current_best is not None
            self._write_summary_csv(int(current_best["iteration"]))
            self._validated_iterations.add(iteration)
            return summary
