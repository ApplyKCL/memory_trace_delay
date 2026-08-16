"""Unit tests for training-time checkpoint validation and best-model tracking."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

module_path = Path(__file__).parents[1] / "scripts" / "rsl_rl" / "training_validation.py"
spec = importlib.util.spec_from_file_location("training_validation", module_path)
assert spec is not None and spec.loader is not None
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)


def make_settings(tmp_path: Path) -> object:
    return module.ValidationSettings(
        method_key="trace08",
        task="MemoryTrace-Trace-Velocity-Flat-Unitree-Go2-v0",
        training_seed=42,
        train_delay_min=0,
        train_delay_max=4,
        evaluation_seeds=(901,),
        test_delays=(0, 2),
        interval=100,
        start_iteration=100,
        num_envs=8,
        num_episodes=16,
        device="cuda:3",
        lock_path=tmp_path / "validation.lock",
    )


def test_validation_schedule_includes_periodic_and_final_checkpoints() -> None:
    assert not module.should_validate(50, final_iteration=499, interval=100, start_iteration=100)
    assert module.should_validate(100, final_iteration=499, interval=100, start_iteration=100)
    assert module.should_validate(400, final_iteration=499, interval=100, start_iteration=100)
    assert module.should_validate(499, final_iteration=499, interval=100, start_iteration=100)


def test_summary_requires_complete_seed_delay_grid(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    with pytest.raises(ValueError, match="Incomplete validation grid"):
        module.summarize_results(
            settings,
            tmp_path / "model_100.pt",
            100,
            [{"test_delay_steps": 0, "evaluation_seed": 901, "mean_return": 10.0, "failure_rate": 0.0}],
        )


def test_validator_updates_best_model_during_training(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    commands = []

    def fake_command_runner(command: list[str]) -> None:
        commands.append(command)
        output_template = command[command.index("--output") + 1]
        delays = [int(command[index + 1]) for index, value in enumerate(command) if value == "--test_delay"]
        seed = int(command[command.index("--seed") + 1])
        checkpoint = Path(command[command.index("--checkpoint") + 1])
        iteration = int(checkpoint.stem.removeprefix("model_"))
        for delay in delays:
            output = Path(output_template.format(test_delay=delay))
            output.write_text(
                json.dumps(
                    {
                        "test_delay_steps": delay,
                        "evaluation_seed": seed,
                        "mean_return": float(iteration / 10 - delay),
                        "failure_rate": float(delay / 10),
                    }
                ),
                encoding="utf-8",
            )

    validator = module.TrainingValidator(
        settings,
        run_dir=run_dir,
        python_executable="python",
        evaluator_script=Path(__file__).parents[1] / "scripts" / "rsl_rl" / "evaluate_delay.py",
        command_runner=fake_command_runner,
    )
    checkpoint_100 = run_dir / "model_100.pt"
    checkpoint_200 = run_dir / "model_200.pt"
    checkpoint_100.write_bytes(b"checkpoint-100")
    checkpoint_200.write_bytes(b"checkpoint-200")

    first = validator.validate_if_due(checkpoint_100, iteration=100, final_iteration=499)
    second = validator.validate_if_due(checkpoint_200, iteration=200, final_iteration=499)

    assert first is not None and first["became_best"]
    assert second is not None and second["became_best"]
    assert second["validation_wall_time_seconds"] >= 0.0
    assert len(commands) == 2
    assert (run_dir / "best_model.pt").read_bytes() == b"checkpoint-200"
    manifest = json.loads((run_dir / "best_checkpoint.json").read_text(encoding="utf-8"))
    assert manifest["iteration"] == 200
    assert manifest["selection_rule"] == module.SELECTION_RULE
    assert (run_dir / "validation" / "checkpoint_summary.csv").is_file()


def test_non_due_checkpoint_does_not_launch_validation(tmp_path: Path) -> None:
    calls = []
    validator = module.TrainingValidator(
        make_settings(tmp_path),
        run_dir=tmp_path,
        python_executable="python",
        evaluator_script=Path(__file__).parents[1] / "scripts" / "rsl_rl" / "evaluate_delay.py",
        command_runner=calls.append,
    )
    checkpoint = tmp_path / "model_50.pt"
    checkpoint.touch()

    assert validator.validate_if_due(checkpoint, iteration=50, final_iteration=499) is None
    assert not calls
