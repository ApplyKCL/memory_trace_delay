"""Unit tests for the JSON-driven experiment runner."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

module_path = Path(__file__).parents[1] / "scripts" / "run_experiments.py"
spec = importlib.util.spec_from_file_location("run_experiments", module_path)
assert spec is not None and spec.loader is not None
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
repo_root = Path(__file__).parents[1]


def sample_method() -> dict:
    return {
        "key": "trace08",
        "task": "MemoryTrace-Trace-Velocity-Flat-Unitree-Go2-v0",
        "experiment_name": "trace-test",
    }


def test_project_experiment_config_is_valid() -> None:
    config = module.load_config(repo_root / "configs" / "experiments" / "all_algorithms.json")
    assert [method["key"] for method in config["methods"]] == [
        "train0",
        "train2",
        "train0to4",
        "stack5",
        "trace08",
    ]


def test_merge_concatenates_common_and_method_extra_args() -> None:
    merged = module.merge_stage_config(
        {"num_envs": 64, "device": "cuda:0", "extra_args": ["--logger", "tensorboard"]},
        {"device": "cuda:1", "extra_args": ["agent.policy.test=true"]},
    )
    assert merged["num_envs"] == 64
    assert merged["device"] == "cuda:1"
    assert merged["extra_args"] == ["--logger", "tensorboard", "agent.policy.test=true"]


def test_train_command_contains_structured_and_special_configuration(tmp_path: Path) -> None:
    train_cfg = {
        "num_envs": 128,
        "max_iterations": 2,
        "delay_min": 0,
        "delay_max": 4,
        "device": "cuda:2",
        "headless": True,
        "run_name": "trace_seed{training_seed}",
        "extra_args": ["--logger", "tensorboard", "env.custom=true"],
    }
    command, run_name = module.build_train_command(tmp_path, "python", sample_method(), train_cfg, 43)

    assert run_name == "trace_seed43"
    assert command[command.index("--device") + 1] == "cuda:2"
    assert command[command.index("--train_delay_max") + 1] == "4"
    assert "--headless" in command
    assert command[-3:] == ["--logger", "tensorboard", "env.custom=true"]


def test_output_filename_keeps_training_and_evaluation_seeds(tmp_path: Path) -> None:
    evaluation_cfg = {
        "output_dir": "results/trace",
        "result_filename": (
            "trace08_train0to4_trainseed{training_seed}_test{test_delay}_"
            "evalseed{evaluation_seed}.json"
        ),
    }
    output = module.render_output_path(
        tmp_path,
        sample_method(),
        evaluation_cfg,
        training_seed=43,
        evaluation_seed=124,
        test_delay=6,
    )
    assert output.name == "trace08_train0to4_trainseed43_test6_evalseed124.json"


def test_latest_checkpoint_uses_run_then_iteration(tmp_path: Path) -> None:
    older_run = tmp_path / "2026-08-01_trace"
    newer_run = tmp_path / "2026-08-02_trace"
    older_run.mkdir()
    newer_run.mkdir()
    older = older_run / "model_999.pt"
    newer_low = newer_run / "model_10.pt"
    newer_high = newer_run / "model_20.pt"
    for path in (older, newer_low, newer_high):
        path.touch()

    assert module.select_latest_checkpoint([older, newer_low, newer_high]) == newer_high


def test_invalid_delay_range_is_rejected(tmp_path: Path) -> None:
    train_cfg = {
        "num_envs": 1,
        "max_iterations": 1,
        "delay_min": 4,
        "delay_max": 2,
        "run_name": "invalid",
        "extra_args": [],
    }
    with pytest.raises(ValueError, match="delay_max"):
        module.build_train_command(tmp_path, "python", sample_method(), train_cfg, 42)
