"""Unit tests for the JSON-driven experiment runner."""

from __future__ import annotations

import importlib.util
import time
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
        "stack5_train0",
        "stack5_train2",
        "stack5",
        "stack2",
        "stack8",
        "trace08_train0",
        "trace08_train2",
        "trace08",
        "trace05",
        "trace0875",
    ]

    assert module.load_execution_settings(config) == module.ExecutionSettings((0, 1, 2), 3, (0, 1, 2, 3), 3, 4)

    assert config["common"]["training"]["seeds"] == [42, 43, 44]
    assert config["common"]["training"]["max_iterations"] == 500
    assert config["common"]["training"]["validation"] == {
        "enabled": True,
        "seeds": [901, 902],
        "test_delays": [0, 2, 4, 6],
        "interval": 100,
        "start_iteration": 100,
        "num_envs": 256,
        "num_episodes": 256,
        "lock_file": "logs/rsl_rl/.locks/training_validation_gpu3.lock",
    }
    assert config["common"]["evaluation"]["training_seeds"] == [42, 43, 44]
    assert config["common"]["evaluation"]["seeds"] == [123, 124, 125]
    assert config["common"]["evaluation"]["test_delays"] == list(range(7))

    methods = {method["key"]: method for method in config["methods"]}
    expected_fixed_delays = {
        "train0": (0, 0),
        "train2": (2, 2),
        "stack5_train0": (0, 0),
        "stack5_train2": (2, 2),
        "trace08_train0": (0, 0),
        "trace08_train2": (2, 2),
    }
    for key, delay_range in expected_fixed_delays.items():
        training = methods[key]["training"]
        assert (training["delay_min"], training["delay_max"]) == delay_range
        assert training["seeds"] == [42, 43, 44]
        assert training["enabled"]

    for key in ("train0to4", "stack5", "trace08"):
        assert methods[key]["training"]["seeds"] == [42, 43, 44, 45, 46]
        assert methods[key]["evaluation"]["training_seeds"] == [42, 43, 44, 45, 46]

    ablation_overrides = {
        "stack2": "env.observations.policy.history_length=2",
        "stack8": "env.observations.policy.history_length=8",
        "trace05": "env.memory_trace.decay=0.5",
        "trace0875": "env.memory_trace.decay=0.875",
    }
    for key, override in ablation_overrides.items():
        assert methods[key]["training"]["seeds"] == [42, 43, 44]
        assert methods[key]["training"]["extra_args"] == [override]
        assert methods[key]["evaluation"]["extra_args"] == [override]


def test_duplicate_gpu_is_rejected() -> None:
    with pytest.raises(ValueError, match="duplicates"):
        module.load_execution_settings({"execution": {"gpus": [0, 0]}})


def test_validation_gpu_must_not_overlap_training_pool() -> None:
    with pytest.raises(ValueError, match="reserved"):
        module.load_execution_settings(
            {"execution": {"training_gpus": [0, 1], "validation_gpu": 1, "evaluation_gpus": [0, 1]}}
        )


def test_scheduler_replaces_device_without_mutating_command() -> None:
    command = ("python", "train.py", "--device", "cuda:0", "--headless")
    assigned = module.command_for_gpu(command, 3)
    assert command[3] == "cuda:0"
    assert assigned[3] == "cuda:3"


def test_dry_run_assigns_jobs_round_robin_to_worker_gpus(tmp_path: Path) -> None:
    jobs = [
        module.CommandJob(f"train:trace08:seed-{seed}", ("python", "train.py", "--device", "cuda:0"))
        for seed in (42, 43, 44)
    ]
    results = module.run_job_batch(
        jobs,
        tmp_path,
        gpus=(0, 1, 2, 3),
        workers=2,
        dry_run=True,
        continue_on_error=False,
    )
    assert [result.gpu for result in results] == [0, 1, 0]
    assert [result.command[-1] for result in results] == ["cuda:0", "cuda:1", "cuda:0"]


def test_parallel_workers_keep_one_active_job_per_gpu(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    active_gpus: set[str] = set()
    seen_gpus: set[str] = set()
    overlapping_gpus: set[str] = set()
    lock = module.threading.Lock()

    def fake_run(command: tuple[str, ...], _repo_root: Path, *, dry_run: bool) -> None:
        assert not dry_run
        device = command[command.index("--device") + 1]
        with lock:
            if device in active_gpus:
                overlapping_gpus.add(device)
            active_gpus.add(device)
            seen_gpus.add(device)
        time.sleep(0.01)
        with lock:
            active_gpus.remove(device)

    monkeypatch.setattr(module, "run_command", fake_run)
    jobs = [module.CommandJob(f"job-{index}", ("python", "train.py", "--device", "cuda:0")) for index in range(8)]
    results = module.run_job_batch(
        jobs,
        tmp_path,
        gpus=(0, 1, 2, 3),
        workers=4,
        dry_run=False,
        continue_on_error=False,
    )

    assert len(results) == 8
    assert seen_gpus == {"cuda:0", "cuda:1", "cuda:2", "cuda:3"}
    assert not overlapping_gpus


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


def test_train_command_enables_online_validation(tmp_path: Path) -> None:
    train_cfg = {
        "num_envs": 128,
        "max_iterations": 500,
        "delay_min": 0,
        "delay_max": 4,
        "device": "cuda:0",
        "headless": True,
        "run_name": "trace_seed{training_seed}",
        "validation": {
            "enabled": True,
            "seeds": [901, 902],
            "test_delays": [0, 2, 4, 6],
            "interval": 100,
            "start_iteration": 100,
            "num_envs": 256,
            "num_episodes": 256,
            "device": "cuda:3",
            "lock_file": "logs/validation.lock",
        },
        "extra_args": [],
    }

    command, _ = module.build_train_command(tmp_path, "python", sample_method(), train_cfg, 42)

    assert "--training_validation" in command
    assert command[command.index("--validation_method") + 1] == "trace08"
    assert command[command.index("--validation_device") + 1] == "cuda:3"
    assert [command[index + 1] for index, value in enumerate(command) if value == "--validation_seed"] == [
        "901",
        "902",
    ]


def test_output_filename_keeps_training_and_evaluation_seeds(tmp_path: Path) -> None:
    evaluation_cfg = {
        "output_dir": "results/trace",
        "result_filename": (
            "trace08_train0to4_trainseed{training_seed}_test{test_delay}_evalseed{evaluation_seed}.json"
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


def test_best_checkpoint_uses_newest_completed_validation_run(tmp_path: Path) -> None:
    experiment = tmp_path / "logs" / "rsl_rl" / "trace-test"
    old_run = experiment / "2026-08-01_trace_seed42"
    new_run = experiment / "2026-08-02_trace_seed42"
    for run in (old_run, new_run):
        run.mkdir(parents=True)
        (run / "best_model.pt").touch()
        (run / "best_checkpoint.json").write_text("{}", encoding="utf-8")

    checkpoint = module.find_best_checkpoint(tmp_path, "trace-test", "trace_seed42")

    assert checkpoint == (new_run / "best_model.pt").resolve()


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


def test_analysis_command_includes_engineering_inputs(tmp_path: Path) -> None:
    command = module.build_analysis_command(
        tmp_path,
        "python",
        {
            "benchmark_dir": "results/benchmarks",
            "benchmark_backend": "torchscript",
            "benchmark_device": "cpu",
            "engineering_csv": "results/engineering_summary.csv",
        },
    )

    assert command[command.index("--benchmark_dir") + 1] == "results/benchmarks"
    assert command[command.index("--benchmark_backend") + 1] == "torchscript"
    assert command[command.index("--benchmark_device") + 1] == "cpu"
    assert command[command.index("--engineering_csv") + 1] == "results/engineering_summary.csv"
