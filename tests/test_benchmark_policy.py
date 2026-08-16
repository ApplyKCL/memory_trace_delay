"""Unit tests for exported-policy parameter and latency benchmarking."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch

module_path = Path(__file__).parents[1] / "scripts" / "benchmark_policy.py"
spec = importlib.util.spec_from_file_location("benchmark_policy", module_path)
assert spec is not None and spec.loader is not None
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


@pytest.mark.parametrize(
    ("method_key", "expected"),
    [
        ("train0to4", ("none", None)),
        ("trace08", ("trace", None)),
        ("stack5", ("stack", 5)),
    ],
)
def test_memory_mode_is_inferred_from_method_key(method_key: str, expected: tuple) -> None:
    assert module.resolve_memory(method_key, "auto", None) == expected


def test_expected_input_dimensions_match_policy_memory() -> None:
    assert module.expected_input_dim("none", 48, None) == 48
    assert module.expected_input_dim("trace", 48, None) == 96
    assert module.expected_input_dim("stack", 48, 5) == 240


def test_backend_is_inferred_from_export_extension() -> None:
    assert module.resolve_backend(Path("policy.pt"), "auto") == "torchscript"
    assert module.resolve_backend(Path("policy.onnx"), "auto") == "onnx"


def test_torchscript_benchmark_counts_parameters_and_runs(tmp_path: Path) -> None:
    model = torch.nn.Sequential(torch.nn.Linear(4, 3), torch.nn.Tanh(), torch.nn.Linear(3, 2)).eval()
    exported = torch.jit.trace(model, torch.zeros((1, 4)))
    model_path = tmp_path / "policy.pt"
    torch.jit.save(exported, model_path)

    result = module.benchmark_torchscript(
        model_path,
        device="cpu",
        input_dim=4,
        memory_mode="none",
        base_obs_dim=4,
        history_length=None,
        trace_decay=0.8,
        warmup=1,
        repetitions=5,
    )

    assert result["parameter_count"] == 23
    assert result["macs_per_inference"] == 18
    assert result["input_shape"] == [1, 4]
    assert result["model_latency"]["p95_ms"] >= 0.0
    assert result["end_to_end_latency"]["p95_ms"] >= 0.0


def test_latency_arguments_are_validated() -> None:
    with pytest.raises(ValueError, match="repetitions"):
        module.time_callable(lambda: None, warmup=0, repetitions=0)
