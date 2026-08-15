"""Unit tests for the pure PyTorch exponential memory trace."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch

module_path = Path(__file__).parents[1] / "source" / "memory_trace_delay" / "memory_trace_delay" / "memory_trace.py"
spec = importlib.util.spec_from_file_location("memory_trace", module_path)
assert spec is not None and spec.loader is not None
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
MemoryTraceBuffer = module.MemoryTraceBuffer


def test_decay_zero_is_exact_identity() -> None:
    buffer = MemoryTraceBuffer(num_envs=2, obs_dim=2, decay=0.0)
    observations = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    assert torch.equal(buffer.push(observations), observations)


def test_exponential_trace_values() -> None:
    buffer = MemoryTraceBuffer(num_envs=1, obs_dim=1, decay=0.5)
    first = buffer.push(torch.tensor([[10.0]]))
    second = buffer.push(torch.tensor([[6.0]]))
    assert torch.equal(first, torch.tensor([[5.0]]))
    assert torch.equal(second, torch.tensor([[5.5]]))


def test_each_environment_has_independent_trace() -> None:
    buffer = MemoryTraceBuffer(num_envs=2, obs_dim=1, decay=0.5)
    buffer.push(torch.tensor([[10.0], [100.0]]))
    output = buffer.push(torch.tensor([[20.0], [200.0]]))
    assert torch.equal(output, torch.tensor([[12.5], [125.0]]))


def test_partial_reset_clears_only_selected_environment() -> None:
    buffer = MemoryTraceBuffer(num_envs=2, obs_dim=1, decay=0.5)
    buffer.push(torch.tensor([[10.0], [100.0]]))
    buffer.reset(env_ids=[0])
    output = buffer.push(torch.tensor([[2.0], [20.0]]))
    assert torch.equal(output, torch.tensor([[1.0], [35.0]]))


def test_boolean_reset_mask_is_supported() -> None:
    buffer = MemoryTraceBuffer(num_envs=3, obs_dim=1, decay=0.5)
    buffer.push(torch.tensor([[2.0], [4.0], [6.0]]))
    buffer.reset(env_ids=torch.tensor([True, False, True]))
    assert torch.equal(buffer.trace, torch.tensor([[0.0], [2.0], [0.0]]))


def test_returned_trace_does_not_alias_internal_state() -> None:
    buffer = MemoryTraceBuffer(num_envs=1, obs_dim=1, decay=0.5)
    output = buffer.push(torch.tensor([[4.0]]))
    output.zero_()
    assert torch.equal(buffer.trace, torch.tensor([[2.0]]))


def test_invalid_configuration_and_shape_raise() -> None:
    with pytest.raises(ValueError):
        MemoryTraceBuffer(num_envs=0, obs_dim=1, decay=0.5)
    with pytest.raises(ValueError):
        MemoryTraceBuffer(num_envs=1, obs_dim=0, decay=0.5)
    with pytest.raises(ValueError):
        MemoryTraceBuffer(num_envs=1, obs_dim=1, decay=1.0)
    with pytest.raises(TypeError):
        MemoryTraceBuffer(num_envs=1, obs_dim=1, decay=0.5, dtype=torch.int64)

    buffer = MemoryTraceBuffer(num_envs=2, obs_dim=3, decay=0.5)
    with pytest.raises(ValueError):
        buffer.push(torch.zeros(2, 4))
    with pytest.raises(ValueError):
        buffer.push(torch.zeros(2, 3, dtype=torch.float64))


def test_output_does_not_track_gradients() -> None:
    buffer = MemoryTraceBuffer(num_envs=1, obs_dim=1, decay=0.5)
    observations = torch.tensor([[2.0]], requires_grad=True)
    assert not buffer.push(observations).requires_grad


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_cuda_trace_stays_on_gpu() -> None:
    buffer = MemoryTraceBuffer(num_envs=2, obs_dim=2, decay=0.5, device="cuda")
    observations = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device="cuda")
    output = buffer.push(observations)
    assert output.device.type == "cuda"
    assert buffer.trace.device.type == "cuda"
