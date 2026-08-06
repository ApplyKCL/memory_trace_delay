"""Unit tests for the pure PyTorch action-delay FIFO."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch

module_path = Path(__file__).parents[1] / "source" / "memory_trace_delay" / "memory_trace_delay" / "action_delay.py"
spec = importlib.util.spec_from_file_location("action_delay", module_path)
assert spec is not None and spec.loader is not None
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
ActionDelayBuffer = module.ActionDelayBuffer


def _actions(*values: float, device: str = "cpu") -> torch.Tensor:
    return torch.tensor(values, device=device, dtype=torch.float32).unsqueeze(-1)


def test_delay_zero_is_exact_identity() -> None:
    buffer = ActionDelayBuffer(num_envs=3, action_dim=2, min_delay=0, max_delay=0)
    buffer.reset()
    actions = torch.tensor([[1.0, -1.0], [2.0, -2.0], [3.0, -3.0]])
    assert torch.equal(buffer.push(actions), actions)


def test_delay_two_zero_fills_then_returns_fifo_values() -> None:
    buffer = ActionDelayBuffer(num_envs=1, action_dim=1, min_delay=2, max_delay=2)
    buffer.reset()
    outputs = [buffer.push(_actions(value)).item() for value in (10.0, 20.0, 30.0, 40.0)]
    assert outputs == [0.0, 0.0, 10.0, 20.0]


def test_each_environment_uses_its_own_delay() -> None:
    buffer = ActionDelayBuffer(num_envs=3, action_dim=1, min_delay=0, max_delay=2)
    buffer.reset(delay_steps=[0, 1, 2])
    assert torch.equal(buffer.push(_actions(10.0, 20.0, 30.0)), _actions(10.0, 0.0, 0.0))
    assert torch.equal(buffer.push(_actions(11.0, 21.0, 31.0)), _actions(11.0, 20.0, 0.0))
    assert torch.equal(buffer.push(_actions(12.0, 22.0, 32.0)), _actions(12.0, 21.0, 30.0))


def test_partial_reset_clears_only_selected_environment() -> None:
    buffer = ActionDelayBuffer(num_envs=2, action_dim=1, min_delay=1, max_delay=2)
    buffer.reset(delay_steps=[1, 1])
    buffer.push(_actions(10.0, 20.0))
    buffer.push(_actions(11.0, 21.0))
    buffer.reset(env_ids=[0], delay_steps=2)
    assert torch.equal(buffer.push(_actions(12.0, 22.0)), _actions(0.0, 21.0))
    assert torch.equal(buffer.delay_steps, torch.tensor([2, 1]))


def test_delay_is_episode_fixed_and_sampled_inclusively() -> None:
    torch.manual_seed(7)
    buffer = ActionDelayBuffer(num_envs=64, action_dim=1, min_delay=0, max_delay=3)
    sampled = buffer.reset()
    for value in range(5):
        buffer.push(torch.full((64, 1), float(value)))
    assert torch.all((sampled >= 0) & (sampled <= 3))
    assert set(sampled.tolist()) == {0, 1, 2, 3}
    assert torch.equal(buffer.delay_steps, sampled)


def test_fixed_zero_delay_does_not_consume_rng_state() -> None:
    torch.manual_seed(1234)
    expected = torch.rand(5)
    torch.manual_seed(1234)
    buffer = ActionDelayBuffer(num_envs=4, action_dim=2, min_delay=0, max_delay=0)
    buffer.reset()
    assert torch.equal(torch.rand(5), expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_cuda_input_storage_and_output_stay_on_gpu() -> None:
    buffer = ActionDelayBuffer(num_envs=2, action_dim=2, min_delay=0, max_delay=1, device="cuda")
    buffer.reset(delay_steps=[0, 1])
    output = buffer.push(torch.tensor([[1.0, 2.0], [3.0, 4.0]], device="cuda"))
    assert output.device.type == "cuda"
    assert buffer.delay_steps.device.type == "cuda"
    assert torch.equal(output, torch.tensor([[1.0, 2.0], [0.0, 0.0]], device="cuda"))
