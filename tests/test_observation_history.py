"""Unit tests for the pure PyTorch observation-history buffer."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch


module_path = (
    Path(__file__).parents[1]
    / "source"
    / "memory_trace_delay"
    / "memory_trace_delay"
    / "observation_history.py"
)

spec = importlib.util.spec_from_file_location(
    "observation_history",
    module_path,
)

assert spec is not None
assert spec.loader is not None

module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

ObservationHistoryBuffer = module.ObservationHistoryBuffer


def _obs(*values: float, device: str = "cpu") -> torch.Tensor:
    return torch.tensor(
        values,
        dtype=torch.float32,
        device=device,
    ).unsqueeze(-1)


def test_history_length_one_is_identity() -> None:
    buffer = ObservationHistoryBuffer(
        num_envs=2,
        obs_dim=2,
        history_length=1,
    )

    observations = torch.tensor(
        [
            [1.0, 2.0],
            [3.0, 4.0],
        ]
    )

    stacked = buffer.push(observations)

    assert torch.equal(
        stacked,
        observations,
    )


def test_history_is_zero_filled_then_newest_first() -> None:
    buffer = ObservationHistoryBuffer(
        num_envs=1,
        obs_dim=1,
        history_length=3,
    )

    buffer.reset()

    output0 = buffer.push(_obs(10.0))
    output1 = buffer.push(_obs(20.0))
    output2 = buffer.push(_obs(30.0))
    output3 = buffer.push(_obs(40.0))

    assert torch.equal(
        output0,
        torch.tensor([[10.0, 0.0, 0.0]]),
    )

    assert torch.equal(
        output1,
        torch.tensor([[20.0, 10.0, 0.0]]),
    )

    assert torch.equal(
        output2,
        torch.tensor([[30.0, 20.0, 10.0]]),
    )

    assert torch.equal(
        output3,
        torch.tensor([[40.0, 30.0, 20.0]]),
    )


def test_each_environment_has_independent_history() -> None:
    buffer = ObservationHistoryBuffer(
        num_envs=2,
        obs_dim=1,
        history_length=3,
    )

    buffer.reset()

    buffer.push(_obs(10.0, 100.0))
    output = buffer.push(_obs(20.0, 200.0))

    expected = torch.tensor(
        [
            [20.0, 10.0, 0.0],
            [200.0, 100.0, 0.0],
        ]
    )

    assert torch.equal(
        output,
        expected,
    )


def test_partial_reset_clears_only_selected_environment() -> None:
    buffer = ObservationHistoryBuffer(
        num_envs=2,
        obs_dim=1,
        history_length=3,
    )

    buffer.reset()

    buffer.push(_obs(10.0, 100.0))
    buffer.push(_obs(20.0, 200.0))

    buffer.reset(env_ids=[0])

    output = buffer.push(
        _obs(30.0, 300.0)
    )

    expected = torch.tensor(
        [
            [30.0, 0.0, 0.0],
            [300.0, 200.0, 100.0],
        ]
    )

    assert torch.equal(
        output,
        expected,
    )


def test_multi_dimensional_observation_is_flattened_correctly() -> None:
    buffer = ObservationHistoryBuffer(
        num_envs=1,
        obs_dim=2,
        history_length=2,
    )

    first = torch.tensor(
        [[1.0, 2.0]]
    )

    second = torch.tensor(
        [[3.0, 4.0]]
    )

    buffer.push(first)
    output = buffer.push(second)

    expected = torch.tensor(
        [[3.0, 4.0, 1.0, 2.0]]
    )

    assert torch.equal(
        output,
        expected,
    )


def test_stacked_dimension_is_correct() -> None:
    buffer = ObservationHistoryBuffer(
        num_envs=4,
        obs_dim=48,
        history_length=5,
    )

    assert buffer.stacked_dim == 240

    output = buffer.push(
        torch.zeros(4, 48)
    )

    assert output.shape == (4, 240)


def test_invalid_observation_shape_raises() -> None:
    buffer = ObservationHistoryBuffer(
        num_envs=2,
        obs_dim=3,
        history_length=5,
    )

    with pytest.raises(ValueError):
        buffer.push(
            torch.zeros(2, 4)
        )


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is not available",
)
def test_cuda_history_stays_on_gpu() -> None:
    buffer = ObservationHistoryBuffer(
        num_envs=2,
        obs_dim=2,
        history_length=3,
        device="cuda",
    )

    observations = torch.tensor(
        [
            [1.0, 2.0],
            [3.0, 4.0],
        ],
        device="cuda",
    )

    output = buffer.push(observations)

    assert output.device.type == "cuda"
    assert output.shape == (2, 6)