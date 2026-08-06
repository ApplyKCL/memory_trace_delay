# Copyright (c) 2026, Memory Trace Delay Project Contributors.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Pure PyTorch FIFO buffering for per-environment action delays."""

from __future__ import annotations

from collections.abc import Sequence

import torch


class ActionDelayBuffer:
    """Delay batched actions independently for each parallel environment.

    At environment step t, :meth:`push` returns action[t - delay]. Histories
    are zero-filled at reset, and delays change only when :meth:`reset` is called.
    """

    def __init__(
        self,
        num_envs: int,
        action_dim: int,
        min_delay: int = 0,
        max_delay: int = 0,
        *,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        # Check if the number of environments and action dimensions are positive, and if the delay values are valid.
        if num_envs <= 0:
            raise ValueError(f"num_envs must be positive, received {num_envs}.")
        if action_dim <= 0:
            raise ValueError(f"action_dim must be positive, received {action_dim}.")
        if min_delay < 0:
            raise ValueError(f"min_delay must be non-negative, received {min_delay}.")
        if max_delay < min_delay:
            raise ValueError(f"max_delay must be >= min_delay, received {min_delay=} and {max_delay=}.")

        # num_envs: Number of parallel environments.
        # action_dim: Dimensionality of the action space.
        # min_delay: Minimum delay (inclusive) for any environment.
        # max_delay: Maximum delay (inclusive) for any environment.
        # device: The device on which to store the buffer (e.g., 'cpu' or 'cuda').
        # dtype: The data type for the stored actions (e.g., torch.float32).
        self.num_envs = num_envs
        self.action_dim = action_dim
        self.min_delay = min_delay
        self.max_delay = max_delay
        self.device = torch.device(device)
        self.dtype = dtype

        # One extra ring slot stores the current action, so d=0 reads the value
        # written by the same push call without a separate code path.
        self._history = torch.zeros((max_delay + 1, num_envs, action_dim), device=self.device, dtype=self.dtype)
        self._delay_steps = torch.full((num_envs,), min_delay, device=self.device, dtype=torch.long)
        self._env_ids = torch.arange(num_envs, device=self.device)
        if self.device.type == "cuda" and self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())

        self._write_index = 0

    @property
    def delay_steps(self) -> torch.Tensor:
        """Return a copy of the current episode-fixed delays."""
        return self._delay_steps.clone()

    def reset(
        self,
        env_ids: Sequence[int] | torch.Tensor | slice | None = None,
        *,
        delay_steps: int | Sequence[int] | torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Clear selected histories and assign their next episode delays."""

        # Resolve the environment IDs to a tensor of indices.
        # If env_ids is None, it defaults to all environments.
        # If env_ids is a slice, it converts it to the corresponding indices.
        # If env_ids is a tensor or sequence, it checks for validity and converts it to a tensor of long type.
        ids = self._resolve_env_ids(env_ids)
        if ids.numel() == 0:
            return self._delay_steps[ids].clone()

        # Clear every ring slot only for reset environments. The shared write
        # index and every non-reset environment remain untouched.
        self._history[:, ids] = 0.0

        if delay_steps is None:
            if self.min_delay == self.max_delay:
                # Fixed d=0 must not consume RNG state relative to the baseline.
                self._delay_steps[ids] = self.min_delay
            else:
                self._delay_steps[ids] = torch.randint(
                    self.min_delay, self.max_delay + 1, (ids.numel(),), device=self.device
                )
        else:
            delays = torch.as_tensor(delay_steps, device=self.device, dtype=torch.long).flatten()
            if delays.numel() == 1:
                delays = delays.expand(ids.numel())
            elif delays.numel() != ids.numel():
                raise ValueError(f"Expected one delay or {ids.numel()} delays, received {delays.numel()}.")
            if torch.any(delays < 0) or torch.any(delays > self.max_delay):
                raise ValueError(f"delay_steps must be in [0, {self.max_delay}].")
            self._delay_steps[ids] = delays

        return self._delay_steps[ids].clone()

    def push(self, actions: torch.Tensor) -> torch.Tensor:
        """Store current actions and return each environment's delayed action."""
        expected_shape = (self.num_envs, self.action_dim)
        # check the shape, device, and dtype of the input actions tensor
        if tuple(actions.shape) != expected_shape:
            raise ValueError(f"Expected action shape {expected_shape}, received {tuple(actions.shape)}.")
        if actions.device != self.device:
            raise ValueError(f"Expected actions on {self.device}, received {actions.device}.")
        if actions.dtype != self.dtype:
            raise ValueError(f"Expected dtype {self.dtype}, received {actions.dtype}.")

        # There is no need to track gradients for the delayed action
        # Since they are not used for backpropagation
        with torch.no_grad():
            # Copy actions into the buffer without retaining autograd history.
            self._history[self._write_index].copy_(actions)
            read_indices = torch.remainder(self._write_index - self._delay_steps, self.max_delay + 1)
            delayed_actions = self._history[read_indices, self._env_ids]

        # Increment the write index in a circular manner to maintain the ring buffer structure.
        self._write_index = (self._write_index + 1) % (self.max_delay + 1)
        # The push returns the delayed actions, but store the current actions
        return delayed_actions

    def _resolve_env_ids(
        self,
        env_ids: Sequence[int] | torch.Tensor | slice | None,
    ) -> torch.Tensor:
        # check if env_ids is None, a slice, or a tensor/sequence of indices
        if env_ids is None:
            return self._env_ids
        if isinstance(env_ids, slice):
            return self._env_ids[env_ids]

        ids = torch.as_tensor(env_ids, device=self.device)
        # chceck if the tensor is of boolean type, and if so, convert it to indices
        if ids.dtype == torch.bool:
            if ids.numel() != self.num_envs:
                raise ValueError(f"Boolean env_ids must contain {self.num_envs} entries.")
            ids = ids.flatten().nonzero(as_tuple=False).squeeze(-1)
        else:
            ids = ids.to(dtype=torch.long).flatten()

        if ids.numel() > 0 and (torch.any(ids < 0) or torch.any(ids >= self.num_envs)):
            raise IndexError(f"env_ids must be in [0, {self.num_envs - 1}].")
        return ids
