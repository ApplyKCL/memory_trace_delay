# Copyright (c) 2026, Memory Trace Delay Project Contributors.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Pure PyTorch observation-history buffering for vectorized environments."""

from __future__ import annotations

from collections.abc import Sequence

import torch


class ObservationHistoryBuffer:
    """Store recent observations for each parallel environment.

    The returned stack is ordered from newest to oldest:

        [obs_t, obs_{t-1}, ..., obs_{t-history_length+1}]

    Histories are zero-filled after reset.
    """

    def __init__(
        self,
        num_envs: int,
        obs_dim: int,
        history_length: int,
        *,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        if num_envs <= 0:
            raise ValueError(
                f"num_envs must be positive, received {num_envs}."
            )
        if obs_dim <= 0:
            raise ValueError(
                f"obs_dim must be positive, received {obs_dim}."
            )
        if history_length <= 0:
            raise ValueError(
                "history_length must be positive, "
                f"received {history_length}."
            )

        self.num_envs = num_envs
        self.obs_dim = obs_dim
        self.history_length = history_length
        self.device = torch.device(device)
        self.dtype = dtype

        if self.device.type == "cuda" and self.device.index is None:
            self.device = torch.device(
                "cuda",
                torch.cuda.current_device(),
            )

        # Shape:
        # [history slot, environment, observation dimension]
        self._history = torch.zeros(
            (
                history_length,
                num_envs,
                obs_dim,
            ),
            device=self.device,
            dtype=self.dtype,
        )

        self._env_ids = torch.arange(
            num_envs,
            device=self.device,
        )

        # Next ring slot to be overwritten.
        self._write_index = 0

    @property
    def stacked_dim(self) -> int:
        """Return the flattened stacked-observation dimension."""
        return self.obs_dim * self.history_length

    def reset(
        self,
        env_ids: Sequence[int] | torch.Tensor | slice | None = None,
    ) -> None:
        """Clear observation histories for selected environments."""

        ids = self._resolve_env_ids(env_ids)

        if ids.numel() == 0:
            return

        self._history[:, ids] = 0.0

    def push(self, observations: torch.Tensor) -> torch.Tensor:
        """Store observations and return flattened newest-to-oldest history."""

        expected_shape = (
            self.num_envs,
            self.obs_dim,
        )

        if tuple(observations.shape) != expected_shape:
            raise ValueError(
                f"Expected observation shape {expected_shape}, "
                f"received {tuple(observations.shape)}."
            )

        if observations.device != self.device:
            raise ValueError(
                f"Expected observations on {self.device}, "
                f"received {observations.device}."
            )

        if observations.dtype != self.dtype:
            raise ValueError(
                f"Expected dtype {self.dtype}, "
                f"received {observations.dtype}."
            )

        with torch.no_grad():
            self._history[self._write_index].copy_(observations)

            offsets = torch.arange(
                self.history_length,
                device=self.device,
            )

            read_indices = torch.remainder(
                self._write_index - offsets,
                self.history_length,
            )

            # [history, env, obs]
            ordered_history = self._history[read_indices]

            # [env, history, obs]
            ordered_history = ordered_history.permute(1, 0, 2)

            # [env, history * obs]
            stacked = ordered_history.reshape(
                self.num_envs,
                self.stacked_dim,
            )

        self._write_index = (
            self._write_index + 1
        ) % self.history_length

        return stacked

    def _resolve_env_ids(
        self,
        env_ids: Sequence[int] | torch.Tensor | slice | None,
    ) -> torch.Tensor:
        """Convert environment selection to integer indices."""

        if env_ids is None:
            return self._env_ids

        if isinstance(env_ids, slice):
            return self._env_ids[env_ids]

        ids = torch.as_tensor(
            env_ids,
            device=self.device,
        )

        if ids.dtype == torch.bool:
            if ids.numel() != self.num_envs:
                raise ValueError(
                    "Boolean env_ids must contain "
                    f"{self.num_envs} entries."
                )

            ids = (
                ids.flatten()
                .nonzero(as_tuple=False)
                .squeeze(-1)
            )
        else:
            ids = ids.to(
                dtype=torch.long
            ).flatten()

        if ids.numel() > 0 and (
            torch.any(ids < 0)
            or torch.any(ids >= self.num_envs)
        ):
            raise IndexError(
                "env_ids must be in "
                f"[0, {self.num_envs - 1}]."
            )

        return ids