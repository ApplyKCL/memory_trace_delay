"""Pure PyTorch exponential memory traces for vectorized environments."""

from __future__ import annotations

from collections.abc import Sequence

import torch


class MemoryTraceBuffer:
    """Maintain an exponential observation trace for each environment.

    Every call to :meth:`push` applies

        trace_t = decay * trace_(t-1) + (1 - decay) * observation_t

    independently to each parallel environment. Traces start at zero and are
    cleared only for environments passed to :meth:`reset`.
    """

    def __init__(
        self,
        num_envs: int,
        obs_dim: int,
        decay: float,
        *,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        if num_envs <= 0:
            raise ValueError(f"num_envs must be positive, received {num_envs}.")
        if obs_dim <= 0:
            raise ValueError(f"obs_dim must be positive, received {obs_dim}.")
        if not 0.0 <= decay < 1.0:
            raise ValueError(f"decay must be in [0, 1), received {decay}.")
        if not dtype.is_floating_point:
            raise TypeError(f"dtype must be floating point, received {dtype}.")

        self.num_envs = num_envs
        self.obs_dim = obs_dim
        self.decay = float(decay)
        self.device = torch.device(device)
        self.dtype = dtype

        if self.device.type == "cuda" and self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())

        self._trace = torch.zeros(
            (num_envs, obs_dim),
            device=self.device,
            dtype=self.dtype,
        )
        self._env_ids = torch.arange(num_envs, device=self.device)

    @property
    def trace(self) -> torch.Tensor:
        """Return a copy of the current trace state."""
        return self._trace.clone()

    def reset(
        self,
        env_ids: Sequence[int] | torch.Tensor | slice | None = None,
    ) -> torch.Tensor:
        """Clear selected trace states and return their new values."""
        ids = self._resolve_env_ids(env_ids)
        if ids.numel() > 0:
            self._trace[ids] = 0.0
        return self._trace[ids].clone()

    def push(self, observations: torch.Tensor) -> torch.Tensor:
        """Update all traces and return the new trace values."""
        expected_shape = (self.num_envs, self.obs_dim)
        if tuple(observations.shape) != expected_shape:
            raise ValueError(f"Expected observation shape {expected_shape}, received {tuple(observations.shape)}.")
        if observations.device != self.device:
            raise ValueError(f"Expected observations on {self.device}, received {observations.device}.")
        if observations.dtype != self.dtype:
            raise ValueError(f"Expected dtype {self.dtype}, received {observations.dtype}.")

        with torch.no_grad():
            self._trace.mul_(self.decay).add_(observations, alpha=1.0 - self.decay)
            return self._trace.clone()

    def _resolve_env_ids(
        self,
        env_ids: Sequence[int] | torch.Tensor | slice | None,
    ) -> torch.Tensor:
        """Convert an environment selection to integer indices."""
        if env_ids is None:
            return self._env_ids
        if isinstance(env_ids, slice):
            return self._env_ids[env_ids]

        ids = torch.as_tensor(env_ids, device=self.device)
        if ids.dtype == torch.bool:
            if ids.numel() != self.num_envs:
                raise ValueError(f"Boolean env_ids must contain {self.num_envs} entries.")
            ids = ids.flatten().nonzero(as_tuple=False).squeeze(-1)
        else:
            ids = ids.to(dtype=torch.long).flatten()

        if ids.numel() > 0 and (torch.any(ids < 0) or torch.any(ids >= self.num_envs)):
            raise IndexError(f"env_ids must be in [0, {self.num_envs - 1}].")
        return ids
