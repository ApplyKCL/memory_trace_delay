"""RSL-RL observation wrapper for exponential memory traces."""

from __future__ import annotations

import torch
from tensordict import TensorDict

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

from .memory_trace import MemoryTraceBuffer


class MemoryTraceRslRlVecEnvWrapper(RslRlVecEnvWrapper):
    """Append an exponential trace to the policy observation."""

    def __init__(
        self,
        env,
        *,
        decay: float,
        clip_actions: float | None = None,
    ) -> None:
        super().__init__(env, clip_actions=clip_actions)

        raw_observations = super().get_observations()
        policy_obs = raw_observations["policy"]

        self.memory_trace_buffer = MemoryTraceBuffer(
            num_envs=self.num_envs,
            obs_dim=policy_obs.shape[-1],
            decay=decay,
            device=policy_obs.device,
            dtype=policy_obs.dtype,
        )
        self._latest_observations = self._augment(raw_observations)

    def _augment(self, observations: TensorDict) -> TensorDict:
        raw_policy_obs = observations["policy"]
        trace = self.memory_trace_buffer.push(raw_policy_obs)

        augmented = observations.clone()
        augmented["policy"] = torch.cat((raw_policy_obs, trace), dim=-1)
        return augmented

    def get_observations(self) -> TensorDict:
        """Return cached observations without updating the trace twice."""
        return self._latest_observations.clone()

    def reset(self) -> tuple[TensorDict, dict]:
        observations, extras = super().reset()
        self.memory_trace_buffer.reset()
        self._latest_observations = self._augment(observations)
        return self._latest_observations.clone(), extras

    def step(self, actions: torch.Tensor) -> tuple[TensorDict, torch.Tensor, torch.Tensor, dict]:
        observations, rewards, dones, extras = super().step(actions)

        done_ids = dones.to(dtype=torch.bool).nonzero(as_tuple=False).squeeze(-1)
        if done_ids.numel() > 0:
            # The parent environment has already reset these environments. Drop
            # their previous-episode trace before adding the reset observation.
            self.memory_trace_buffer.reset(done_ids)

        self._latest_observations = self._augment(observations)
        return self._latest_observations.clone(), rewards, dones, extras


def wrap_rsl_rl_env(env, env_cfg, clip_actions: float | None):
    """Select the standard or Memory Trace RSL-RL wrapper."""
    trace_cfg = getattr(env_cfg, "memory_trace", None)
    if trace_cfg is not None and trace_cfg.enabled:
        return MemoryTraceRslRlVecEnvWrapper(
            env,
            decay=trace_cfg.decay,
            clip_actions=clip_actions,
        )
    return RslRlVecEnvWrapper(env, clip_actions=clip_actions)
