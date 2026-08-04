# Copyright (c) 2026, Memory Trace Delay Project Contributors.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Manager-based RL environment that delays raw policy actions."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.envs.common import VecEnvStepReturn

from .action_delay import ActionDelayBuffer


class ActionDelayManagerBasedRLEnv(ManagerBasedRLEnv):
    """Delay raw actions immediately before ActionManager.process_action."""

    def __init__(self, cfg, render_mode: str | None = None, **kwargs) -> None:
        self._action_delay_buffer: ActionDelayBuffer | None = None
        self._last_policy_action: torch.Tensor | None = None
        self._last_delayed_action: torch.Tensor | None = None
        super().__init__(cfg=cfg, render_mode=render_mode, **kwargs)

        self._action_delay_buffer = ActionDelayBuffer(
            num_envs=self.num_envs,
            action_dim=self.action_manager.total_action_dim,
            min_delay=cfg.action_delay.min_steps,
            max_delay=cfg.action_delay.max_steps,
            device=self.device,
            dtype=self.action_manager.action.dtype
        )

    @property
    def action_delay_buffer(self) -> ActionDelayBuffer:
        if self._action_delay_buffer is None:
            raise RuntimeError("Action delay buffer is not initialized.")
        return self._action_delay_buffer

    @property
    def action_delay_steps(self) -> torch.Tensor:
        return self.action_delay_buffer.delay_steps

    @property
    def last_policy_action(self) -> torch.Tensor | None:
        return self._last_policy_action

    @property
    def last_delayed_action(self) -> torch.Tensor | None:
        return self._last_delayed_action

    def step(self, action: torch.Tensor) -> VecEnvStepReturn:
        policy_action = action.to(self.device)
        delayed_action = self.action_delay_buffer.push(policy_action)
        self._last_policy_action = policy_action.detach()
        self._last_delayed_action = delayed_action

        # The first operation in the Isaac Lab v2.3.2 parent step is
        # ActionManager.process_action(delayed_action).
        return super().step(delayed_action)

    def _reset_idx(self, env_ids: Sequence[int]) -> None:
        super()._reset_idx(env_ids)
        if self._action_delay_buffer is not None:
            self._action_delay_buffer.reset(env_ids)
