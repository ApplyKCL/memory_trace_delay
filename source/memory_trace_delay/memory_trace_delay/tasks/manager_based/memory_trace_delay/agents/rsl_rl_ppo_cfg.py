# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""RSL-RL PPO configuration for the external Go2 baseline."""

from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.locomotion.velocity.config.go2.agents.rsl_rl_ppo_cfg import (
    UnitreeGo2FlatPPORunnerCfg,
)


@configclass
class MemoryTraceDelayGo2PPORunnerCfg(UnitreeGo2FlatPPORunnerCfg):
    """Official Go2 flat PPO configuration with an isolated log directory."""

    def __post_init__(self) -> None:
        super().__post_init__()

        # Only change the log directory name.
        # PPO hyperparameters and network architecture remain unchanged.
        self.experiment_name = "memory_trace_delay_go2_flat_baseline"
