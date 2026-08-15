# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""External Go2 flat-terrain environment with episode-fixed action delay."""

from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.locomotion.velocity.config.go2.flat_env_cfg import (
    UnitreeGo2FlatEnvCfg,
)


@configclass
class ActionDelayCfg:
    """Uniform raw-action delay range, measured in environment steps."""

    min_steps: int = 0
    max_steps: int = 3


@configclass
class MemoryTraceCfg:
    """Exponential policy-observation trace configuration."""

    enabled: bool = False
    decay: float = 0.8


@configclass
class MemoryTraceDelayGo2EnvCfg(UnitreeGo2FlatEnvCfg):
    """Official Go2 flat environment with only raw action delay added."""

    action_delay: ActionDelayCfg = ActionDelayCfg()
    memory_trace: MemoryTraceCfg = MemoryTraceCfg()

    def __post_init__(self) -> None:
        super().__post_init__()


@configclass
class MemoryTraceDelayGo2StackEnvCfg(MemoryTraceDelayGo2EnvCfg):
    """Go2 delay environment with stacked policy-observation history."""

    def __post_init__(self) -> None:
        super().__post_init__()

        self.observations.policy.history_length = 5
        self.observations.policy.flatten_history_dim = True


@configclass
class MemoryTraceDelayGo2TraceEnvCfg(MemoryTraceDelayGo2EnvCfg):
    """Go2 delay environment with a single exponential observation trace."""

    def __post_init__(self) -> None:
        super().__post_init__()

        self.memory_trace.enabled = True
        # 1 / (1 - 0.8) = 5 environment steps, matching Stack-5's
        # approximate memory horizon for the controlled comparison.
        self.memory_trace.decay = 0.8
