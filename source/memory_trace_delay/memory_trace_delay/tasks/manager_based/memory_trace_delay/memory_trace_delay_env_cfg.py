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
class MemoryTraceDelayGo2EnvCfg(UnitreeGo2FlatEnvCfg):
    """Official Go2 flat environment with only raw action delay added."""

    action_delay: ActionDelayCfg = ActionDelayCfg()

    def __post_init__(self) -> None:
        super().__post_init__()
