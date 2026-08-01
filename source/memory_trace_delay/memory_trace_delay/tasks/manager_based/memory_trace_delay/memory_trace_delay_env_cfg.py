# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""External Go2 flat-terrain environment configuration.

This initial configuration intentionally adds no action delay or memory trace.
It provides a controlled external-project baseline matching the official
Isaac Lab Unitree Go2 flat velocity-tracking environment.
"""

from isaaclab.utils import configclass
from isaaclab_tasks.manager_based.locomotion.velocity.config.go2.flat_env_cfg import (
    UnitreeGo2FlatEnvCfg,
)


@configclass
class MemoryTraceDelayGo2EnvCfg(UnitreeGo2FlatEnvCfg):
    """Unmodified external wrapper around the official Go2 flat environment."""

    def __post_init__(self) -> None:
        super().__post_init__()
