# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents


gym.register(
    id="MemoryTrace-Velocity-Flat-Unitree-Go2-v0",
    entry_point="memory_trace_delay.action_delay_env:ActionDelayManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.memory_trace_delay_env_cfg:"
            "MemoryTraceDelayGo2EnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:"
            "MemoryTraceDelayGo2PPORunnerCfg"
        ),
    },
)

gym.register(
    id="MemoryTrace-Stack-Velocity-Flat-Unitree-Go2-v0",
    entry_point="memory_trace_delay.action_delay_env:ActionDelayManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.memory_trace_delay_env_cfg:"
            "MemoryTraceDelayGo2StackEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:"
            "MemoryTraceDelayGo2StackPPORunnerCfg"
        ),
    },
)