"""Check registration and inheritance of the external Go2 task."""

# Isaac Sim must be launched before importing Isaac Lab task modules.
from isaaclab.app import AppLauncher

app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

try:
    import gymnasium as gym
    from memory_trace_delay.tasks.manager_based.memory_trace_delay.agents.rsl_rl_ppo_cfg import (
        MemoryTraceDelayGo2PPORunnerCfg,
        MemoryTraceDelayGo2StackPPORunnerCfg,
        MemoryTraceDelayGo2TracePPORunnerCfg,
    )
    from memory_trace_delay.tasks.manager_based.memory_trace_delay.memory_trace_delay_env_cfg import (
        MemoryTraceDelayGo2EnvCfg,
        MemoryTraceDelayGo2StackEnvCfg,
        MemoryTraceDelayGo2TraceEnvCfg,
    )

    from isaaclab_tasks.manager_based.locomotion.velocity.config.go2.agents.rsl_rl_ppo_cfg import (
        UnitreeGo2FlatPPORunnerCfg,
    )
    from isaaclab_tasks.manager_based.locomotion.velocity.config.go2.flat_env_cfg import (
        UnitreeGo2FlatEnvCfg,
    )

    print("\n===== REGISTRATION =====")
    task_ids = (
        "MemoryTrace-Velocity-Flat-Unitree-Go2-v0",
        "MemoryTrace-Stack-Velocity-Flat-Unitree-Go2-v0",
        "MemoryTrace-Trace-Velocity-Flat-Unitree-Go2-v0",
    )
    for task_id in task_ids:
        registered = task_id in gym.registry
        print(f"{task_id}: {registered}")
        if not registered:
            raise RuntimeError(f"Task is not registered: {task_id}")

    print("\n===== INHERITANCE =====")
    print(
        "Environment inherits official Go2:",
        issubclass(MemoryTraceDelayGo2EnvCfg, UnitreeGo2FlatEnvCfg),
    )
    print(
        "PPO inherits official Go2 PPO:",
        issubclass(
            MemoryTraceDelayGo2PPORunnerCfg,
            UnitreeGo2FlatPPORunnerCfg,
        ),
    )
    print(
        "Stack environment inherits delay Go2:",
        issubclass(MemoryTraceDelayGo2StackEnvCfg, MemoryTraceDelayGo2EnvCfg),
    )
    print(
        "Trace environment inherits delay Go2:",
        issubclass(MemoryTraceDelayGo2TraceEnvCfg, MemoryTraceDelayGo2EnvCfg),
    )

    ppo_cfg = MemoryTraceDelayGo2PPORunnerCfg()

    print("\n===== PPO CONFIG =====")
    print("Experiment name:", ppo_cfg.experiment_name)
    print("Steps per environment:", ppo_cfg.num_steps_per_env)
    print("Maximum iterations:", ppo_cfg.max_iterations)
    print("Stack experiment:", MemoryTraceDelayGo2StackPPORunnerCfg().experiment_name)
    print("Trace experiment:", MemoryTraceDelayGo2TracePPORunnerCfg().experiment_name)

finally:
    simulation_app.close()
