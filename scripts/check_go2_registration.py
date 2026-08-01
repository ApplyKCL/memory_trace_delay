"""Check registration and inheritance of the external Go2 task."""

# Isaac Sim must be launched before importing Isaac Lab task modules.
from isaaclab.app import AppLauncher

app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

try:
    import gymnasium as gym
    import memory_trace_delay.tasks

    from memory_trace_delay.tasks.manager_based.memory_trace_delay.memory_trace_delay_env_cfg import (
        MemoryTraceDelayGo2EnvCfg,
    )
    from memory_trace_delay.tasks.manager_based.memory_trace_delay.agents.rsl_rl_ppo_cfg import (
        MemoryTraceDelayGo2PPORunnerCfg,
    )

    from isaaclab_tasks.manager_based.locomotion.velocity.config.go2.flat_env_cfg import (
        UnitreeGo2FlatEnvCfg,
    )
    from isaaclab_tasks.manager_based.locomotion.velocity.config.go2.agents.rsl_rl_ppo_cfg import (
        UnitreeGo2FlatPPORunnerCfg,
    )

    task_id = "MemoryTrace-Velocity-Flat-Unitree-Go2-v0"

    print("\n===== REGISTRATION =====")
    registered = task_id in gym.registry
    print("Registered:", registered)

    if registered:
        spec = gym.spec(task_id)
        print("Task ID:", spec.id)
        print("Entry point:", spec.entry_point)
        print("Environment config:", spec.kwargs.get("env_cfg_entry_point"))
        print("RSL-RL config:", spec.kwargs.get("rsl_rl_cfg_entry_point"))

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

    ppo_cfg = MemoryTraceDelayGo2PPORunnerCfg()

    print("\n===== PPO CONFIG =====")
    print("Experiment name:", ppo_cfg.experiment_name)
    print("Steps per environment:", ppo_cfg.num_steps_per_env)
    print("Maximum iterations:", ppo_cfg.max_iterations)

finally:
    simulation_app.close()
