"""Run a short Isaac Lab Memory Trace integration check."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Verify the policy-observation memory trace.")
parser.add_argument("--steps", type=int, default=5, help="Number of environment steps to verify.")
parser.add_argument("--num_envs", type=int, default=4, help="Number of Go2 environments.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import memory_trace_delay.tasks  # noqa: F401
import torch
from memory_trace_delay.rsl_rl_trace_wrapper import MemoryTraceRslRlVecEnvWrapper

from isaaclab_tasks.utils import parse_env_cfg


def main() -> None:
    task_id = "MemoryTrace-Trace-Velocity-Flat-Unitree-Go2-v0"
    env_cfg = parse_env_cfg(task_id, device=args_cli.device, num_envs=args_cli.num_envs)
    env_cfg.action_delay.min_steps = 0
    env_cfg.action_delay.max_steps = 0
    env_cfg.observations.policy.enable_corruption = False

    raw_env = gym.make(task_id, cfg=env_cfg)
    env = MemoryTraceRslRlVecEnvWrapper(
        raw_env,
        decay=env_cfg.memory_trace.decay,
    )

    try:
        observations = env.get_observations()
        policy_obs = observations["policy"]
        raw_dim = policy_obs.shape[-1] // 2

        if policy_obs.shape[-1] != 2 * raw_dim:
            raise AssertionError(f"Expected [raw, trace] observation, received {policy_obs.shape}.")

        raw_obs = policy_obs[:, :raw_dim]
        trace = policy_obs[:, raw_dim:]
        torch.testing.assert_close(trace, (1.0 - env_cfg.memory_trace.decay) * raw_obs)

        print(
            f"[MEMORY_TRACE] raw_dim={raw_dim} augmented_dim={policy_obs.shape[-1]} "
            f"decay={env_cfg.memory_trace.decay}"
        )

        previous_trace = trace
        with torch.inference_mode():
            for step in range(args_cli.steps):
                actions = torch.zeros(raw_env.action_space.shape, device=raw_env.unwrapped.device)
                observations, _, dones, _ = env.step(actions)
                policy_obs = observations["policy"]
                raw_obs = policy_obs[:, :raw_dim]
                trace = policy_obs[:, raw_dim:]

                expected = env_cfg.memory_trace.decay * previous_trace + (
                    1.0 - env_cfg.memory_trace.decay
                ) * raw_obs
                active = ~dones.to(dtype=torch.bool)
                if active.any():
                    torch.testing.assert_close(trace[active], expected[active])

                print(
                    f"[MEMORY_TRACE] step={step} "
                    f"raw={raw_obs[0, 0].item():.4f} trace={trace[0, 0].item():.4f}"
                )
                previous_trace = trace

        print("[MEMORY_TRACE] PASS")
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
