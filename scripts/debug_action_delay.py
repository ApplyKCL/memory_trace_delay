"""Run a short Isaac Lab action-delay integration check."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Verify raw-to-delayed action flow.")
parser.add_argument("--delay", type=int, default=2, help="Fixed delay in environment steps.")
parser.add_argument("--steps", type=int, default=5, help="Number of steps to verify.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of Go2 environments.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import memory_trace_delay.tasks  # noqa: F401
import torch

from isaaclab_tasks.utils import parse_env_cfg


def main() -> None:
    task_id = "MemoryTrace-Velocity-Flat-Unitree-Go2-v0"
    env_cfg = parse_env_cfg(task_id, device=args_cli.device, num_envs=args_cli.num_envs)
    env_cfg.action_delay.min_steps = args_cli.delay
    env_cfg.action_delay.max_steps = args_cli.delay

    env = gym.make(task_id, cfg=env_cfg)
    env.reset()
    core = env.unwrapped

    print(f"[ACTION_DELAY] episode_delays={core.action_delay_steps.tolist()}")
    with torch.inference_mode():
        for step in range(args_cli.steps):
            raw_value = 0.01 * (step + 1)
            raw_action = torch.full(env.action_space.shape, raw_value, device=core.device)
            env.step(raw_action)
            expected_value = 0.0 if step < args_cli.delay else 0.01 * (step - args_cli.delay + 1)
            expected = torch.full_like(core.last_delayed_action, expected_value)
            print(
                f"[ACTION_DELAY] step={step} delay={args_cli.delay} "
                f"raw={core.last_policy_action[0, 0].item():.3f} "
                f"delayed={core.last_delayed_action[0, 0].item():.3f} "
                f"manager={core.action_manager.action[0, 0].item():.3f} "
                f"expected={expected_value:.3f}"
            )
            torch.testing.assert_close(core.last_delayed_action, expected)
            torch.testing.assert_close(core.action_manager.action, expected)

    print("[ACTION_DELAY] PASS")
    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
