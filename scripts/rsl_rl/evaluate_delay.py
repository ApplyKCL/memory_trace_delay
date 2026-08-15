# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# Copyright (c) 2026, Memory Trace Delay Project Contributors.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Evaluate an RSL-RL policy under a fixed action delay."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

# Local imports must be available before launching Isaac Sim.
import cli_args  # isort: skip


parser = argparse.ArgumentParser(description="Evaluate an RSL-RL checkpoint under a fixed action delay.")
parser.add_argument(
    "--task",
    type=str,
    default="MemoryTrace-Velocity-Flat-Unitree-Go2-v0",
    help="Registered Gym task name.",
)
parser.add_argument(
    "--agent",
    type=str,
    default="rsl_rl_cfg_entry_point",
    help="RSL-RL agent configuration entry point.",
)
parser.add_argument(
    "--num_envs",
    type=int,
    default=256,
    help="Number of parallel evaluation environments.",
)
parser.add_argument(
    "--num_episodes",
    type=int,
    default=1024,
    help="Number of completed episodes to collect.",
)
parser.add_argument(
    "--test_delay",
    type=int,
    required=True,
    help="Fixed action delay used during evaluation, measured in environment steps.",
)
parser.add_argument(
    "--seed",
    type=int,
    default=123,
    help="Evaluation seed.",
)
parser.add_argument(
    "--method",
    type=str,
    default="unspecified",
    help="Method label stored in the result metadata.",
)
parser.add_argument(
    "--training_seed",
    type=int,
    default=None,
    help="Seed used to train the evaluated checkpoint.",
)
parser.add_argument(
    "--train_delay_min",
    type=int,
    default=None,
    help="Minimum action delay used to train the checkpoint.",
)
parser.add_argument(
    "--train_delay_max",
    type=int,
    default=None,
    help="Maximum action delay used to train the checkpoint.",
)
parser.add_argument(
    "--output",
    type=str,
    required=True,
    help="Path of the output JSON file.",
)

cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)

args_cli, hydra_args = parser.parse_known_args()

if args_cli.checkpoint is None:
    parser.error("--checkpoint is required.")

if args_cli.test_delay < 0:
    parser.error("--test_delay must be non-negative.")

if args_cli.num_envs <= 0:
    parser.error("--num_envs must be positive.")

if args_cli.num_episodes <= 0:
    parser.error("--num_episodes must be positive.")

# This script performs numerical evaluation only.
args_cli.headless = True

# Hydra must only receive configuration overrides that were not consumed above.
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


"""Imports below this line require Isaac Sim to be running."""

import gymnasium as gym
import memory_trace_delay.tasks  # noqa: F401
import torch
from memory_trace_delay.rsl_rl_trace_wrapper import wrap_rsl_rl_env
from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config


def mean_and_std(values: list[float]) -> tuple[float, float]:
    """Return population mean and standard deviation."""

    tensor = torch.tensor(values, dtype=torch.float64)
    return tensor.mean().item(), tensor.std(unbiased=False).item()


@hydra_task_config(args_cli.task, args_cli.agent)
def main(
    env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg,
    agent_cfg: RslRlBaseRunnerCfg,
) -> None:
    """Evaluate the selected policy checkpoint."""

    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)

    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = agent_cfg.seed

    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device
        agent_cfg.device = args_cli.device

    # Use a fixed delay for every environment and every episode.
    env_cfg.action_delay.min_steps = args_cli.test_delay
    env_cfg.action_delay.max_steps = args_cli.test_delay

    # Match the deterministic parts of the official Go2 play configuration.
    env_cfg.observations.policy.enable_corruption = False

    if hasattr(env_cfg, "events"):
        if hasattr(env_cfg.events, "base_external_force_torque"):
            env_cfg.events.base_external_force_torque = None
        if hasattr(env_cfg.events, "push_robot"):
            env_cfg.events.push_robot = None

    resume_path = retrieve_file_path(args_cli.checkpoint)
    print(f"[INFO] Checkpoint: {resume_path}")
    print(f"[INFO] Test delay: {args_cli.test_delay}")
    print(f"[INFO] Environments: {args_cli.num_envs}")
    print(f"[INFO] Requested episodes: {args_cli.num_episodes}")

    env = None

    try:
        env = gym.make(args_cli.task, cfg=env_cfg)

        if isinstance(env.unwrapped, DirectMARLEnv):
            env = multi_agent_to_single_agent(env)

        env = wrap_rsl_rl_env(
            env,
            env_cfg,
            clip_actions=agent_cfg.clip_actions,
        )

        if agent_cfg.class_name == "OnPolicyRunner":
            runner = OnPolicyRunner(
                env,
                agent_cfg.to_dict(),
                log_dir=None,
                device=agent_cfg.device,
            )
        elif agent_cfg.class_name == "DistillationRunner":
            runner = DistillationRunner(
                env,
                agent_cfg.to_dict(),
                log_dir=None,
                device=agent_cfg.device,
            )
        else:
            raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")

        runner.load(resume_path)

        policy = runner.get_inference_policy(device=env.unwrapped.device)

        try:
            policy_nn = runner.alg.policy
        except AttributeError:
            policy_nn = runner.alg.actor_critic

        core = env.unwrapped
        device = core.device
        num_envs = core.num_envs

        print(
            "[CHECK] action delay:",
            core.action_delay_steps[: min(8, num_envs)].tolist(),
        )

        # Per-environment accumulators for the current episode.
        episode_returns = torch.zeros(num_envs, device=device)
        episode_lengths = torch.zeros(
            num_envs,
            device=device,
            dtype=torch.long,
        )
        episode_pre_action_xy_errors = torch.zeros(num_envs, device=device)
        episode_pre_action_yaw_errors = torch.zeros(num_envs, device=device)

        # Completed episode results.
        completed_returns: list[float] = []
        completed_lengths: list[float] = []
        completed_pre_action_xy_errors: list[float] = []
        completed_pre_action_yaw_errors: list[float] = []
        completed_failures: list[float] = []
        completed_timeouts: list[float] = []

        obs = env.get_observations()

        vector_steps = 0
        episodes_per_batch = math.ceil(args_cli.num_episodes / args_cli.num_envs)
        max_vector_steps = 2 * episodes_per_batch * env.max_episode_length

        while (
            simulation_app.is_running()
            and len(completed_returns) < args_cli.num_episodes
            and vector_steps < max_vector_steps
        ):
            with torch.inference_mode():
                # Measure the state seen by the policy before applying its action.
                # This intentionally reports pre-action tracking error because the
                # parent environment auto-resets terminated environments in step().
                command = core.command_manager.get_command("base_velocity")
                robot_data = core.scene["robot"].data

                xy_error = torch.linalg.vector_norm(
                    command[:, :2] - robot_data.root_lin_vel_b[:, :2],
                    dim=1,
                )
                yaw_error = torch.abs(command[:, 2] - robot_data.root_ang_vel_b[:, 2])

                actions = policy(obs)
                obs, rewards, dones, extras = env.step(actions)

                episode_returns += rewards
                episode_lengths += 1
                episode_pre_action_xy_errors += xy_error
                episode_pre_action_yaw_errors += yaw_error

                done_mask = dones.to(dtype=torch.bool)
                done_ids = done_mask.nonzero(as_tuple=False).squeeze(-1)

                if done_ids.numel() > 0:
                    if "time_outs" in extras:
                        timeouts = extras["time_outs"].to(
                            device=device,
                            dtype=torch.bool,
                        )
                    elif hasattr(core, "termination_manager"):
                        timeouts = core.termination_manager.time_outs.to(device=device, dtype=torch.bool)
                    else:
                        timeouts = torch.zeros(
                            num_envs,
                            device=device,
                            dtype=torch.bool,
                        )

                    done_lengths = episode_lengths[done_ids]
                    safe_lengths = done_lengths.clamp_min(1).to(dtype=episode_returns.dtype)
                    batch_metrics = (
                        torch.stack(
                            (
                                episode_returns[done_ids],
                                done_lengths.to(dtype=episode_returns.dtype),
                                episode_pre_action_xy_errors[done_ids] / safe_lengths,
                                episode_pre_action_yaw_errors[done_ids] / safe_lengths,
                                timeouts[done_ids].to(dtype=episode_returns.dtype),
                            ),
                            dim=1,
                        )
                        .detach()
                        .cpu()
                    )

                    completed_returns.extend(batch_metrics[:, 0].tolist())
                    completed_lengths.extend(batch_metrics[:, 1].tolist())
                    completed_pre_action_xy_errors.extend(batch_metrics[:, 2].tolist())
                    completed_pre_action_yaw_errors.extend(batch_metrics[:, 3].tolist())
                    batch_timeouts = batch_metrics[:, 4].tolist()
                    completed_timeouts.extend(batch_timeouts)
                    completed_failures.extend(1.0 - value for value in batch_timeouts)

                    episode_returns[done_ids] = 0.0
                    episode_lengths[done_ids] = 0
                    episode_pre_action_xy_errors[done_ids] = 0.0
                    episode_pre_action_yaw_errors[done_ids] = 0.0

                    policy_nn.reset(dones)

                    completed_count = min(
                        len(completed_returns),
                        args_cli.num_episodes,
                    )
                    print(
                        f"\r[INFO] Completed episodes: {completed_count}/{args_cli.num_episodes}",
                        end="",
                        flush=True,
                    )

            vector_steps += 1

        print()

        if len(completed_returns) < args_cli.num_episodes:
            raise RuntimeError(
                "Evaluation ended before enough episodes were collected. "
                f"Collected {len(completed_returns)} of "
                f"{args_cli.num_episodes} episodes."
            )

        # A final vector step may finish more episodes than requested.
        completed_returns = completed_returns[: args_cli.num_episodes]
        completed_lengths = completed_lengths[: args_cli.num_episodes]
        completed_pre_action_xy_errors = completed_pre_action_xy_errors[: args_cli.num_episodes]
        completed_pre_action_yaw_errors = completed_pre_action_yaw_errors[: args_cli.num_episodes]
        completed_failures = completed_failures[: args_cli.num_episodes]
        completed_timeouts = completed_timeouts[: args_cli.num_episodes]

        mean_return, std_return = mean_and_std(completed_returns)
        mean_length, std_length = mean_and_std(completed_lengths)
        mean_pre_action_xy_error, std_pre_action_xy_error = mean_and_std(completed_pre_action_xy_errors)
        mean_pre_action_yaw_error, std_pre_action_yaw_error = mean_and_std(completed_pre_action_yaw_errors)
        failure_rate, _ = mean_and_std(completed_failures)
        timeout_rate, _ = mean_and_std(completed_timeouts)

        result = {
            "task": args_cli.task,
            "checkpoint": str(Path(resume_path).resolve()),
            "method": args_cli.method,
            "seed": agent_cfg.seed,
            "training_seed": args_cli.training_seed,
            "evaluation_seed": agent_cfg.seed,
            "train_delay_min": args_cli.train_delay_min,
            "train_delay_max": args_cli.train_delay_max,
            "trace_enabled": bool(
                getattr(getattr(env_cfg, "memory_trace", None), "enabled", False)
            ),
            "trace_decay": getattr(
                getattr(env_cfg, "memory_trace", None),
                "decay",
                None,
            ),
            "num_envs": args_cli.num_envs,
            "num_episodes": args_cli.num_episodes,
            "test_delay_steps": args_cli.test_delay,
            "step_dt_seconds": float(core.step_dt),
            "test_delay_seconds": float(args_cli.test_delay * core.step_dt),
            "tracking_error_sample_phase": "pre_action",
            "mean_return": mean_return,
            "std_return": std_return,
            "mean_episode_length": mean_length,
            "std_episode_length": std_length,
            "mean_pre_action_xy_velocity_error": mean_pre_action_xy_error,
            "std_pre_action_xy_velocity_error": std_pre_action_xy_error,
            "mean_pre_action_yaw_velocity_error": mean_pre_action_yaw_error,
            "std_pre_action_yaw_velocity_error": std_pre_action_yaw_error,
            "failure_rate": failure_rate,
            "timeout_rate": timeout_rate,
            "vector_steps": vector_steps,
        }

        output_path = Path(args_cli.output).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with output_path.open("w", encoding="utf-8") as file:
            json.dump(result, file, indent=2)

        print(json.dumps(result, indent=2))
        print(f"[INFO] Saved evaluation to: {output_path}")

    finally:
        if env is not None:
            env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
