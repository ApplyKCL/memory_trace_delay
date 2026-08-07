"""Aggregate MLP and Stack-PPO delay-evaluation results."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description="Aggregate action-delay evaluation results.")

parser.add_argument(
    "--mlp_dir",
    type=str,
    default="results/mlp",
    help="Directory containing MLP evaluation JSON files.",
)

parser.add_argument(
    "--stack_dir",
    type=str,
    default="results/stack",
    help="Directory containing Stack-PPO evaluation JSON files.",
)

parser.add_argument(
    "--output_csv",
    type=str,
    default="results/combined_summary.csv",
    help="Output summary CSV.",
)

parser.add_argument(
    "--figure_dir",
    type=str,
    default="results/figures",
    help="Output directory for figures.",
)

args = parser.parse_args()


METHOD_LABELS = {
    "train0": "Fixed delay 0 MLP",
    "train2": "Fixed delay 2 MLP",
    "train0to4": "Random delay 0-4 MLP",
    "stack5": "Random delay 0-4 Stack-5",
}


def parse_mlp_filename(path: Path) -> tuple[str, int, int]:
    """Parse metadata from an MLP evaluation filename."""

    pattern = re.compile(r"^(train0to4|train0|train2)_test(\d+)_seed(\d+)\.json$")

    match = pattern.match(path.name)

    if match is None:
        raise ValueError(f"Unexpected MLP filename: {path.name}")

    method = match.group(1)
    test_delay = int(match.group(2))
    seed = int(match.group(3))

    return method, test_delay, seed


def parse_stack_filename(path: Path) -> tuple[str, int, int]:
    """Parse metadata from a Stack-5 evaluation filename."""

    pattern = re.compile(r"^stack5_train0to4_test(\d+)_seed(\d+)\.json$")

    match = pattern.match(path.name)

    if match is None:
        raise ValueError(f"Unexpected Stack filename: {path.name}")

    test_delay = int(match.group(1))
    seed = int(match.group(2))

    return "stack5", test_delay, seed


def read_json(path: Path) -> dict:
    """Read one JSON result file."""

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def make_row(
    result: dict,
    method_key: str,
    test_delay: int,
    seed: int,
) -> dict:
    """Convert one evaluation JSON into one summary row."""

    return {
        "method_key": method_key,
        "method": METHOD_LABELS[method_key],
        "test_delay": test_delay,
        "seed": seed,
        "mean_return": result["mean_return"],
        "std_return": result["std_return"],
        "mean_episode_length": result["mean_episode_length"],
        "mean_xy_error": result["mean_pre_action_xy_velocity_error"],
        "mean_yaw_error": result["mean_pre_action_yaw_velocity_error"],
        "failure_rate": result["failure_rate"],
        "timeout_rate": result["timeout_rate"],
        "num_episodes": result["num_episodes"],
        "checkpoint": result["checkpoint"],
    }


def load_mlp_results(directory: Path) -> list[dict]:
    """Load all valid MLP evaluation results."""

    rows = []

    for path in sorted(directory.glob("*.json")):
        try:
            method_key, test_delay, seed = parse_mlp_filename(path)
        except ValueError:
            print(f"[SKIP] {path}")
            continue

        result = read_json(path)

        rows.append(
            make_row(
                result,
                method_key,
                test_delay,
                seed,
            )
        )

    return rows


def load_stack_results(directory: Path) -> list[dict]:
    """Load all valid Stack-5 evaluation results."""

    rows = []

    for path in sorted(directory.glob("*.json")):
        try:
            method_key, test_delay, seed = parse_stack_filename(path)
        except ValueError:
            print(f"[SKIP] {path}")
            continue

        result = read_json(path)

        rows.append(
            make_row(
                result,
                method_key,
                test_delay,
                seed,
            )
        )

    return rows


def save_csv(rows: list[dict], output_path: Path) -> None:
    """Save combined evaluation results."""

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = [
        "method",
        "test_delay",
        "seed",
        "mean_return",
        "std_return",
        "mean_episode_length",
        "mean_xy_error",
        "mean_yaw_error",
        "failure_rate",
        "timeout_rate",
        "num_episodes",
        "checkpoint",
    ]

    with output_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for row in rows:
            writer.writerow({key: row[key] for key in fieldnames})


def plot_metric(
    rows: list[dict],
    metric: str,
    ylabel: str,
    output_path: Path,
) -> None:
    """Plot one metric against test delay."""

    plt.figure(figsize=(7, 5))

    for method_key, label in METHOD_LABELS.items():
        method_rows = [row for row in rows if row["method_key"] == method_key]

        if not method_rows:
            continue

        method_rows.sort(key=lambda row: row["test_delay"])

        delays = [row["test_delay"] for row in method_rows]

        values = [row[metric] for row in method_rows]

        plt.plot(
            delays,
            values,
            marker="o",
            label=label,
        )

    plt.xlabel("Test action delay (steps)")
    plt.ylabel(ylabel)

    plt.xticks([0, 2, 4, 6])

    plt.grid(
        True,
        alpha=0.3,
    )

    plt.legend()

    plt.tight_layout()

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    plt.savefig(
        output_path,
        dpi=200,
    )

    plt.close()


def print_stack_vs_mlp(rows: list[dict]) -> None:
    """Print Random-MLP vs Stack-5 comparison."""

    print()
    print("Random MLP vs Stack-5")
    print("=" * 78)

    for delay in (0, 2, 4, 6):
        mlp_row = next(
            (row for row in rows if row["method_key"] == "train0to4" and row["test_delay"] == delay),
            None,
        )

        stack_row = next(
            (row for row in rows if row["method_key"] == "stack5" and row["test_delay"] == delay),
            None,
        )

        if mlp_row is None or stack_row is None:
            continue

        print(f"\nTest delay = {delay}")

        print(f"  Return:   MLP={mlp_row['mean_return']:.3f} | Stack={stack_row['mean_return']:.3f}")

        print(f"  XY error: MLP={mlp_row['mean_xy_error']:.3f} | Stack={stack_row['mean_xy_error']:.3f}")

        print(f"  Failure:  MLP={mlp_row['failure_rate']:.3f} | Stack={stack_row['failure_rate']:.3f}")


def main() -> None:
    """Load results, save summary, and generate plots."""

    mlp_dir = Path(args.mlp_dir)
    stack_dir = Path(args.stack_dir)

    rows = []

    rows.extend(load_mlp_results(mlp_dir))

    rows.extend(load_stack_results(stack_dir))

    if not rows:
        raise RuntimeError("No evaluation results were found.")

    method_order = list(METHOD_LABELS)

    rows.sort(
        key=lambda row: (
            method_order.index(row["method_key"]),
            row["test_delay"],
        )
    )

    print(f"[INFO] Loaded {len(rows)} evaluation results.")

    for row in rows:
        print(
            f"{row['method']:28s} | "
            f"test={row['test_delay']} | "
            f"return={row['mean_return']:.3f} | "
            f"xy={row['mean_xy_error']:.3f} | "
            f"failure={row['failure_rate']:.3f}"
        )

    output_csv = Path(args.output_csv)
    figure_dir = Path(args.figure_dir)

    save_csv(
        rows,
        output_csv,
    )

    plot_metric(
        rows,
        metric="mean_return",
        ylabel="Mean episode return",
        output_path=(figure_dir / "return_vs_delay_with_stack.png"),
    )

    plot_metric(
        rows,
        metric="mean_xy_error",
        ylabel="Mean XY velocity tracking error (m/s)",
        output_path=(figure_dir / "xy_error_vs_delay_with_stack.png"),
    )

    plot_metric(
        rows,
        metric="failure_rate",
        ylabel="Failure rate",
        output_path=(figure_dir / "failure_rate_vs_delay_with_stack.png"),
    )

    print_stack_vs_mlp(rows)

    print()
    print(f"[INFO] Saved CSV to: {output_csv}")
    print(f"[INFO] Saved figures to: {figure_dir}")


if __name__ == "__main__":
    main()
