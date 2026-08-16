"""Benchmark exported TorchScript or ONNX policies without launching Isaac Sim."""

from __future__ import annotations

import argparse
import json
import math
import re
import time
from collections.abc import Callable
from pathlib import Path
from statistics import fmean
from typing import Any

import numpy as np


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Measure exported policy size, parameters, MACs, and latency.")
    parser.add_argument("--model", required=True, help="Path to policy.pt or policy.onnx.")
    parser.add_argument("--method", required=True, help="Method key stored in benchmark metadata.")
    parser.add_argument("--training_seed", type=int, default=None)
    parser.add_argument("--backend", choices=("auto", "torchscript", "onnx"), default="auto")
    parser.add_argument("--device", default="cpu", help="cpu or cuda:N.")
    parser.add_argument("--input_dim", type=int, default=None, help="Override the model input dimension.")
    parser.add_argument("--base_obs_dim", type=int, default=48)
    parser.add_argument("--memory_mode", choices=("auto", "none", "trace", "stack"), default="auto")
    parser.add_argument("--history_length", type=int, default=None)
    parser.add_argument("--trace_decay", type=float, default=0.8)
    parser.add_argument("--warmup", type=int, default=1000)
    parser.add_argument("--repetitions", type=int, default=10000)
    parser.add_argument("--output", default=None, help="Output JSON; defaults under results/benchmarks.")
    return parser


def resolve_backend(model_path: Path, requested: str) -> str:
    if requested != "auto":
        return requested
    if model_path.suffix.lower() == ".onnx":
        return "onnx"
    if model_path.suffix.lower() in {".pt", ".jit"}:
        return "torchscript"
    raise ValueError("Cannot infer backend. Use --backend torchscript or --backend onnx.")


def resolve_memory(method_key: str, requested: str, history_length: int | None) -> tuple[str, int | None]:
    if requested == "auto":
        stack_match = re.match(r"stack(\d+)", method_key)
        if stack_match is not None:
            return "stack", int(stack_match.group(1))
        if method_key.startswith("trace"):
            return "trace", None
        return "none", None
    if requested == "stack":
        if history_length is None or history_length <= 0:
            raise ValueError("--history_length must be positive for stack memory.")
        return requested, history_length
    return requested, history_length


def expected_input_dim(memory_mode: str, base_obs_dim: int, history_length: int | None) -> int:
    if base_obs_dim <= 0:
        raise ValueError("base_obs_dim must be positive.")
    if memory_mode == "trace":
        return 2 * base_obs_dim
    if memory_mode == "stack":
        if history_length is None or history_length <= 0:
            raise ValueError("history_length must be positive for stack memory.")
        return history_length * base_obs_dim
    return base_obs_dim


def latency_summary(samples_ms: list[float]) -> dict[str, float]:
    if not samples_ms:
        raise ValueError("Latency sample list must not be empty.")
    return {
        "mean_ms": fmean(samples_ms),
        "p50_ms": float(np.percentile(samples_ms, 50)),
        "p95_ms": float(np.percentile(samples_ms, 95)),
        "p99_ms": float(np.percentile(samples_ms, 99)),
    }


def time_callable(
    function: Callable[[], Any],
    *,
    warmup: int,
    repetitions: int,
    synchronize: Callable[[], None] | None = None,
) -> dict[str, float]:
    if warmup < 0 or repetitions <= 0:
        raise ValueError("warmup must be non-negative and repetitions must be positive.")
    for _ in range(warmup):
        function()
    if synchronize is not None:
        synchronize()

    samples_ms = []
    for _ in range(repetitions):
        start = time.perf_counter_ns()
        function()
        if synchronize is not None:
            synchronize()
        samples_ms.append((time.perf_counter_ns() - start) / 1.0e6)
    return latency_summary(samples_ms)


def torch_preprocessor(
    torch_module,
    *,
    memory_mode: str,
    base_obs_dim: int,
    history_length: int | None,
    trace_decay: float,
    device: str,
):
    torch = torch_module
    observation = torch.zeros((1, base_obs_dim), dtype=torch.float32, device=device)
    if memory_mode == "trace":
        trace = torch.zeros_like(observation)

        def prepare():
            trace.mul_(trace_decay).add_(observation, alpha=1.0 - trace_decay)
            return torch.cat((observation, trace), dim=-1)

        return prepare
    if memory_mode == "stack":
        assert history_length is not None
        history = torch.zeros((1, history_length, base_obs_dim), dtype=torch.float32, device=device)

        def prepare():
            history.copy_(torch.roll(history, shifts=-1, dims=1))
            history[:, -1].copy_(observation)
            return history.flatten(start_dim=1)

        return prepare
    return lambda: observation


def numpy_preprocessor(
    *,
    memory_mode: str,
    base_obs_dim: int,
    history_length: int | None,
    trace_decay: float,
):
    observation = np.zeros((1, base_obs_dim), dtype=np.float32)
    if memory_mode == "trace":
        trace = np.zeros_like(observation)

        def prepare():
            trace[:] = trace_decay * trace + (1.0 - trace_decay) * observation
            return np.concatenate((observation, trace), axis=-1)

        return prepare
    if memory_mode == "stack":
        assert history_length is not None
        history = np.zeros((1, history_length, base_obs_dim), dtype=np.float32)

        def prepare():
            history[:, :-1] = history[:, 1:]
            history[:, -1] = observation
            return history.reshape(1, -1)

        return prepare
    return lambda: observation


def benchmark_torchscript(
    model_path: Path,
    *,
    device: str,
    input_dim: int,
    memory_mode: str,
    base_obs_dim: int,
    history_length: int | None,
    trace_decay: float,
    warmup: int,
    repetitions: int,
) -> dict[str, Any]:
    import torch

    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA was requested but is not available: {device}")
    model = torch.jit.load(str(model_path), map_location=device).eval()
    parameters = list(model.parameters())
    parameter_count = sum(parameter.numel() for parameter in parameters)
    macs = sum(parameter.numel() for parameter in parameters if parameter.ndim == 2)
    prepare = torch_preprocessor(
        torch,
        memory_mode=memory_mode,
        base_obs_dim=base_obs_dim,
        history_length=history_length,
        trace_decay=trace_decay,
        device=device,
    )
    ready_input = torch.zeros((1, input_dim), dtype=torch.float32, device=device)
    synchronize = (lambda: torch.cuda.synchronize(device)) if device.startswith("cuda") else None

    with torch.inference_mode():
        model_latency = time_callable(
            lambda: model(ready_input),
            warmup=warmup,
            repetitions=repetitions,
            synchronize=synchronize,
        )
        end_to_end_latency = time_callable(
            lambda: model(prepare()),
            warmup=warmup,
            repetitions=repetitions,
            synchronize=synchronize,
        )
    return {
        "parameter_count": parameter_count,
        "macs_per_inference": macs,
        "input_shape": [1, input_dim],
        "model_latency": model_latency,
        "end_to_end_latency": end_to_end_latency,
    }


def onnx_input_shape(session, input_dim: int) -> list[int]:
    shape = []
    for index, dimension in enumerate(session.get_inputs()[0].shape):
        if index == len(session.get_inputs()[0].shape) - 1:
            shape.append(input_dim)
        elif isinstance(dimension, int) and dimension > 0:
            shape.append(dimension)
        else:
            shape.append(1)
    return shape


def benchmark_onnx(
    model_path: Path,
    *,
    device: str,
    input_dim: int,
    memory_mode: str,
    base_obs_dim: int,
    history_length: int | None,
    trace_decay: float,
    warmup: int,
    repetitions: int,
) -> dict[str, Any]:
    try:
        import onnx
        import onnxruntime as ort
        from onnx import numpy_helper
    except ImportError as error:
        raise RuntimeError("ONNX benchmarking requires onnx and onnxruntime.") from error

    requested_provider = "CUDAExecutionProvider" if device.startswith("cuda") else "CPUExecutionProvider"
    if requested_provider not in ort.get_available_providers():
        raise RuntimeError(f"ONNX Runtime provider is unavailable: {requested_provider}")
    if device.startswith("cuda"):
        _, _, device_index = device.partition(":")
        provider_options = {"device_id": int(device_index or "0")}
        providers: list[Any] = [(requested_provider, provider_options)]
    else:
        providers = [requested_provider]
    session = ort.InferenceSession(str(model_path), providers=providers)
    if len(session.get_inputs()) != 1:
        raise ValueError("Only single-input exported policies are supported.")
    input_info = session.get_inputs()[0]
    input_shape = onnx_input_shape(session, input_dim)
    ready_input = np.zeros(input_shape, dtype=np.float32)
    prepare = numpy_preprocessor(
        memory_mode=memory_mode,
        base_obs_dim=base_obs_dim,
        history_length=history_length,
        trace_decay=trace_decay,
    )
    model = onnx.load(str(model_path), load_external_data=False)
    initializer_arrays = [numpy_helper.to_array(initializer) for initializer in model.graph.initializer]
    parameter_count = sum(array.size for array in initializer_arrays)
    macs = sum(array.size for array in initializer_arrays if array.ndim == 2)

    def model_call():
        return session.run(None, {input_info.name: ready_input})

    def end_to_end_call():
        return session.run(None, {input_info.name: prepare().reshape(input_shape)})

    return {
        "parameter_count": int(parameter_count),
        "macs_per_inference": int(macs),
        "input_shape": input_shape,
        "model_latency": time_callable(model_call, warmup=warmup, repetitions=repetitions),
        "end_to_end_latency": time_callable(end_to_end_call, warmup=warmup, repetitions=repetitions),
    }


def default_output_path(repo_root: Path, method_key: str, backend: str, device: str, training_seed: int | None) -> Path:
    safe_device = device.replace(":", "-")
    seed_suffix = f"_trainseed{training_seed}" if training_seed is not None else ""
    return repo_root / "results" / "benchmarks" / f"{method_key}{seed_suffix}_{backend}_{safe_device}.json"


def main() -> None:
    args = build_parser().parse_args()
    if not 0.0 <= args.trace_decay < 1.0:
        raise ValueError("trace_decay must satisfy 0 <= decay < 1.")
    if args.input_dim is not None and args.input_dim <= 0:
        raise ValueError("input_dim must be positive.")

    repo_root = Path(__file__).resolve().parents[1]
    model_path = Path(args.model).expanduser().resolve()
    if not model_path.is_file():
        raise FileNotFoundError(f"Policy model does not exist: {model_path}")
    backend = resolve_backend(model_path, args.backend)
    memory_mode, history_length = resolve_memory(args.method, args.memory_mode, args.history_length)
    inferred_input_dim = expected_input_dim(memory_mode, args.base_obs_dim, history_length)
    input_dim = args.input_dim or inferred_input_dim
    if input_dim != inferred_input_dim:
        raise ValueError(
            f"Configured input_dim={input_dim} disagrees with {memory_mode} "
            f"preprocessing dimension {inferred_input_dim}."
        )

    common = {
        "device": args.device,
        "input_dim": input_dim,
        "memory_mode": memory_mode,
        "base_obs_dim": args.base_obs_dim,
        "history_length": history_length,
        "trace_decay": args.trace_decay,
        "warmup": args.warmup,
        "repetitions": args.repetitions,
    }
    if backend == "torchscript":
        measurements = benchmark_torchscript(model_path, **common)
    else:
        measurements = benchmark_onnx(model_path, **common)

    end_to_end_mean = measurements["end_to_end_latency"]["mean_ms"]
    result = {
        "method_key": args.method,
        "training_seed": args.training_seed,
        "model_path": str(model_path),
        "backend": backend,
        "device": args.device,
        "model_size_bytes": model_path.stat().st_size,
        "memory_mode": memory_mode,
        "base_obs_dim": args.base_obs_dim,
        "history_length": history_length,
        "trace_decay": args.trace_decay if memory_mode == "trace" else None,
        "warmup": args.warmup,
        "repetitions": args.repetitions,
        **measurements,
        "end_to_end_inferences_per_second": 1000.0 / end_to_end_mean if end_to_end_mean > 0.0 else math.inf,
    }
    output_path = (
        Path(args.output).expanduser()
        if args.output
        else default_output_path(repo_root, args.method, backend, args.device, args.training_seed)
    )
    output_path = output_path if output_path.is_absolute() else repo_root / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(result, file, indent=2)
    print(json.dumps(result, indent=2))
    print(f"[INFO] Saved benchmark: {output_path}")


if __name__ == "__main__":
    main()
