from __future__ import annotations

import time
from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager
from typing import Any

import torch


def _profile_stats(values: list[float]) -> tuple[float, float]:
    t = torch.tensor(values)
    return float(t.mean().item()), float(t.std(unbiased=False).item())


def _n_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


@contextmanager
def _stable_cuda_benchmark(device: torch.device, enabled: bool):
    """Temporarily fix CUDA backend heuristics/math mode for reproducible microbenchmarks."""
    if not enabled or device.type != "cuda":
        yield
        return

    prev_cudnn_benchmark = torch.backends.cudnn.benchmark
    prev_cudnn_deterministic = torch.backends.cudnn.deterministic
    prev_matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
    prev_cudnn_tf32 = torch.backends.cudnn.allow_tf32
    try:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        yield
    finally:
        torch.backends.cudnn.benchmark = prev_cudnn_benchmark
        torch.backends.cudnn.deterministic = prev_cudnn_deterministic
        torch.backends.cuda.matmul.allow_tf32 = prev_matmul_tf32
        torch.backends.cudnn.allow_tf32 = prev_cudnn_tf32


def _first_tensor(obj: Any) -> torch.Tensor | None:
    if torch.is_tensor(obj):
        return obj
    if isinstance(obj, Mapping):
        for value in obj.values():
            t = _first_tensor(value)
            if t is not None:
                return t
        return None
    if isinstance(obj, Sequence) and not isinstance(obj, (str, bytes)):
        for value in obj:
            t = _first_tensor(value)
            if t is not None:
                return t
    return None


def _move_to_device_dtype(obj: Any, device: torch.device, dtype: torch.dtype) -> Any:
    if torch.is_tensor(obj):
        if obj.is_floating_point():
            return obj.to(device=device, dtype=dtype)
        return obj.to(device=device)
    if isinstance(obj, Mapping):
        return {k: _move_to_device_dtype(v, device=device, dtype=dtype) for k, v in obj.items()}
    if isinstance(obj, tuple):
        return tuple(_move_to_device_dtype(v, device=device, dtype=dtype) for v in obj)
    if isinstance(obj, list):
        return [_move_to_device_dtype(v, device=device, dtype=dtype) for v in obj]
    return obj


def _default_forward(model: torch.nn.Module, batch: Any) -> Any:
    if isinstance(batch, Mapping):
        return model(**batch)
    if isinstance(batch, tuple):
        return model(*batch)
    if isinstance(batch, list):
        return model(*batch)
    return model(batch)


def _iter_tensors(obj: Any):
    if torch.is_tensor(obj):
        yield obj
        return
    if isinstance(obj, Mapping):
        for value in obj.values():
            yield from _iter_tensors(value)
        return
    if isinstance(obj, Sequence) and not isinstance(obj, (str, bytes)):
        for value in obj:
            yield from _iter_tensors(value)


def _default_loss(output: Any) -> torch.Tensor:
    tensors = [t for t in _iter_tensors(output) if torch.is_tensor(t) and t.is_floating_point()]
    if len(tensors) == 0:
        raise TypeError(
            "Default loss cannot be computed: forward output contains no floating-point tensors. "
            "Provide a custom loss_fn(output) -> scalar tensor."
        )
    return sum(t.square().mean() for t in tensors)


def _resolve_device_dtype(
    model: torch.nn.Module,
    batch: Any,
    device: torch.device | None,
    dtype: torch.dtype | None,
) -> tuple[torch.device, torch.dtype]:
    if device is None:
        try:
            device = next(model.parameters()).device
        except StopIteration:
            t = _first_tensor(batch)
            device = t.device if t is not None else torch.device("cpu")

    if dtype is None:
        dtypes = [p.dtype for p in model.parameters() if p.is_floating_point()]
        if len(dtypes) > 0:
            dtype = dtypes[0]
        else:
            t = _first_tensor(batch)
            dtype = t.dtype if t is not None and t.is_floating_point() else torch.float32

    return device, dtype


def _prime_eval_cache(
    model: torch.nn.Module,
    batch: Any,
    device: torch.device,
    forward_fn: Callable[[torch.nn.Module, Any], Any],
) -> None:
    """Populate eval caches once outside measured/profiled loops."""
    model.eval()
    with torch.no_grad():
        _ = forward_fn(model, batch)
    _sync(device)


def _run_timing(
    model: torch.nn.Module,
    batch: Any,
    device: torch.device,
    n_steps: int,
    warmup_steps: int,
    forward_fn: Callable[[torch.nn.Module, Any], Any],
    loss_fn: Callable[[Any], torch.Tensor],
) -> dict[str, float]:
    model.train()
    train_forward_ms: list[float] = []
    backward_ms: list[float] = []
    use_cuda_events = device.type == "cuda"
    if use_cuda_events:
        fwd_start = torch.cuda.Event(enable_timing=True)
        fwd_end = torch.cuda.Event(enable_timing=True)
        bwd_start = torch.cuda.Event(enable_timing=True)
        bwd_end = torch.cuda.Event(enable_timing=True)
    for i in range(warmup_steps + n_steps):
        model.zero_grad(set_to_none=True)
        if use_cuda_events:
            fwd_start.record()
            out = forward_fn(model, batch)
            fwd_end.record()
            fwd_end.synchronize()
            forward_ms = float(fwd_start.elapsed_time(fwd_end))
        else:
            _sync(device)
            t0 = time.perf_counter()
            out = forward_fn(model, batch)
            _sync(device)
            t1 = time.perf_counter()
            forward_ms = (t1 - t0) * 1e3

        loss = loss_fn(out)
        if use_cuda_events:
            bwd_start.record()
            loss.backward()
            bwd_end.record()
            bwd_end.synchronize()
            backward_step_ms = float(bwd_start.elapsed_time(bwd_end))
        else:
            _sync(device)
            t2 = time.perf_counter()
            loss.backward()
            _sync(device)
            t3 = time.perf_counter()
            backward_step_ms = (t3 - t2) * 1e3

        if i >= warmup_steps:
            train_forward_ms.append(forward_ms)
            backward_ms.append(backward_step_ms)

    model.eval()
    _prime_eval_cache(model, batch, device, forward_fn=forward_fn)
    val_forward_ms: list[float] = []
    if use_cuda_events:
        eval_start = torch.cuda.Event(enable_timing=True)
        eval_end = torch.cuda.Event(enable_timing=True)
    with torch.no_grad():
        for i in range(warmup_steps + n_steps):
            if use_cuda_events:
                eval_start.record()
                _ = forward_fn(model, batch)
                eval_end.record()
                eval_end.synchronize()
                forward_ms = float(eval_start.elapsed_time(eval_end))
            else:
                _sync(device)
                t0 = time.perf_counter()
                _ = forward_fn(model, batch)
                _sync(device)
                t1 = time.perf_counter()
                forward_ms = (t1 - t0) * 1e3
            if i >= warmup_steps:
                val_forward_ms.append(forward_ms)

    f_m, f_s = _profile_stats(train_forward_ms)
    b_m, b_s = _profile_stats(backward_ms)
    v_m, v_s = _profile_stats(val_forward_ms)
    return {
        "train_forward_ms_mean": f_m,
        "train_forward_ms_std": f_s,
        "backward_ms_mean": b_m,
        "backward_ms_std": b_s,
        "val_forward_ms_mean": v_m,
        "val_forward_ms_std": v_s,
    }


def _run_module_pair_timing_interleaved(
    lhs: torch.nn.Module,
    rhs: torch.nn.Module,
    batch: Any,
    device: torch.device,
    n_steps: int,
    warmup_steps: int,
    lhs_forward_fn: Callable[[torch.nn.Module, Any], Any],
    rhs_forward_fn: Callable[[torch.nn.Module, Any], Any],
    lhs_loss_fn: Callable[[Any], torch.Tensor],
    rhs_loss_fn: Callable[[Any], torch.Tensor],
) -> tuple[dict[str, float], dict[str, float]]:
    """Benchmark two modules in one alternating loop to reduce drift bias."""
    lhs.train()
    rhs.train()

    stats = {
        "lhs": {"train_forward_ms": [], "backward_ms": [], "val_forward_ms": []},
        "rhs": {"train_forward_ms": [], "backward_ms": [], "val_forward_ms": []},
    }
    models = {
        "lhs": (lhs, lhs_forward_fn, lhs_loss_fn),
        "rhs": (rhs, rhs_forward_fn, rhs_loss_fn),
    }

    use_cuda_events = device.type == "cuda"
    if use_cuda_events:
        events = {
            "lhs": {
                "fwd": (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)),
                "bwd": (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)),
                "eval": (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)),
            },
            "rhs": {
                "fwd": (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)),
                "bwd": (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)),
                "eval": (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)),
            },
        }

    def _timed_forward(side: str, model: torch.nn.Module, forward_fn: Callable[[torch.nn.Module, Any], Any]):
        if use_cuda_events:
            start, end = events[side]["fwd"]
            start.record()
            out = forward_fn(model, batch)
            end.record()
            end.synchronize()
            return out, float(start.elapsed_time(end))

        _sync(device)
        t0 = time.perf_counter()
        out = forward_fn(model, batch)
        _sync(device)
        t1 = time.perf_counter()
        return out, (t1 - t0) * 1e3

    def _timed_eval_forward(side: str, model: torch.nn.Module, forward_fn: Callable[[torch.nn.Module, Any], Any]):
        if use_cuda_events:
            start, end = events[side]["eval"]
            start.record()
            out = forward_fn(model, batch)
            end.record()
            end.synchronize()
            return out, float(start.elapsed_time(end))

        _sync(device)
        t0 = time.perf_counter()
        out = forward_fn(model, batch)
        _sync(device)
        t1 = time.perf_counter()
        return out, (t1 - t0) * 1e3

    def _timed_backward(side: str, loss: torch.Tensor) -> float:
        if use_cuda_events:
            start, end = events[side]["bwd"]
            start.record()
            loss.backward()
            end.record()
            end.synchronize()
            return float(start.elapsed_time(end))

        _sync(device)
        t0 = time.perf_counter()
        loss.backward()
        _sync(device)
        t1 = time.perf_counter()
        return (t1 - t0) * 1e3

    for i in range(warmup_steps + n_steps):
        order = ("lhs", "rhs") if i % 2 == 0 else ("rhs", "lhs")
        for side in order:
            model, forward_fn, loss_fn = models[side]
            model.zero_grad(set_to_none=True)
            out, forward_ms = _timed_forward(side, model, forward_fn)
            loss = loss_fn(out)
            backward_ms = _timed_backward(side, loss)
            if i >= warmup_steps:
                stats[side]["train_forward_ms"].append(forward_ms)
                stats[side]["backward_ms"].append(backward_ms)

    lhs.eval()
    rhs.eval()
    _prime_eval_cache(lhs, batch, device, forward_fn=lhs_forward_fn)
    _prime_eval_cache(rhs, batch, device, forward_fn=rhs_forward_fn)
    with torch.no_grad():
        for i in range(warmup_steps + n_steps):
            order = ("lhs", "rhs") if i % 2 == 0 else ("rhs", "lhs")
            for side in order:
                model, forward_fn, _ = models[side]
                _, forward_ms = _timed_eval_forward(side, model, forward_fn)
                if i >= warmup_steps:
                    stats[side]["val_forward_ms"].append(forward_ms)

    lhs_train_mean, lhs_train_std = _profile_stats(stats["lhs"]["train_forward_ms"])
    lhs_bwd_mean, lhs_bwd_std = _profile_stats(stats["lhs"]["backward_ms"])
    lhs_eval_mean, lhs_eval_std = _profile_stats(stats["lhs"]["val_forward_ms"])
    rhs_train_mean, rhs_train_std = _profile_stats(stats["rhs"]["train_forward_ms"])
    rhs_bwd_mean, rhs_bwd_std = _profile_stats(stats["rhs"]["backward_ms"])
    rhs_eval_mean, rhs_eval_std = _profile_stats(stats["rhs"]["val_forward_ms"])

    lhs_timing = {
        "train_forward_ms_mean": lhs_train_mean,
        "train_forward_ms_std": lhs_train_std,
        "backward_ms_mean": lhs_bwd_mean,
        "backward_ms_std": lhs_bwd_std,
        "val_forward_ms_mean": lhs_eval_mean,
        "val_forward_ms_std": lhs_eval_std,
    }
    rhs_timing = {
        "train_forward_ms_mean": rhs_train_mean,
        "train_forward_ms_std": rhs_train_std,
        "backward_ms_mean": rhs_bwd_mean,
        "backward_ms_std": rhs_bwd_std,
        "val_forward_ms_mean": rhs_eval_mean,
        "val_forward_ms_std": rhs_eval_std,
    }
    return lhs_timing, rhs_timing


def _profile_ops(
    model: torch.nn.Module,
    batch: Any,
    device: torch.device,
    mode: str,
    active_steps: int,
    warmup_steps: int,
    forward_fn: Callable[[torch.nn.Module, Any], Any],
    loss_fn: Callable[[Any], torch.Tensor],
) -> dict[str, float]:
    activities = [torch.profiler.ProfilerActivity.CPU]
    use_cuda = device.type == "cuda"
    if use_cuda:
        activities.append(torch.profiler.ProfilerActivity.CUDA)

    model.train(mode == "train")
    if mode == "eval":
        _prime_eval_cache(model, batch, device, forward_fn=forward_fn)
    prof_warmup_steps = warmup_steps if mode == "train" else 0
    profile_kwargs = {
        "activities": activities,
        "record_shapes": False,
        "with_stack": False,
    }
    if prof_warmup_steps > 0:
        profile_kwargs["schedule"] = torch.profiler.schedule(
            wait=0,
            warmup=prof_warmup_steps,
            active=active_steps,
            repeat=1,
        )
    with torch.profiler.profile(**profile_kwargs) as prof:
        for _ in range(prof_warmup_steps + active_steps):
            if mode == "train":
                model.zero_grad(set_to_none=True)
                out = forward_fn(model, batch)
                loss = loss_fn(out)
                loss.backward()
            else:
                with torch.no_grad():
                    _ = forward_fn(model, batch)
            prof.step()
    _sync(device)

    op_to_ms_per_step: dict[str, float] = {}
    for item in prof.key_averages():
        op_name = item.key
        if not op_name.startswith("aten::"):
            continue
        if use_cuda:
            self_time_us = getattr(item, "self_cuda_time_total", None)
            if self_time_us is None:
                self_time_us = getattr(item, "self_device_time_total", 0.0)
        else:
            self_time_us = getattr(item, "self_cpu_time_total", 0.0)
        self_time_us = float(self_time_us)
        if self_time_us <= 0:
            continue
        op_to_ms_per_step[op_name] = self_time_us / 1000.0 / active_steps
    return op_to_ms_per_step


def _fmt_ms(mean_ms: float, std_ms: float) -> str:
    return f"{mean_ms:7.3f}±{std_ms:6.3f} ms"


def _print_top_ops(title: str, ops: dict[str, float], top_k: int) -> None:
    print(title)
    print(f"{'Op':<30} {'ms/step':>12}")
    print("-" * 44)
    for op_name, ms in sorted(ops.items(), key=lambda kv: kv[1], reverse=True)[:top_k]:
        print(f"{op_name:<30.30} {ms:10.3f}")


def profile_module(
    name: str,
    module: torch.nn.Module,
    batch: Any,
    *,
    profile_active_steps: int = 25,
    profile_warmup_steps: int = 5,
    mode: str = "eval",
    top_k: int = 10,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
    forward_fn: Callable[[torch.nn.Module, Any], Any] | None = None,
    loss_fn: Callable[[Any], torch.Tensor] | None = None,
    stabilize_cuda: bool = True,
    print_report: bool = True,
) -> dict[str, object]:
    """Profile a single module with generic input/output handling.

    Args:
        name: Display name.
        module: Module to profile.
        batch: Input batch passed to the model. Supported types: tensor, tuple/list, dict.
        profile_active_steps: Number of measured iterations.
        profile_warmup_steps: Warmup iterations.
        mode: One of {"eval", "train", "both"} for op-level profiling.
        top_k: Number of top ops to print per mode.
        device: Optional target device; inferred if None.
        dtype: Optional floating dtype; inferred if None.
        forward_fn: Optional custom forward wrapper, called as `forward_fn(module, batch)`.
        loss_fn: Optional custom scalar loss for backward-mode profiling.
        stabilize_cuda: If True, temporarily disables autotune/TF32 and enables deterministic CuDNN.
        print_report: Whether to print a report.

    Returns:
        Dict containing timing and op profile details.
    """
    if mode not in {"eval", "train", "both"}:
        raise ValueError("mode must be one of: eval, train, both")

    if isinstance(device, str):
        device = torch.device(device)

    device, dtype = _resolve_device_dtype(module, batch, device=device, dtype=dtype)
    module = module.to(device=device, dtype=dtype)
    batch = _move_to_device_dtype(batch, device=device, dtype=dtype)

    if forward_fn is None:
        forward_fn = _default_forward
    if loss_fn is None:
        loss_fn = _default_loss

    n_steps = profile_active_steps
    warmup_steps = profile_warmup_steps
    with _stable_cuda_benchmark(device=device, enabled=stabilize_cuda):
        timing = _run_timing(
            module,
            batch=batch,
            device=device,
            n_steps=n_steps,
            warmup_steps=warmup_steps,
            forward_fn=forward_fn,
            loss_fn=loss_fn,
        )

        profile_modes = ["eval", "train"] if mode == "both" else [mode]
        mode_results: dict[str, dict[str, object]] = {}
        for profile_mode in profile_modes:
            ops = _profile_ops(
                module,
                batch=batch,
                device=device,
                mode=profile_mode,
                active_steps=profile_active_steps,
                warmup_steps=profile_warmup_steps,
                forward_fn=forward_fn,
                loss_fn=loss_fn,
            )
            mode_results[profile_mode] = {"ops": ops}

    input_tensor = _first_tensor(batch)
    batch_size = int(input_tensor.shape[0]) if input_tensor is not None and input_tensor.ndim >= 1 else -1
    input_dim = int(input_tensor.shape[-1]) if input_tensor is not None and input_tensor.ndim >= 1 else -1

    result: dict[str, object] = {
        "name": name,
        "module": module,
        "timing": timing,
        "modes": mode_results,
        "n_parameters": _n_parameters(module),
        "device": device,
        "dtype": dtype,
        "batch_size": batch_size,
        "input_dim": input_dim,
    }

    if print_report:
        print(f"Module: {name}")
        print(f"Device: {device} | dtype: {dtype} | batch: {batch_size} | input dim: {input_dim}")
        print(f"Timing steps: warmup={warmup_steps}, measured={n_steps}")
        print()
        print("Average Runtime (ms, mean±std)")
        print(f"{'Module':<12} {'#params':>10} {'TrainFwd':>18} {'Backward':>18} {'ValFwd':>18}")
        print("-" * 78)
        print(
            f"{name:<12} {_n_parameters(module):10d} "
            f"{_fmt_ms(timing['train_forward_ms_mean'], timing['train_forward_ms_std']):>18} "
            f"{_fmt_ms(timing['backward_ms_mean'], timing['backward_ms_std']):>18} "
            f"{_fmt_ms(timing['val_forward_ms_mean'], timing['val_forward_ms_std']):>18}"
        )
        for profile_mode in profile_modes:
            print()
            _print_top_ops(f"[{profile_mode}] Top ops for {name}", mode_results[profile_mode]["ops"], top_k=top_k)

    return result


def profile(
    name: str,
    module: torch.nn.Module,
    batch: Any,
    **kwargs,
) -> dict[str, object]:
    """Public generic profiler entrypoint.

    This is a thin wrapper over `profile_module(...)` so callers can consistently use `profile(...)`
    for any module/batch combination.
    """
    return profile_module(name=name, module=module, batch=batch, **kwargs)


def run_module_pair_profile(
    lhs_name: str,
    lhs: torch.nn.Module,
    rhs_name: str,
    rhs: torch.nn.Module,
    x: Any,
    *,
    group_name: str | None = None,
    profile_active_steps: int = 25,
    profile_warmup_steps: int = 5,
    mode: str = "eval",
    top_k: int = 10,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
    lhs_forward_fn: Callable[[torch.nn.Module, Any], Any] | None = None,
    rhs_forward_fn: Callable[[torch.nn.Module, Any], Any] | None = None,
    lhs_loss_fn: Callable[[Any], torch.Tensor] | None = None,
    rhs_loss_fn: Callable[[Any], torch.Tensor] | None = None,
    stabilize_cuda: bool = True,
    interleave_timing: bool = True,
) -> dict[str, object]:
    """Run timing + op profiling for two modules on the same input structure.

    The input ``x`` may be a Tensor, tuple/list, or dict depending on the module signature.
    Timing is interleaved by default to reduce bias from clock, thermal, and allocator drift.
    """
    if mode not in {"eval", "train", "both"}:
        raise ValueError("mode must be one of: eval, train, both")

    if isinstance(device, str):
        device = torch.device(device)

    device, dtype = _resolve_device_dtype(lhs, x, device=device, dtype=dtype)
    lhs = lhs.to(device=device, dtype=dtype)
    rhs = rhs.to(device=device, dtype=dtype)
    x = _move_to_device_dtype(x, device=device, dtype=dtype)

    if lhs_forward_fn is None:
        lhs_forward_fn = _default_forward
    if rhs_forward_fn is None:
        rhs_forward_fn = _default_forward
    if lhs_loss_fn is None:
        lhs_loss_fn = _default_loss
    if rhs_loss_fn is None:
        rhs_loss_fn = _default_loss

    with _stable_cuda_benchmark(device=device, enabled=stabilize_cuda):
        if interleave_timing:
            lhs_timing, rhs_timing = _run_module_pair_timing_interleaved(
                lhs=lhs,
                rhs=rhs,
                batch=x,
                device=device,
                n_steps=profile_active_steps,
                warmup_steps=profile_warmup_steps,
                lhs_forward_fn=lhs_forward_fn,
                rhs_forward_fn=rhs_forward_fn,
                lhs_loss_fn=lhs_loss_fn,
                rhs_loss_fn=rhs_loss_fn,
            )
        else:
            lhs_timing = _run_timing(
                lhs,
                batch=x,
                device=device,
                n_steps=profile_active_steps,
                warmup_steps=profile_warmup_steps,
                forward_fn=lhs_forward_fn,
                loss_fn=lhs_loss_fn,
            )
            rhs_timing = _run_timing(
                rhs,
                batch=x,
                device=device,
                n_steps=profile_active_steps,
                warmup_steps=profile_warmup_steps,
                forward_fn=rhs_forward_fn,
                loss_fn=rhs_loss_fn,
            )

        profile_modes = ["eval", "train"] if mode == "both" else [mode]
        lhs_mode_results: dict[str, dict[str, object]] = {}
        rhs_mode_results: dict[str, dict[str, object]] = {}
        for profile_mode in profile_modes:
            lhs_mode_results[profile_mode] = {
                "ops": _profile_ops(
                    lhs,
                    batch=x,
                    device=device,
                    mode=profile_mode,
                    active_steps=profile_active_steps,
                    warmup_steps=profile_warmup_steps,
                    forward_fn=lhs_forward_fn,
                    loss_fn=lhs_loss_fn,
                )
            }
            rhs_mode_results[profile_mode] = {
                "ops": _profile_ops(
                    rhs,
                    batch=x,
                    device=device,
                    mode=profile_mode,
                    active_steps=profile_active_steps,
                    warmup_steps=profile_warmup_steps,
                    forward_fn=rhs_forward_fn,
                    loss_fn=rhs_loss_fn,
                )
            }

    input_tensor = _first_tensor(x)
    batch_size = int(input_tensor.shape[0]) if input_tensor is not None and input_tensor.ndim >= 1 else -1
    input_dim = int(input_tensor.shape[-1]) if input_tensor is not None and input_tensor.ndim >= 1 else -1

    lhs_result = {
        "name": lhs_name,
        "module": lhs,
        "timing": lhs_timing,
        "modes": lhs_mode_results,
        "n_parameters": _n_parameters(lhs),
        "device": device,
        "dtype": dtype,
        "batch_size": batch_size,
        "input_dim": input_dim,
    }
    rhs_result = {
        "name": rhs_name,
        "module": rhs,
        "timing": rhs_timing,
        "modes": rhs_mode_results,
        "n_parameters": _n_parameters(rhs),
        "device": device,
        "dtype": dtype,
        "batch_size": batch_size,
        "input_dim": input_dim,
    }

    lhs_timing = lhs_result["timing"]
    rhs_timing = rhs_result["timing"]

    print(f"Device: {device} | dtype: {dtype}")
    if group_name is not None:
        print(f"Group: {group_name} | input dim: {input_dim} | batch: {batch_size}")
    else:
        print(f"Input dim: {input_dim} | batch: {batch_size}")
    print(f"Compare: {lhs_name} vs {rhs_name}")
    print(f"Timing steps: warmup={profile_warmup_steps}, measured={profile_active_steps}")
    print(f"Timing order: {'interleaved' if interleave_timing else 'sequential'}")
    print(f"Profiler steps: warmup={profile_warmup_steps}, active={profile_active_steps}")
    print()
    print("Average Runtime (ms, mean±std)")
    print(
        f"{'Module':<12} {'#params':>10} {'Batch':>8} {'InputDim':>10} {'TrainFwd':>18} {'Backward':>18} {'ValFwd':>18}"
    )
    print("-" * 108)
    print(
        f"{lhs_name:<12} {lhs_result['n_parameters']:10d} {batch_size:8d} {input_dim:10d} "
        f"{_fmt_ms(lhs_timing['train_forward_ms_mean'], lhs_timing['train_forward_ms_std']):>18} "
        f"{_fmt_ms(lhs_timing['backward_ms_mean'], lhs_timing['backward_ms_std']):>18} "
        f"{_fmt_ms(lhs_timing['val_forward_ms_mean'], lhs_timing['val_forward_ms_std']):>18}"
    )
    print(
        f"{rhs_name:<12} {rhs_result['n_parameters']:10d} {batch_size:8d} {input_dim:10d} "
        f"{_fmt_ms(rhs_timing['train_forward_ms_mean'], rhs_timing['train_forward_ms_std']):>18} "
        f"{_fmt_ms(rhs_timing['backward_ms_mean'], rhs_timing['backward_ms_std']):>18} "
        f"{_fmt_ms(rhs_timing['val_forward_ms_mean'], rhs_timing['val_forward_ms_std']):>18}"
    )

    profile_modes = ["eval", "train"] if mode == "both" else [mode]
    all_mode_results: dict[str, dict[str, object]] = {}
    for profile_mode in profile_modes:
        lhs_ops = lhs_result["modes"][profile_mode]["ops"]
        rhs_ops = rhs_result["modes"][profile_mode]["ops"]

        keys = set(lhs_ops) | set(rhs_ops)
        deltas: list[tuple[float, str, float, float, float]] = []
        for op_name in keys:
            lhs_ms = lhs_ops.get(op_name, 0.0)
            rhs_ms = rhs_ops.get(op_name, 0.0)
            delta = lhs_ms - rhs_ms
            if delta <= 1e-3:
                continue
            ratio = lhs_ms / rhs_ms if rhs_ms > 1e-12 else float("inf")
            deltas.append((delta, op_name, lhs_ms, rhs_ms, ratio))
        deltas.sort(reverse=True)

        print()
        print(f"[{profile_mode}] Where {lhs_name} is slower than {rhs_name} (top aten ops, ms/step)")
        print(f"{'Op':<30} {lhs_name + ' (ms)':>12} {rhs_name + ' (ms)':>12} {'Delta':>10} {'Ratio':>8}")
        print("-" * 86)
        for delta, op_name, lhs_ms, rhs_ms, ratio in deltas[:top_k]:
            ratio_str = "inf" if ratio == float("inf") else f"{ratio:.2f}x"
            print(f"{op_name:<30.30} {lhs_ms:10.3f} {rhs_ms:10.3f} {delta:8.3f} {ratio_str:>8}")

        print()
        _print_top_ops(f"[{profile_mode}] Top ops for {lhs_name}", lhs_ops, top_k=top_k)
        print()
        _print_top_ops(f"[{profile_mode}] Top ops for {rhs_name}", rhs_ops, top_k=top_k)

        all_mode_results[profile_mode] = {"lhs_ops": lhs_ops, "rhs_ops": rhs_ops, "slowdowns": deltas}

    return {
        "lhs_timing": lhs_timing,
        "rhs_timing": rhs_timing,
        "lhs_result": lhs_result,
        "rhs_result": rhs_result,
        "modes": all_mode_results,
    }
