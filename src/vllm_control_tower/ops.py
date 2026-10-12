"""Operations shared by the CLI and the TUI: estimate, start and stop deployments.

Everything takes an Entry: a whole deployment, or one of its models on its own.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

from .config import Deployment, ServeConfig
from .estimator import DeploymentEstimate, Estimate, discount_running, estimate, estimate_deployment
from .hardware import GpuSpec, detect_gpus, lookup
from .model_info import ModelInfoError, get_model_info
from .process import ProcessManager, State

Entry = Deployment | ServeConfig


def members(entry: Entry) -> list[ServeConfig]:
    return entry.members if isinstance(entry, Deployment) else [entry]


def is_up(pm: ProcessManager, name: str) -> bool:
    return pm.status(name, check_health=False).state not in (State.STOPPED, State.STALE)


def resolve_gpu(
    entry: Entry, pm: ProcessManager, gpu: str | None = None, num_gpus: int | None = None
) -> tuple[GpuSpec, int]:
    """The GPU to estimate on. Local GPUs give back the VRAM this entry's own running models hold."""
    if gpu:
        spec = lookup(gpu)
        if spec is None:
            raise ValueError(f"unknown GPU {gpu!r}; run `vct gpus --known` for the list")
        shards = max(
            int(c.arg("tensor-parallel-size", 1)) * int(c.arg("pipeline-parallel-size", 1))
            for c in members(entry)
        )
        return spec, num_gpus or shards
    local = detect_gpus()
    if not local:
        raise RuntimeError("no local NVIDIA GPU detected; pick a GPU model to estimate for")
    # Assume a homogeneous node and budget against the most constrained card.
    tightest = replace(min(local, key=lambda g: g.free_bytes), name=local[0].name)
    running = [c for c in members(entry) if is_up(pm, c.name)]
    return discount_running(tightest, running), num_gpus or len(local)


def estimate_entry(
    entry: Entry,
    pm: ProcessManager,
    *,
    runtime_dir: Path,
    gpu: str | None = None,
    num_gpus: int | None = None,
    offline: bool = False,
    prompt_tokens: int | None = None,
    output_tokens: int | None = None,
) -> Estimate | DeploymentEstimate:
    spec, n = resolve_gpu(entry, pm, gpu, num_gpus)

    def one(cfg: ServeConfig) -> Estimate:
        info = get_model_info(cfg.model, runtime_dir=runtime_dir, offline=offline)
        return estimate(
            cfg, info, spec, n, avg_prompt_tokens=prompt_tokens, avg_output_tokens=output_tokens
        )

    if isinstance(entry, ServeConfig):
        return one(entry)
    results: list[tuple[ServeConfig, Estimate | str]] = []
    for cfg in entry.members:
        try:
            results.append((cfg, one(cfg)))
        except ModelInfoError as e:
            results.append((cfg, str(e)))
    return estimate_deployment(entry.name, results, spec, n)


Progress = Callable[[str, str], None]  # (member name, message)


def wait_healthy(
    pm: ProcessManager,
    name: str,
    timeout: float,
    on_progress: Progress | None = None,
    should_stop: Callable[[], bool] | None = None,
) -> State:
    last = ""
    deadline = time.time() + timeout
    while time.time() < deadline:
        st = pm.status(name)
        if st.state in (State.RUNNING, State.STALE, State.STOPPED):
            return st.state
        if should_stop and should_stop():
            return st.state
        tail = pm.tail(name, 1)
        if on_progress and tail and tail[-1] != last:
            last = tail[-1]
            on_progress(name, last)
        time.sleep(1)
    return pm.status(name).state


def start_deployment(
    pm: ProcessManager,
    entry: Entry,
    *,
    timeout: float = 900,
    wait_last: bool = True,
    on_event: Progress | None = None,
    on_progress: Progress | None = None,
    should_stop: Callable[[], bool] | None = None,
) -> bool:
    """Start models in order, each after the previous one is healthy. Returns True on success.

    vLLM sizes its KV cache from the free memory it sees while profiling, so models sharing
    a GPU must not load at the same time.
    """
    emit = on_event or (lambda _n, _m: None)
    todo = members(entry)
    for i, cfg in enumerate(todo):
        if is_up(pm, cfg.name):
            emit(cfg.name, "already running")
            continue
        try:
            rec = pm.start(cfg)
        except RuntimeError as e:
            emit(cfg.name, f"error: {e}")
            return False
        emit(cfg.name, f"started pid {rec.pid} on :{cfg.port}")
        if i == len(todo) - 1 and not wait_last:
            break
        state = wait_healthy(pm, cfg.name, timeout, on_progress, should_stop)
        if state is State.RUNNING:
            emit(cfg.name, "healthy")
        elif state in (State.STALE, State.STOPPED):
            pm.pid_file(cfg.name).unlink(missing_ok=True)
            emit(cfg.name, "error: exited during startup")
            return False
        else:
            emit(cfg.name, f"error: not healthy after {timeout:.0f}s")
            return False
    return True


def stop_entry(pm: ProcessManager, entry: Entry, timeout: float = 30) -> list[tuple[str, bool]]:
    """Stop a model, or a deployment's models in reverse start order."""
    return [(cfg.name, pm.stop(cfg.name, timeout)) for cfg in reversed(members(entry))]
