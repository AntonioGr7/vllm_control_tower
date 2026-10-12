"""Rich renderables shared by the CLI and the TUI."""

from __future__ import annotations

from rich.console import Group, RenderableType
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from .config import GroupConfig, ServeConfig
from .estimator import Estimate, GroupEstimate, device_set
from .hardware import GiB, GpuSpec
from .process import State, Status

STATE_STYLE = {
    State.RUNNING: ("●", "bold green"),
    State.STARTING: ("◐", "bold yellow"),
    State.UNHEALTHY: ("●", "bold red"),
    State.STALE: ("✕", "red"),
    State.STOPPED: ("○", "grey50"),
    State.PARTIAL: ("◑", "bold dark_orange"),
}


def gib(n: float) -> str:
    return f"{n / GiB:.2f} GiB"


def human_tokens(n: float) -> str:
    if n >= 1e6:
        return f"{n / 1e6:.1f}M"
    if n >= 1e3:
        return f"{n / 1e3:.1f}k"
    return f"{n:.0f}"


def state_badge(state: State) -> Text:
    icon, style = STATE_STYLE[state]
    return Text(f"{icon} {state.value}", style=style)


def bar(fraction: float, width: int, color: str, empty: str = "grey23") -> Text:
    fraction = min(max(fraction, 0.0), 1.0)
    full = fraction * width
    whole = int(full)
    t = Text("█" * whole, style=color)
    partial = " ▏▎▍▌▋▊▉"[int((full - whole) * 8)] if whole < width else ""
    if partial.strip():
        t.append(partial, style=f"{color} on {empty}")
        whole += 1
    t.append(" " * (width - whole), style=f"on {empty}")
    return t


def memory_bar(est: Estimate, width: int = 60) -> RenderableType:
    """Stacked horizontal bar of per-GPU VRAM usage, with legend."""
    segments = est.memory_segments()
    total = max(est.total_bytes, 1)
    line = Text()
    used = 0
    for seg in segments:
        cells = round(seg.bytes / total * width)
        cells = min(cells, width - used)
        line.append("█" * cells, style=seg.color)
        used += cells
    line.append("█" * (width - used), style="grey23")

    budget_pos = min(int(est.gpu_memory_utilization * width), width - 1)
    marker = Text(
        " " * budget_pos
        + "▲ "
        + f"budget {gib(est.budget_bytes)} ({est.gpu_memory_utilization:.0%})",
        style="yellow",
    )

    legend = Table.grid(padding=(0, 2))
    legend.add_column()
    legend.add_column(justify="right")
    legend.add_column(justify="right", style="grey62")
    for seg in segments:
        legend.add_row(
            Text("■ ", style=seg.color) + Text(seg.label),
            gib(seg.bytes),
            f"{seg.bytes / total:.0%}",
        )
    return Group(line, marker, Text(), legend)


def throughput_chart(est: Estimate, width: int = 28) -> RenderableType:
    """Horizontal bar chart of aggregate decode tok/s by batch size."""
    t = Table(box=None, padding=(0, 1), show_header=True, header_style="bold")
    t.add_column("batch", justify="right", no_wrap=True)
    t.add_column("aggregate decode", no_wrap=True)
    t.add_column("tok/s", justify="right", style="bold", no_wrap=True)
    t.add_column("per seq", justify="right", no_wrap=True)
    t.add_column("bound", style="grey62", no_wrap=True)
    peak = max((p.tokens_per_s for p in est.curve), default=1)
    for p in est.curve:
        color = {"memory": "dodger_blue1", "compute": "orange1", "overhead": "grey62"}[p.bound]
        marker = " ◀" if p.batch == est.effective_batch else ""
        t.add_row(
            str(p.batch),
            bar(p.tokens_per_s / peak, width, color),
            f"{p.tokens_per_s:,.0f}",
            f"{p.per_seq_tokens_per_s:,.1f}",
            p.bound + marker,
        )
    return t


def estimate_summary(est: Estimate) -> RenderableType:
    m = est.model
    left = Table.grid(padding=(0, 2))
    left.add_column(style="grey62")
    left.add_column()
    left.add_row("model", f"{m.model_id}  [grey62]({m.architecture})[/]")
    params = f"{m.param_count / 1e9:.2f}B"
    if m.is_moe:
        params += f" total / {m.active_param_count / 1e9:.2f}B active ({m.num_experts} experts, top-{m.experts_per_token})"
    left.add_row("parameters", params + f"  [grey62]{m.params_source}[/]")
    left.add_row(
        "layers / heads",
        f"{m.num_layers} layers, {m.num_heads} q heads, {m.num_kv_heads} kv heads, head_dim {m.head_dim}",
    )
    left.add_row(
        "hardware",
        f"{est.num_gpus}× {est.gpu.name}  [grey62]{gib(est.total_bytes)}, {est.gpu.bandwidth_gbs:,.0f} GB/s, {est.gpu.tflops_16:,.0f} TFLOPS[/]",
    )
    left.add_row("parallelism", f"tp={est.tp} pp={est.pp}")
    left.add_row("dtype / kv dtype", f"{est.dtype} / {est.kv_dtype}")
    left.add_row("max-model-len", f"{est.max_model_len:,}")
    left.add_row(
        "max-num-seqs",
        f"{est.max_num_seqs}   [grey62]batched tokens {est.max_num_batched_tokens:,}[/]",
    )
    left.add_row("cuda graphs", "off (enforce-eager)" if est.enforce_eager else "on")

    kv = Table.grid(padding=(0, 2))
    kv.add_column(style="grey62")
    kv.add_column(style="bold")
    kv.add_row("KV per token", f"{est.kv_bytes_per_token / 1024:,.1f} KiB")
    kv.add_row("KV cache", gib(max(est.kv_cache_bytes, 0)) + " / GPU")
    kv.add_row("KV capacity", f"{est.kv_capacity_tokens:,} tokens")
    conc_style = (
        "red" if est.max_concurrency < 1 else "yellow" if est.max_concurrency < 2 else "green"
    )
    kv.add_row(
        "max concurrency",
        Text(f"{est.max_concurrency:.1f}× @ {est.max_model_len:,} tok", style=conc_style),
    )
    kv.add_row("", "")
    kv.add_row("workload", f"{est.avg_prompt_tokens} in / {est.avg_output_tokens} out")
    kv.add_row("single stream", f"{est.single_stream_tps:,.1f} tok/s")
    kv.add_row(f"batch {est.effective_batch}", f"{est.batch_tps:,.0f} tok/s")
    kv.add_row("prefill", f"{est.prefill_tps:,.0f} tok/s")
    kv.add_row("TTFT (idle)", f"{est.ttft_ms:,.0f} ms")
    kv.add_row("requests/s", f"{est.requests_per_s:,.2f}")

    grid = Table.grid(expand=True, padding=(0, 4))
    grid.add_column(ratio=3)
    grid.add_column(ratio=2)
    grid.add_row(left, kv)
    return grid


def diagnostics(est: Estimate) -> RenderableType | None:
    lines = [Text("✖ " + e, style="bold red") for e in est.errors]
    lines += [Text("⚠ " + w, style="yellow") for w in est.warnings]
    lines += [Text("ℹ " + n, style="grey62") for n in est.notes]
    return Group(*lines) if lines else None


def estimate_report(est: Estimate) -> RenderableType:
    verdict = (
        Text(" FITS ", style="bold black on green")
        if est.fits
        else Text(" DOES NOT FIT ", style="bold white on red")
    )
    parts: list[RenderableType] = [
        estimate_summary(est),
        Text(),
        Text("VRAM per GPU", style="bold"),
        memory_bar(est),
        Text(),
        Text("Decode throughput (roofline estimate)", style="bold"),
        throughput_chart(est),
    ]
    diag = diagnostics(est)
    if diag:
        parts += [Text(), diag]
    return Panel(
        Group(*parts),
        title=Text.assemble(f" {est.config_name} ", verdict),
        border_style="green" if est.fits else "red",
    )


def gpu_table(gpus: list[GpuSpec]) -> RenderableType:
    t = Table(header_style="bold", box=None, padding=(0, 2))
    for col in ("#", "GPU", "memory", "", "util", "temp", "power", "bandwidth", "compute"):
        t.add_column(col)
    for g in gpus:
        frac = g.used_bytes / g.memory_bytes if g.memory_bytes else 0
        color = "green" if frac < 0.6 else "yellow" if frac < 0.85 else "red"
        t.add_row(
            str(g.index),
            g.name,
            f"{g.used_bytes / GiB:.1f}/{g.memory_bytes / GiB:.1f} GiB",
            bar(frac, 20, color),
            f"{g.utilization}%" if g.utilization is not None else "-",
            f"{g.temperature}°C" if g.temperature is not None else "-",
            f"{g.power_w:.0f} W" if g.power_w is not None else "-",
            f"{g.bandwidth_gbs:,.0f} GB/s",
            f"sm_{g.compute_capability[0]}{g.compute_capability[1]}"
            if g.compute_capability
            else "-",
        )
    return t


def config_details(cfg: ServeConfig, status: Status | None = None) -> RenderableType:
    t = Table.grid(padding=(0, 2))
    t.add_column(style="grey62")
    t.add_column()
    if status:
        badge = state_badge(status.state)
        if status.record and status.state is not State.STALE:
            badge.append(f"  pid {status.record.pid}", style="grey62")
            if status.uptime_s is not None:
                badge.append(f"  up {_duration(status.uptime_s)}", style="grey62")
        t.add_row("status", badge)
    if cfg.description:
        t.add_row("description", cfg.description)
    t.add_row("model", Text(cfg.model, style="bold cyan"))
    t.add_row("endpoint", f"{cfg.base_url}/v1")
    t.add_row("file", str(cfg.path) if cfg.path else "-")
    for k, v in cfg.vllm.args.items():
        t.add_row(f"  --{k}", str(v))
    for f in cfg.vllm.flags:
        t.add_row(f"  --{f}", "✓")
    if cfg.env:
        t.add_row("env", ", ".join(cfg.env))
    cmd = Text(" ".join(cfg.command()), style="grey70")
    return Group(t, Text(), Text("command", style="grey62"), cmd)


def _duration(s: float) -> str:
    s = int(s)
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    return f"{h}h{m:02d}m" if h else f"{m}m{sec:02d}s"


MEMBER_COLORS = [
    "dodger_blue1",
    "orange1",
    "spring_green2",
    "medium_purple1",
    "deep_pink2",
    "gold1",
]


def group_memory_bar(ge: GroupEstimate, width: int = 60) -> RenderableType:
    """One stacked bar per device set: each member's claimed budget, other processes, free."""
    total = max(ge.gpu.memory_bytes, 1)
    sets: dict[str, list[tuple[int, ServeConfig, Estimate]]] = {}
    for i, (cfg, est) in enumerate(ge.members):
        if isinstance(est, Estimate):
            sets.setdefault(device_set(cfg), []).append((i, cfg, est))
    parts: list[RenderableType] = []
    for devices, items in sets.items():
        if len(sets) > 1:
            parts.append(Text(f"CUDA_VISIBLE_DEVICES={devices}", style="grey62"))
        segs = [
            (c.short_name, e.budget_bytes, MEMBER_COLORS[i % len(MEMBER_COLORS)])
            for i, c, e in items
        ]
        contexts = sum(e.cuda_context_bytes for _, _, e in items)
        segs.append((f"CUDA contexts ({len(items)}×)", contexts, "grey42"))
        if ge.gpu.used_bytes:
            segs.append(("other processes", ge.gpu.used_bytes, "grey50"))
        claimed = sum(b for _, b, _ in segs)
        segs.append(("free", max(total - claimed, 0), "grey23"))
        line = Text()
        used = 0
        for _, b, color in segs:
            cells = min(round(b / total * width), width - used)
            line.append("█" * cells, style=color if claimed <= total else f"{color} on red")
            used += cells
        if claimed > total:
            line.append(" overcommitted", style="bold red")
        legend = Table.grid(padding=(0, 2))
        legend.add_column()
        legend.add_column(justify="right")
        legend.add_column(justify="right", style="grey62")
        for label, b, color in segs:
            legend.add_row(Text("■ ", style=color) + Text(label), gib(b), f"{b / total:.0%}")
        parts += [line, legend, Text()]
    return Group(*parts)


def group_member_table(ge: GroupEstimate) -> RenderableType:
    t = Table(box=None, padding=(0, 2), header_style="bold")
    t.add_column("member", no_wrap=True)
    for col in ("util", "budget", "weights", "KV tokens", "concurrency", "tok/s 1 · batch"):
        t.add_column(col, justify="right", no_wrap=True)
    t.add_column("", no_wrap=True)
    for i, (cfg, est) in enumerate(ge.members):
        name = Text("■ ", style=MEMBER_COLORS[i % len(MEMBER_COLORS)]) + Text(
            cfg.short_name, style="bold"
        )
        if isinstance(est, str):
            t.add_row(name, "", "", "", "", "", "", Text("✖ " + est, style="red"))
            continue
        t.add_row(
            name,
            f"{est.gpu_memory_utilization:.2f}",
            gib(est.budget_bytes),
            gib(est.weight_bytes),
            f"{est.kv_capacity_tokens:,}",
            f"{est.max_concurrency:.1f}×",
            f"{est.single_stream_tps:,.0f} · {est.batch_tps:,.0f}",
            Text("✓", style="green") if est.fits else Text("✖", style="red"),
        )
    return t


def group_estimate_report(ge: GroupEstimate) -> RenderableType:
    verdict = (
        Text(" FITS ", style="bold black on green")
        if ge.fits
        else Text(" DOES NOT FIT ", style="bold white on red")
    )
    g = ge.gpu
    head = Text.assemble(
        ("hardware  ", "grey62"),
        f"{ge.num_gpus}× {g.name}  ",
        (f"{gib(g.memory_bytes)}, {g.bandwidth_gbs:,.0f} GB/s", "grey62"),
    )
    parts: list[RenderableType] = [
        head,
        Text(),
        Text("VRAM claimed per GPU", style="bold"),
        group_memory_bar(ge),
        group_member_table(ge),
    ]
    lines = [Text("✖ " + e, style="bold red") for e in ge.errors]
    lines += [Text("⚠ " + w, style="yellow") for w in ge.warnings]
    lines += [Text("ℹ " + n, style="grey62") for n in ge.notes]
    if lines:
        parts += [Text(), *lines]
    summary = Panel(
        Group(*parts),
        title=Text.assemble(f" {ge.name} (group) ", verdict),
        border_style="green" if ge.fits else "red",
    )
    details = [estimate_report(e) for _, e in ge.members if isinstance(e, Estimate)]
    return Group(summary, *details)


def group_details(group: GroupConfig, statuses: dict[str, Status]) -> RenderableType:
    t = Table.grid(padding=(0, 2))
    t.add_column(style="grey62")
    t.add_column()
    if group.description:
        t.add_row("description", group.description)
    t.add_row("file", str(group.path) if group.path else "-")
    t.add_row("start order", " → ".join(m.short_name for m in group.members))
    members = Table(box=None, padding=(0, 2), header_style="bold")
    for col in ("member", "status", "model", "endpoint", "util"):
        members.add_column(col, no_wrap=True)
    for i, m in enumerate(group.members):
        st = statuses.get(m.name)
        members.add_row(
            Text("■ ", style=MEMBER_COLORS[i % len(MEMBER_COLORS)])
            + Text(m.short_name, style="bold"),
            state_badge(st.state if st else State.STOPPED),
            Text(m.model, style="cyan"),
            f"{m.base_url}/v1",
            str(m.arg("gpu-memory-utilization", 0.9)),
        )
    cmds = [Text()]
    for m in group.members:
        cmds += [Text(m.short_name, style="grey62"), Text(" ".join(m.command()), style="grey70")]
    return Group(t, Text(), members, *cmds)


def group_state(group: GroupConfig, statuses: dict[str, Status]) -> State:
    """Running only when every member is; starting if any is starting; stopped when all are."""
    states = [
        statuses[m.name].state if m.name in statuses else State.STOPPED for m in group.members
    ]
    if all(s is State.RUNNING for s in states):
        return State.RUNNING
    if any(s in (State.STARTING, State.RUNNING) for s in states):
        return State.STARTING if any(s is State.STARTING for s in states) else State.PARTIAL
    if any(s is State.STALE for s in states):
        return State.STALE
    return State.STOPPED
