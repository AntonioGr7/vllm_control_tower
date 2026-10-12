"""`vct` command line. Run without a subcommand to open the TUI."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from . import config as config_mod
from . import ops
from .config import GroupConfig, ServeConfig, find_entry, load_catalog
from .estimator import GroupEstimate
from .hardware import detect_gpus, known_gpu_names
from .model_info import ModelInfoError
from .process import ProcessManager, State
from .render import (
    MEMBER_COLORS,
    config_details,
    estimate_report,
    gpu_table,
    group_details,
    group_estimate_report,
    group_state,
    state_badge,
)

app = typer.Typer(
    help="vLLM Control Tower: serve, stop and size vLLM configs from YAML.",
    no_args_is_help=False,
    add_completion=True,
    rich_markup_mode="rich",
)
console = Console()

ConfigDir = Annotated[
    Path,
    typer.Option("--config-dir", "-c", envvar="VCT_CONFIG_DIR", help="Folder of YAML configs."),
]
RuntimeDir = Annotated[
    Path,
    typer.Option(
        "--runtime-dir", envvar="VCT_RUNTIME_DIR", help="Where PID files, logs and caches live."
    ),
]


class Ctx:
    config_dir: Path = config_mod.DEFAULT_CONFIG_DIR
    runtime_dir: Path = config_mod.DEFAULT_RUNTIME_DIR

    @property
    def pm(self) -> ProcessManager:
        return ProcessManager(self.runtime_dir)


ctx = Ctx()


@app.callback(invoke_without_command=True)
def main(
    typer_ctx: typer.Context,
    config_dir: ConfigDir = config_mod.DEFAULT_CONFIG_DIR,
    runtime_dir: RuntimeDir = config_mod.DEFAULT_RUNTIME_DIR,
) -> None:
    ctx.config_dir = config_dir
    ctx.runtime_dir = runtime_dir
    if typer_ctx.invoked_subcommand is None:
        tui()


def _get(name: str) -> ServeConfig | GroupConfig:
    try:
        return find_entry(name, ctx.config_dir)
    except KeyError:
        cat = load_catalog(ctx.config_dir)
        names = ", ".join(e.name for e in cat.entries) or "none"
        console.print(f"[red]No config named {name!r} in {ctx.config_dir}[/] (available: {names})")
        raise typer.Exit(1) from None
    except config_mod.ConfigError as e:
        console.print(f"[red]{e}[/]")
        raise typer.Exit(1) from None


def _statuses(entry: ServeConfig | GroupConfig) -> dict:
    pm = ctx.pm
    return {m.name: pm.status(m.name) for m in ops.members(entry)}


@app.command("tui")
def tui() -> None:
    """Open the interactive control panel."""
    from .tui import ControlTower

    ControlTower(ctx.config_dir, ctx.runtime_dir).run()


@app.command("list")
def list_configs() -> None:
    """List configs and groups with their state."""
    cat = load_catalog(ctx.config_dir)
    t = Table(header_style="bold", box=None, padding=(0, 2))
    for col in ("name", "state", "model", "endpoint"):
        t.add_column(col, no_wrap=True)
    for entry in cat.entries:
        statuses = _statuses(entry)
        if isinstance(entry, GroupConfig):
            t.add_row(
                f"[bold]{entry.name}[/] [grey62]group[/]",
                state_badge(group_state(entry, statuses)),
                f"{len(entry.members)} models",
                "",
            )
            for i, m in enumerate(entry.members):
                color = MEMBER_COLORS[i % len(MEMBER_COLORS)]
                t.add_row(
                    f"  [{color}]■[/] {m.name}",
                    state_badge(statuses[m.name].state),
                    m.model,
                    m.base_url,
                )
        else:
            t.add_row(
                f"[bold]{entry.name}[/]",
                state_badge(statuses[entry.name].state),
                entry.model,
                entry.base_url,
            )
    console.print(t if cat.entries else f"[yellow]No configs in {ctx.config_dir}/[/]")
    for e in cat.errors:
        console.print(f"[red]✖ {e}[/]")


@app.command()
def show(name: str) -> None:
    """Show a config (or group) and the exact vLLM command(s) it runs."""
    entry = _get(name)
    if isinstance(entry, GroupConfig):
        console.print(group_details(entry, _statuses(entry)))
    else:
        console.print(config_details(entry, ctx.pm.status(entry.name)))


@app.command()
def validate() -> None:
    """Validate every YAML file in the config folder."""
    cat = load_catalog(ctx.config_dir)
    for entry in cat.entries:
        kind = f" (group of {len(entry.members)})" if isinstance(entry, GroupConfig) else ""
        console.print(f"[green]✓[/] {entry.path}{kind}")
    errors = cat.errors
    for e in errors:
        console.print(f"[red]✖ {e}[/]")
    raise typer.Exit(1 if errors else 0)


@app.command()
def estimate(
    name: str,
    gpu: Annotated[
        str | None,
        typer.Option(
            "--gpu",
            "-g",
            help="Estimate for a GPU model instead of the local one, e.g. H100, A100-40GB, L4.",
        ),
    ] = None,
    num_gpus: Annotated[
        int | None, typer.Option("--num-gpus", "-n", help="GPUs on the node.")
    ] = None,
    prompt_tokens: Annotated[
        int | None, typer.Option("--prompt", help="Average prompt tokens.")
    ] = None,
    output_tokens: Annotated[
        int | None, typer.Option("--output", help="Average output tokens.")
    ] = None,
    offline: Annotated[bool, typer.Option(help="Only use the local Hugging Face cache.")] = False,
) -> None:
    """Estimate VRAM, KV cache capacity and tokens/s for a config or group."""
    entry = _get(name)
    try:
        with console.status(f"Reading model metadata for {name}…"):
            result = ops.estimate_entry(
                entry,
                ctx.pm,
                runtime_dir=ctx.runtime_dir,
                gpu=gpu,
                num_gpus=num_gpus,
                offline=offline,
                prompt_tokens=prompt_tokens,
                output_tokens=output_tokens,
            )
    except (ValueError, RuntimeError, ModelInfoError) as e:
        console.print(f"[red]{e}[/]")
        raise typer.Exit(1) from None
    if isinstance(result, GroupEstimate):
        console.print(group_estimate_report(result))
    else:
        console.print(estimate_report(result))
    raise typer.Exit(0 if result.fits else 2)


@app.command()
def start(
    name: str,
    wait: Annotated[
        bool, typer.Option("--wait/--no-wait", help="Block until /health answers.")
    ] = True,
    timeout: Annotated[float, typer.Option(help="Seconds to wait for health.")] = 900,
) -> None:
    """Start serving a config in the background. Groups start member by member."""
    entry = _get(name)
    pm = ctx.pm
    if isinstance(entry, GroupConfig):
        _start_group(entry, wait, timeout)
        return
    cfg = entry
    try:
        rec = pm.start(cfg)
    except RuntimeError as e:
        console.print(f"[red]{e}[/]")
        raise typer.Exit(1) from None
    console.print(f"[green]▶[/] started [bold]{cfg.name}[/] (pid {rec.pid}) → {cfg.base_url}/v1")
    console.print(f"  logs: {rec.log_file}")
    if not wait:
        return
    with console.status("Waiting for vLLM to load the model…") as status:
        state = ops.wait_healthy(
            pm, cfg.name, timeout, lambda _n, line: status.update(f"[grey62]{line[:120]}[/]")
        )
    _report_health(cfg, state, timeout)


def _report_health(cfg: ServeConfig, state: State, timeout: float) -> None:
    pm = ctx.pm
    if state is State.RUNNING:
        console.print(f"[green]●[/] {cfg.name} is healthy → {cfg.base_url}/v1")
    elif state in (State.STALE, State.STOPPED):
        console.print(f"[red]✖ {cfg.name} exited during startup. Last log lines:[/]")
        for line in pm.tail(cfg.name, 25):
            console.print(f"  [grey62]{line}[/]", markup=False, highlight=False)
        pm.pid_file(cfg.name).unlink(missing_ok=True)
        raise typer.Exit(1)
    else:
        console.print(f"[yellow]Timed out after {timeout:.0f}s; {cfg.name} is still starting.[/]")


def _start_group(group: GroupConfig, wait: bool, timeout: float) -> None:
    pm = ctx.pm
    console.print(
        f"Starting group [bold]{group.name}[/]: " + " → ".join(m.short_name for m in group.members)
    )
    failed: list[str] = []
    with console.status("") as status:

        def event(name: str, msg: str) -> None:
            if msg.startswith("error"):
                failed.append(name)
                console.print(f"[red]✖[/] {name}: {msg.removeprefix('error: ')}")
            elif msg == "healthy":
                cfg = next(m for m in group.members if m.name == name)
                console.print(f"[green]●[/] {name} is healthy → {cfg.base_url}/v1")
            else:
                console.print(f"[green]▶[/] {name}: {msg}")
                status.update(f"Waiting for {name} to load…")

        ok = ops.start_group(
            pm,
            group,
            timeout=timeout,
            wait_last=wait,
            on_event=event,
            on_progress=lambda n, line: status.update(f"[grey62]{n}: {line[:110]}[/]"),
        )
    if not ok:
        for name in failed:
            console.print(f"[red]Last log lines of {name}:[/]")
            for line in pm.tail(name, 25):
                console.print(f"  [grey62]{line}[/]", markup=False, highlight=False)
        console.print(
            f"[yellow]Group {group.name} is partially up; `vct stop {group.name}` stops the rest.[/]"
        )
        raise typer.Exit(1)


@app.command()
def stop(
    name: Annotated[str | None, typer.Argument(help="Config to stop.")] = None,
    all_: Annotated[bool, typer.Option("--all", help="Stop every running config.")] = False,
    timeout: Annotated[float, typer.Option(help="Seconds before SIGKILL.")] = 30,
) -> None:
    """Stop a running config or group (SIGTERM, then SIGKILL after the timeout)."""
    pm = ctx.pm
    if all_:
        names = [s.name for s in pm.running()]
    elif name:
        names = [m.name for m in reversed(ops.members(_get(name)))]
    else:
        names = []
    if not names:
        console.print("[yellow]Nothing to stop. Pass a name or --all.[/]")
        raise typer.Exit(1) from None
    for n in names:
        with console.status(f"Stopping {n}…"):
            was = pm.stop(n, timeout)
        console.print(f"[red]■[/] stopped {n}" if was else f"[grey62]{n} was not running[/]")


@app.command()
def status() -> None:
    """Show running servers and GPU usage."""
    pm = ctx.pm
    running = pm.running()
    if running:
        t = Table(header_style="bold", box=None, padding=(0, 2))
        for col in ("name", "state", "pid", "endpoint", "model"):
            t.add_column(col)
        for s in running:
            st = pm.status(s.name)
            t.add_row(
                s.name, state_badge(st.state), str(s.record.pid), s.record.base_url, s.record.model
            )
        console.print(t)
    else:
        console.print("[grey62]No servers running.[/]")
    gpus = detect_gpus()
    if gpus:
        console.print()
        console.print(gpu_table(gpus))


@app.command()
def logs(
    name: str,
    lines: Annotated[int, typer.Option("--tail", "-n")] = 100,
    follow: Annotated[bool, typer.Option("--follow", "-f")] = False,
) -> None:
    """Print (and optionally follow) a server's log."""
    pm = ctx.pm
    log = pm.log_file(name)
    if not log.exists():
        console.print(f"[yellow]No log for {name} at {log}[/]")
        raise typer.Exit(1) from None
    for line in pm.tail(name, lines):
        print(line)
    if not follow:
        return
    with log.open() as f:
        f.seek(0, 2)
        try:
            while True:
                line = f.readline()
                if line:
                    print(line, end="")
                else:
                    time.sleep(0.3)
        except KeyboardInterrupt:
            pass


@app.command()
def gpus(
    known: Annotated[bool, typer.Option(help="List GPU models usable with --gpu.")] = False,
) -> None:
    """Show local GPUs."""
    if known:
        console.print(", ".join(known_gpu_names()))
        return
    found = detect_gpus()
    console.print(
        gpu_table(found) if found else "[yellow]No NVIDIA GPU detected (NVML unavailable).[/]"
    )


if __name__ == "__main__":
    app()
