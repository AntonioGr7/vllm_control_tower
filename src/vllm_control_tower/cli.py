"""`vct` command line. Run without a subcommand to open the TUI."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from . import config as config_mod
from .config import ServeConfig, find_config, load_configs
from .estimator import estimate as run_estimate
from .hardware import detect_gpus, known_gpu_names, resolve_hardware
from .model_info import ModelInfoError, get_model_info
from .process import ProcessManager, State
from .render import config_details, estimate_report, gpu_table, state_badge

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


def _get(name: str) -> ServeConfig:
    try:
        return find_config(name, ctx.config_dir)
    except KeyError:
        configs, _ = load_configs(ctx.config_dir)
        names = ", ".join(c.name for c in configs) or "none"
        console.print(f"[red]No config named {name!r} in {ctx.config_dir}[/] (available: {names})")
        raise typer.Exit(1) from None
    except config_mod.ConfigError as e:
        console.print(f"[red]{e}[/]")
        raise typer.Exit(1) from None


@app.command("tui")
def tui() -> None:
    """Open the interactive control panel."""
    from .tui import ControlTower

    ControlTower(ctx.config_dir, ctx.runtime_dir).run()


@app.command("list")
def list_configs() -> None:
    """List configs and their state."""
    configs, errors = load_configs(ctx.config_dir)
    t = Table(header_style="bold", box=None, padding=(0, 2))
    for col in ("name", "state", "model", "endpoint"):
        t.add_column(col, no_wrap=True)
    pm = ctx.pm
    for c in configs:
        st = pm.status(c.name)
        t.add_row(f"[bold]{c.name}[/]", state_badge(st.state), c.model, c.base_url)
    console.print(t if configs else f"[yellow]No configs in {ctx.config_dir}/[/]")
    for e in errors:
        console.print(f"[red]✖ {e}[/]")


@app.command()
def show(name: str) -> None:
    """Show a config and the exact vLLM command it runs."""
    cfg = _get(name)
    console.print(config_details(cfg, ctx.pm.status(cfg.name)))


@app.command()
def validate() -> None:
    """Validate every YAML file in the config folder."""
    configs, errors = load_configs(ctx.config_dir)
    for c in configs:
        console.print(f"[green]✓[/] {c.path}")
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
    """Estimate VRAM, KV cache capacity and tokens/s for a config."""
    cfg = _get(name)
    try:
        spec, n = resolve_hardware(gpu, num_gpus)
        if gpu and num_gpus is None:
            # A GPU from the spec table: assume the node has as many as the config shards over.
            n = int(cfg.arg("tensor-parallel-size", 1)) * int(cfg.arg("pipeline-parallel-size", 1))
    except (ValueError, RuntimeError) as e:
        console.print(f"[red]{e}[/]")
        raise typer.Exit(1) from None
    with console.status(f"Reading model metadata for {cfg.model}…"):
        try:
            info = get_model_info(cfg.model, runtime_dir=ctx.runtime_dir, offline=offline)
        except ModelInfoError as e:
            console.print(f"[red]{e}[/]")
            raise typer.Exit(1) from None
    est = run_estimate(
        cfg, info, spec, n, avg_prompt_tokens=prompt_tokens, avg_output_tokens=output_tokens
    )
    console.print(estimate_report(est))
    raise typer.Exit(0 if est.fits else 2)


@app.command()
def start(
    name: str,
    wait: Annotated[
        bool, typer.Option("--wait/--no-wait", help="Block until /health answers.")
    ] = True,
    timeout: Annotated[float, typer.Option(help="Seconds to wait for health.")] = 900,
) -> None:
    """Start serving a config in the background."""
    cfg = _get(name)
    pm = ctx.pm
    try:
        rec = pm.start(cfg)
    except RuntimeError as e:
        console.print(f"[red]{e}[/]")
        raise typer.Exit(1) from None
    console.print(f"[green]▶[/] started [bold]{cfg.name}[/] (pid {rec.pid}) → {cfg.base_url}/v1")
    console.print(f"  logs: {rec.log_file}")
    if not wait:
        return
    last = ""
    with console.status("Waiting for vLLM to load the model…") as status:
        deadline = time.time() + timeout
        while time.time() < deadline:
            st = pm.status(cfg.name)
            if st.state is State.RUNNING:
                console.print(f"[green]●[/] {cfg.name} is healthy")
                return
            if st.state in (State.STALE, State.STOPPED):
                console.print(f"[red]✖ {cfg.name} exited during startup. Last log lines:[/]")
                for line in pm.tail(cfg.name, 25):
                    console.print(f"  [grey62]{line}[/]", markup=False, highlight=False)
                pm.pid_file(cfg.name).unlink(missing_ok=True)
                raise typer.Exit(1) from None
            tail = pm.tail(cfg.name, 1)
            if tail and tail[-1] != last:
                last = tail[-1]
                status.update(f"[grey62]{last[:120]}[/]")
            time.sleep(1)
    console.print(f"[yellow]Timed out after {timeout:.0f}s; {cfg.name} is still starting.[/]")


@app.command()
def stop(
    name: Annotated[str | None, typer.Argument(help="Config to stop.")] = None,
    all_: Annotated[bool, typer.Option("--all", help="Stop every running config.")] = False,
    timeout: Annotated[float, typer.Option(help="Seconds before SIGKILL.")] = 30,
) -> None:
    """Stop a running config (SIGTERM, then SIGKILL after the timeout)."""
    pm = ctx.pm
    names = [s.name for s in pm.running()] if all_ else [name] if name else []
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
