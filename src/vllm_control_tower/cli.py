"""`vct` command line. Run without a subcommand to open the TUI."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Annotated

import httpx
import typer
from rich.console import Console
from rich.table import Table

from . import config as config_mod
from . import ops
from .config import Deployment, ServeConfig, load_catalog
from .estimator import DeploymentEstimate
from .hardware import detect_gpus, known_gpu_names
from .model_info import ModelInfoError
from .process import ProcessManager, State
from .render import (
    MEMBER_COLORS,
    bench_report,
    config_details,
    deployment_details,
    deployment_estimate_report,
    deployment_state,
    estimate_report,
    gpu_table,
    measured_sections,
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


def _get(name: str) -> Deployment | ServeConfig:
    """A deployment by name, one of its models as `deployment.model`, or a YAML path."""
    try:
        cat = load_catalog(ctx.config_dir)
        found = cat.deployment(name) or cat.server(name)
        if found is None and Path(name).is_file():
            found = config_mod.load_file(Path(name))
    except config_mod.ConfigError as e:
        console.print(f"[red]{e}[/]")
        raise typer.Exit(1) from None
    if found is None:
        names = ", ".join(d.name for d in cat.deployments) or "none"
        console.print(
            f"[red]No deployment named {name!r} in {ctx.config_dir}[/] (available: {names})"
        )
        raise typer.Exit(1)
    return found


def _statuses(entry: Deployment | ServeConfig) -> dict:
    pm = ctx.pm
    return {m.name: pm.status(m.name) for m in ops.members(entry)}


@app.command("tui")
def tui() -> None:
    """Open the interactive control panel."""
    from .tui import ControlTower

    ControlTower(ctx.config_dir, ctx.runtime_dir).run()


@app.command("list")
def list_configs() -> None:
    """List deployments and their state."""
    cat = load_catalog(ctx.config_dir)
    t = Table(header_style="bold", box=None, padding=(0, 2))
    for col in ("name", "state", "model", "endpoint"):
        t.add_column(col, no_wrap=True)
    for dep in cat.deployments:
        statuses = _statuses(dep)
        state = state_badge(deployment_state(dep, statuses))
        n = len(dep.members)
        t.add_row(f"[bold]{dep.name}[/]", state, f"[grey62]{n} model{'s' if n > 1 else ''}[/]", "")
        for i, m in enumerate(dep.members):
            color = MEMBER_COLORS[i % len(MEMBER_COLORS)]
            t.add_row(
                f"  [{color}]■[/] {m.short_name}",
                state_badge(statuses[m.name].state),
                m.model,
                m.base_url,
            )
    console.print(t if cat.deployments else f"[yellow]No deployments in {ctx.config_dir}/[/]")
    for e in cat.errors:
        console.print(f"[red]✖ {e}[/]")


@app.command()
def show(name: str) -> None:
    """Show a deployment (or one of its models) and the exact vLLM commands."""
    entry = _get(name)
    if isinstance(entry, Deployment):
        console.print(deployment_details(entry, _statuses(entry)))
    else:
        console.print(config_details(entry, ctx.pm.status(entry.name)))


@app.command()
def validate() -> None:
    """Validate every YAML file in the config folder."""
    cat = load_catalog(ctx.config_dir)
    for dep in cat.deployments:
        n = len(dep.members)
        console.print(f"[green]✓[/] {dep.path} [grey62]({n} model{'s' if n > 1 else ''})[/]")
    for e in cat.errors:
        console.print(f"[red]✖ {e}[/]")
    raise typer.Exit(1 if cat.errors else 0)


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
    """Estimate VRAM, KV cache capacity and tokens/s for a deployment."""
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
    if isinstance(result, DeploymentEstimate):
        console.print(deployment_estimate_report(result))
        pairs = result.members
    else:
        console.print(estimate_report(result))
        pairs = [(entry, result)]
    if gpu is None:  # measurements only describe the local GPU
        for section in measured_sections(pairs, ctx.runtime_dir):
            console.print(section)
    raise typer.Exit(0 if result.fits else 2)


@app.command()
def start(
    name: str,
    wait: Annotated[
        bool, typer.Option("--wait/--no-wait", help="Block until the last model is healthy.")
    ] = True,
    timeout: Annotated[float, typer.Option(help="Seconds to wait for health, per model.")] = 900,
) -> None:
    """Start a deployment in the background, one model at a time."""
    if not _start(_get(name), wait, timeout):
        raise typer.Exit(1)


def _start(entry: Deployment | ServeConfig, wait: bool, timeout: float) -> bool:
    """Start models in order with progress output. Returns True when all came up."""
    pm = ctx.pm
    models = ops.members(entry)
    if len(models) > 1:
        console.print(
            f"Starting [bold]{entry.name}[/]: " + " → ".join(m.short_name for m in models)
        )
    failed: list[str] = []
    with console.status("") as status:

        def event(model: str, msg: str) -> None:
            cfg = next(m for m in models if m.name == model)
            if msg.startswith("error"):
                failed.append(model)
                console.print(f"[red]✖[/] {model}: {msg.removeprefix('error: ')}")
            elif msg == "healthy":
                console.print(f"[green]●[/] {model} is healthy → {cfg.base_url}/v1")
            elif msg == "already running":
                console.print(f"[green]●[/] {model} is already running")
            else:
                console.print(f"[green]▶[/] {model}: {msg}")
                console.print(f"  logs: {pm.log_file(model)}", style="grey62")
                status.update(f"Waiting for {model} to load…")

        ok = ops.start_deployment(
            pm,
            entry,
            timeout=timeout,
            wait_last=wait,
            on_event=event,
            on_progress=lambda n, line: status.update(f"[grey62]{n}: {line[:110]}[/]"),
        )
    if ok:
        return True
    for model in failed:
        console.print(f"[red]Last log lines of {model}:[/]")
        for line in pm.tail(model, 25):
            console.print(f"  [grey62]{line}[/]", markup=False, highlight=False)
    if len(models) > 1 and any(ops.is_up(pm, m.name) for m in models):
        console.print(
            f"[yellow]{entry.name} is partially up; `vct stop {entry.name}` stops the rest.[/]"
        )
    return False


@app.command()
def bench(
    name: Annotated[
        str | None,
        typer.Argument(
            help="Deployment (or deployment.model). Omit to open the TUI in bench mode."
        ),
    ] = None,
    keep: Annotated[
        bool,
        typer.Option("--keep", help="Headless: leave models running if bench had to start them."),
    ] = False,
    prompt_tokens: Annotated[
        int | None, typer.Option("--prompt", help="Prompt tokens per request.")
    ] = None,
    output_tokens: Annotated[
        int | None, typer.Option("--output", help="Generated tokens per request (default ≤128).")
    ] = None,
    max_concurrency: Annotated[
        int, typer.Option(help="Highest number of concurrent requests to try.")
    ] = 16,
    timeout: Annotated[float, typer.Option(help="Seconds to wait for each model to start.")] = 900,
) -> None:
    """Benchmark deployments right after they start.

    Without a name, opens the TUI in bench mode: every deployment you start there is
    benchmarked as soon as its models are healthy. With a name, runs headless: starts the
    deployment if needed, measures it, prints the comparison and stops what it started.
    Results are saved and calibrate future estimates for this GPU.
    """
    if name is None:
        from .tui import ControlTower

        ControlTower(
            ctx.config_dir,
            ctx.runtime_dir,
            bench_options={
                "prompt_tokens": prompt_tokens,
                "output_tokens": output_tokens,
                "max_concurrency": max_concurrency,
            },
        ).run()
        return
    entry = _get(name)
    pm = ctx.pm
    models = ops.members(entry)
    started = [m for m in models if not ops.is_up(pm, m.name)]
    if started and not _start(entry, wait=True, timeout=timeout):
        _stop_models(started, keep)
        raise typer.Exit(1)
    try:
        for cfg in models:
            if ops.wait_healthy(pm, cfg.name, timeout) is not State.RUNNING:
                console.print(f"[red]✖ {cfg.name} is not healthy; skipping[/]")
                continue
            _bench_model(cfg, prompt_tokens, output_tokens, max_concurrency)
    finally:
        _stop_models(started, keep)


def _bench_model(
    cfg: ServeConfig, prompt_tokens: int | None, output_tokens: int | None, cap: int
) -> None:
    console.rule(f"[bold]{cfg.name}[/]")
    with console.status("") as status:
        try:
            out = ops.run_benchmark(
                cfg,
                ctx.pm,
                runtime_dir=ctx.runtime_dir,
                prompt_tokens=prompt_tokens,
                output_tokens=output_tokens,
                max_concurrency=cap,
                on_progress=lambda msg: status.update(f"{cfg.name}: {msg}…"),
            )
        except (RuntimeError, httpx.HTTPError) as e:
            console.print(f"[red]✖ benchmark of {cfg.name} failed: {e}[/]")
            return
    if out.estimate_error:
        console.print(f"[yellow]No estimate to compare with: {out.estimate_error}[/]")
    console.print(bench_report(out.result, out.raw))
    console.print(f"  saved {out.path}", style="grey62")
    if out.calibration:
        cal = out.calibration
        mode = "eager" if cal.eager else "CUDA graphs"
        console.print(
            f"  calibrated {cal.gpu} ({mode}): +{cal.extra_step_s * 1e3:.0f} ms per decode step "
            "in future estimates",
            style="grey62",
        )


def _stop_models(models: list[ServeConfig], keep: bool) -> None:
    if not models:
        return
    if keep:
        console.print(f"[grey62]left running: {', '.join(m.name for m in models)}[/]")
        return
    for m in reversed(models):
        with console.status(f"Stopping {m.name}…"):
            ctx.pm.stop(m.name)
        console.print(
            f"[red]■[/] stopped {m.name} [grey62](started by bench; --keep leaves it up)[/]"
        )


@app.command()
def stop(
    name: Annotated[
        str | None, typer.Argument(help="Deployment (or deployment.model) to stop.")
    ] = None,
    all_: Annotated[bool, typer.Option("--all", help="Stop everything that is running.")] = False,
    timeout: Annotated[float, typer.Option(help="Seconds before SIGKILL.")] = 30,
) -> None:
    """Stop a deployment, models in reverse order (SIGTERM, then SIGKILL after the timeout)."""
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
    """Print (and optionally follow) the log of a deployment's model."""
    pm = ctx.pm
    models = ops.members(_get(name))
    if len(models) > 1 and follow:
        names = ", ".join(m.name for m in models)
        console.print(f"[yellow]{name} has {len(models)} models; follow one of: {names}[/]")
        raise typer.Exit(1)
    for m in models:
        if len(models) > 1:
            console.rule(m.name, style="grey50")
        log = pm.log_file(m.name)
        if not log.exists():
            console.print(f"[yellow]No log for {m.name} yet ({log})[/]")
            continue
        for line in pm.tail(m.name, lines):
            print(line)
    if not follow or not pm.log_file(models[0].name).exists():
        return
    with pm.log_file(models[0].name).open() as f:
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
