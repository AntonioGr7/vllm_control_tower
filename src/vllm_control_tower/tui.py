"""Textual control panel: pick a config with the arrows, serve it, stop it, watch it."""

from __future__ import annotations

from collections import deque
from dataclasses import replace
from pathlib import Path

import httpx
from rich.console import Group
from rich.text import Text
from textual import on, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Grid, Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widget import Widget
from textual.widgets import (
    Button,
    Footer,
    Header,
    Label,
    ListItem,
    ListView,
    OptionList,
    RichLog,
    Static,
    TabbedContent,
    TabPane,
)
from textual.widgets.option_list import Option

from .config import ServeConfig, load_configs
from .estimator import Estimate, estimate
from .hardware import GiB, GpuSpec, detect_gpus, known_gpu_names, lookup
from .metrics import MetricsTracker
from .model_info import ModelInfoError, get_model_info
from .process import ProcessManager, State, Status
from .render import STATE_STYLE, bar, config_details, estimate_report

TABS = ["overview", "estimate", "monitor", "logs"]
LOCAL = "__local__"


class ConfigItem(ListItem):
    def __init__(self, cfg: ServeConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.label = Static(self._text(State.STOPPED))

    def compose(self) -> ComposeResult:
        yield self.label

    def set_state(self, state: State) -> None:
        self.label.update(self._text(state))

    def _text(self, state: State) -> Text:
        icon, style = STATE_STYLE[state]
        t = Text(f" {icon} ", style=style)
        t.append(self.cfg.name, style="bold")
        t.append(f"\n   :{self.cfg.port}  {self.cfg.model.split('/')[-1]}", style="grey50")
        return t


class Chart(Widget):
    """Zero-based block chart; newest sample on the right."""

    DEFAULT_CSS = "Chart { height: 3; width: 1fr; }"
    BLOCKS = " ▁▂▃▄▅▆▇█"

    def __init__(self, color: str, vmax: float | None = None) -> None:
        super().__init__()
        self.color = color
        self.vmax = vmax
        self.values: list[float] = []

    def set(self, values: list[float], vmax: float | None = None) -> None:
        self.values = values
        if vmax is not None:
            self.vmax = vmax
        self.refresh()

    def render(self) -> Text:
        width, height = self.size.width, self.size.height
        vals = self.values[-width:]
        vals = [0.0] * (width - len(vals)) + vals
        top = self.vmax or max(vals, default=0) or 1
        out = Text()
        for row in range(height):
            floor = (height - 1 - row) * 8
            for v in vals:
                level = int(min(max(v / top, 0), 1) * height * 8)
                fill = min(max(level - floor, 0), 8)
                if fill:
                    out.append(self.BLOCKS[fill], style=self.color)
                elif row == height - 1:
                    out.append("▁", style="#303030")
                else:
                    out.append(" ")
            if row < height - 1:
                out.append("\n")
        return out


class MetricCard(Vertical):
    """Title + current value + chart."""

    def __init__(
        self, title: str, fmt: str, color: str, id: str, vmax: float | None = None
    ) -> None:
        super().__init__(id=id, classes="card")
        self.title = title
        self.fmt = fmt
        self.color = color
        self.vmax = vmax

    def compose(self) -> ComposeResult:
        yield Label(self.title, classes="card-title")
        yield Static("–", classes="card-value")
        yield Chart(self.color, self.vmax)

    def push(self, values: list[float], current: float | None, vmax: float | None = None) -> None:
        self.query_one(Chart).set(values, vmax)
        text = "–" if current is None else self.fmt.format(current)
        self.query_one(".card-value", Static).update(Text(text, style=f"bold {self.color}"))


class ConfirmScreen(ModalScreen[bool]):
    BINDINGS = [
        ("escape", "dismiss(False)", "Cancel"),
        ("y", "dismiss(True)", "Yes"),
        ("n", "dismiss(False)", "No"),
    ]

    def __init__(self, message: str) -> None:
        super().__init__()
        self.message = message

    def compose(self) -> ComposeResult:
        with Vertical(id="dialog"):
            yield Label(self.message, id="dialog-text")
            with Horizontal(id="dialog-buttons"):
                yield Button("Stop [y]", variant="error", id="yes")
                yield Button("Cancel [n]", id="no")

    def on_mount(self) -> None:
        self.query_one("#no", Button).focus()

    @on(Button.Pressed)
    def _pressed(self, event: Button.Pressed) -> None:
        self.dismiss(event.button.id == "yes")


class HardwareScreen(ModalScreen[str | None]):
    """Choose which GPU the estimates are computed for."""

    BINDINGS = [("escape", "dismiss(None)", "Cancel")]

    def __init__(self, local: list[GpuSpec], current: str) -> None:
        super().__init__()
        self.local = local
        self.current = current

    def compose(self) -> ComposeResult:
        options = []
        if self.local:
            g = self.local[0]
            options.append(
                Option(
                    f"● Local: {len(self.local)}× {g.name} ({g.memory_bytes / GiB:.0f} GiB)",
                    id=LOCAL,
                )
            )
        for name in known_gpu_names():
            spec = lookup(name)
            options.append(
                Option(
                    f"  {name:<14} {spec.memory_bytes / GiB:>4.0f} GiB  {spec.bandwidth_gbs:>5,.0f} GB/s",
                    id=name,
                )
            )
        with Vertical(id="hw-dialog"):
            yield Label(
                "Estimate for which GPU?  [dim](table GPUs: count = tensor × pipeline parallel)[/]"
            )
            yield OptionList(*options, id="hw-list")

    def on_mount(self) -> None:
        ol = self.query_one(OptionList)
        try:
            ol.highlighted = ol.get_option_index(self.current)
        except Exception:
            ol.highlighted = 0
        ol.focus()

    @on(OptionList.OptionSelected)
    def _selected(self, event: OptionList.OptionSelected) -> None:
        self.dismiss(event.option.id)


class ControlTower(App):
    TITLE = "vLLM Control Tower"
    CSS = """
    Screen { layout: vertical; }
    #body { height: 1fr; }
    #sidebar { width: 36; border-right: tall $primary-background; }
    .section-title { color: $text-muted; text-style: bold; padding: 1 1 0 1; }
    #configs { height: 1fr; background: $surface; }
    #configs > ListItem { padding: 0 0 1 0; }
    #gpu-panel { height: auto; max-height: 12; padding: 0 1 1 1; }
    #main { width: 1fr; }
    TabPane { padding: 0 1; }
    #monitor-grid { grid-size: 2 3; grid-gutter: 1 2; padding: 1 0; height: auto; }
    .card { height: 9; border: round $primary-background; padding: 0 1; }
    .card-title { color: $text-muted; }
    .card-value { height: 1; }
    .card Chart { margin-top: 1; }
    #monitor-status { padding: 1 0 0 0; color: $text-muted; }
    #log-view { height: 1fr; }
    ConfirmScreen, HardwareScreen { align: center middle; }
    #dialog { width: 60; height: auto; border: thick $error; background: $surface; padding: 1 2; }
    #dialog-text { width: 100%; content-align: center middle; padding-bottom: 1; }
    #dialog-buttons { height: auto; align: center middle; }
    #dialog-buttons Button { margin: 0 1; }
    #hw-dialog { width: 84; height: 80%; border: thick $primary; background: $surface; padding: 1 2; }
    #hw-list { height: 1fr; margin-top: 1; }
    """

    BINDINGS = [
        Binding("s", "start", "Serve"),
        Binding("x", "stop", "Stop"),
        Binding("e", "show_tab('estimate')", "Estimate"),
        Binding("h", "hardware", "Hardware"),
        Binding("r", "reload", "Reload"),
        Binding("left", "tab(-1)", "◀ Tab", show=False),
        Binding("right", "tab(1)", "Tab ▶", show=False),
        Binding("1", "show_tab('overview')", show=False),
        Binding("2", "show_tab('estimate')", show=False),
        Binding("3", "show_tab('monitor')", show=False),
        Binding("4", "show_tab('logs')", show=False),
        Binding("q", "quit", "Quit"),
    ]

    def __init__(self, config_dir: Path, runtime_dir: Path) -> None:
        super().__init__()
        self.config_dir = config_dir
        self.runtime_dir = runtime_dir
        self.pm = ProcessManager(runtime_dir)
        self.configs: list[ServeConfig] = []
        self.config_errors: list[str] = []
        self.selected: str | None = None
        self.statuses: dict[str, Status] = {}
        self.trackers: dict[str, MetricsTracker] = {}
        self.estimates: dict[tuple[str, str], Estimate | str] = {}
        self.hardware = LOCAL
        self.local_gpus: list[GpuSpec] = []
        self.gpu_util: deque[float] = deque(maxlen=120)
        self.gpu_mem: deque[float] = deque(maxlen=120)
        self._log_offset: tuple[str | None, int] = (None, 0)
        self._http = httpx.Client(timeout=2.0)

    # ---- layout -----------------------------------------------------------

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Horizontal(id="body"):
            with Vertical(id="sidebar"):
                yield Label("CONFIGS", classes="section-title")
                yield ListView(id="configs")
                yield Label("GPU", classes="section-title")
                yield Static(id="gpu-panel")
            with TabbedContent(id="main", initial="overview"):
                with TabPane("Overview", id="overview"):
                    yield VerticalScroll(Static(id="overview-body"))
                with TabPane("Estimate", id="estimate"):
                    yield VerticalScroll(Static(id="estimate-body"))
                with TabPane("Monitor", id="monitor"):
                    with VerticalScroll():
                        yield Static(id="monitor-status")
                        with Grid(id="monitor-grid"):
                            yield MetricCard("Generation tok/s", "{:,.1f}", "#00d75f", id="m-gen")
                            yield MetricCard("Prompt tok/s", "{:,.1f}", "#0087ff", id="m-prompt")
                            yield MetricCard(
                                "KV cache usage", "{:.1%}", "#ffaf00", id="m-kv", vmax=100
                            )
                            yield MetricCard(
                                "Requests running / waiting", "{:,.0f}", "#af87ff", id="m-reqs"
                            )
                            yield MetricCard(
                                "GPU utilization", "{:.0f}%", "#ff005f", id="m-gpu", vmax=100
                            )
                            yield MetricCard("GPU memory", "{:.2f} GiB", "#ffd700", id="m-vram")
                with TabPane("Logs", id="logs"):
                    yield RichLog(
                        id="log-view", max_lines=2000, wrap=True, markup=False, highlight=True
                    )
        yield Footer()

    def on_mount(self) -> None:
        self.sub_title = str(self.config_dir.resolve())
        self.load()
        self.set_interval(2.0, self.poll)
        self.set_interval(1.0, self.refresh_logs)
        self.poll()
        self.query_one("#configs", ListView).focus()

    def on_unmount(self) -> None:
        self._http.close()

    # ---- data loading -------------------------------------------------------

    def load(self) -> None:
        configs, errors = load_configs(self.config_dir)
        self.configs = configs
        self.config_errors = [str(e) for e in errors]
        lv = self.query_one("#configs", ListView)
        lv.clear()
        for cfg in configs:
            item = ConfigItem(cfg)
            lv.append(item)
            st = self.statuses.get(cfg.name)
            if st:
                item.set_state(st.state)
        for err in self.config_errors:
            self.notify(err, title="Invalid config", severity="error", timeout=10)
        if not configs:
            self.query_one("#overview-body", Static).update(
                Text(f"No YAML configs found in {self.config_dir.resolve()}/", style="yellow")
            )
            return
        names = [c.name for c in configs]
        index = names.index(self.selected) if self.selected in names else 0
        lv.index = index
        self.select(names[index])

    def config(self, name: str | None) -> ServeConfig | None:
        return next((c for c in self.configs if c.name == name), None)

    def select(self, name: str) -> None:
        if name == self.selected and self._log_offset[0] == name:
            return
        self.selected = name
        self.render_overview()
        self.render_estimate()
        self.render_monitor()
        self._log_offset = (None, 0)
        self.refresh_logs()

    @on(ListView.Highlighted, "#configs")
    def _highlighted(self, event: ListView.Highlighted) -> None:
        if isinstance(event.item, ConfigItem):
            self.select(event.item.cfg.name)

    @on(ListView.Selected, "#configs")
    def _selected(self, event: ListView.Selected) -> None:
        self.action_start()

    # ---- rendering ----------------------------------------------------------

    def render_overview(self) -> None:
        cfg = self.config(self.selected)
        if cfg:
            self.query_one("#overview-body", Static).update(
                config_details(cfg, self.statuses.get(cfg.name))
            )

    def hardware_label(self) -> str:
        return "local GPU" if self.hardware == LOCAL else self.hardware

    def render_estimate(self) -> None:
        cfg = self.config(self.selected)
        if not cfg:
            return
        body = self.query_one("#estimate-body", Static)
        key = (cfg.name, self.hardware)
        result = self.estimates.get(key)
        hint = Text(
            f"Estimating for {self.hardware_label()} · press h to change hardware", style="grey50"
        )
        if result is None:
            body.update(
                Group(hint, Text(), Text(f"Computing estimate for {cfg.model}…", style="italic"))
            )
            self.compute_estimate(cfg, self.hardware)
        elif isinstance(result, str):
            body.update(Group(hint, Text(), Text(result, style="red")))
        else:
            body.update(Group(hint, estimate_report(result)))

    @work(thread=True, group="estimate")
    def compute_estimate(self, cfg: ServeConfig, hardware: str) -> None:
        key = (cfg.name, hardware)
        try:
            info = get_model_info(cfg.model, runtime_dir=self.runtime_dir)
            shards = int(cfg.arg("tensor-parallel-size", 1)) * int(
                cfg.arg("pipeline-parallel-size", 1)
            )
            if hardware == LOCAL:
                gpus = detect_gpus()
                if not gpus:
                    raise RuntimeError(
                        "No local GPU detected; press h to pick one from the spec table."
                    )
                tightest = min(gpus, key=lambda g: g.free_bytes)
                spec, n = replace(tightest, name=gpus[0].name), len(gpus)
            else:
                spec, n = lookup(hardware), shards
            result: Estimate | str = estimate(cfg, info, spec, n)
        except (ModelInfoError, RuntimeError, ValueError) as e:
            result = str(e)
        self.estimates[key] = result
        self.call_from_thread(self._estimate_ready, key)

    def _estimate_ready(self, key: tuple[str, str]) -> None:
        if key == (self.selected, self.hardware):
            self.render_estimate()

    def render_monitor(self) -> None:
        cfg = self.config(self.selected)
        if not cfg:
            return
        st = self.statuses.get(cfg.name)
        tracker = self.trackers.get(cfg.name)
        status = self.query_one("#monitor-status", Static)
        if not st or st.state in (State.STOPPED, State.STALE):
            status.update(Text(f"{cfg.name} is not running. Press s to serve it.", style="grey50"))
        elif tracker and tracker.error and st.state is not State.RUNNING:
            status.update(
                Text(f"{cfg.name} is starting… waiting for {cfg.base_url}/metrics", style="yellow")
            )
        elif tracker and tracker.error:
            status.update(Text(f"metrics unavailable: {tracker.error}", style="red"))
        else:
            status.update(Text(f"Live metrics from {cfg.base_url}/metrics", style="green"))

        last = tracker.snapshots[-1] if tracker and tracker.snapshots else None
        self.query_one("#m-gen", MetricCard).push(
            tracker.series("gen_tps") if tracker else [], last.gen_tps if last else None
        )
        self.query_one("#m-prompt", MetricCard).push(
            tracker.series("prompt_tps") if tracker else [], last.prompt_tps if last else None
        )
        self.query_one("#m-kv", MetricCard).push(
            [v * 100 for v in tracker.series("kv_usage")] if tracker else [],
            last.kv_usage if last else None,
        )
        reqs = self.query_one("#m-reqs", MetricCard)
        reqs.push(tracker.series("running") if tracker else [], None)
        if last:
            reqs.query_one(".card-value", Static).update(
                Text(
                    f"{last.running:.0f} running / {last.waiting:.0f} waiting", style="bold #af87ff"
                )
            )
        self.query_one("#m-gpu", MetricCard).push(
            list(self.gpu_util), self.gpu_util[-1] if self.gpu_util else None
        )
        total = sum(g.memory_bytes for g in self.local_gpus) / GiB or None
        self.query_one("#m-vram", MetricCard).push(
            list(self.gpu_mem), self.gpu_mem[-1] if self.gpu_mem else None, total
        )

    def render_gpus(self) -> None:
        panel = self.query_one("#gpu-panel", Static)
        if not self.local_gpus:
            panel.update(Text("No NVIDIA GPU detected", style="grey50"))
            return
        lines: list[Text] = []
        for g in self.local_gpus:
            frac = g.used_bytes / g.memory_bytes if g.memory_bytes else 0
            color = "green" if frac < 0.6 else "yellow" if frac < 0.85 else "red"
            name = g.name.replace("NVIDIA ", "").replace("GeForce ", "")
            lines.append(Text(f"{g.index} {name}", style="bold"))
            mem = Text("mem ")
            mem.append_text(bar(frac, 18, color))
            mem.append(f" {g.used_bytes / GiB:.1f}/{g.memory_bytes / GiB:.0f}G")
            lines.append(mem)
            util = Text("sm  ")
            util.append_text(bar((g.utilization or 0) / 100, 18, "deep_pink2"))
            util.append(f" {g.utilization or 0}%")
            extra = f"    {g.temperature}°C" if g.temperature is not None else ""
            extra += f"  {g.power_w:.0f} W" if g.power_w is not None else ""
            lines += [util, Text(extra, style="grey50")]
        panel.update(Group(*lines))

    # ---- polling ------------------------------------------------------------

    @work(thread=True, exclusive=True, group="poll")
    def poll(self) -> None:
        statuses = {c.name: self.pm.status(c.name) for c in self.configs}
        for name, st in statuses.items():
            if st.state in (State.RUNNING, State.STARTING, State.UNHEALTHY) and st.record:
                tracker = self.trackers.get(name)
                if tracker is None or tracker.base_url != st.record.base_url:
                    tracker = self.trackers[name] = MetricsTracker(st.record.base_url)
                tracker.poll(self._http)
            elif name in self.trackers and st.state is State.STOPPED:
                del self.trackers[name]
        gpus = detect_gpus()
        self.call_from_thread(self._poll_done, statuses, gpus)

    def _poll_done(self, statuses: dict[str, Status], gpus: list[GpuSpec]) -> None:
        changed = {
            n
            for n, s in statuses.items()
            if self.statuses.get(n, Status(n, State.STOPPED)).state != s.state
        }
        for name in changed:
            old = self.statuses.get(name)
            new = statuses[name]
            if old and old.state is State.STARTING and new.state is State.RUNNING:
                self.notify(
                    f"{name} is ready at {new.record.base_url}/v1",
                    title="Serving",
                    severity="information",
                )
            elif old and old.state in (State.STARTING, State.RUNNING) and new.state is State.STALE:
                self.notify(
                    f"{name} exited unexpectedly; see the Logs tab",
                    title="Crashed",
                    severity="error",
                    timeout=10,
                )
        self.statuses = statuses
        self.local_gpus = gpus
        if gpus:
            self.gpu_util.append(float(sum(g.utilization or 0 for g in gpus) / len(gpus)))
            self.gpu_mem.append(sum(g.used_bytes for g in gpus) / GiB)
        for item in self.query_one("#configs", ListView).query(ConfigItem):
            st = statuses.get(item.cfg.name)
            item.set_state(st.state if st else State.STOPPED)
        if self.selected in changed:
            self.render_overview()
        self.render_monitor()
        self.render_gpus()

    def refresh_logs(self) -> None:
        name = self.selected
        log = self.query_one("#log-view", RichLog)
        if not name:
            return
        path = self.pm.log_file(name)
        current, offset = self._log_offset
        if current != name:
            log.clear()
            if not path.exists():
                log.write(Text(f"No logs yet for {name}.", style="grey50"))
                self._log_offset = (name, 0)
                return
            for line in self.pm.tail(name, 500):
                log.write(line)
            self._log_offset = (name, path.stat().st_size)
            return
        if not path.exists():
            return
        size = path.stat().st_size
        if size < offset:  # rotated by a restart
            self._log_offset = (None, 0)
            self.refresh_logs()
            return
        if size == offset:
            return
        with path.open("rb") as f:
            f.seek(offset)
            chunk = f.read(size - offset).decode(errors="replace")
        for line in chunk.splitlines():
            log.write(line)
        self._log_offset = (name, size)

    # ---- actions ------------------------------------------------------------

    def action_tab(self, step: int) -> None:
        tc = self.query_one(TabbedContent)
        tc.active = TABS[(TABS.index(tc.active) + step) % len(TABS)]

    def action_show_tab(self, tab: str) -> None:
        self.query_one(TabbedContent).active = tab

    def action_reload(self) -> None:
        self.estimates.clear()
        self._log_offset = (None, 0)
        self.load()
        self.notify(f"Loaded {len(self.configs)} configs")

    def action_hardware(self) -> None:
        def chosen(value: str | None) -> None:
            if value:
                self.hardware = value
                self.action_show_tab("estimate")
                self.render_estimate()

        self.push_screen(HardwareScreen(detect_gpus(), self.hardware), chosen)

    def action_start(self) -> None:
        cfg = self.config(self.selected)
        if not cfg:
            return
        st = self.statuses.get(cfg.name)
        if st and st.state in (State.RUNNING, State.STARTING, State.UNHEALTHY):
            self.notify(f"{cfg.name} is already {st.state.value}", severity="warning")
            return
        try:
            rec = self.pm.start(cfg)
        except RuntimeError as e:
            self.notify(str(e), title=f"Cannot start {cfg.name}", severity="error", timeout=10)
            return
        self.notify(f"pid {rec.pid} · loading {cfg.model}", title=f"Starting {cfg.name}")
        self.statuses[cfg.name] = Status(cfg.name, State.STARTING, rec)
        self._log_offset = (None, 0)
        self.action_show_tab("logs")
        self.poll()

    def action_stop(self) -> None:
        cfg = self.config(self.selected)
        if not cfg:
            return
        st = self.statuses.get(cfg.name)
        if not st or st.state is State.STOPPED:
            self.notify(f"{cfg.name} is not running", severity="warning")
            return

        def confirmed(ok: bool | None) -> None:
            if ok:
                self.stop_server(cfg.name)

        self.push_screen(
            ConfirmScreen(f"Stop {cfg.name}?\n\n{cfg.model} on :{cfg.port}"), confirmed
        )

    @work(thread=True)
    def stop_server(self, name: str) -> None:
        self.call_from_thread(self.notify, f"Sending SIGTERM to {name}…", title="Stopping")
        was = self.pm.stop(name)
        self.call_from_thread(
            self.notify, f"{name} stopped" if was else f"{name} was not running", title="Stopped"
        )
        self.call_from_thread(self.poll)
