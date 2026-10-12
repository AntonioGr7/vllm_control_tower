"""Start, stop and track vLLM server processes via PID files under the runtime dir."""

from __future__ import annotations

import contextlib
import json
import os
import signal
import socket
import subprocess
import time
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path

import httpx

from .config import DEFAULT_RUNTIME_DIR, ServeConfig


class State(str, Enum):
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    UNHEALTHY = "unhealthy"
    STALE = "stale"  # PID file left behind by a process that died


@dataclass
class ProcessRecord:
    name: str
    pid: int
    pgid: int
    port: int
    base_url: str
    model: str
    command: list[str]
    started_at: float
    log_file: str


@dataclass
class Status:
    name: str
    state: State
    record: ProcessRecord | None = None

    @property
    def uptime_s(self) -> float | None:
        return time.time() - self.record.started_at if self.record else None


class ProcessManager:
    def __init__(self, runtime_dir: Path = DEFAULT_RUNTIME_DIR):
        self.runtime_dir = runtime_dir
        self.pid_dir = runtime_dir / "pids"
        self.log_dir = runtime_dir / "logs"

    def pid_file(self, name: str) -> Path:
        return self.pid_dir / f"{name}.json"

    def log_file(self, name: str) -> Path:
        return self.log_dir / f"{name}.log"

    def read_record(self, name: str) -> ProcessRecord | None:
        f = self.pid_file(name)
        if not f.exists():
            return None
        try:
            return ProcessRecord(**json.loads(f.read_text()))
        except (TypeError, ValueError):
            return None

    def status(self, name: str, *, check_health: bool = True) -> Status:
        rec = self.read_record(name)
        if rec is None:
            return Status(name, State.STOPPED)
        if not _alive(rec.pid):
            return Status(name, State.STALE, rec)
        if not check_health:
            return Status(name, State.RUNNING, rec)
        if health(rec.base_url):
            return Status(name, State.RUNNING, rec)
        # vLLM takes a while to load weights and capture CUDA graphs before /health answers.
        return Status(
            name, State.STARTING if time.time() - rec.started_at < 900 else State.UNHEALTHY, rec
        )

    def running(self) -> list[Status]:
        if not self.pid_dir.exists():
            return []
        out = [self.status(p.stem, check_health=False) for p in sorted(self.pid_dir.glob("*.json"))]
        return [s for s in out if s.state is not State.STALE]

    def start(self, cfg: ServeConfig) -> ProcessRecord:
        current = self.status(cfg.name, check_health=False)
        if current.state is State.STALE:
            self.pid_file(cfg.name).unlink(missing_ok=True)
        elif current.state is not State.STOPPED:
            raise RuntimeError(f"{cfg.name} is already running (pid {current.record.pid})")
        for other in self.running():
            if other.record and other.record.port == cfg.port:
                raise RuntimeError(f"port {cfg.port} is already used by {other.name}")
        if _port_in_use(cfg.host, cfg.port):
            raise RuntimeError(f"port {cfg.port} is already in use by another process")

        self.pid_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        log = self.log_file(cfg.name)
        if log.exists():
            log.replace(log.with_suffix(".log.prev"))

        cmd = cfg.command()
        env = {**os.environ, **cfg.resolved_env()}
        env.setdefault("PYTHONUNBUFFERED", "1")
        with log.open("ab") as fh:
            fh.write(f"$ {' '.join(cmd)}\n".encode())
            fh.flush()
            try:
                proc = subprocess.Popen(
                    cmd,
                    stdout=fh,
                    stderr=subprocess.STDOUT,
                    stdin=subprocess.DEVNULL,
                    env=env,
                    start_new_session=True,  # own process group, survives the TUI exiting
                )
            except FileNotFoundError as e:
                raise RuntimeError(
                    f"cannot run {cmd[0]!r}: install vLLM (uv sync --extra vllm) or set vllm.executable"
                ) from e

        rec = ProcessRecord(
            name=cfg.name,
            pid=proc.pid,
            pgid=os.getpgid(proc.pid),
            port=cfg.port,
            base_url=cfg.base_url,
            model=cfg.model,
            command=cmd,
            started_at=time.time(),
            log_file=str(log),
        )
        self.pid_file(cfg.name).write_text(json.dumps(asdict(rec), indent=2))
        return rec

    def stop(self, name: str, timeout: float = 30.0) -> bool:
        """SIGTERM the process group, escalate to SIGKILL after `timeout`. Returns True if it was running."""
        rec = self.read_record(name)
        if rec is None:
            return False
        was_running = _alive(rec.pid)
        if was_running:
            _signal_group(rec.pgid, signal.SIGTERM)
            deadline = time.time() + timeout
            while time.time() < deadline and _group_alive(rec.pgid):
                time.sleep(0.25)
            if _group_alive(rec.pgid):
                _signal_group(rec.pgid, signal.SIGKILL)
                time.sleep(0.5)
        self.pid_file(name).unlink(missing_ok=True)
        return was_running

    def wait_healthy(self, name: str, timeout: float = 900.0, poll: float = 2.0) -> Status:
        deadline = time.time() + timeout
        while True:
            st = self.status(name)
            if st.state in (State.RUNNING, State.STALE, State.STOPPED) or time.time() > deadline:
                return st
            time.sleep(poll)

    def tail(self, name: str, lines: int = 200) -> list[str]:
        log = self.log_file(name)
        if not log.exists():
            return []
        with log.open("rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            f.seek(max(size - 256 * 1024, 0))
            data = f.read().decode(errors="replace")
        return data.splitlines()[-lines:]


def health(base_url: str, timeout: float = 1.5) -> bool:
    try:
        return httpx.get(f"{base_url}/health", timeout=timeout).status_code == 200
    except httpx.HTTPError:
        return False


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    # A zombie still answers kill(0); treat it as dead.
    try:
        with open(f"/proc/{pid}/stat") as f:
            return f.read().split(")")[-1].split()[0] != "Z"
    except OSError:
        return True


def _group_alive(pgid: int) -> bool:
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    # Reap our own children so they don't linger as zombies.
    try:
        while os.waitpid(-pgid, os.WNOHANG)[0]:
            pass
    except ChildProcessError:
        pass
    try:
        os.killpg(pgid, 0)
        return True
    except ProcessLookupError:
        return False


def _signal_group(pgid: int, sig: signal.Signals) -> None:
    with contextlib.suppress(ProcessLookupError):
        os.killpg(pgid, sig)


def _port_in_use(host: str, port: int) -> bool:
    """Try to bind like vLLM will; connect() probes give false positives under WSL2."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind((host, port))
        except OSError:
            return True
    return False
