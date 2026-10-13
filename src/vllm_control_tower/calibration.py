"""Per-GPU corrections to the throughput model, learned from `vct bench` runs.

The roofline model leaves out the fixed cost of each decode step: kernel launches, the
Python scheduler loop, host<->device copies. That cost depends on the machine (driver,
CPU, WSL vs native) and on whether vLLM runs eagerly or with CUDA graphs, so it is
measured rather than guessed, and stored per GPU name and execution mode.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from .config import DEFAULT_RUNTIME_DIR


@dataclass
class Calibration:
    gpu: str
    eager: bool
    extra_step_s: float  # measured step time minus the modeled step time
    layers: int  # of the benchmarked model; eager overhead scales with kernel count
    source: str  # deployment.model that was benchmarked
    measured_at: str

    def extra_for(self, layers: int) -> float:
        """Extra seconds per decode step for a model with `layers` layers."""
        if self.eager and self.layers:
            # Eager mode launches every kernel separately, roughly proportional to depth.
            return self.extra_step_s * layers / self.layers
        return self.extra_step_s


def key(gpu: str, eager: bool) -> str:
    return f"{gpu}|{'eager' if eager else 'graphs'}"


def _path(runtime_dir: Path) -> Path:
    return runtime_dir / "calibration.json"


def load(gpu: str, eager: bool, runtime_dir: Path = DEFAULT_RUNTIME_DIR) -> Calibration | None:
    try:
        data = json.loads(_path(runtime_dir).read_text())
        return Calibration(**data[key(gpu, eager)])
    except (OSError, ValueError, KeyError, TypeError):
        return None


def save(cal: Calibration, runtime_dir: Path = DEFAULT_RUNTIME_DIR) -> None:
    path = _path(runtime_dir)
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        data = {}
    data[key(cal.gpu, cal.eager)] = asdict(cal)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))
