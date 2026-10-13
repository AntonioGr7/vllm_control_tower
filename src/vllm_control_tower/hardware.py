"""GPU discovery (NVML) and a small table of known GPU specs for estimates."""

from __future__ import annotations

import re
import threading
from dataclasses import dataclass, replace

GiB = 1024**3


@dataclass(frozen=True)
class GpuSpec:
    name: str
    memory_bytes: int
    bandwidth_gbs: float  # peak DRAM bandwidth, GB/s
    tflops_16: float  # dense fp16/bf16 tensor TFLOPS (fp32 accumulate)
    compute_capability: tuple[int, int] | None = None
    # Live readings, only for locally detected GPUs.
    index: int | None = None
    used_bytes: int = 0
    utilization: int | None = None
    temperature: int | None = None
    power_w: float | None = None
    source: str = "table"
    # CPU and GPU share system RAM (DGX Spark / GB10): memory comes from /proc/meminfo.
    unified_memory: bool = False

    @property
    def free_bytes(self) -> int:
        return max(self.memory_bytes - self.used_bytes, 0)

    @property
    def supports_bf16(self) -> bool:
        return self.compute_capability is None or self.compute_capability >= (8, 0)


# name, memory GiB, bandwidth GB/s, dense 16-bit tensor TFLOPS, compute capability
_KNOWN: list[tuple[str, float, float, float, tuple[int, int]]] = [
    ("B200", 180, 8000, 2250, (10, 0)),
    ("GB10", 128, 273, 100, (12, 1)),  # DGX Spark, unified LPDDR5x; TFLOPS approximate
    ("H200", 141, 4800, 989, (9, 0)),
    ("H100 SXM", 80, 3350, 989, (9, 0)),
    ("H100 NVL", 94, 3900, 835, (9, 0)),
    ("H100 PCIe", 80, 2000, 756, (9, 0)),
    ("H100", 80, 3350, 989, (9, 0)),
    ("GH200", 96, 4000, 989, (9, 0)),
    ("A100 80GB", 80, 2039, 312, (8, 0)),
    ("A100 40GB", 40, 1555, 312, (8, 0)),
    ("A100", 80, 2039, 312, (8, 0)),
    ("L40S", 48, 864, 362, (8, 9)),
    ("L40", 48, 864, 181, (8, 9)),
    ("L4", 24, 300, 121, (8, 9)),
    ("A10G", 24, 600, 70, (8, 6)),
    ("A10", 24, 600, 125, (8, 6)),
    ("A6000", 48, 768, 155, (8, 6)),
    ("RTX 6000 Ada", 48, 960, 364, (8, 9)),
    ("RTX PRO 6000", 96, 1792, 500, (12, 0)),
    ("T4", 16, 320, 65, (7, 5)),
    ("V100", 32, 900, 125, (7, 0)),
    ("RTX 5090", 32, 1792, 209, (12, 0)),
    ("RTX 5080", 16, 960, 113, (12, 0)),
    ("RTX 4090", 24, 1008, 165, (8, 9)),
    ("RTX 4080", 16, 717, 97, (8, 9)),
    ("RTX 4070", 12, 504, 58, (8, 9)),
    ("RTX 4060", 8, 272, 30, (8, 9)),
    ("RTX 3090", 24, 936, 71, (8, 6)),
    ("RTX 3080", 10, 760, 60, (8, 6)),
    ("RTX 3070", 8, 448, 41, (8, 6)),
    ("RTX 3060", 12, 360, 26, (8, 6)),
    ("RTX 3050", 8, 224, 18, (8, 6)),
    ("RTX 2050", 4, 112, 12, (8, 6)),
]


def _norm(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", name.lower()).strip()


def lookup(name: str) -> GpuSpec | None:
    """Find a known GPU by (partial) name, e.g. 'h100', 'A100 40GB', 'NVIDIA GeForce RTX 4090'."""
    query = set(_norm(name).split())
    # Most specific key first so "H100 PCIe" wins over "H100".
    for key, mem, bw, tf, cc in sorted(_KNOWN, key=lambda k: -len(_norm(k[0]).split())):
        if set(_norm(key).split()) <= query:
            return GpuSpec(key, int(mem * GiB), bw, tf, cc)
    # Short queries like "4090" or "pcie": accept a unique key containing them.
    hits = [k for k in _KNOWN if query <= set(_norm(k[0]).split())]
    if len(hits) == 1:
        key, mem, bw, tf, cc = hits[0]
        return GpuSpec(key, int(mem * GiB), bw, tf, cc)
    return None


def known_gpu_names() -> list[str]:
    return [k[0] for k in _KNOWN]


# NVML is process-global: initialise it once and never shut it down mid-run. The TUI
# polls GPUs from one thread while estimates run in another; per-call init/shutdown let
# one thread tear NVML down under the other and made static readings flap.
_nvml_lock = threading.Lock()
_nvml_state: dict[str, object] = {"module": None, "failed": False}
# Static per-device facts (bandwidth, compute, capability), read once per process.
_static: dict[int, tuple[float | None, float | None, tuple[int, int] | None]] = {}


def _nvml():
    if _nvml_state["module"] is None and not _nvml_state["failed"]:
        try:
            import pynvml as nv

            nv.nvmlInit()
            _nvml_state["module"] = nv
        except Exception:
            _nvml_state["failed"] = True
    return _nvml_state["module"]


def detect_gpus() -> list[GpuSpec]:
    """Return locally visible NVIDIA GPUs, or [] if NVML is unavailable."""
    with _nvml_lock:
        nv = _nvml()
        if nv is None:
            return []
        try:
            return [_read_gpu(nv, i) for i in range(nv.nvmlDeviceGetCount())]
        except Exception:
            return []


def _read_gpu(nv, i: int) -> GpuSpec:
    h = nv.nvmlDeviceGetHandleByIndex(i)
    name = nv.nvmlDeviceGetName(h)
    name = name.decode() if isinstance(name, bytes) else name
    # Unified-memory GPUs (GB10) have no dedicated VRAM and NVML reports it as unsupported;
    # the GPU allocates from system RAM, so budget against that instead.
    mem = _try(lambda: nv.nvmlDeviceGetMemoryInfo(h))
    unified = mem is None or not mem.total
    if unified:
        total, used = _system_memory()
    else:
        total, used = mem.total, mem.used
    known = lookup(name)
    if i not in _static or None in _static[i]:
        cc = _try(lambda: tuple(nv.nvmlDeviceGetCudaComputeCapability(h)))
        bandwidth = _nvml_bandwidth(nv, h)
        tflops = known.tflops_16 if known else _nvml_tflops(nv, h, cc)
        _static[i] = (bandwidth, tflops, cc)
    bandwidth, tflops, cc = _static[i]
    return GpuSpec(
        name=name,
        memory_bytes=total,
        bandwidth_gbs=bandwidth or (known.bandwidth_gbs if known else 300.0),
        tflops_16=tflops or 30.0,
        compute_capability=cc or (known.compute_capability if known else None),
        index=i,
        used_bytes=used,
        utilization=_try(lambda: nv.nvmlDeviceGetUtilizationRates(h).gpu),
        temperature=_try(lambda: nv.nvmlDeviceGetTemperature(h, nv.NVML_TEMPERATURE_GPU)),
        power_w=_try(lambda: nv.nvmlDeviceGetPowerUsage(h) / 1000),
        source="nvml",
        unified_memory=unified,
    )


def _system_memory() -> tuple[int, int]:
    """(total, used) system RAM from /proc/meminfo; page cache counts as free."""
    info: dict[str, int] = {}
    with open("/proc/meminfo") as f:
        for line in f:
            key, _, rest = line.partition(":")
            info[key] = int(rest.split()[0]) * 1024  # values are in kB
    total = info["MemTotal"]
    return total, total - info.get("MemAvailable", info.get("MemFree", 0))


def resolve_hardware(gpu: str | None = None, num_gpus: int | None = None) -> tuple[GpuSpec, int]:
    """Pick the GPU model to estimate for: an explicit name from the table, or the local GPU."""
    if gpu:
        spec = lookup(gpu)
        if spec is None:
            raise ValueError(f"unknown GPU {gpu!r}; known: {', '.join(known_gpu_names())}")
        return spec, num_gpus or 1
    local = detect_gpus()
    if not local:
        raise RuntimeError(
            "no local NVIDIA GPU detected; pass a GPU name explicitly (e.g. --gpu H100)"
        )
    # Assume a homogeneous node and budget against the most constrained card.
    tightest = min(local, key=lambda g: g.free_bytes)
    return replace(tightest, name=local[0].name), num_gpus or len(local)


def _nvml_bandwidth(nv, h) -> float | None:
    # Reported max memory clock * 2 (double data rate) * bus width bytes works for GDDR6/6X and HBM.
    bus = _try(lambda: nv.nvmlDeviceGetMemoryBusWidth(h))
    clock = _try(lambda: nv.nvmlDeviceGetMaxClockInfo(h, nv.NVML_CLOCK_MEM))
    if not bus or not clock:
        return None
    return clock * 1e6 * 2 * bus / 8 / 1e9


def _nvml_tflops(nv, h, cc) -> float | None:
    cores = _try(lambda: nv.nvmlDeviceGetNumGpuCores(h))
    clock = _try(lambda: nv.nvmlDeviceGetMaxClockInfo(h, nv.NVML_CLOCK_SM))
    if not cores or not clock:
        return None
    # Roughly 4x the FP32 rate for 16-bit tensor math with fp32 accumulate on consumer parts.
    return cores * 2 * clock * 1e6 * 4 / 1e12


def _try(fn):
    try:
        return fn()
    except Exception:
        return None
