from types import SimpleNamespace

import pytest

from vllm_control_tower import hardware
from vllm_control_tower.hardware import GiB, lookup


class FakeNvml:
    """Just enough of pynvml for _read_gpu; memory_info=None mimics GB10's NOT_SUPPORTED."""

    NVML_TEMPERATURE_GPU = 0
    NVML_CLOCK_MEM = 1
    NVML_CLOCK_SM = 2

    def __init__(self, name: str, memory_info):
        self.name = name
        self.memory_info = memory_info

    def nvmlDeviceGetHandleByIndex(self, i):
        return i

    def nvmlDeviceGetName(self, h):
        return self.name

    def nvmlDeviceGetMemoryInfo(self, h):
        if self.memory_info is None:
            raise RuntimeError("NVML_ERROR_NOT_SUPPORTED")
        return self.memory_info

    def nvmlDeviceGetCudaComputeCapability(self, h):
        return (12, 1)

    def __getattr__(self, attr):
        def unsupported(*args):
            raise RuntimeError("NVML_ERROR_NOT_SUPPORTED")

        return unsupported


@pytest.fixture(autouse=True)
def _clear_static_cache():
    hardware._static.clear()
    yield
    hardware._static.clear()


def test_unified_memory_falls_back_to_system_ram(monkeypatch):
    monkeypatch.setattr(hardware, "_system_memory", lambda: (119 * GiB, 9 * GiB))
    gpu = hardware._read_gpu(FakeNvml("NVIDIA GB10", None), 0)
    assert gpu.unified_memory
    assert gpu.memory_bytes == 119 * GiB
    assert gpu.free_bytes == 110 * GiB
    # Specs NVML can't report come from the table.
    assert gpu.bandwidth_gbs == lookup("GB10").bandwidth_gbs
    assert gpu.compute_capability == (12, 1)


def test_dedicated_vram_still_uses_nvml(monkeypatch):
    monkeypatch.setattr(hardware, "_system_memory", lambda: pytest.fail("read system RAM"))
    mem = SimpleNamespace(total=24 * GiB, used=2 * GiB)
    gpu = hardware._read_gpu(FakeNvml("NVIDIA GeForce RTX 4090", mem), 0)
    assert not gpu.unified_memory
    assert gpu.memory_bytes == 24 * GiB
    assert gpu.used_bytes == 2 * GiB


def test_system_memory_counts_page_cache_as_free(tmp_path, monkeypatch):
    meminfo = tmp_path / "meminfo"
    meminfo.write_text(
        "MemTotal:       124000000 kB\nMemFree:          2000000 kB\nMemAvailable:   100000000 kB\n"
    )
    real_open = open
    monkeypatch.setattr(
        "builtins.open",
        lambda path, *a, **k: real_open(meminfo if path == "/proc/meminfo" else path, *a, **k),
    )
    total, used = hardware._system_memory()
    assert total == 124000000 * 1024
    assert used == 24000000 * 1024


def test_lookup_gb10_by_nvml_name():
    spec = lookup("NVIDIA GB10")
    assert spec is not None and spec.name == "GB10"
    assert spec.compute_capability == (12, 1)
