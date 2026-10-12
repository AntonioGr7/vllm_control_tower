from vllm_control_tower import ops
from vllm_control_tower.config import Deployment, load_catalog, load_file
from vllm_control_tower.estimator import discount_running, estimate, estimate_deployment
from vllm_control_tower.hardware import GiB, GpuSpec
from vllm_control_tower.model_info import parse_config
from vllm_control_tower.process import ProcessManager, State

from .conftest import QWEN3_06B


def _group(write_config, port_a=8010, port_b=8011, exe="vllm", util_a=0.5, util_b=0.3):
    return write_config(
        "duo",
        {
            "name": "duo",
            "env": {"SHARED": "1", "OVERRIDE": "group"},
            "host": "127.0.0.1",
            "models": [
                {
                    "name": "chat",
                    "model": "Qwen/Qwen3-0.6B",
                    "port": port_a,
                    "env": {"OVERRIDE": "member"},
                    "vllm": {
                        "executable": exe,
                        "args": {"gpu-memory-utilization": util_a, "max-model-len": 2048},
                    },
                },
                {
                    "name": "embed",
                    "model": "Qwen/Qwen3-0.6B",
                    "port": port_b,
                    "vllm": {
                        "executable": exe,
                        "args": {"gpu-memory-utilization": util_b, "max-model-len": 1024},
                    },
                },
            ],
        },
    )


def test_deployment_loading(write_config):
    _group(write_config)
    write_config("solo", {"models": [{"model": "m", "port": 9000}]})
    cat = load_catalog(write_config.dir)
    assert cat.errors == []
    assert [d.name for d in cat.deployments] == ["duo", "solo"]
    assert [m.name for m in cat.servers] == ["duo.chat", "duo.embed", "solo.main"]
    duo, solo = cat.deployments
    assert (len(duo.members), len(solo.members)) == (2, 1)
    chat, embed = duo.members
    assert chat.deployment == "duo" and chat.short_name == "chat"
    assert duo.member("chat") is chat and duo.member("duo.embed") is embed
    assert chat.resolved_env() == {"SHARED": "1", "OVERRIDE": "member"}
    assert embed.resolved_env() == {"SHARED": "1", "OVERRIDE": "group"}
    assert isinstance(load_file(duo.path), Deployment)


def test_rejects_shared_ports(write_config):
    _group(write_config, port_a=8010, port_b=8010)
    cat = load_catalog(write_config.dir)
    assert any("distinct ports" in str(e) for e in cat.errors)


def _estimates(group, gpu):
    info = parse_config("Qwen/Qwen3-0.6B", QWEN3_06B)
    return [(m, estimate(m, info, gpu)) for m in group.members]


def test_estimate_checks_combined_budget(write_config):
    group = load_file(_group(write_config, util_a=0.5, util_b=0.3))
    idle = GpuSpec("RTX 2050", 4 * GiB, 96, 12, (8, 6))
    assert not estimate_deployment("duo", _estimates(group, idle), idle).errors

    busy = GpuSpec("RTX 2050", 4 * GiB, 96, 12, (8, 6), used_bytes=int(1 * GiB), source="nvml")
    ge = estimate_deployment("duo", _estimates(group, busy), busy)
    # 3 GiB free - 2 CUDA contexts (0.2 GiB each) = 2.6 GiB -> 0.65 of 4 GiB, minus margin
    assert any("share the same GPU" in e and "<= 0.64" in e for e in ge.errors)
    assert not ge.fits


def test_members_on_different_devices_dont_share(write_config):
    path = _group(write_config, util_a=0.8, util_b=0.8)
    group = load_file(path)
    group.members[0].env["CUDA_VISIBLE_DEVICES"] = "0"
    group.members[1].env["CUDA_VISIBLE_DEVICES"] = "1"
    gpu = GpuSpec("A10", 24 * GiB, 600, 125, (8, 6))
    assert not estimate_deployment("duo", _estimates(group, gpu), gpu, 2).errors


def test_running_members_memory_is_given_back():
    gpu = GpuSpec("RTX 2050", 4 * GiB, 96, 12, (8, 6), used_bytes=int(2.6 * GiB), source="nvml")
    from vllm_control_tower.config import ServeConfig

    cfg = ServeConfig(name="x", model="m", vllm={"args": {"gpu-memory-utilization": 0.5}})
    assert discount_running(gpu, [cfg]).used_bytes == int(2.6 * GiB) - 2 * GiB
    assert discount_running(gpu, []).used_bytes == gpu.used_bytes


def test_start_in_order_and_stop_in_reverse(write_config, fake_vllm_cmd, tmp_path):
    import socket

    ports = []
    for _ in range(2):
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            ports.append(s.getsockname()[1])
    group = load_file(_group(write_config, *ports, exe=fake_vllm_cmd))
    pm = ProcessManager(tmp_path / "rt")
    events: list[tuple[str, str]] = []
    try:
        assert ops.start_deployment(
            pm, group, timeout=15, on_event=lambda n, m: events.append((n, m))
        )
        assert [n for n, m in events if m == "healthy"] == ["duo.chat", "duo.embed"]
        # chat was healthy before embed was even spawned
        order = [(n, m.split()[0]) for n, m in events]
        assert order.index(("duo.chat", "healthy")) < order.index(("duo.embed", "started"))
        assert all(pm.status(m.name).state is State.RUNNING for m in group.members)
        # starting again is a no-op
        events.clear()
        assert ops.start_deployment(
            pm, group, timeout=15, on_event=lambda n, m: events.append((n, m))
        )
        assert {m for _, m in events} == {"already running"}
    finally:
        results = ops.stop_entry(pm, group, timeout=5)
    assert results == [("duo.embed", True), ("duo.chat", True)]
    assert all(pm.status(m.name).state is State.STOPPED for m in group.members)
