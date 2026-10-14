import contextlib
import json
import time

from vllm_control_tower.config import load_file
from vllm_control_tower.metrics import MetricsTracker, parse_prometheus
from vllm_control_tower.process import ProcessManager, State


def test_parse_prometheus_sums_labels():
    text = 'a{x="1"} 1\na{x="2"} 2.5\n# comment\nb 3e2\n'
    assert parse_prometheus(text) == {"a": 3.5, "b": 300.0}


def test_start_health_metrics_stop(write_config, fake_vllm_cmd, free_port, tmp_path):
    path = write_config(
        "fake",
        {
            "models": [
                {"model": "fake/model", "port": free_port, "vllm": {"executable": fake_vllm_cmd}}
            ]
        },
    )
    cfg = load_file(path).members[0]
    pm = ProcessManager(tmp_path / "rt")

    rec = pm.start(cfg)
    try:
        assert pm.status(cfg.name, check_health=False).state is State.RUNNING
        st = pm.wait_healthy(cfg.name, timeout=15, poll=0.2)
        assert st.state is State.RUNNING
        assert any("fake vllm serving" in line for line in pm.tail(cfg.name))

        tracker = MetricsTracker(cfg.base_url)
        assert tracker.poll() is not None
        time.sleep(0.3)
        snap = tracker.poll()
        assert snap.running == 2
        assert snap.kv_usage == 0.25
        assert snap.gen_tps > 0
    finally:
        assert pm.stop(cfg.name, timeout=5)
    assert pm.status(cfg.name).state is State.STOPPED
    assert not pm.pid_file(cfg.name).exists()
    assert not pm.stop(cfg.name)
    assert rec.pid > 0


def test_stale_pid_file_is_detected_and_cleared(write_config, fake_vllm_cmd, free_port, tmp_path):
    path = write_config(
        "fake",
        {
            "models": [
                {"model": "fake/model", "port": free_port, "vllm": {"executable": fake_vllm_cmd}}
            ]
        },
    )
    cfg = load_file(path).members[0]
    pm = ProcessManager(tmp_path / "rt")
    rec = pm.start(cfg)
    pm.wait_healthy(cfg.name, timeout=15, poll=0.2)
    import os
    import signal

    os.killpg(rec.pgid, signal.SIGKILL)
    deadline = time.time() + 5
    while pm.status(cfg.name).state is not State.STALE and time.time() < deadline:
        time.sleep(0.1)
        with contextlib.suppress(ChildProcessError):
            os.waitpid(rec.pid, os.WNOHANG)
    assert pm.status(cfg.name).state is State.STALE
    # Starting again clears the stale record.
    pm.start(cfg)
    pm.wait_healthy(cfg.name, timeout=15, poll=0.2)
    assert pm.stop(cfg.name, timeout=5)


def _docker_calls(tmp_path) -> list[list[str]]:
    lines = (tmp_path / "docker" / "calls.jsonl").read_text().splitlines()
    return [json.loads(line) for line in lines]


def _docker_cfg(write_config, fake_docker_cmd, port, tmp_path):
    path = write_config(
        "boxed",
        {
            "env": {"HF_TOKEN": "t"},
            "models": [
                {
                    "model": "fake/model",
                    "port": port,
                    "vllm": {
                        "image": "vllm/vllm-openai:test",
                        "docker": {"executable": fake_docker_cmd, "hf_cache": str(tmp_path / "hf")},
                    },
                }
            ],
        },
    )
    return load_file(path).members[0]


def test_docker_start_health_stop(write_config, fake_docker_cmd, free_port, tmp_path):
    cfg = _docker_cfg(write_config, fake_docker_cmd, free_port, tmp_path)
    pm = ProcessManager(tmp_path / "rt")
    rec = pm.start(cfg)
    try:
        assert rec.container == "vct-boxed.main"
        assert pm.read_record(cfg.name).container == "vct-boxed.main"
        assert (tmp_path / "hf").is_dir()  # created by us, not by docker as root
        assert pm.wait_healthy(cfg.name, timeout=15, poll=0.2).state is State.RUNNING
    finally:
        assert pm.stop(cfg.name, timeout=5)
    calls = _docker_calls(tmp_path)
    assert calls[0] == ["rm", "-f", "vct-boxed.main"]  # a leftover container is cleared
    assert calls[1][0] == "run"
    assert calls[2] == ["stop", "-t", "5", "vct-boxed.main"]
    assert pm.status(cfg.name).state is State.STOPPED


def test_docker_orphan_container_is_removed_on_start(
    write_config, fake_docker_cmd, free_port, tmp_path
):
    """`docker run` died but its container kept serving: the next start clears it."""
    cfg = _docker_cfg(write_config, fake_docker_cmd, free_port, tmp_path)
    pm = ProcessManager(tmp_path / "rt")
    pm.start(cfg)
    pm.wait_healthy(cfg.name, timeout=15, poll=0.2)
    pm.pid_file(cfg.name).unlink()  # vct lost track of it; the port is still held
    rec = pm.start(cfg)  # would fail with "port in use" if the orphan survived
    try:
        assert pm.wait_healthy(cfg.name, timeout=15, poll=0.2).state is State.RUNNING
        assert rec.pid > 0
    finally:
        pm.stop(cfg.name, timeout=5)


def test_refuses_double_start_and_busy_port(write_config, fake_vllm_cmd, free_port, tmp_path):
    a = load_file(
        write_config(
            "a",
            {"models": [{"model": "m", "port": free_port, "vllm": {"executable": fake_vllm_cmd}}]},
        )
    ).members[0]
    b = load_file(
        write_config(
            "b",
            {"models": [{"model": "m", "port": free_port, "vllm": {"executable": fake_vllm_cmd}}]},
        )
    ).members[0]
    pm = ProcessManager(tmp_path / "rt")
    pm.start(a)
    try:
        pm.wait_healthy(a.name, timeout=15, poll=0.2)
        for cfg, msg in ((a, "already running"), (b, "port")):
            try:
                pm.start(cfg)
            except RuntimeError as e:
                assert msg in str(e)
            else:
                raise AssertionError("expected RuntimeError")
    finally:
        pm.stop(a.name, timeout=5)
