import contextlib
import time

from vllm_control_tower.config import load_config
from vllm_control_tower.metrics import MetricsTracker, parse_prometheus
from vllm_control_tower.process import ProcessManager, State


def test_parse_prometheus_sums_labels():
    text = 'a{x="1"} 1\na{x="2"} 2.5\n# comment\nb 3e2\n'
    assert parse_prometheus(text) == {"a": 3.5, "b": 300.0}


def test_start_health_metrics_stop(write_config, fake_vllm_cmd, free_port, tmp_path):
    path = write_config(
        "fake", {"model": "fake/model", "port": free_port, "vllm": {"executable": fake_vllm_cmd}}
    )
    cfg = load_config(path)
    pm = ProcessManager(tmp_path / "rt")

    rec = pm.start(cfg)
    try:
        assert pm.status("fake", check_health=False).state is State.RUNNING
        st = pm.wait_healthy("fake", timeout=15, poll=0.2)
        assert st.state is State.RUNNING
        assert any("fake vllm serving" in line for line in pm.tail("fake"))

        tracker = MetricsTracker(cfg.base_url)
        assert tracker.poll() is not None
        time.sleep(0.3)
        snap = tracker.poll()
        assert snap.running == 2
        assert snap.kv_usage == 0.25
        assert snap.gen_tps > 0
    finally:
        assert pm.stop("fake", timeout=5)
    assert pm.status("fake").state is State.STOPPED
    assert not pm.pid_file("fake").exists()
    assert not pm.stop("fake")
    assert rec.pid > 0


def test_stale_pid_file_is_detected_and_cleared(write_config, fake_vllm_cmd, free_port, tmp_path):
    path = write_config(
        "fake", {"model": "fake/model", "port": free_port, "vllm": {"executable": fake_vllm_cmd}}
    )
    cfg = load_config(path)
    pm = ProcessManager(tmp_path / "rt")
    rec = pm.start(cfg)
    pm.wait_healthy("fake", timeout=15, poll=0.2)
    import os
    import signal

    os.killpg(rec.pgid, signal.SIGKILL)
    deadline = time.time() + 5
    while pm.status("fake").state is not State.STALE and time.time() < deadline:
        time.sleep(0.1)
        with contextlib.suppress(ChildProcessError):
            os.waitpid(rec.pid, os.WNOHANG)
    assert pm.status("fake").state is State.STALE
    # Starting again clears the stale record.
    pm.start(cfg)
    pm.wait_healthy("fake", timeout=15, poll=0.2)
    assert pm.stop("fake", timeout=5)


def test_refuses_double_start_and_busy_port(write_config, fake_vllm_cmd, free_port, tmp_path):
    a = load_config(
        write_config("a", {"model": "m", "port": free_port, "vllm": {"executable": fake_vllm_cmd}})
    )
    b = load_config(
        write_config("b", {"model": "m", "port": free_port, "vllm": {"executable": fake_vllm_cmd}})
    )
    pm = ProcessManager(tmp_path / "rt")
    pm.start(a)
    try:
        pm.wait_healthy("a", timeout=15, poll=0.2)
        for cfg, msg in ((a, "already running"), (b, "port")):
            try:
                pm.start(cfg)
            except RuntimeError as e:
                assert msg in str(e)
            else:
                raise AssertionError("expected RuntimeError")
    finally:
        pm.stop("a", timeout=5)
