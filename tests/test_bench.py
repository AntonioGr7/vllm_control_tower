import asyncio
import json

import pytest
from typer.testing import CliRunner

from vllm_control_tower import bench, calibration
from vllm_control_tower.cli import app
from vllm_control_tower.config import ServeConfig, load_file
from vllm_control_tower.estimator import estimate
from vllm_control_tower.hardware import GiB, GpuSpec
from vllm_control_tower.model_info import parse_config
from vllm_control_tower.process import ProcessManager, State

from .conftest import QWEN3_06B

RTX2050 = GpuSpec("RTX 2050", 4 * GiB, 96, 12, (8, 6))


def test_parse_memory_from_vllm_log():
    lines = [
        "(EngineCore pid=1) INFO 10-12 13:57:39 [model_runner.py:428] Model loading took 0.74 GiB memory and 90.7 seconds",
        "(EngineCore pid=1) INFO 10-12 13:57:52 [gpu_worker.py:640] Available KV cache memory: 2.33 GiB",
        "(EngineCore pid=1) INFO 10-12 13:57:52 [kv_cache_utils.py:2395] GPU KV cache size: 21,824 tokens, Maximum concurrency for 8,192 tokens per request: 2.66x",
    ]
    m = bench.parse_memory(lines)
    assert (m.weights_gib, m.kv_cache_gib, m.kv_tokens) == (0.74, 2.33, 21_824)
    assert (m.max_concurrency_len, m.max_concurrency) == (8192, 2.66)


def _deployment(write_config, exe, port, **model):
    return load_file(
        write_config(
            "fake",
            {
                "models": [
                    {"model": "fake/model", "port": port, "vllm": {"executable": exe}, **model}
                ]
            },
        )
    ).members[0]


def test_bench_generation_against_fake_server(write_config, fake_vllm_cmd, free_port, tmp_path):
    cfg = _deployment(write_config, fake_vllm_cmd, free_port)
    pm = ProcessManager(tmp_path / "rt")
    pm.start(cfg)
    try:
        st = pm.wait_healthy(cfg.name, timeout=15, poll=0.2)
        assert st.state is State.RUNNING, (st.state, pm.tail(cfg.name))
        result = asyncio.run(
            bench.run(
                cfg,
                concurrencies=[1, 4],
                prompt_tokens=40,
                output_tokens=20,
                gpu="Fake GPU",
                log_lines=pm.tail(cfg.name),
            )
        )
    finally:
        pm.stop(cfg.name, timeout=5)
    assert result.kind == "generate" and result.memory.kv_tokens == 12_345
    one, four = result.scenarios
    assert one.errors == four.errors == 0
    # The fake server emits one token every 4 ms per request.
    assert 2.0 < one.tpot_ms < 15.0
    assert one.per_seq_tps > 60
    # Requests run in parallel, so 4 at once give well over 2x the aggregate rate.
    assert four.agg_tps > 2 * one.agg_tps
    path = bench.save(result, tmp_path / "rt")
    assert json.loads(path.read_text())["model"] == "fake.main"
    assert bench.load("fake.main", tmp_path / "rt").scenario(4).concurrency == 4


def test_bench_embeddings_against_fake_server(write_config, fake_vllm_cmd, free_port, tmp_path):
    cfg = _deployment(
        write_config,
        fake_vllm_cmd,
        free_port,
        vllm={"executable": fake_vllm_cmd, "args": {"runner": "pooling", "max-model-len": 64}},
    )
    pm = ProcessManager(tmp_path / "rt")
    pm.start(cfg)
    try:
        st = pm.wait_healthy(cfg.name, timeout=15, poll=0.2)
        assert st.state is State.RUNNING, (st.state, pm.tail(cfg.name))
        result = asyncio.run(
            bench.run(
                cfg,
                concurrencies=[1, 8],
                prompt_tokens=512,
                output_tokens=0,
                gpu="Fake GPU",
                log_lines=[],
            )
        )
    finally:
        pm.stop(cfg.name, timeout=5)
    assert result.kind == "embed" and result.prompt_tokens == 62  # capped by max-model-len
    assert all(s.items_per_s > 0 for s in result.scenarios)


def _qwen_estimate(**kw):
    info = parse_config("Qwen/Qwen3-0.6B", QWEN3_06B)
    info.weight_bytes = 596_049_920 * 2
    cfg = ServeConfig(
        name="q.main",
        model="Qwen/Qwen3-0.6B",
        vllm={"args": {"gpu-memory-utilization": 0.8, "max-model-len": 4096, "max-num-seqs": 16}},
    )
    return estimate(cfg, info, RTX2050, **kw)


def test_fit_and_apply_calibration(tmp_path):
    raw = _qwen_estimate()
    modeled = {p.batch: p.step_ms for p in raw.curve}
    measured = bench.ModelBench(
        model="q.main",
        model_id="Qwen/Qwen3-0.6B",
        kind="generate",
        gpu="RTX 2050",
        eager=False,
        config_hash="x",
        prompt_tokens=512,
        output_tokens=128,
        measured_at="2026-10-13T12:00:00+00:00",
        memory=bench.MemoryFacts(),
        scenarios=[
            bench.ScenarioResult(c, c, 0, 1.0, 0, 0, 0, tpot_ms=modeled[c] + 90.0)
            for c in (1, 2, 4)
        ],
    )
    cal = bench.fit_calibration(measured, raw)
    assert cal.extra_step_s == pytest.approx(0.090)
    calibration.save(cal, tmp_path)
    loaded = calibration.load("RTX 2050", False, tmp_path)
    assert loaded == cal
    assert calibration.load("RTX 2050", True, tmp_path) is None  # eager is calibrated separately

    tuned = _qwen_estimate(calibration=loaded)
    assert tuned.calibrated and not raw.calibrated
    for p_raw, p_tuned in zip(raw.curve, tuned.curve, strict=True):
        assert p_tuned.step_ms == pytest.approx(p_raw.step_ms + 90.0)


def test_eager_calibration_scales_with_depth():
    cal = calibration.Calibration("g", True, 0.1, 28, "m", "t")
    assert cal.extra_for(56) == pytest.approx(0.2)
    graphs = calibration.Calibration("g", False, 0.1, 28, "m", "t")
    assert graphs.extra_for(56) == pytest.approx(0.1)


def test_pick_concurrencies_respects_kv_and_max_seqs():
    raw = _qwen_estimate()
    assert bench.pick_concurrencies(raw, 16, None, 640, cap=16)[0] == 1
    assert max(bench.pick_concurrencies(raw, 16, None, 640, cap=4)) <= 4
    assert bench.pick_concurrencies(raw, 16, 2_000, 640, cap=16) == [1, 2]


def test_cli_bench_starts_measures_and_stops(write_config, fake_vllm_cmd, free_port, tmp_path):
    _deployment(write_config, fake_vllm_cmd, free_port)
    rt = tmp_path / "rt"
    result = CliRunner().invoke(
        app,
        ["--config-dir", str(write_config.dir), "--runtime-dir", str(rt), "bench", "fake",
         "--prompt", "16", "--output", "8", "--max-concurrency", "2"],
    )  # fmt: skip
    assert result.exit_code == 0, result.output
    assert "fake.main · benchmark" in result.output
    assert "stopped fake.main" in result.output  # bench started it, so it stops it
    assert (rt / "bench" / "fake.main.json").exists()
    assert not ProcessManager(rt).pid_file("fake.main").exists()
