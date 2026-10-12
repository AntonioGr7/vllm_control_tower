import json

import pytest

from vllm_control_tower.config import ServeConfig
from vllm_control_tower.estimator import estimate
from vllm_control_tower.hardware import GiB, GpuSpec, lookup
from vllm_control_tower.model_info import get_model_info, parse_config

from .conftest import QWEN3_06B


def _cfg(**args) -> ServeConfig:
    return ServeConfig(name="t", model="Qwen/Qwen3-0.6B", vllm={"args": args})


@pytest.fixture
def qwen():
    info = parse_config("Qwen/Qwen3-0.6B", QWEN3_06B)
    info.weight_bytes = 751_632_384 * 2
    info.param_count = 751_632_384
    info.params_source = "safetensors"
    return info


RTX2050 = GpuSpec("RTX 2050", 4 * GiB, 96, 12, (8, 6))


def test_kv_bytes_per_token(qwen):
    est = estimate(_cfg(**{"max-model-len": 4096, "gpu-memory-utilization": 0.8}), qwen, RTX2050)
    # 2 (K,V) * 28 layers * 8 kv heads * 128 head_dim * 2 bytes
    assert est.kv_bytes_per_token == 2 * 28 * 8 * 128 * 2
    assert est.fits
    assert est.kv_capacity_tokens % 16 == 0
    assert est.kv_capacity_tokens == pytest.approx(
        est.kv_cache_bytes / est.kv_bytes_per_token, abs=16
    )


def test_fp8_kv_cache_doubles_capacity(qwen):
    base = estimate(_cfg(**{"max-model-len": 4096}), qwen, RTX2050)
    fp8 = estimate(_cfg(**{"max-model-len": 4096, "kv-cache-dtype": "fp8"}), qwen, RTX2050)
    assert fp8.kv_bytes_per_token == base.kv_bytes_per_token / 2
    assert fp8.kv_capacity_tokens == pytest.approx(base.kv_capacity_tokens * 2, abs=16)


def test_tensor_parallel_splits_weights_and_kv(qwen):
    h100 = lookup("H100")
    tp1 = estimate(_cfg(**{"max-model-len": 8192}), qwen, h100, 2)
    tp2 = estimate(_cfg(**{"max-model-len": 8192, "tensor-parallel-size": 2}), qwen, h100, 2)
    assert tp2.weight_bytes == tp1.weight_bytes // 2
    assert tp2.kv_bytes_per_token == tp1.kv_bytes_per_token / 2


def test_too_many_gpus_is_an_error(qwen):
    est = estimate(_cfg(**{"tensor-parallel-size": 4}), qwen, RTX2050, 1)
    assert not est.fits
    assert any("needs 4 GPUs" in e for e in est.errors)


def test_context_too_long_for_kv_cache(qwen):
    est = estimate(_cfg(**{"max-model-len": 40960, "gpu-memory-utilization": 0.8}), qwen, RTX2050)
    assert not est.fits
    assert any("set max-model-len <=" in e for e in est.errors)


def test_utilization_above_free_memory_is_flagged(qwen):
    busy = GpuSpec("RTX 2050", 4 * GiB, 96, 12, (8, 6), used_bytes=int(0.6 * GiB), source="nvml")
    est = estimate(_cfg(**{"max-model-len": 2048, "gpu-memory-utilization": 0.9}), qwen, busy)
    assert any("gpu-memory-utilization=0.9" in e for e in est.errors)
    ok = estimate(_cfg(**{"max-model-len": 2048, "gpu-memory-utilization": 0.8}), qwen, busy)
    assert ok.fits


def test_throughput_is_memory_bound_and_grows_with_batch(qwen):
    est = estimate(_cfg(**{"max-model-len": 4096, "gpu-memory-utilization": 0.8}), qwen, RTX2050)
    # Batch 1 decode can't beat reading the weights once per token.
    assert est.single_stream_tps < 96e9 / est.weight_bytes
    assert est.curve[0].bound == "memory"
    tps = [p.tokens_per_s for p in est.curve]
    assert tps == sorted(tps)
    assert est.batch_tps > est.single_stream_tps


def test_bf16_falls_back_on_turing(qwen):
    t4 = lookup("T4")
    est = estimate(_cfg(**{"max-model-len": 4096}), qwen, t4)
    assert est.dtype == "float16"
    assert any("bf16" in w for w in est.warnings)


def test_moe_active_params():
    cfg = dict(QWEN3_06B, num_experts=8, num_experts_per_tok=2, moe_intermediate_size=512)
    info = parse_config("moe", cfg)
    assert info.is_moe
    assert info.active_param_count < info.param_count


def test_local_model_dir(tmp_path):
    import struct

    model_dir = tmp_path / "m"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(json.dumps(QWEN3_06B))
    header = json.dumps(
        {"w": {"dtype": "BF16", "shape": [10, 10], "data_offsets": [0, 200]}}
    ).encode()
    (model_dir / "model.safetensors").write_bytes(
        struct.pack("<Q", len(header)) + header + b"\0" * 200
    )
    info = get_model_info(str(model_dir), runtime_dir=tmp_path / "rt")
    assert info.param_count == 100
    assert info.weight_bytes == 200
    assert info.params_source == "safetensors"


def test_gpu_lookup():
    assert lookup("NVIDIA A100-SXM4-40GB").name == "A100 40GB"
    assert lookup("NVIDIA H100 PCIe").name == "H100 PCIe"
    assert lookup("4090").name == "RTX 4090"
    assert lookup("NVIDIA H100 80GB HBM3").name == "H100"
    assert lookup("unknown thing") is None


def test_tied_lm_head_copy_is_not_counted(tmp_path):
    import struct

    model_dir = tmp_path / "tied"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(json.dumps(QWEN3_06B))  # tie_word_embeddings: true
    tensors = {
        "model.embed_tokens.weight": {"dtype": "BF16", "shape": [10, 10], "data_offsets": [0, 200]},
        "lm_head.weight": {"dtype": "BF16", "shape": [10, 10], "data_offsets": [200, 400]},
        "model.layers.0.mlp.weight": {
            "dtype": "F8_E4M3",
            "shape": [10, 5],
            "data_offsets": [400, 450],
        },
    }
    header = json.dumps(tensors).encode()
    (model_dir / "model.safetensors").write_bytes(
        struct.pack("<Q", len(header)) + header + b"\0" * 450
    )
    info = get_model_info(str(model_dir), runtime_dir=tmp_path / "rt")
    assert info.dtype_counts == {"BF16": 100, "F8_E4M3": 50}
    assert info.weight_bytes == 250


def test_matches_vllm_on_rtx2050():
    """vLLM 0.30, Qwen3-0.6B-FP8, util 0.8, eager: 0.74 GiB weights, 2.33 GiB / 21,824 KV tokens."""
    info = parse_config(
        "Qwen/Qwen3-0.6B-FP8", {**QWEN3_06B, "quantization_config": {"quant_method": "fp8"}}
    )
    info.dtype_counts = {"BF16": 155_674_880, "F8_E4M3": 440_401_920}
    info.weight_bytes = 155_674_880 * 2 + 440_401_920
    cfg = ServeConfig(
        name="fp8",
        model="Qwen/Qwen3-0.6B-FP8",
        vllm={
            "args": {"gpu-memory-utilization": 0.8, "max-model-len": 8192, "max-num-seqs": 16},
            "flags": ["enforce-eager"],
        },
    )
    est = estimate(cfg, info, RTX2050)
    assert est.fits
    assert est.kv_cache_bytes / GiB == pytest.approx(2.33, abs=0.08)
    assert est.kv_capacity_tokens == pytest.approx(21_824, rel=0.04)
