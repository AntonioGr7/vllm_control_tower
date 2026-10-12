import socket
import sys
from pathlib import Path

import pytest
import yaml

FAKE_VLLM = Path(__file__).parent / "fake_vllm.py"

QWEN3_06B = {
    "architectures": ["Qwen3ForCausalLM"],
    "model_type": "qwen3",
    "head_dim": 128,
    "hidden_size": 1024,
    "intermediate_size": 3072,
    "max_position_embeddings": 40960,
    "num_attention_heads": 16,
    "num_hidden_layers": 28,
    "num_key_value_heads": 8,
    "tie_word_embeddings": True,
    "torch_dtype": "bfloat16",
    "vocab_size": 151936,
}


@pytest.fixture
def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def write_config(tmp_path):
    config_dir = tmp_path / "configs"
    config_dir.mkdir()

    def write(name: str, data: dict) -> Path:
        path = config_dir / f"{name}.yaml"
        path.write_text(yaml.safe_dump(data, sort_keys=False))
        return path

    write.dir = config_dir
    return write


@pytest.fixture
def fake_vllm_cmd(tmp_path) -> str:
    """An executable wrapper so the config's `executable` points at the fake server."""
    wrapper = tmp_path / "vllm"
    wrapper.write_text(f'#!/bin/sh\nexec {sys.executable} {FAKE_VLLM} "$@"\n')
    wrapper.chmod(0o755)
    return str(wrapper)
