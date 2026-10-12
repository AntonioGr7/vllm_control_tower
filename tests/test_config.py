import pytest

from vllm_control_tower.config import find_config, load_configs


def test_load_and_command(write_config, monkeypatch):
    monkeypatch.setenv("MY_TOKEN", "secret")
    write_config(
        "small",
        {
            "model": "Qwen/Qwen3-0.6B",
            "port": 8123,
            "env": {"HF_TOKEN": "${MY_TOKEN}", "OTHER": "${UNSET_VAR:-fallback}"},
            "vllm": {
                "executable": "/opt/vllm",
                "args": {
                    "--max-model-len": 4096,
                    "gpu_memory_utilization": 0.8,
                    "enable-prefix-caching": True,
                },
                "flags": ["--enforce-eager"],
                "extra_args": ["--seed", "1"],
            },
        },
    )
    configs, errors = load_configs(write_config.dir)
    assert errors == []
    (cfg,) = configs
    assert cfg.name == "small"  # defaults to the file stem
    assert cfg.arg("max_model_len") == 4096
    assert cfg.arg("--gpu-memory-utilization") == 0.8
    assert cfg.has_flag("enforce_eager")
    assert cfg.resolved_env() == {"HF_TOKEN": "secret", "OTHER": "fallback"}
    assert cfg.command() == [
        "/opt/vllm", "serve", "Qwen/Qwen3-0.6B",
        "--host", "127.0.0.1", "--port", "8123",
        "--max-model-len", "4096",
        "--gpu-memory-utilization", "0.8",
        "--enable-prefix-caching",
        "--enforce-eager",
        "--seed", "1",
    ]  # fmt: skip


def test_invalid_and_duplicate_configs(write_config):
    write_config("good", {"name": "dup", "model": "a"})
    write_config("also", {"name": "dup", "model": "b"})
    (write_config.dir / "broken.yaml").write_text("model: [unclosed")
    write_config("missing", {"port": 1})
    configs, errors = load_configs(write_config.dir)
    messages = " | ".join(str(e) for e in errors)
    assert "invalid YAML" in messages
    assert "model" in messages  # missing required field
    assert "duplicate name 'dup'" in messages
    assert len(configs) == 2


def test_find_config_unknown(write_config):
    write_config("a", {"model": "x"})
    assert find_config("a", write_config.dir).model == "x"
    with pytest.raises(KeyError):
        find_config("nope", write_config.dir)
