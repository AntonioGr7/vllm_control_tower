import pytest

from vllm_control_tower.config import find_deployment, load_catalog


def _single(model: str, **kw) -> dict:
    return {"models": [{"model": model, **kw}]}


def test_single_model_deployment_and_command(write_config, monkeypatch):
    monkeypatch.setenv("MY_TOKEN", "secret")
    write_config(
        "small",
        {
            "env": {"HF_TOKEN": "${MY_TOKEN}", "OTHER": "${UNSET_VAR:-fallback}"},
            "models": [
                {
                    "model": "Qwen/Qwen3-0.6B",
                    "port": 8123,
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
                }
            ],
        },
    )
    cat = load_catalog(write_config.dir)
    assert cat.errors == []
    (dep,) = cat.deployments
    assert dep.name == "small"  # defaults to the file stem
    assert len(dep.members) == 1
    (cfg,) = dep.members
    assert cfg.name == "small.main"  # a lone model defaults to "main"
    assert cfg.short_name == "main" and cfg.deployment == "small"
    assert cat.server("small.main") is cfg
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


def test_docker_command(write_config, monkeypatch, tmp_path):
    monkeypatch.setenv("MY_TOKEN", "secret")
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf"))
    write_config(
        "spark",
        {
            "env": {"HF_TOKEN": "${MY_TOKEN}", "HF_HOME": "/elsewhere"},
            "models": [
                {
                    "name": "chat",
                    "model": "nvidia/Qwen3.8-27B-NVFP4",
                    "port": 8030,
                    "vllm": {
                        "image": "vllm/vllm-openai:qwen38",
                        "args": {"max-model-len": 4096},
                        "docker": {"executable": "/usr/bin/docker", "args": ["--shm-size", "16g"]},
                    },
                },
                {
                    "name": "local weights",
                    "model": "/models/embed",
                    "port": 8031,
                    "vllm": {"image": "nvcr.io/nvidia/vllm:26.05-py3"},
                },
            ],
        },
    )
    chat, local = load_catalog(write_config.dir).deployments[0].members
    assert chat.container == "vct-spark.chat"
    assert chat.command() == [
        "/usr/bin/docker", "run", "--rm", "--init",
        "--name", "vct-spark.chat", "--gpus", "all", "--ipc", "host", "--network", "host",
        "-v", "/elsewhere:/root/.cache/huggingface",  # HF_HOME from `env:` wins
        "-e", "PYTHONUNBUFFERED=1", "-e", "HF_TOKEN",  # names only: no secrets in the log
        "--shm-size", "16g",
        "--entrypoint", "vllm", "vllm/vllm-openai:qwen38",
        "serve", "nvidia/Qwen3.8-27B-NVFP4", "--host", "127.0.0.1", "--port", "8030",
        "--max-model-len", "4096",
    ]  # fmt: skip
    assert chat.resolved_env()["HF_TOKEN"] == "secret"  # docker reads it from its environment

    cmd = local.command()
    assert local.container == "vct-spark.local-weights"
    assert cmd[0].endswith("docker") and cmd[1] == "run"  # resolved on PATH, like vllm
    assert "/models/embed:/models/embed:ro" in cmd  # local weights are visible inside
    assert "/elsewhere:/root/.cache/huggingface" in cmd
    assert cmd[cmd.index("--entrypoint") :][:3] == [
        "--entrypoint",
        "vllm",
        "nvcr.io/nvidia/vllm:26.05-py3",
    ]


def test_hf_cache_defaults(monkeypatch, tmp_path):
    from vllm_control_tower.config import ServeConfig

    cfg = ServeConfig(name="x", model="m", vllm={"image": "img"})
    monkeypatch.delenv("HF_HOME", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    assert cfg.hf_cache() == tmp_path / ".cache/huggingface"
    monkeypatch.setenv("HF_HOME", "/hf")
    assert str(cfg.hf_cache()) == "/hf"
    assert ServeConfig(name="y", model="m").container is None


def test_invalid_and_duplicate_deployments(write_config):
    write_config("good", {"name": "dup", **_single("a")})
    write_config("also", {"name": "dup", **_single("b")})
    (write_config.dir / "broken.yaml").write_text("models: [unclosed")
    write_config("missing", {"models": [{"port": 1}]})
    write_config("flat", {"model": "x", "port": 1})  # old flat format
    write_config("empty", {"models": []})
    cat = load_catalog(write_config.dir)
    messages = " | ".join(str(e) for e in cat.errors)
    assert "invalid YAML" in messages
    assert "Field required" in messages and "model" in messages
    assert "duplicate name 'dup'" in messages
    assert "missing `models:` list" in messages
    assert "non-empty `models:` list" in messages
    assert len(cat.deployments) == 2


def test_find_deployment(write_config):
    write_config("a", _single("x"))
    assert find_deployment("a", write_config.dir).members[0].model == "x"
    with pytest.raises(KeyError):
        find_deployment("nope", write_config.dir)
