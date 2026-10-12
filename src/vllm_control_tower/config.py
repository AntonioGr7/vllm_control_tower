"""Serving configurations: one YAML file per config inside a configs folder."""

from __future__ import annotations

import json
import os
import re
import shutil
import sys
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, ValidationError, field_validator

DEFAULT_CONFIG_DIR = Path(os.environ.get("VCT_CONFIG_DIR", "configs"))
DEFAULT_RUNTIME_DIR = Path(os.environ.get("VCT_RUNTIME_DIR", ".vct"))

_ENV_VAR = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


class VllmSection(BaseModel):
    executable: str = "vllm"
    # Any `vllm serve` option, written with or without leading dashes.
    args: dict[str, Any] = Field(default_factory=dict)
    # Boolean switches such as `enforce-eager`.
    flags: list[str] = Field(default_factory=list)
    # Raw extra arguments appended verbatim.
    extra_args: list[str] = Field(default_factory=list)

    @field_validator("args", mode="before")
    @classmethod
    def _normalize_args(cls, v: dict[str, Any] | None) -> dict[str, Any]:
        return {_norm(k): val for k, val in (v or {}).items()}

    @field_validator("flags", mode="before")
    @classmethod
    def _normalize_flags(cls, v: list[str] | None) -> list[str]:
        return [_norm(f) for f in (v or [])]


class EstimateHints(BaseModel):
    """Workload assumptions used by the throughput estimator."""

    avg_prompt_tokens: int = 512
    avg_output_tokens: int = 256


class ServeConfig(BaseModel):
    name: str
    description: str = ""
    model: str
    host: str = "127.0.0.1"
    port: int = 8000
    env: dict[str, str] = Field(default_factory=dict)
    vllm: VllmSection = Field(default_factory=VllmSection)
    estimate: EstimateHints = Field(default_factory=EstimateHints)

    # Set by the loader, not part of the YAML.
    path: Path | None = Field(default=None, exclude=True)

    def arg(self, name: str, default: Any = None) -> Any:
        return self.vllm.args.get(_norm(name), default)

    def has_flag(self, name: str) -> bool:
        return _norm(name) in self.vllm.flags

    @property
    def base_url(self) -> str:
        host = "127.0.0.1" if self.host in ("0.0.0.0", "::") else self.host
        return f"http://{host}:{self.port}"

    def resolved_env(self) -> dict[str, str]:
        return {k: _expand(str(v)) for k, v in self.env.items()}

    def command(self) -> list[str]:
        cmd = [resolve_executable(self.vllm.executable), "serve", self.model]
        cmd += ["--host", self.host, "--port", str(self.port)]
        for key, value in self.vllm.args.items():
            if key in ("host", "port"):
                continue
            if value is None or value is False:
                continue
            if value is True:
                cmd.append(f"--{key}")
            elif isinstance(value, (dict, list)):
                cmd += [f"--{key}", json.dumps(value)]
            else:
                cmd += [f"--{key}", str(value)]
        cmd += [f"--{f}" for f in self.vllm.flags]
        cmd += [_expand(a) for a in self.vllm.extra_args]
        return cmd


class ConfigError(Exception):
    def __init__(self, path: Path, message: str):
        super().__init__(f"{path}: {message}")
        self.path = path
        self.message = message


def _norm(key: str) -> str:
    return key.lstrip("-").replace("_", "-")


def _expand(value: str) -> str:
    def sub(m: re.Match[str]) -> str:
        return os.environ.get(m.group(1), m.group(2) or "")

    return _ENV_VAR.sub(sub, value)


def resolve_executable(name: str) -> str:
    """Prefer the executable from this project's venv, then PATH."""
    if os.sep in name:
        return name
    venv_bin = Path(sys.prefix) / "bin" / name
    if venv_bin.exists():
        return str(venv_bin)
    return shutil.which(name) or name


def load_config(path: Path) -> ServeConfig:
    try:
        data = yaml.safe_load(path.read_text()) or {}
    except yaml.YAMLError as e:
        raise ConfigError(path, f"invalid YAML: {e}") from e
    if not isinstance(data, dict):
        raise ConfigError(path, "top level must be a mapping")
    data.setdefault("name", path.stem)
    try:
        cfg = ServeConfig.model_validate(data)
    except ValidationError as e:
        raise ConfigError(path, str(e)) from e
    cfg.path = path
    return cfg


def load_configs(
    config_dir: Path = DEFAULT_CONFIG_DIR,
) -> tuple[list[ServeConfig], list[ConfigError]]:
    """Load every *.yaml / *.yml in the folder. Invalid files are returned as errors."""
    configs: list[ServeConfig] = []
    errors: list[ConfigError] = []
    if not config_dir.is_dir():
        return configs, errors
    files = sorted([*config_dir.glob("*.yaml"), *config_dir.glob("*.yml")])
    for f in files:
        try:
            configs.append(load_config(f))
        except ConfigError as e:
            errors.append(e)
    seen: dict[str, Path] = {}
    for c in configs:
        if c.name in seen:
            errors.append(
                ConfigError(c.path, f"duplicate name {c.name!r} (also in {seen[c.name]})")
            )
        seen[c.name] = c.path
    return configs, errors


def find_config(name: str, config_dir: Path = DEFAULT_CONFIG_DIR) -> ServeConfig:
    configs, _ = load_configs(config_dir)
    for c in configs:
        if c.name == name:
            return c
    candidate = Path(name)
    if candidate.is_file():
        return load_config(candidate)
    raise KeyError(name)
