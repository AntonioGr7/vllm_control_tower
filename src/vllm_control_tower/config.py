"""Deployments: one YAML file per deployment inside a configs folder.

A deployment has a name, shared defaults and a `models:` list with one or more models.
Each model is one vLLM server (a ServeConfig), addressed as `<deployment>.<model>`.
Models start in the order listed, each after the previous is healthy, and stop in
reverse order.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import sys
from dataclasses import dataclass, field
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
    deployment: str | None = Field(default=None, exclude=True)

    @property
    def short_name(self) -> str:
        """The model's name inside its deployment."""
        return self.name.removeprefix(f"{self.deployment}.") if self.deployment else self.name

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


class Deployment(BaseModel):
    """One or more models served together, e.g. a single chat model, or chat + embeddings."""

    name: str
    description: str = ""
    # Defaults applied to every model; a model's own values win.
    host: str | None = None
    env: dict[str, str] = Field(default_factory=dict)
    members: list[ServeConfig] = Field(alias="models", min_length=1)

    path: Path | None = Field(default=None, exclude=True)

    @classmethod
    def from_yaml(cls, data: dict[str, Any]) -> Deployment:
        raw_models = data.get("models")
        if not isinstance(raw_models, list) or not raw_models:
            raise ValueError("a deployment needs a non-empty `models:` list")
        shared_env = data.get("env") or {}
        members = []
        for i, raw in enumerate(raw_models):
            if not isinstance(raw, dict):
                raise ValueError(f"models[{i}] must be a mapping")
            m = dict(raw)
            default = "main" if len(raw_models) == 1 else f"m{i}"
            m["name"] = f"{data['name']}.{m.get('name', default)}"
            m["env"] = {**shared_env, **(m.get("env") or {})}
            if data.get("host"):
                m.setdefault("host", data["host"])
            members.append(m)
        return cls.model_validate({**data, "models": members})

    @property
    def single(self) -> bool:
        return len(self.members) == 1

    @property
    def ports(self) -> list[int]:
        return [m.port for m in self.members]

    def member(self, name: str) -> ServeConfig | None:
        """A model by full (`dep.model`) or short (`model`) name."""
        return next((m for m in self.members if name in (m.name, m.short_name)), None)


@dataclass
class Catalog:
    deployments: list[Deployment] = field(default_factory=list)
    errors: list[ConfigError] = field(default_factory=list)

    @property
    def servers(self) -> list[ServeConfig]:
        """Every vLLM server across all deployments."""
        return [m for d in self.deployments for m in d.members]

    def deployment(self, name: str) -> Deployment | None:
        return next((d for d in self.deployments if d.name == name), None)

    def server(self, name: str) -> ServeConfig | None:
        return next((m for m in self.servers if m.name == name), None)


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


def _read_yaml(path: Path) -> dict[str, Any]:
    try:
        data = yaml.safe_load(path.read_text()) or {}
    except yaml.YAMLError as e:
        raise ConfigError(path, f"invalid YAML: {e}") from e
    if not isinstance(data, dict):
        raise ConfigError(path, "top level must be a mapping")
    data.setdefault("name", path.stem)
    return data


def load_file(path: Path) -> Deployment:
    data = _read_yaml(path)
    if "models" not in data:
        raise ConfigError(path, "missing `models:` list (even a single model goes under `models:`)")
    try:
        dep = Deployment.from_yaml(data)
    except (ValidationError, ValueError) as e:
        raise ConfigError(path, str(e)) from e
    dep.path = path
    names = [m.name for m in dep.members]
    if len(set(names)) != len(names):
        raise ConfigError(path, "model names must be unique within a deployment")
    if len(set(dep.ports)) != len(dep.ports):
        raise ConfigError(path, f"models need distinct ports, got {dep.ports}")
    for m in dep.members:
        m.path = path
        m.deployment = dep.name
    return dep


def load_catalog(config_dir: Path = DEFAULT_CONFIG_DIR) -> Catalog:
    """Load every *.yaml / *.yml in the folder. Invalid files are reported as errors."""
    cat = Catalog()
    if not config_dir.is_dir():
        return cat
    files = sorted([*config_dir.glob("*.yaml"), *config_dir.glob("*.yml")])
    for f in files:
        try:
            cat.deployments.append(load_file(f))
        except ConfigError as e:
            cat.errors.append(e)
    seen: dict[str, Path | None] = {}
    for d in cat.deployments:
        if d.name in seen:
            cat.errors.append(
                ConfigError(d.path, f"duplicate name {d.name!r} (also in {seen[d.name]})")
            )
        seen[d.name] = d.path
    return cat


def find_deployment(name: str, config_dir: Path = DEFAULT_CONFIG_DIR) -> Deployment:
    """A deployment by name, or by path to its YAML file."""
    dep = load_catalog(config_dir).deployment(name)
    if dep:
        return dep
    candidate = Path(name)
    if candidate.is_file():
        return load_file(candidate)
    raise KeyError(name)
