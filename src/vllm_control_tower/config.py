"""Serving configurations: YAML files inside a configs folder.

A file holds either one model (a ServeConfig) or a group: a `models:` list of ServeConfigs
that are started together, in order, and stopped together. Group members are addressed as
`<group>.<member>`.
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
    group: str | None = Field(default=None, exclude=True)

    @property
    def short_name(self) -> str:
        return self.name.removeprefix(f"{self.group}.") if self.group else self.name

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


class GroupConfig(BaseModel):
    """Several models served side by side, e.g. a chat model plus an embedding model."""

    name: str
    description: str = ""
    # Defaults applied to every member; a member's own values win.
    host: str | None = None
    env: dict[str, str] = Field(default_factory=dict)
    members: list[ServeConfig] = Field(default_factory=list, alias="models", min_length=1)

    path: Path | None = Field(default=None, exclude=True)

    @classmethod
    def from_yaml(cls, data: dict[str, Any]) -> GroupConfig:
        group_name = data.get("name")
        shared_env = data.get("env") or {}
        members = []
        for i, raw in enumerate(data.get("models") or []):
            if not isinstance(raw, dict):
                raise ValueError(f"models[{i}] must be a mapping")
            m = dict(raw)
            m["name"] = f"{group_name}.{m.get('name', f'm{i}')}"
            m["env"] = {**shared_env, **(m.get("env") or {})}
            if data.get("host"):
                m.setdefault("host", data["host"])
            members.append(m)
        return cls.model_validate({**data, "models": members})

    @property
    def ports(self) -> list[int]:
        return [m.port for m in self.members]


@dataclass
class Catalog:
    configs: list[ServeConfig] = field(
        default_factory=list
    )  # every runnable model, members included
    groups: list[GroupConfig] = field(default_factory=list)
    errors: list[ConfigError] = field(default_factory=list)

    @property
    def entries(self) -> list[ServeConfig | GroupConfig]:
        """Top-level items in file order: standalone configs and groups (members follow their group)."""
        items: list[tuple[str, ServeConfig | GroupConfig]] = [
            (str(c.path), c) for c in self.configs if c.group is None
        ]
        items += [(str(g.path), g) for g in self.groups]
        return [item for _, item in sorted(items, key=lambda t: t[0])]

    def config(self, name: str) -> ServeConfig | None:
        return next((c for c in self.configs if c.name == name), None)

    def group(self, name: str) -> GroupConfig | None:
        return next((g for g in self.groups if g.name == name), None)


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


def load_file(path: Path) -> ServeConfig | GroupConfig:
    data = _read_yaml(path)
    try:
        if "models" in data:
            item: ServeConfig | GroupConfig = GroupConfig.from_yaml(data)
        else:
            item = ServeConfig.model_validate(data)
    except (ValidationError, ValueError) as e:
        raise ConfigError(path, str(e)) from e
    item.path = path
    if isinstance(item, GroupConfig):
        names = [m.name for m in item.members]
        if len(set(names)) != len(names):
            raise ConfigError(path, "member names must be unique within a group")
        if len(set(item.ports)) != len(item.ports):
            raise ConfigError(path, f"members need distinct ports, got {item.ports}")
        for m in item.members:
            m.path = path
            m.group = item.name
    return item


def load_config(path: Path) -> ServeConfig:
    item = load_file(path)
    if isinstance(item, GroupConfig):
        raise ConfigError(path, "is a group config; use load_file()")
    return item


def load_catalog(config_dir: Path = DEFAULT_CONFIG_DIR) -> Catalog:
    """Load every *.yaml / *.yml in the folder. Invalid files are reported as errors."""
    cat = Catalog()
    if not config_dir.is_dir():
        return cat
    files = sorted([*config_dir.glob("*.yaml"), *config_dir.glob("*.yml")])
    for f in files:
        try:
            item = load_file(f)
        except ConfigError as e:
            cat.errors.append(e)
            continue
        if isinstance(item, GroupConfig):
            cat.groups.append(item)
            cat.configs.extend(item.members)
        else:
            cat.configs.append(item)
    seen: dict[str, Path | None] = {}
    for name, path in [(c.name, c.path) for c in cat.configs] + [
        (g.name, g.path) for g in cat.groups
    ]:
        if name in seen:
            cat.errors.append(ConfigError(path, f"duplicate name {name!r} (also in {seen[name]})"))
        seen[name] = path
    return cat


def load_configs(
    config_dir: Path = DEFAULT_CONFIG_DIR,
) -> tuple[list[ServeConfig], list[ConfigError]]:
    """Every runnable config (group members included) plus load errors."""
    cat = load_catalog(config_dir)
    return cat.configs, cat.errors


def find_config(name: str, config_dir: Path = DEFAULT_CONFIG_DIR) -> ServeConfig:
    cfg = load_catalog(config_dir).config(name)
    if cfg:
        return cfg
    candidate = Path(name)
    if candidate.is_file():
        return load_config(candidate)
    raise KeyError(name)


def find_entry(name: str, config_dir: Path = DEFAULT_CONFIG_DIR) -> ServeConfig | GroupConfig:
    """A config or a group by name (or a path to a YAML file)."""
    cat = load_catalog(config_dir)
    item = cat.group(name) or cat.config(name)
    if item:
        return item
    candidate = Path(name)
    if candidate.is_file():
        return load_file(candidate)
    raise KeyError(name)
