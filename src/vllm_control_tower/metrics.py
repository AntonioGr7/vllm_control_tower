"""Scrape vLLM's Prometheus /metrics endpoint and keep short rolling histories."""

from __future__ import annotations

import re
import time
from collections import deque
from dataclasses import dataclass, field

import httpx

_LINE = re.compile(r"^([a-zA-Z_:][a-zA-Z0-9_:]*)(\{[^}]*\})?\s+([-+0-9.eEinfNa]+)")

# Metric names changed across vLLM versions; first match wins.
RUNNING = ("vllm:num_requests_running",)
WAITING = ("vllm:num_requests_waiting",)
KV_USAGE = ("vllm:kv_cache_usage_perc", "vllm:gpu_cache_usage_perc")
GEN_TOKENS = ("vllm:generation_tokens_total",)
PROMPT_TOKENS = ("vllm:prompt_tokens_total",)
PREFIX_HITS = ("vllm:prefix_cache_hits_total", "vllm:gpu_prefix_cache_hits_total")
PREFIX_QUERIES = ("vllm:prefix_cache_queries_total", "vllm:gpu_prefix_cache_queries_total")


def parse_prometheus(text: str) -> dict[str, float]:
    """Sum samples per metric name across label sets."""
    out: dict[str, float] = {}
    for line in text.splitlines():
        if not line or line[0] == "#":
            continue
        m = _LINE.match(line)
        if not m:
            continue
        try:
            value = float(m.group(3))
        except ValueError:
            continue
        out[m.group(1)] = out.get(m.group(1), 0.0) + value
    return out


def _pick(samples: dict[str, float], names: tuple[str, ...]) -> float | None:
    for n in names:
        if n in samples:
            return samples[n]
    return None


@dataclass
class Snapshot:
    t: float
    running: float = 0
    waiting: float = 0
    kv_usage: float = 0  # 0..1
    gen_tps: float = 0
    prompt_tps: float = 0
    prefix_hit_rate: float | None = None


@dataclass
class MetricsTracker:
    base_url: str
    history: int = 120
    snapshots: deque[Snapshot] = field(default_factory=deque)
    _prev: tuple[float, dict[str, float]] | None = None
    error: str | None = None

    def poll(self, client: httpx.Client | None = None) -> Snapshot | None:
        try:
            get = client.get if client else httpx.get
            resp = get(f"{self.base_url}/metrics", timeout=2.0)
            resp.raise_for_status()
        except httpx.HTTPError as e:
            self.error = str(e) or type(e).__name__
            return None
        self.error = None
        now = time.time()
        samples = parse_prometheus(resp.text)
        snap = Snapshot(
            t=now,
            running=_pick(samples, RUNNING) or 0,
            waiting=_pick(samples, WAITING) or 0,
            kv_usage=_pick(samples, KV_USAGE) or 0,
        )
        if self._prev:
            t0, prev = self._prev
            dt = max(now - t0, 1e-6)
            snap.gen_tps = _rate(samples, prev, GEN_TOKENS, dt)
            snap.prompt_tps = _rate(samples, prev, PROMPT_TOKENS, dt)
            hits = _delta(samples, prev, PREFIX_HITS)
            queries = _delta(samples, prev, PREFIX_QUERIES)
            if queries:
                snap.prefix_hit_rate = hits / queries
        self._prev = (now, samples)
        self.snapshots.append(snap)
        while len(self.snapshots) > self.history:
            self.snapshots.popleft()
        return snap

    def series(self, attr: str) -> list[float]:
        return [getattr(s, attr) or 0.0 for s in self.snapshots]


def _delta(cur: dict[str, float], prev: dict[str, float], names: tuple[str, ...]) -> float:
    a, b = _pick(cur, names), _pick(prev, names)
    if a is None or b is None:
        return 0.0
    return max(a - b, 0.0)  # counters reset on restart


def _rate(
    cur: dict[str, float], prev: dict[str, float], names: tuple[str, ...], dt: float
) -> float:
    return _delta(cur, prev, names) / dt
