"""Measure a running model with short synthetic requests, to set against the estimate.

Generation models get streamed /v1/completions requests at increasing concurrency, using the
same prompt/output lengths and batch sizes as the estimate. Prompts are exact token-id lists
(built with vLLM's /tokenize) with a unique prefix each, so prefix caching cannot inflate the
numbers, and `ignore_eos` makes every request generate exactly the requested tokens.
Pooling (embedding) models get /v1/embeddings requests instead.

What vLLM actually allocated (weights, KV cache) is read from its startup log.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
import secrets
import statistics
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import httpx

from .calibration import Calibration
from .config import ServeConfig, _expand
from .estimator import Estimate

TIMEOUT = httpx.Timeout(connect=5.0, read=600.0, write=30.0, pool=None)
FILLER = (
    "The control tower watches every runway, tracks each arrival and departure, and keeps "
    "a log of weather, fuel, crews and gates so that the whole airport moves on time. "
)


@dataclass
class RequestResult:
    ok: bool
    latency: float = 0.0
    ttft: float = 0.0
    output_tokens: int = 0
    error: str | None = None


@dataclass
class ScenarioResult:
    concurrency: int
    requests: int
    errors: int
    wall_s: float
    agg_tps: float  # generated tokens/s across all requests (embeddings: prompt tokens/s)
    per_seq_tps: float  # median decode speed of one request after its first token
    ttft_ms: float  # median time to first token
    tpot_ms: float  # median time per output token after the first
    items_per_s: float = 0.0  # embeddings only


@dataclass
class MemoryFacts:
    """What vLLM reported at startup."""

    weights_gib: float | None = None
    kv_cache_gib: float | None = None
    kv_tokens: int | None = None
    max_concurrency: float | None = None
    max_concurrency_len: int | None = None


@dataclass
class ModelBench:
    model: str  # deployment.model
    model_id: str
    kind: str  # "generate" or "embed"
    gpu: str
    eager: bool
    config_hash: str
    prompt_tokens: int
    output_tokens: int
    measured_at: str
    memory: MemoryFacts
    scenarios: list[ScenarioResult] = field(default_factory=list)

    def scenario(self, concurrency: int) -> ScenarioResult | None:
        return next((s for s in self.scenarios if s.concurrency == concurrency), None)


Progress = Callable[[str], None]


def config_hash(cfg: ServeConfig) -> str:
    """Identifies the exact vLLM command a benchmark was taken with."""
    cmd = cfg.command()[1:]  # the executable path may differ between machines
    return hashlib.sha1(json.dumps([cmd, cfg.resolved_env()]).encode()).hexdigest()[:12]


def served_name(cfg: ServeConfig) -> str:
    name = cfg.arg("served-model-name")
    if isinstance(name, list):
        name = name[0] if name else None
    return str(name) if name else cfg.model


def _headers(cfg: ServeConfig) -> dict[str, str]:
    key = cfg.arg("api-key")
    return {"Authorization": f"Bearer {_expand(str(key))}"} if key else {}


# ---- vLLM startup log --------------------------------------------------------------

_MEMORY_PATTERNS = {
    "weights_gib": re.compile(r"Model loading took ([\d.]+) GiB"),
    "kv_cache_gib": re.compile(r"Available KV cache memory: ([\d.]+) GiB"),
    "kv_tokens": re.compile(r"GPU KV cache size: ([\d,]+) tokens"),
    "concurrency": re.compile(r"Maximum concurrency for ([\d,]+) tokens per request: ([\d.]+)x"),
}


def parse_memory(lines: list[str]) -> MemoryFacts:
    facts = MemoryFacts()
    for line in lines:
        if m := _MEMORY_PATTERNS["weights_gib"].search(line):
            facts.weights_gib = float(m.group(1))
        elif m := _MEMORY_PATTERNS["kv_cache_gib"].search(line):
            facts.kv_cache_gib = float(m.group(1))
        if m := _MEMORY_PATTERNS["kv_tokens"].search(line):
            facts.kv_tokens = int(m.group(1).replace(",", ""))
        if m := _MEMORY_PATTERNS["concurrency"].search(line):
            facts.max_concurrency_len = int(m.group(1).replace(",", ""))
            facts.max_concurrency = float(m.group(2))
    return facts


# ---- requests ----------------------------------------------------------------------


class Bench:
    def __init__(self, cfg: ServeConfig, base_url: str | None = None):
        self.cfg = cfg
        self.base_url = base_url or cfg.base_url
        self.model = served_name(cfg)
        self.client = httpx.AsyncClient(timeout=TIMEOUT, headers=_headers(cfg))
        self._filler: list[int] | None = None

    async def close(self) -> None:
        await self.client.aclose()

    async def tokenize(self, text: str) -> list[int]:
        r = await self.client.post(
            f"{self.base_url}/tokenize",
            json={"model": self.model, "prompt": text, "add_special_tokens": False},
        )
        r.raise_for_status()
        return r.json()["tokens"]

    async def prompt(self, length: int) -> list[int]:
        """`length` token ids with a unique prefix, so no two requests share cached blocks."""
        if self._filler is None:
            self._filler = await self.tokenize(FILLER * 8)
        prefix = await self.tokenize(f"Request {secrets.token_hex(6)}. ")
        ids = list(prefix)
        while len(ids) < length:
            ids += self._filler
        return ids[:length]

    async def generate(self, prompt: list[int], max_tokens: int) -> RequestResult:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "min_tokens": max_tokens,
            "ignore_eos": True,
            "temperature": 0.8,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        start = time.perf_counter()
        ttft = None
        chunks = 0
        usage_tokens = None
        try:
            async with self.client.stream(
                "POST", f"{self.base_url}/v1/completions", json=payload
            ) as r:
                if r.status_code != 200:
                    body = (await r.aread()).decode(errors="replace")[:200]
                    return RequestResult(False, error=f"HTTP {r.status_code}: {body}")
                async for line in r.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    data = line[6:].strip()
                    if data == "[DONE]":
                        break
                    obj = json.loads(data)
                    if obj.get("choices") and obj["choices"][0].get("text"):
                        chunks += 1
                        if ttft is None:
                            ttft = time.perf_counter() - start
                    if obj.get("usage"):
                        usage_tokens = obj["usage"].get("completion_tokens")
        except (httpx.HTTPError, json.JSONDecodeError) as e:
            return RequestResult(False, error=str(e) or type(e).__name__)
        latency = time.perf_counter() - start
        return RequestResult(True, latency, ttft or latency, usage_tokens or chunks)

    async def embed(self, prompt: list[int]) -> RequestResult:
        start = time.perf_counter()
        try:
            r = await self.client.post(
                f"{self.base_url}/v1/embeddings", json={"model": self.model, "input": [prompt]}
            )
            if r.status_code != 200:
                return RequestResult(False, error=f"HTTP {r.status_code}: {r.text[:200]}")
        except httpx.HTTPError as e:
            return RequestResult(False, error=str(e) or type(e).__name__)
        latency = time.perf_counter() - start
        return RequestResult(True, latency, latency, 0)

    async def scenario(
        self, concurrency: int, prompt_tokens: int, output_tokens: int, kind: str
    ) -> ScenarioResult:
        prompts = [await self.prompt(prompt_tokens) for _ in range(concurrency)]
        start = time.perf_counter()
        if kind == "embed":
            results = await asyncio.gather(*(self.embed(p) for p in prompts))
        else:
            results = await asyncio.gather(*(self.generate(p, output_tokens) for p in prompts))
        wall = time.perf_counter() - start
        ok = [r for r in results if r.ok]
        errors = [r.error for r in results if not r.ok]
        if not ok:
            raise RuntimeError(f"all {concurrency} requests failed: {errors[0]}")
        if kind == "embed":
            return ScenarioResult(
                concurrency=concurrency,
                requests=len(results),
                errors=len(errors),
                wall_s=wall,
                agg_tps=len(ok) * prompt_tokens / wall,
                per_seq_tps=0.0,
                ttft_ms=statistics.median(r.latency for r in ok) * 1e3,
                tpot_ms=0.0,
                items_per_s=len(ok) / wall,
            )
        decode = [
            ((r.latency - r.ttft) / (r.output_tokens - 1), r)
            for r in ok
            if r.output_tokens > 1 and r.latency > r.ttft
        ]
        tpot = statistics.median(d for d, _ in decode) if decode else 0.0
        return ScenarioResult(
            concurrency=concurrency,
            requests=len(results),
            errors=len(errors),
            wall_s=wall,
            agg_tps=sum(r.output_tokens for r in ok) / wall,
            per_seq_tps=1 / tpot if tpot else 0.0,
            ttft_ms=statistics.median(r.ttft for r in ok) * 1e3,
            tpot_ms=tpot * 1e3,
        )


async def run(
    cfg: ServeConfig,
    *,
    concurrencies: list[int],
    prompt_tokens: int,
    output_tokens: int,
    gpu: str,
    log_lines: list[str],
    on_progress: Progress | None = None,
) -> ModelBench:
    say = on_progress or (lambda _m: None)
    kind = "embed" if str(cfg.arg("runner", "")).lower() == "pooling" else "generate"
    if kind == "embed":
        max_len = int(cfg.arg("max-model-len", 512))
        prompt_tokens = min(prompt_tokens, max_len - 2)
    result = ModelBench(
        model=cfg.name,
        model_id=cfg.model,
        kind=kind,
        gpu=gpu,
        eager=cfg.has_flag("enforce-eager") or bool(cfg.arg("enforce-eager", False)),
        config_hash=config_hash(cfg),
        prompt_tokens=prompt_tokens,
        output_tokens=output_tokens if kind == "generate" else 0,
        measured_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        memory=parse_memory(log_lines),
    )
    bench = Bench(cfg)
    try:
        say("warming up")
        if kind == "embed":
            await bench.embed(await bench.prompt(min(prompt_tokens, 64)))
        else:
            warm = await bench.generate(await bench.prompt(min(prompt_tokens, 64)), 8)
            if not warm.ok:
                raise RuntimeError(f"warm-up request failed: {warm.error}")
        for c in concurrencies:
            say(f"{c} concurrent request{'s' if c > 1 else ''}")
            result.scenarios.append(await bench.scenario(c, prompt_tokens, output_tokens, kind))
    finally:
        await bench.close()
    return result


# ---- results -----------------------------------------------------------------------


def fit_calibration(bench: ModelBench, raw: Estimate) -> Calibration | None:
    """Fixed per-step overhead the roofline model misses: median of measured - modeled."""
    if bench.kind != "generate":
        return None
    modeled = {p.batch: p.step_ms for p in raw.curve}
    diffs = [
        s.tpot_ms - modeled[s.concurrency]
        for s in bench.scenarios
        if s.concurrency in modeled and s.tpot_ms
    ]
    if not diffs:
        return None
    return Calibration(
        gpu=bench.gpu,
        eager=bench.eager,
        extra_step_s=max(statistics.median(diffs), 0.0) / 1e3,
        layers=raw.model.num_layers // max(raw.pp, 1),
        source=bench.model,
        measured_at=bench.measured_at,
    )


def _path(runtime_dir: Path, model: str) -> Path:
    return runtime_dir / "bench" / f"{model}.json"


def save(bench: ModelBench, runtime_dir: Path) -> Path:
    path = _path(runtime_dir, bench.model)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(bench), indent=2))
    return path


def load(model: str, runtime_dir: Path) -> ModelBench | None:
    try:
        data = json.loads(_path(runtime_dir, model).read_text())
        data["memory"] = MemoryFacts(**data["memory"])
        data["scenarios"] = [ScenarioResult(**s) for s in data["scenarios"]]
        return ModelBench(**data)
    except (OSError, ValueError, KeyError, TypeError):
        return None


def pick_concurrencies(
    est: Estimate | None, max_num_seqs: int, kv_tokens: int | None, tokens_per_req: int, cap: int
) -> list[int]:
    """The estimate's batch sizes, limited by what the server can actually hold at once."""
    limit = min(cap, max_num_seqs)
    if kv_tokens and tokens_per_req:
        limit = min(limit, max(kv_tokens // tokens_per_req, 1))
    batches = [p.batch for p in est.curve] if est else [1, 2, 4, 8, 16, 32]
    picked = sorted({b for b in batches if b <= limit} | {1})
    return picked
