"""Back-of-the-envelope memory, KV cache and throughput estimates for a serving config.

The memory model mirrors how vLLM sizes its KV cache at startup:

    budget   = total_vram * gpu_memory_utilization
    kv_cache = budget - weights - activation_peak - cuda_graphs - non_torch_overhead

Throughput uses a roofline model: a decode step is bound by reading the weights plus the
KV cache of the running batch from HBM, or by compute, whichever is slower. Expect the
numbers to be within ~20-30% of reality; benchmark before relying on them.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

from .config import ServeConfig
from .hardware import GiB, GpuSpec
from .model_info import ModelInfo

MiB = 1024**2
BLOCK_SIZE = 16

# Efficiency factors applied to peak hardware numbers.
BANDWIDTH_EFFICIENCY = 0.75
DECODE_MFU = 0.45
PREFILL_MFU = 0.55

DTYPE_SIZES = {
    "float32": 4, "float": 4, "fp32": 4,
    "bfloat16": 2, "bf16": 2, "float16": 2, "half": 2, "fp16": 2,
    "fp8": 1, "fp8_e4m3": 1, "fp8_e5m2": 1, "fp8_ds_mla": 1,
}  # fmt: skip


@dataclass
class MemorySegment:
    label: str
    bytes: int
    color: str


@dataclass
class ThroughputPoint:
    batch: int
    step_ms: float
    tokens_per_s: float  # aggregate decode tokens/s
    per_seq_tokens_per_s: float
    bound: str  # "memory", "compute" or "overhead"


@dataclass
class Estimate:
    config_name: str
    model: ModelInfo
    gpu: GpuSpec
    num_gpus: int
    tp: int
    pp: int
    # Resolved settings
    dtype: str
    kv_dtype: str
    gpu_memory_utilization: float
    max_model_len: int
    max_num_seqs: int
    max_num_batched_tokens: int
    enforce_eager: bool
    # Memory, per GPU
    total_bytes: int
    budget_bytes: int
    weight_bytes: int
    activation_bytes: int
    cuda_graph_bytes: int
    overhead_bytes: int
    kv_cache_bytes: int
    kv_bytes_per_token: float
    kv_capacity_tokens: int
    max_concurrency: float  # full-length requests that fit at once
    # Throughput
    avg_prompt_tokens: int
    avg_output_tokens: int
    effective_batch: int
    single_stream_tps: float
    batch_tps: float
    prefill_tps: float
    ttft_ms: float
    requests_per_s: float
    curve: list[ThroughputPoint] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def fits(self) -> bool:
        return not self.errors

    def memory_segments(self) -> list[MemorySegment]:
        used = (
            self.weight_bytes
            + self.activation_bytes
            + self.cuda_graph_bytes
            + self.overhead_bytes
            + max(self.kv_cache_bytes, 0)
        )
        return [
            MemorySegment("weights", self.weight_bytes, "dodger_blue1"),
            MemorySegment("activations", self.activation_bytes, "medium_purple1"),
            MemorySegment("cuda graphs", self.cuda_graph_bytes, "orchid"),
            MemorySegment("runtime overhead", self.overhead_bytes, "grey62"),
            MemorySegment("KV cache", max(self.kv_cache_bytes, 0), "spring_green2"),
            MemorySegment("reserved / free", max(self.total_bytes - used, 0), "grey23"),
        ]


def _vllm_defaults(gpu: GpuSpec) -> tuple[int, int]:
    """vLLM picks larger scheduler limits for online serving on big GPUs."""
    if gpu.memory_bytes >= 70 * GiB:
        return 1024, 8192
    return 256, 2048


def _resolve_dtype(cfg: ServeConfig, model: ModelInfo, gpu: GpuSpec, warnings: list[str]) -> str:
    dtype = str(cfg.arg("dtype", "auto")).lower()
    if dtype == "auto":
        dtype = (model.torch_dtype or "bfloat16").lower()
        if dtype in ("float32", "float"):
            dtype = "bfloat16"
    if dtype in ("bfloat16", "bf16") and not gpu.supports_bf16:
        warnings.append(
            f"{gpu.name} (compute {gpu.compute_capability}) has no bf16 support; vLLM will need dtype=half."
        )
        dtype = "float16"
    return dtype


def _weight_bytes(cfg: ServeConfig, model: ModelInfo, dtype: str, notes: list[str]) -> int:
    quant = cfg.arg("quantization")
    elem = DTYPE_SIZES.get(dtype, 2)
    embed = model.vocab_size * model.hidden_size * (1 if model.tie_embeddings else 2)

    if model.is_prequantized:
        if model.weight_bytes is None:
            notes.append("prequantized checkpoint without size info; assuming ~4.5 bits/weight")
            return int(model.param_count * 0.56)
        return model.weight_bytes
    if quant in ("fp8", "fbgemm_fp8", "ptpc_fp8"):
        notes.append("online fp8 quantization: linear layers at 1 byte/param, embeddings unchanged")
        return int((model.param_count - embed) * 1 + embed * elem)
    if quant in ("bitsandbytes",):
        return int((model.param_count - embed) * 0.56 + embed * elem)
    if quant:
        notes.append(
            f"quantization={quant} without a prequantized checkpoint; assuming unquantized size"
        )
    return int(model.param_count * elem)


def _kv_bytes_per_token(
    model: ModelInfo, kv_elem: float, tp: int, pp: int, max_model_len: int
) -> float:
    layers_per_stage = model.num_layers / pp
    if model.is_mla:
        # Compressed latent + rope key, replicated on every TP rank.
        per_layer = (model.kv_lora_rank + (model.qk_rope_head_dim or 0)) * kv_elem
        return per_layer * layers_per_stage

    kv_heads_per_gpu = max(1, -(-model.num_kv_heads // tp))
    per_layer = 2 * kv_heads_per_gpu * model.head_dim * kv_elem

    if model.layer_types and model.sliding_window:
        full = sum(1 for t in model.layer_types if t != "sliding_attention")
        sliding = len(model.layer_types) - full
        # Sliding-window layers only keep `window` tokens per sequence.
        frac = min(1.0, model.sliding_window / max(max_model_len, 1))
        return per_layer * (full + sliding * frac) / pp
    return per_layer * layers_per_stage


def estimate(
    cfg: ServeConfig,
    model: ModelInfo,
    gpu: GpuSpec,
    num_gpus: int = 1,
    *,
    avg_prompt_tokens: int | None = None,
    avg_output_tokens: int | None = None,
) -> Estimate:
    warnings: list[str] = []
    errors: list[str] = []
    notes: list[str] = []

    tp = int(cfg.arg("tensor-parallel-size", 1))
    pp = int(cfg.arg("pipeline-parallel-size", 1))
    if tp * pp > num_gpus:
        errors.append(f"needs {tp * pp} GPUs (tp={tp}, pp={pp}) but only {num_gpus} available")
    if model.num_heads % tp:
        errors.append(
            f"num_attention_heads={model.num_heads} is not divisible by tensor-parallel-size={tp}"
        )

    dtype = _resolve_dtype(cfg, model, gpu, warnings)
    act_elem = DTYPE_SIZES.get(dtype, 2)
    kv_dtype = str(cfg.arg("kv-cache-dtype", "auto")).lower()
    kv_elem = act_elem if kv_dtype == "auto" else DTYPE_SIZES.get(kv_dtype, 1)

    util = float(cfg.arg("gpu-memory-utilization", 0.9))
    default_seqs, default_batched = _vllm_defaults(gpu)
    max_num_seqs = int(cfg.arg("max-num-seqs", default_seqs))
    native_len = model.max_position_embeddings or 4096
    max_model_len = int(cfg.arg("max-model-len", native_len))
    max_batched = int(cfg.arg("max-num-batched-tokens", max(default_batched, 0)))
    enforce_eager = cfg.has_flag("enforce-eager") or bool(cfg.arg("enforce-eager", False))
    if max_model_len > native_len:
        warnings.append(f"max-model-len {max_model_len} exceeds the model's native {native_len}")

    # ---- Memory, per GPU -------------------------------------------------
    shards = tp * pp
    total = gpu.memory_bytes
    budget = int(total * util)
    weights_total = _weight_bytes(cfg, model, dtype, notes)
    offload = float(cfg.arg("cpu-offload-gb", 0)) * 1e9
    weights = int(max(weights_total - offload, 0) / shards)

    # Peak activations of the profiling forward pass (max_num_batched_tokens tokens) plus
    # fp32 logits for up to max_num_seqs sampled rows.
    h = model.hidden_size
    inter = (
        (model.moe_intermediate_size * max(model.experts_per_token, 1))
        if model.is_moe
        else model.intermediate_size
    )
    q_kv = (model.num_heads + 2 * model.num_kv_heads) * model.head_dim
    per_token_act = (4 * h + (3 * inter + q_kv) / tp) * act_elem
    logits_rows = min(max_num_seqs, max_batched)
    activations = int(max_batched * per_token_act + logits_rows * model.vocab_size / tp * 4 * 2)

    graphs = 0 if enforce_eager else int(min(1 * GiB, 64 * MiB + model.num_layers / pp * 6 * MiB))
    overhead = int(0.35 * GiB + (0.25 * GiB if shards > 1 else 0))

    kv_cache = budget - weights - activations - graphs - overhead
    kv_mem_override = cfg.arg("kv-cache-memory") or cfg.arg("kv-cache-memory-bytes")
    if kv_mem_override:
        kv_cache = int(_parse_bytes(kv_mem_override))
        notes.append("KV cache size fixed by kv-cache-memory")

    kv_per_token = _kv_bytes_per_token(model, kv_elem, tp, pp, max_model_len)
    kv_tokens = (
        int(max(kv_cache, 0) / kv_per_token) // BLOCK_SIZE * BLOCK_SIZE if kv_per_token else 0
    )
    max_concurrency = kv_tokens / max_model_len if max_model_len else 0.0

    if gpu.used_bytes and budget > gpu.free_bytes:
        max_util = (gpu.free_bytes / total) - 0.01
        errors.append(
            f"only {gpu.free_bytes / GiB:.2f} GiB free on {gpu.name} "
            f"({gpu.used_bytes / GiB:.2f} GiB used by other processes) but "
            f"gpu-memory-utilization={util} requests {budget / GiB:.2f} GiB; use <= {max_util:.2f}"
        )
    if weights > budget:
        errors.append(
            f"weights ({weights / GiB:.2f} GiB per GPU) exceed the memory budget ({budget / GiB:.2f} GiB)"
        )
    elif kv_cache <= 0:
        errors.append(
            "no memory left for the KV cache; lower max-num-batched-tokens/max-num-seqs or use a smaller model"
        )
    elif kv_tokens < max_model_len:
        suggested = max(kv_tokens // 1024 * 1024, BLOCK_SIZE)
        errors.append(
            f"KV cache holds {kv_tokens:,} tokens, less than max-model-len {max_model_len:,}; "
            f"set max-model-len <= {suggested:,}"
        )
    elif max_concurrency < 2:
        warnings.append(
            f"only {max_concurrency:.1f} full-length requests fit in the KV cache at once"
        )

    if kv_per_token and kv_tokens and max_num_seqs * BLOCK_SIZE > kv_tokens:
        warnings.append(
            f"max-num-seqs={max_num_seqs} can never be reached; even 1 block each needs more KV than available"
        )

    # ---- Throughput --------------------------------------------------------
    prompt = avg_prompt_tokens or cfg.estimate.avg_prompt_tokens
    output = avg_output_tokens or cfg.estimate.avg_output_tokens
    seq_tokens = prompt + output
    avg_ctx = prompt + output / 2
    effective_batch = max(1, min(max_num_seqs, kv_tokens // seq_tokens if seq_tokens else 1))

    bw = gpu.bandwidth_gbs * 1e9 * BANDWIDTH_EFFICIENCY
    flops = gpu.tflops_16 * 1e12
    active_per_gpu = model.active_param_count / shards
    attn_flops_per_tok = 4 * model.num_layers * model.num_heads * model.head_dim / shards

    def moe_weight_fraction(batch: int) -> float:
        if not model.is_moe:
            return 1.0
        e, k = model.num_experts, max(model.experts_per_token, 1)
        touched = e * (1 - (1 - k / e) ** batch)
        expert_bytes_share = (
            model.expert_params_per_layer * e * model.num_layers / max(model.param_count, 1)
        )
        return (1 - expert_bytes_share) + expert_bytes_share * touched / e

    base_overhead = 0.0025 + (model.num_layers / pp * 0.00012 if enforce_eager else 0.0)
    tp_overhead = 2 * model.num_layers * 25e-6 if tp > 1 else 0.0

    def step(batch: int) -> ThroughputPoint:
        mem_bytes = weights * moe_weight_fraction(batch) + batch * avg_ctx * kv_per_token
        t_mem = mem_bytes / bw
        t_compute = (
            batch * (2 * active_per_gpu + attn_flops_per_tok * avg_ctx) / (flops * DECODE_MFU)
        )
        t = max(t_mem, t_compute) + base_overhead + tp_overhead
        bound = "memory" if t_mem >= t_compute else "compute"
        if base_overhead + tp_overhead > max(t_mem, t_compute):
            bound = "overhead"
        return ThroughputPoint(batch, t * 1e3, batch / t, 1 / t, bound)

    batches = sorted({1, 2, 4, 8, 16, 32, 64, 128, 256, effective_batch})
    curve = [step(b) for b in batches if b <= max(effective_batch, 1)]
    single = step(1)
    at_batch = step(effective_batch)

    prefill_flops_per_tok = 2 * active_per_gpu + attn_flops_per_tok * prompt / 2
    prefill_tps = flops * PREFILL_MFU / prefill_flops_per_tok
    prefill_tps = min(prefill_tps, max_batched / (weights / bw + base_overhead))
    ttft = (prompt / prefill_tps + base_overhead) * 1e3

    cycle = output * at_batch.step_ms / 1e3 + effective_batch * prompt / prefill_tps
    rps = effective_batch / cycle if cycle else 0.0

    if str(cfg.arg("runner", "")).lower() == "pooling":
        notes.append("pooling model: tokens/s figures assume generation; read them as prefill only")
    if gpu.source == "table":
        notes.append(f"hardware figures for {gpu.name} come from the built-in spec table")
    if model.params_source == "estimated":
        notes.append("parameter count estimated from config.json (safetensors headers unavailable)")

    return Estimate(
        config_name=cfg.name,
        model=model,
        gpu=gpu,
        num_gpus=num_gpus,
        tp=tp,
        pp=pp,
        dtype=dtype,
        kv_dtype=kv_dtype if kv_dtype != "auto" else f"auto ({dtype})",
        gpu_memory_utilization=util,
        max_model_len=max_model_len,
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=max_batched,
        enforce_eager=enforce_eager,
        total_bytes=total,
        budget_bytes=budget,
        weight_bytes=weights,
        activation_bytes=activations,
        cuda_graph_bytes=graphs,
        overhead_bytes=overhead,
        kv_cache_bytes=kv_cache,
        kv_bytes_per_token=kv_per_token,
        kv_capacity_tokens=kv_tokens,
        max_concurrency=max_concurrency,
        avg_prompt_tokens=prompt,
        avg_output_tokens=output,
        effective_batch=effective_batch,
        single_stream_tps=single.tokens_per_s,
        batch_tps=at_batch.tokens_per_s,
        prefill_tps=prefill_tps,
        ttft_ms=ttft,
        requests_per_s=rps,
        curve=curve,
        warnings=warnings,
        errors=errors,
        notes=notes,
    )


def _parse_bytes(value) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    s = str(value).strip().upper().removesuffix("B").removesuffix("I")
    mult = {"K": 1024, "M": MiB, "G": GiB, "T": 1024 * GiB}
    if s and s[-1] in mult:
        return float(s[:-1]) * mult[s[-1]]
    return float(s)


def discount_running(gpu: GpuSpec, running: list[ServeConfig]) -> GpuSpec:
    """Give back the VRAM held by configs that are already up, so they don't count as 'used by others'."""
    if not gpu.used_bytes or not running:
        return gpu
    held = sum(int(gpu.memory_bytes * float(c.arg("gpu-memory-utilization", 0.9))) for c in running)
    return replace(gpu, used_bytes=max(gpu.used_bytes - held, 0))


@dataclass
class GroupEstimate:
    name: str
    gpu: GpuSpec  # with running members' memory already discounted
    num_gpus: int
    members: list[tuple[ServeConfig, Estimate | str]]  # str: why the estimate failed
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def fits(self) -> bool:
        return not self.errors and all(isinstance(e, Estimate) and e.fits for _, e in self.members)


def device_set(cfg: ServeConfig) -> str:
    return cfg.resolved_env().get("CUDA_VISIBLE_DEVICES", "default")


def estimate_group(
    name: str,
    members: list[tuple[ServeConfig, Estimate | str]],
    gpu: GpuSpec,
    num_gpus: int = 1,
) -> GroupEstimate:
    """Check that members sharing a GPU fit together.

    Every vLLM instance claims gpu-memory-utilization × total VRAM for itself and refuses
    to start if that much isn't free, so members on the same device must sum below it.
    """
    ge = GroupEstimate(name, gpu, num_gpus, members)
    by_device: dict[str, list[tuple[ServeConfig, Estimate]]] = {}
    for cfg, est in members:
        if isinstance(est, Estimate):
            by_device.setdefault(device_set(cfg), []).append((cfg, est))
    available = gpu.free_bytes if gpu.used_bytes else gpu.memory_bytes
    for devices, items in by_device.items():
        if len(items) < 2:
            continue
        where = "the same GPU" if devices == "default" else f"CUDA_VISIBLE_DEVICES={devices}"
        total_util = sum(e.gpu_memory_utilization for _, e in items)
        requested = sum(e.budget_bytes for _, e in items)
        names = ", ".join(c.short_name for c, _ in items)
        if requested > available:
            ge.errors.append(
                f"{names} share {where} and request {requested / GiB:.2f} GiB together "
                f"(utilization {total_util:.2f}) but only {available / GiB:.2f} GiB is available; "
                f"keep the sum of gpu-memory-utilization <= {available / gpu.memory_bytes - 0.01:.2f}"
            )
        elif total_util > 0.95:
            ge.warnings.append(f"{names} together claim {total_util:.0%} of VRAM on {where}")
    if len(by_device) == 1 and len(members) > 1:
        ge.notes.append(
            "members start one after another; each waits for the previous one to be healthy "
            "so vLLM's memory profiling sees the right free memory"
        )
    return ge
