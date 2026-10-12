"""Model architecture facts needed for estimates, read from config.json and safetensors headers."""

from __future__ import annotations

import json
import struct
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .config import DEFAULT_RUNTIME_DIR

# Bytes per element for safetensors dtypes.
DTYPE_BYTES = {
    "F64": 8, "I64": 8, "U64": 8,
    "F32": 4, "I32": 4, "U32": 4,
    "F16": 2, "BF16": 2, "I16": 2, "U16": 2,
    "F8_E4M3": 1, "F8_E5M2": 1, "F8_E8M0": 1, "I8": 1, "U8": 1, "BOOL": 1,
    "F4": 0.5, "F6_E2M3": 0.75, "F6_E3M2": 0.75,
}  # fmt: skip


@dataclass
class ModelInfo:
    model_id: str
    architecture: str
    model_type: str
    hidden_size: int
    num_layers: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    intermediate_size: int
    vocab_size: int
    tie_embeddings: bool
    max_position_embeddings: int | None
    torch_dtype: str | None
    quant_method: str | None = None
    # Mixture of experts
    num_experts: int = 0
    experts_per_token: int = 0
    moe_intermediate_size: int = 0
    shared_expert_intermediate_size: int = 0
    # Multi-head latent attention (DeepSeek style)
    kv_lora_rank: int | None = None
    qk_rope_head_dim: int | None = None
    # Per-layer attention kind; sliding layers only keep `sliding_window` tokens.
    layer_types: list[str] | None = None
    sliding_window: int | None = None
    # Weights
    param_count: int = 0
    weight_bytes: int | None = None  # actual checkpoint size, when known
    dtype_counts: dict[str, int] = field(default_factory=dict)
    params_source: str = "estimated"

    @property
    def is_moe(self) -> bool:
        return self.num_experts > 1

    @property
    def is_mla(self) -> bool:
        return self.kv_lora_rank is not None

    @property
    def expert_params_per_layer(self) -> int:
        return 3 * self.hidden_size * self.moe_intermediate_size

    @property
    def active_param_count(self) -> int:
        """Parameters touched per token (differs from total only for MoE)."""
        if not self.is_moe:
            return self.param_count
        idle = (self.num_experts - self.experts_per_token) * self.expert_params_per_layer
        return max(self.param_count - idle * self.num_moe_layers, 0)

    @property
    def num_moe_layers(self) -> int:
        return self.num_layers if self.is_moe else 0

    @property
    def is_prequantized(self) -> bool:
        return self.quant_method is not None


class ModelInfoError(Exception):
    pass


def get_model_info(
    model: str, *, runtime_dir: Path = DEFAULT_RUNTIME_DIR, offline: bool = False
) -> ModelInfo:
    """Resolve a model id (HF repo or local dir) into ModelInfo, caching the result."""
    local = Path(model).expanduser()
    if local.is_dir():
        return _from_local_dir(model, local)

    cache_file = runtime_dir / "cache" / "models" / (model.replace("/", "--") + ".json")
    if cache_file.exists():
        try:
            return ModelInfo(**json.loads(cache_file.read_text()))
        except (TypeError, ValueError):
            pass

    info = _from_hf_cache(model)
    if info is None:
        if offline:
            raise ModelInfoError(f"{model} is not in the local Hugging Face cache")
        info = _from_hub(model)

    cache_file.parent.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(json.dumps(asdict(info), indent=2))
    return info


def parse_config(model_id: str, cfg: dict[str, Any]) -> ModelInfo:
    quant = cfg.get("quantization_config") or {}
    text = cfg.get("text_config") or cfg.get("llm_config") or cfg
    if text is not cfg:
        quant = quant or text.get("quantization_config") or {}

    def g(*keys: str, default: Any = None) -> Any:
        for k in keys:
            if text.get(k) is not None:
                return text[k]
        return default

    hidden = g("hidden_size", "d_model", "n_embd")
    layers = g("num_hidden_layers", "n_layer", "num_layers")
    heads = g("num_attention_heads", "n_head")
    if not (hidden and layers and heads):
        raise ModelInfoError(
            f"{model_id}: config.json lacks hidden_size/num_hidden_layers/num_attention_heads"
        )
    kv_heads = g("num_key_value_heads", "num_kv_heads", "multi_query_group_num", default=heads)
    if g("multi_query", default=False):
        kv_heads = 1
    head_dim = g("head_dim", default=hidden // heads)

    num_experts = g(
        "num_local_experts", "num_experts", "n_routed_experts", "moe_num_experts", default=0
    )
    moe_inter = g("moe_intermediate_size", default=0) if num_experts else 0
    shared_inter = g("shared_expert_intermediate_size", default=0)
    if not shared_inter and g("n_shared_experts"):
        shared_inter = g("n_shared_experts") * (moe_inter or 0)

    info = ModelInfo(
        model_id=model_id,
        architecture=(cfg.get("architectures") or ["?"])[0],
        model_type=cfg.get("model_type", "?"),
        hidden_size=hidden,
        num_layers=layers,
        num_heads=heads,
        num_kv_heads=kv_heads,
        head_dim=head_dim,
        intermediate_size=g("intermediate_size", "ffn_hidden_size", "n_inner", default=4 * hidden),
        vocab_size=g("vocab_size", default=32000),
        tie_embeddings=bool(cfg.get("tie_word_embeddings", text.get("tie_word_embeddings", False))),
        max_position_embeddings=g("max_position_embeddings", "max_sequence_length", "seq_length"),
        torch_dtype=cfg.get("torch_dtype") or cfg.get("dtype") or text.get("torch_dtype"),
        quant_method=quant.get("quant_method"),
        num_experts=num_experts or 0,
        experts_per_token=g("num_experts_per_tok", "moe_topk", "top_k", default=0)
        if num_experts
        else 0,
        moe_intermediate_size=moe_inter or 0,
        shared_expert_intermediate_size=shared_inter or 0,
        kv_lora_rank=g("kv_lora_rank"),
        qk_rope_head_dim=g("qk_rope_head_dim"),
        layer_types=g("layer_types"),
        sliding_window=g("sliding_window") if g("use_sliding_window", default=True) else None,
    )
    info.param_count = estimate_param_count(info)
    return info


def estimate_param_count(m: ModelInfo) -> int:
    """Rough parameter count from the architecture, for when safetensors headers are unavailable."""
    h = m.hidden_size
    attn = h * (m.num_heads * m.head_dim) * 2 + 2 * h * (m.num_kv_heads * m.head_dim)
    if m.is_moe:
        mlp = (
            m.num_experts * m.expert_params_per_layer
            + 3 * h * m.shared_expert_intermediate_size
            + h * m.num_experts
        )
    else:
        mlp = 3 * h * m.intermediate_size
    embed = m.vocab_size * h * (1 if m.tie_embeddings else 2)
    return m.num_layers * (attn + mlp + 2 * h) + embed


def _apply_weights(info: ModelInfo, dtype_counts: dict[str, int], source: str) -> ModelInfo:
    if not dtype_counts:
        return info
    info.dtype_counts = dtype_counts
    info.weight_bytes = int(sum(DTYPE_BYTES.get(dt, 2) * n for dt, n in dtype_counts.items()))
    raw = sum(dtype_counts.values())
    if info.is_prequantized:
        # Packed int4 (GPTQ/AWQ) stores 8 weights per I32; scales inflate F16 counts slightly.
        # The architecture estimate is a better parameter count than the raw element count there.
        packed = any(dt in ("I32", "U32", "U8", "I8") for dt in dtype_counts)
        info.param_count = info.param_count if packed else raw
    else:
        info.param_count = raw
    info.params_source = source
    return info


def _read_safetensors_header(path: Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    with path.open("rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(n))
    for key, t in header.items():
        if key == "__metadata__":
            continue
        numel = 1
        for d in t["shape"]:
            numel *= d
        counts[t["dtype"]] = counts.get(t["dtype"], 0) + numel
    return counts


def _from_local_dir(model_id: str, path: Path) -> ModelInfo:
    cfg_path = path / "config.json"
    if not cfg_path.exists():
        raise ModelInfoError(f"{path} has no config.json")
    info = parse_config(model_id, json.loads(cfg_path.read_text()))
    counts: dict[str, int] = {}
    for st in sorted(path.glob("*.safetensors")):
        for dt, n in _read_safetensors_header(st).items():
            counts[dt] = counts.get(dt, 0) + n
    return _apply_weights(info, counts, "safetensors")


def _from_hf_cache(model_id: str) -> ModelInfo | None:
    try:
        from huggingface_hub import snapshot_download

        snap = Path(snapshot_download(model_id, local_files_only=True))
    except Exception:
        return None
    if not (snap / "config.json").exists():
        return None
    info = parse_config(model_id, json.loads((snap / "config.json").read_text()))
    shards = sorted(snap.glob("*.safetensors"))
    index = snap / "model.safetensors.index.json"
    expected = set(json.loads(index.read_text())["weight_map"].values()) if index.exists() else None
    if shards and (expected is None or expected <= {s.name for s in shards}):
        counts: dict[str, int] = {}
        for st in shards:
            for dt, n in _read_safetensors_header(st).items():
                counts[dt] = counts.get(dt, 0) + n
        return _apply_weights(info, counts, "safetensors")
    return info


def _from_hub(model_id: str) -> ModelInfo:
    try:
        from huggingface_hub import get_safetensors_metadata, hf_hub_download
        from huggingface_hub.utils import disable_progress_bars

        disable_progress_bars()

        cfg = json.loads(Path(hf_hub_download(model_id, "config.json")).read_text())
    except Exception as e:
        raise ModelInfoError(f"cannot fetch config.json for {model_id}: {e}") from e
    info = parse_config(model_id, cfg)
    try:
        meta = get_safetensors_metadata(model_id)
        counts: dict[str, int] = {}
        for fm in meta.files_metadata.values():
            for t in fm.tensors.values():
                numel = 1
                for d in t.shape:
                    numel *= d
                counts[t.dtype] = counts.get(t.dtype, 0) + numel
        return _apply_weights(info, counts, "safetensors")
    except Exception:
        return info
