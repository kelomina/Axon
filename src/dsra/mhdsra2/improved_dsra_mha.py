"""Multi-head DSRA-v2 prototype.

This file implements a streaming, chunk-wise attention module designed for very
long contexts. It keeps only a bounded local KV cache and a fixed number of
per-head global slots on GPU. Optional exact retrieval KV can be supplied by an
external CPU/NVMe memory index.

The implementation is intentionally conservative: Archived DSRA alias / MHDSRA2
removes the archived O(K^3) orthogonal inverse and replaces dense top-k
distributions with gather/scatter top-k operations so the main buffers scale with
O(B * H * C * K) per chunk, not O(B * H * T^2).
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class MHDSRA2Config:
    dim: int
    heads: int = 8
    slots: int = 128
    read_topk: int = 8
    write_topk: int = 4
    local_window: int = 512
    use_local: bool = True
    use_retrieval: bool = True
    tau_init: float = 8.0
    tau_write_init: float = 4.0
    eta: float = 0.25
    max_update: float = 0.50
    forget_base: float = 0.001
    forget_conflict: float = 0.20
    forget_age: float = 0.0002
    usage_decay: float = 0.995
    conf_decay: float = 0.999
    usage_prior: float = 0.25
    retrieval_tau: float = 8.0
    retrieval_query_pooling: str = "mean"
    retrieval_max_tokens: int = 128
    retrieval_attention_topk: Optional[int] = None
    retrieval_neighbor_span: int = 0
    retrieval_neighbor_direction: str = "right"
    retrieval_neighbor_seed_multiplier: int = 1
    retrieval_neighbor_budget_mode: str = "unbounded"
    retrieval_quality_gate_bias: float = 0.0
    retrieval_quality_gate_adapter: bool = False
    age_write_bias: float = 0.02
    conf_read_bias: float = 0.50
    age_read_penalty: float = 0.005
    detach_state: bool = True
    eps: float = 1e-6
    # 训练热路径关闭：last_write_stats 仅在 return_aux=True 时被读取，
    # 训练从不走该分支；开着会每 chunk 白做 ~12 个 reduction kernel（~192/step）。
    compute_write_stats: bool = False
    # CCFM: Context-Conditioned Feature Modulation
    use_context_film: bool = False
    max_contexts: int = 8
    context_film_hidden: Optional[int] = None
    # Momentum-QKV: slow-moving QKV for stable slot reading
    momentum_qkv: bool = False
    momentum_decay: float = 0.9999
    # RoPE position encoding for slot read
    slot_pe: str = "none"  # "none" or "rope"
    write_protection: int = 0  # 写入保护：写入后 N 步内不允许覆盖

    def __post_init__(self) -> None:
        if self.dim % self.heads != 0:
            raise ValueError("dim must be divisible by heads")
        if self.read_topk < 1 or self.write_topk < 1:
            raise ValueError("top-k values must be positive")
        if self.slots < 1:
            raise ValueError("slots must be positive")
        if self.context_film_hidden is not None and self.context_film_hidden < 1:
            raise ValueError("context_film_hidden must be positive or None")
        if self.retrieval_query_pooling not in {"mean", "max_token"}:
            raise ValueError("retrieval_query_pooling must be 'mean' or 'max_token'")
        if int(self.retrieval_max_tokens) < 1:
            raise ValueError("retrieval_max_tokens must be positive")
        self.retrieval_max_tokens = int(self.retrieval_max_tokens)
        if self.retrieval_attention_topk is not None:
            if int(self.retrieval_attention_topk) < 1:
                raise ValueError("retrieval_attention_topk must be positive or None")
            self.retrieval_attention_topk = int(self.retrieval_attention_topk)
        if int(self.retrieval_neighbor_span) < 0:
            raise ValueError("retrieval_neighbor_span must be non-negative")
        self.retrieval_neighbor_span = int(self.retrieval_neighbor_span)
        if self.retrieval_neighbor_direction not in {"right", "left", "both"}:
            raise ValueError("retrieval_neighbor_direction must be 'right', 'left', or 'both'")
        if int(self.retrieval_neighbor_seed_multiplier) < 1:
            raise ValueError("retrieval_neighbor_seed_multiplier must be positive")
        self.retrieval_neighbor_seed_multiplier = int(
            self.retrieval_neighbor_seed_multiplier
        )
        if self.retrieval_neighbor_budget_mode not in {"unbounded", "pair_aware"}:
            raise ValueError(
                "retrieval_neighbor_budget_mode must be 'unbounded' or 'pair_aware'"
            )
        if not isinstance(self.retrieval_quality_gate_adapter, bool):
            raise ValueError("retrieval_quality_gate_adapter must be a bool")


@dataclass
class MHDSRA2State:
    slot_k: torch.Tensor
    slot_v: torch.Tensor
    age: torch.Tensor
    usage: torch.Tensor
    confidence: torch.Tensor
    local_k: Optional[torch.Tensor] = None
    local_v: Optional[torch.Tensor] = None
    position: int = 0
    stage_dominance: Optional[torch.Tensor] = None
    slot_positions: Optional[torch.Tensor] = None  # [B, H, slots] last-write positions for RoPE
    protected_until: Optional[torch.Tensor] = None  # [B, H, slots] position until which slot is protected


class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding for MHDSRA2 slot read.

    Computes cos/sin for each position and applies rotation to queries/slot keys.
    """

    def __init__(self, dim: int, max_len: int = 1000000):
        super().__init__()
        half = dim // 2
        inv_freq = 1.0 / (10000 ** (torch.arange(0, half, dtype=torch.float32) / half))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.max_len = max_len

    def _compute_cis(self, positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (cos, sin) for given positions. positions: [..., 1] with position values."""
        flat_pos = positions.squeeze(-1).float()  # [...]
        freqs = torch.einsum("...i,j->...ij", flat_pos, self.inv_freq)  # [..., half]
        # Flatten the last two dims: interleave cos/sin
        cos = freqs.cos().to(dtype=positions.dtype)
        sin = freqs.sin().to(dtype=positions.dtype)
        # Duplicate to full dim: cos/sin for each pair
        cos = torch.cat([cos, cos], dim=-1)  # [..., dim]
        sin = torch.cat([sin, sin], dim=-1)
        return cos, sin

    @staticmethod
    def rotate_half(x: torch.Tensor) -> torch.Tensor:
        d = x.shape[-1]
        x1, x2 = x[..., : d // 2], x[..., d // 2:]
        return torch.cat((-x2, x1), dim=-1)

    def apply(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        cos, sin = self._compute_cis(positions)
        return x * cos + self.rotate_half(x) * sin


class MultiHeadDSRA2(nn.Module):
    """Streaming multi-head DSRA-v2 layer.

    Forward input shape:  x [B, C, D]
    Forward output shape: y [B, C, D]

    State shapes are bounded by slots K and local_window W. For 2M-token
    sequences, call this module repeatedly on chunks and carry the returned
    state forward.
    """

    def __init__(self, cfg: MHDSRA2Config):
        super().__init__()
        self.cfg = cfg
        self.dim = cfg.dim
        self.heads = cfg.heads
        self.d_head = cfg.dim // cfg.heads
        self.slots = cfg.slots

        self.qkv = nn.Linear(cfg.dim, 3 * cfg.dim, bias=False)
        self.out_proj = nn.Linear(cfg.dim, cfg.dim, bias=False)

        self.slot_k_init = nn.Parameter(
            torch.randn(cfg.heads, cfg.slots, self.d_head) / (self.d_head**0.5)
        )
        self.slot_v_init = nn.Parameter(
            torch.randn(cfg.heads, cfg.slots, self.d_head) / (self.d_head**0.5)
        )

        self.token_write_gate = nn.Linear(self.d_head, 1)
        self.fuse_gate = nn.Linear(self.d_head, 3)
        if cfg.retrieval_quality_gate_adapter:
            self.retrieval_quality_adapter = nn.Linear(4, 1)
            nn.init.zeros_(self.retrieval_quality_adapter.weight)
            nn.init.zeros_(self.retrieval_quality_adapter.bias)
        else:
            self.retrieval_quality_adapter = None

        # CCFM: Context-Conditioned Feature Modulation (per-feature FiLM)
        if cfg.use_context_film:
            self.context_embed = nn.Embedding(cfg.max_contexts, cfg.dim)
            film_hidden = self._resolve_context_film_hidden(cfg)
            # FiLM network outputs 6*dim values: scale_q, bias_q, scale_k, bias_k, scale_v, bias_v
            # Each is a per-feature vector (not scalar), applied as: q_i * (1+tanh(s_i)) + b_i
            self.film_net = nn.Sequential(
                nn.Linear(cfg.dim, film_hidden),
                nn.ReLU(),
                nn.Linear(film_hidden, 6 * cfg.dim),
            )
            self._active_context_id = 0
        else:
            self.context_embed = None
            self.film_net = None

        # Momentum-QKV: slow-moving QKV for stable slot reading
        if cfg.momentum_qkv:
            self.qkv_slow = nn.Linear(cfg.dim, 3 * cfg.dim, bias=False)
            self.qkv_slow.weight.data.copy_(self.qkv.weight.data)
            self.qkv_slow.requires_grad_(False)
        else:
            self.qkv_slow = None

        # RoPE for slot read
        if cfg.slot_pe == "rope":
            self.rotary = RotaryEmbedding(self.d_head)
        else:
            self.rotary = None

        self.log_tau_read = nn.Parameter(torch.log(torch.tensor(float(cfg.tau_init))))
        self.log_tau_write = nn.Parameter(torch.log(torch.tensor(float(cfg.tau_write_init))))
        self._local_mask_cache_key: Optional[tuple[int, int, int, str, int | None, torch.dtype]] = None
        self._local_mask_cache: Optional[torch.Tensor] = None

    @staticmethod
    def _resolve_context_film_hidden(cfg: MHDSRA2Config) -> int:
        """Resolve the CCFM FiLM hidden width from explicit or scaled config.

        中文说明:
        - 调用方 / Called by: `__init__`
        - 调用对象 / Calls: 内置 `min`, `max`
        - 作用 / Purpose: 让 CCFM 的 FiLM 网络宽度随模型维度缩放，避免固定 hidden=8
          成为大维度模型的表达瓶颈；旧 checkpoint 可显式传 `context_film_hidden=8`
        - 参数 / Parameters: `cfg` 是当前 MHDSRA2 配置
        - 返回 / Returns: 正整数 hidden width
        - 错误处理 / Error handling: 非法显式值由 `MHDSRA2Config.__post_init__` 拦截
        - 关键词 / Keywords:
          ccfm|film|hidden|context|scale|mhdsra2|config|modulation|上下文

        English documentation:
        Function name:
            _resolve_context_film_hidden
        Purpose:
            Use an explicit FiLM hidden size when provided, otherwise scale it
            with model dimension using a bounded rule.
        """
        if cfg.context_film_hidden is not None:
            return int(cfg.context_film_hidden)
        return min(max(cfg.dim // 4, 8), 128)

    def set_context(self, context_id: int) -> None:
        """Set active context ID for CCFM modulation. Context IDs range from 0 to max_contexts-1.

        中文说明:
        - 调用方 / Called by: training loop, inference entry point.
        - 调用对象 / Calls: none; only sets an internal integer.
        - 作用 / Purpose: 切换当前层的上下文调制参数。
        - 变量 / Variables: ``context_id`` 是上下文索引。
        - 接入 / Integration: 训练循环在切换任务/阶段时调用本函数。
        - 错误处理 / Error handling: 越界时由 nn.Embedding 抛出 IndexError。
        - 关键词 / Keywords:
          context|film|modulation|ccfm|task_switch|mhdsra2|layer|set|set_context|上下文

        English documentation:
        Function name:
            set_context
        Purpose:
            Set the active context ID for CCFM modulation.
        Called by:
            Training loop and inference entry point.
        Calls:
            None.
        Parameters:
            - context_id: integer context index [0, max_contexts).
        Integration:
            Call before forward pass to switch task/stage conditioning.
        English keywords:
            context, film, modulation, ccfm, task_switch, set_context
        """
        self._active_context_id = int(context_id)

    def update_momentum(self) -> None:
        """EMA update of slow QKV from fast QKV. Call after optimizer.step()."""
        if self.qkv_slow is not None:
            with torch.no_grad():
                decay = self.cfg.momentum_decay
                self.qkv_slow.weight.data.mul_(decay).add_(self.qkv.weight.data, alpha=1 - decay)

    def init_state(self, batch_size: int, device=None, dtype=None) -> MHDSRA2State:
        """Initialize bounded MHDSRA2 state with the default empty-slot state.

        中文说明:
        - 调用方 / Called by: `forward`, `forward_step`
        - 调用对象 / Calls: `F.normalize`, `torch.zeros`, `torch.full_like`
        - 作用 / Purpose: 创建初始 slot 状态，保持既有 confidence=0.5 默认口径，
          并在启用 write_protection 时初始化写入保护时间表
        - 变量 / Variables:
          `k/v` 是初始 slot 键值, `age/usage` 为零, `confidence=0.5` 沿用原始基线
        - 接入 / Integration: 每次新序列处理开始时自动调用
        - 错误处理 / Error handling: 维度/设备参数错误由 PyTorch 抛出
        """
        cfg = self.cfg
        k = F.normalize(self.slot_k_init, dim=-1).unsqueeze(0).expand(batch_size, -1, -1, -1)
        v = self.slot_v_init.unsqueeze(0).expand(batch_size, -1, -1, -1)
        k = k.to(device=device, dtype=dtype)
        v = v.to(device=device, dtype=dtype)
        zeros = torch.zeros(batch_size, cfg.heads, cfg.slots, device=device, dtype=torch.float32)
        conf = torch.full_like(zeros, 0.5)
        slot_positions = torch.zeros(batch_size, cfg.heads, cfg.slots, device=device, dtype=torch.long) if cfg.slot_pe == "rope" else None
        protected_until = None
        if cfg.write_protection > 0:
            protected_until = torch.full(
                (batch_size, cfg.heads, cfg.slots),
                -cfg.write_protection,
                dtype=torch.long,
                device=device,
            )
        return MHDSRA2State(
            k.contiguous(), v.contiguous(), zeros, zeros.clone(), conf,
            stage_dominance=None,
            slot_positions=slot_positions,
            protected_until=protected_until,
        )

    def _to_heads(self, x: torch.Tensor) -> torch.Tensor:
        b, t, _ = x.shape
        return x.view(b, t, self.heads, self.d_head).transpose(1, 2).contiguous()

    def _from_heads(self, x: torch.Tensor) -> torch.Tensor:
        b, h, t, d = x.shape
        return x.transpose(1, 2).contiguous().view(b, t, h * d)

    @staticmethod
    def _gather_slots(slots: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        b, h, k, d = slots.shape
        t, r = idx.shape[2], idx.shape[3]
        expanded = slots.unsqueeze(2).expand(b, h, t, k, d)
        gather_idx = idx.unsqueeze(-1).expand(b, h, t, r, d)
        return torch.gather(expanded, dim=3, index=gather_idx)

    @staticmethod
    def _scatter_mass(idx: torch.Tensor, weights: torch.Tensor, slots: int) -> torch.Tensor:
        b, h, t, r = idx.shape
        idx_flat = idx.reshape(b * h, t * r, 1)
        src = weights.reshape(b * h, t * r, 1).to(dtype=torch.float32)
        out = torch.zeros(b * h, slots, 1, device=weights.device, dtype=torch.float32)
        out.scatter_add_(1, idx_flat, src)
        return out.view(b, h, slots)

    @staticmethod
    def _scatter_values(
        idx: torch.Tensor, weights: torch.Tensor, values: torch.Tensor, slots: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        b, h, t, r = idx.shape
        d = values.shape[-1]
        idx_flat = idx.reshape(b * h, t * r)
        src = (weights.unsqueeze(-1) * values.unsqueeze(3)).reshape(b * h, t * r, d)
        # Ensure dtype matches target tensor for scatter_add_
        src = src.to(dtype=values.dtype)
        out = torch.zeros(b * h, slots, d, device=values.device, dtype=values.dtype)
        out.scatter_add_(1, idx_flat.unsqueeze(-1).expand(-1, -1, d), src)

        mass_src = weights.reshape(b * h, t * r, 1).to(dtype=torch.float32)
        mass = torch.zeros(b * h, slots, 1, device=values.device, dtype=torch.float32)
        mass.scatter_add_(1, idx_flat.unsqueeze(-1), mass_src)
        return out.view(b, h, slots, d), mass.view(b, h, slots, 1)

    def _causal_prefix_mask(
        self, t_q: int, t_k: int, prefix_len: int, device, dtype
    ) -> torch.Tensor:
        device_index = getattr(device, "index", None)
        cache_key = (int(t_q), int(t_k), int(prefix_len), str(device), device_index)
        if self._local_mask_cache_key == cache_key and self._local_mask_cache is not None:
            return self._local_mask_cache
        q_pos = torch.arange(t_q, device=device).unsqueeze(1) + prefix_len
        k_pos = torch.arange(t_k, device=device).unsqueeze(0)
        # 原始 float(-inf) mask（base 对照用）。
        mask = torch.zeros(t_q, t_k, device=device, dtype=dtype)
        mask = mask.masked_fill(k_pos > q_pos, float("-inf"))
        mask = mask.unsqueeze(0).unsqueeze(0)
        self._local_mask_cache_key = cache_key
        self._local_mask_cache = mask
        return mask

    def _slot_read(
        self, q: torch.Tensor, state: MHDSRA2State
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        cfg = self.cfg
        slot_k = state.slot_k.to(dtype=q.dtype)
        slot_v = state.slot_v.to(dtype=q.dtype)

        # RoPE: apply position-aware rotation to q and slot_k
        if cfg.slot_pe == "rope" and self.rotary is not None and state.slot_positions is not None:
            B, H, T, D = q.shape
            # q positions: global token indices within this chunk
            q_pos = torch.arange(state.position, state.position + T, device=q.device, dtype=torch.float32)
            q_pos = q_pos.view(1, 1, T, 1).expand(B, H, T, 1)  # [B, H, T, 1]
            # k positions: last-write positions for each slot
            k_pos = state.slot_positions.to(device=q.device, dtype=torch.float32).unsqueeze(-1)  # [B, H, S, 1]
            q = self.rotary.apply(q, q_pos)
            slot_k = self.rotary.apply(slot_k, k_pos)
            qn = F.normalize(q, dim=-1)
            sk = F.normalize(slot_k, dim=-1)
        else:
            qn = F.normalize(q, dim=-1)
            sk = F.normalize(slot_k, dim=-1)

        tau = self.log_tau_read.exp().to(dtype=q.dtype)
        logits = torch.einsum("bhtd,bhkd->bhtk", qn, sk) * tau
        logits = logits + cfg.conf_read_bias * state.confidence.to(dtype=q.dtype).unsqueeze(2)
        logits = logits - cfg.age_read_penalty * torch.log1p(state.age).to(dtype=q.dtype).unsqueeze(2)

        r = min(cfg.read_topk, cfg.slots)
        top_logits, top_idx = torch.topk(logits, r, dim=-1)
        probs = F.softmax(top_logits, dim=-1)
        selected_v = self._gather_slots(slot_v, top_idx)
        out = (probs.unsqueeze(-1) * selected_v).sum(dim=3)

        read_mass = self._scatter_mass(top_idx, probs, cfg.slots)
        aux = {
            "read_idx": top_idx,
            "read_probs": probs,
            "read_mass": read_mass,
            "read_logits_top": top_logits,
        }
        return out, aux

    def _local_attention(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, state: MHDSRA2State
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        cfg = self.cfg
        if not cfg.use_local or cfg.local_window <= 0:
            return torch.zeros_like(q), None, None

        if state.local_k is not None and state.local_v is not None:
            prev_k = state.local_k.to(device=q.device, dtype=q.dtype)
            prev_v = state.local_v.to(device=q.device, dtype=q.dtype)
            if prev_k.shape[2] > cfg.local_window:
                prev_k = prev_k[:, :, -cfg.local_window :, :]
                prev_v = prev_v[:, :, -cfg.local_window :, :]
            k_cat = torch.cat([prev_k, k], dim=2)
            v_cat = torch.cat([prev_v, v], dim=2)
            prefix = prev_k.shape[2]
        else:
            k_cat = k
            v_cat = v
            prefix = 0

        mask = self._causal_prefix_mask(q.shape[2], k_cat.shape[2], prefix, q.device, q.dtype)
        out = F.scaled_dot_product_attention(q, k_cat, v_cat, attn_mask=mask, is_causal=False)

        keep = min(cfg.local_window, k_cat.shape[2])
        new_k = k_cat[:, :, -keep:, :]
        new_v = v_cat[:, :, -keep:, :]
        if cfg.detach_state:
            new_k = new_k.detach()
            new_v = new_v.detach()
        return out, new_k.contiguous(), new_v.contiguous()

    def _apply_retrieval_attention_topk(
        self,
        scaled_logits: torch.Tensor,
        valid_view: torch.Tensor,
    ) -> torch.Tensor:
        """Mask retrieval attention logits to the strongest valid candidates.

        中文说明:
        - 调用方 / Called by: `_retrieval_attention`。
        - 调用对象 / Calls: `torch.topk`, `Tensor.scatter_`, `masked_fill`。
        - 作用 / Purpose: 在 retrieval 已经返回较大候选池后，只让每个 query
          的最高分候选参与 softmax，避免 128 个候选共同归一化导致关键 token
          权重被摊薄；默认 `retrieval_attention_topk=None` 时完全保留旧行为。
        - 参数 / Parameters: `scaled_logits` 是温度缩放和 padding mask 之后的
          `[B,H,T,R]` 分数；`valid_view` 是可广播到同形状的 bool 有效候选 mask。
        - 返回 / Returns: 形状不变的 logits；未入选 top-k 或无效候选被置为当前
          dtype 可表示最小值，后续 softmax 后再乘 mask 并重归一化。
        - 错误处理 / Error handling: 非正 top-k 已在 `MHDSRA2Config.__post_init__`
          拒绝；空候选行保持全 mask，由后续归一化路径输出零向量。
        - 副作用 / Side effects: 无，只返回新 logits。
        - 关键词 / Keywords:
          retrieval_attention_topk|softmax_dilution|topk_mask|paged_recall|mhdsra2|注意力稀释

        English documentation:
        Function name:
            _apply_retrieval_attention_topk
        Purpose:
            Keep retrieval recall broad while limiting softmax normalization to
            the strongest valid candidates for each query.
        """
        topk = self.cfg.retrieval_attention_topk
        if topk is None or int(topk) >= scaled_logits.shape[-1]:
            return scaled_logits

        selected_count = min(int(topk), scaled_logits.shape[-1])
        _, top_idx = torch.topk(scaled_logits, selected_count, dim=-1)
        topk_mask = torch.zeros_like(scaled_logits, dtype=torch.bool)
        topk_mask.scatter_(-1, top_idx, True)
        keep_mask = topk_mask & valid_view.to(device=scaled_logits.device, dtype=torch.bool)
        return scaled_logits.masked_fill(
            ~keep_mask,
            torch.finfo(scaled_logits.dtype).min,
        )

    def _retrieval_attention(
        self,
        q: torch.Tensor,
        retrieved_k: Optional[torch.Tensor],
        retrieved_v: Optional[torch.Tensor],
        retrieved_mask: Optional[torch.Tensor] = None,
        return_weights: bool = False,
        return_scores: bool = False,
    ) -> (
        torch.Tensor
        | tuple[torch.Tensor, torch.Tensor | None]
        | tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]
    ):
        """Read exact external K/V memory with sharp score-normalized attention.

        中文说明:
        - 调用方 / Called by: `forward`
        - 调用对象 / Calls: `torch.einsum`, `F.softmax`
        - 作用 / Purpose: 对外部 paged recall 返回的 K/V 执行 retrieval 分支读出；
          使用带温度的 softmax 让最高相似 token 胜出；当显式设置
          `retrieval_attention_topk` 时，softmax 只在最高分有效候选上归一化，
          避免召回候选过多时把 exact match 的权重摊薄
        - 变量 / Variables:
          `q` 为当前 query heads, `retrieved_k/retrieved_v` 为外部记忆召回的键和值,
          `retrieved_mask` 标记 batch padding 后仍有效的召回 token,
          `logits` 为缩放点积分数, `tau` 为 retrieval softmax 温度, `weights` 为 token 权重；
          `return_scores` 仅在训练/诊断显式请求时返回 mask 后 raw score，默认不改变旧 API。
        - 接入 / Integration: 调用 `forward(..., retrieved_k=..., retrieved_v=...)` 时自动启用；
          可通过 `MHDSRA2Config.retrieval_tau` 调整 retrieval 分支锐度，
          通过 `MHDSRA2Config.retrieval_attention_topk` 显式开启 softmax 候选裁剪
        - 错误处理 / Error handling: 关闭 retrieval 或缺少 K/V 时返回零张量；非法 K/V 维度抛出 `ValueError`
        - 关键词 / Keywords:
          retrieval_attention|paged_recall|softmax|retrieval_tau|exact_match|distractor|external_memory|mhdsra2|recall|检索注意力
        """
        cfg = self.cfg
        if (not cfg.use_retrieval) or retrieved_k is None or retrieved_v is None:
            output = torch.zeros_like(q)
            if return_scores:
                return output, None, None
            return (output, None) if return_weights else output
        scale = self.d_head**-0.5
        tau = torch.tensor(float(cfg.retrieval_tau), device=q.device, dtype=q.dtype)
        if retrieved_k.dim() == 4:
            logits = torch.einsum("bhtd,bhrd->bhtr", q, retrieved_k.to(dtype=q.dtype)) * scale
            valid = self._retrieval_valid_mask(retrieved_k, retrieved_mask).to(device=q.device)
            valid_view = valid.view(valid.shape[0], 1, 1, valid.shape[1])
            masked_logits = (tau * logits).masked_fill(
                ~valid_view,
                torch.finfo(logits.dtype).min,
            )
            attention_logits = self._apply_retrieval_attention_topk(masked_logits, valid_view)
            weights = F.softmax(attention_logits, dim=-1)
            weights = weights * valid_view.to(dtype=q.dtype)
            denom = weights.sum(dim=-1, keepdim=True).clamp_min(cfg.eps)
            weights = weights / denom
            output = torch.einsum("bhtr,bhrd->bhtd", weights, retrieved_v.to(dtype=q.dtype))
            if return_scores:
                return output, weights, masked_logits
            return (output, weights) if return_weights else output
        if retrieved_k.dim() == 5:
            logits = torch.einsum("bhtd,bhtrd->bhtr", q, retrieved_k.to(dtype=q.dtype)) * scale
            valid = self._retrieval_valid_mask(retrieved_k, retrieved_mask).to(device=q.device)
            if valid.dim() == 2:
                valid_view = valid.view(valid.shape[0], 1, 1, valid.shape[1])
            else:
                valid_view = valid.unsqueeze(1)
            masked_logits = (tau * logits).masked_fill(
                ~valid_view,
                torch.finfo(logits.dtype).min,
            )
            attention_logits = self._apply_retrieval_attention_topk(masked_logits, valid_view)
            weights = F.softmax(attention_logits, dim=-1)
            weights = weights * valid_view.to(dtype=q.dtype)
            denom = weights.sum(dim=-1, keepdim=True).clamp_min(cfg.eps)
            weights = weights / denom
            output = torch.einsum("bhtr,bhtrd->bhtd", weights, retrieved_v.to(dtype=q.dtype))
            if return_scores:
                return output, weights, masked_logits
            return (output, weights) if return_weights else output
        raise ValueError("retrieved_k/v must be [B,H,R,d] or [B,H,T,R,d]")

    def _slot_write(
        self, k: torch.Tensor, v: torch.Tensor, state: MHDSRA2State, read_mass: torch.Tensor,
        stage_id: int | None = None,
    ) -> MHDSRA2State:
        """Update slot memory and expose overwrite diagnostics for one chunk.

        中文说明:
        - 调用方 / Called by: `forward`
        - 调用对象 / Calls: `F.normalize`, `_scatter_values`, `F.cosine_similarity`, `torch.maximum`
        - 作用 / Purpose: 根据当前 chunk 的 K/V、读出质量和新旧冲突强度更新 slot；
          当 correction token 先读到旧槽位时，用 `read_mass` 强化同槽写回并遗忘旧值。
          当前实现是带门控的 blended update，不是严格正交投影更新。
        - 变量 / Variables:
          `k/v` 是当前 chunk 的 head-space token 表示, `state` 是写入前状态,
          `read_mass` 是本 chunk 对旧 slot 的读出分布, `read_hint` 是写路由的同槽提示,
          `mass/reinforced_mass` 是基础写入质量与读回强化后的写入质量,
          `write_gate/forget` 是逐 slot 写入与遗忘门
        - 接入 / Integration: 调用 `forward(..., return_aux=True)` 后读取 `aux["write_stats"]`
          可获得 overwrite、forget gate 与 read/write mass 诊断数据
        - 错误处理 / Error handling: 依赖上游张量维度与数值稳定性检查；异常直接向上抛出
        - 关键词 / Keywords:
          slot_write|overwrite|correction|latest_wins|forget_gate|write_mass|read_mass|state_update|mhdsra2|覆盖写入
        """
        cfg = self.cfg
        batch_size, heads, seq_len, d_head = k.shape
        slot_k = state.slot_k.to(dtype=k.dtype)
        slot_v = state.slot_v.to(dtype=v.dtype)
        kn = F.normalize(k, dim=-1)
        sk = F.normalize(slot_k, dim=-1)
        sim = torch.einsum("bhtd,bhkd->bhtk", kn, sk)
        max_sim = sim.max(dim=-1).values
        novelty = (1.0 - max_sim).clamp(0.0, 1.0)

        tau = self.log_tau_write.exp().to(dtype=k.dtype)
        read_hint = read_mass.to(dtype=k.dtype).unsqueeze(2).clamp(0.0, 1.0)
        usage_penalty = cfg.usage_prior * torch.log1p(state.usage).to(
            dtype=k.dtype
        ).unsqueeze(2)
        write_logits = sim * tau
        write_logits = write_logits - usage_penalty * (1.0 - read_hint)
        write_logits = write_logits + cfg.age_write_bias * torch.log1p(state.age).to(
            dtype=k.dtype
        ).unsqueeze(2)
        write_logits = write_logits + tau * read_hint

        w_top = min(cfg.write_topk, cfg.slots)
        top_logits, top_idx = torch.topk(write_logits, w_top, dim=-1)
        route = F.softmax(top_logits, dim=-1)
        base_token_gate = torch.sigmoid(self.token_write_gate(k)).squeeze(-1)
        selected_sim = sim.gather(3, top_idx).clamp(0.0, 1.0)
        selected_read_hint = read_hint.expand(-1, -1, seq_len, -1).gather(3, top_idx)
        overwrite_gate = torch.maximum(selected_sim, selected_read_hint)
        write_drive = torch.maximum(novelty.unsqueeze(-1), overwrite_gate)
        token_gate = base_token_gate
        weights = route * token_gate.unsqueeze(-1) * write_drive

        # Apply write protection: prevent writing to slots that are still protected
        if cfg.write_protection > 0 and state.protected_until is not None:
            # For each token in the chunk, calculate its position
            batch_size, heads, seq_len, write_topk = top_idx.shape
            token_positions = torch.arange(state.position, state.position + seq_len, 
                                            device=k.device, dtype=torch.long)
            protected_topk = state.protected_until.unsqueeze(2).expand(
                -1, -1, seq_len, -1
            ).gather(3, top_idx)
            token_pos_topk = token_positions.view(1, 1, seq_len, 1).expand_as(protected_topk)
            selected_protected = (protected_topk > token_pos_topk).to(dtype=k.dtype)
            # Zero out weights for protected slots
            weights = weights * (1.0 - selected_protected)

        agg_k, mass = self._scatter_values(top_idx, weights, k, cfg.slots)
        agg_v, _ = self._scatter_values(top_idx, weights, v, cfg.slots)
        mass_safe = mass.clamp_min(cfg.eps).to(dtype=k.dtype)
        new_k = agg_k / mass_safe
        new_v = agg_v / mass_safe
        has_write = (mass > cfg.eps).to(dtype=k.dtype)

        conflict = (
            (1.0 - F.cosine_similarity(new_k, slot_k, dim=-1, eps=cfg.eps))
            .clamp(0.0, 2.0)
            .unsqueeze(-1)
        )
        read_mass_boost = read_mass.to(dtype=k.dtype).unsqueeze(-1).clamp(0.0, 1.0)
        reinforced_mass = mass + read_mass_boost * has_write
        write_gate = (1.0 - torch.exp(-cfg.eta * reinforced_mass)).to(dtype=k.dtype).clamp(
            0.0, cfg.max_update
        ) * has_write

        age_term = cfg.forget_age * torch.log1p(state.age).unsqueeze(-1).to(dtype=k.dtype)
        forget = cfg.forget_base + age_term + cfg.forget_conflict * write_gate * conflict
        forget = torch.maximum(forget, write_gate * has_write)
        forget = forget.clamp(0.0, 0.95)

        slot_k_next = (1.0 - forget) * slot_k + write_gate * new_k
        slot_k_next = F.normalize(slot_k_next, dim=-1)
        slot_v_next = (1.0 - forget) * slot_v + write_gate * new_v

        wg32 = write_gate.squeeze(-1).to(dtype=torch.float32)
        fg32 = forget.squeeze(-1).to(dtype=torch.float32)
        age_next = (state.age + k.shape[2]).to(dtype=torch.float32)
        age_next = age_next * (1.0 - wg32).clamp(0.0, 1.0)
        usage_next = cfg.usage_decay * state.usage + read_mass + mass.squeeze(-1).to(
            dtype=torch.float32
        )
        conf_new = (1.0 - conflict.squeeze(-1).clamp(0.0, 1.0)).to(dtype=torch.float32)
        conf_next = cfg.conf_decay * state.confidence * (1.0 - fg32) + wg32 * conf_new
        conf_next = conf_next.clamp(0.0, 1.0)

        # RoPE: update slot positions for written slots
        slot_positions_next = state.slot_positions
        if cfg.slot_pe == "rope" and state.slot_positions is not None:
            wrote_mask = (mass.squeeze(-1) > cfg.eps).to(device=k.device)
            new_pos = torch.full_like(state.slot_positions, state.position + k.shape[2], dtype=torch.long)
            slot_positions_next = torch.where(wrote_mask, new_pos, state.slot_positions.to(device=k.device))

        # 方案 3: 写入保护 - 更新 protected_until
        protected_until_next = state.protected_until
        if cfg.write_protection > 0:
            wrote_mask = (mass.squeeze(-1) > cfg.eps).to(device=k.device)
            if protected_until_next is None:
                # 正确初始化 protected_until，不依赖可能为 None 的 slot_positions
                protected_until_next = torch.full(
                    (batch_size, cfg.heads, cfg.slots), 
                    -cfg.write_protection, 
                    dtype=torch.long, 
                    device=k.device
                )
            # 安全地创建 new_protection
            new_protection = torch.full_like(protected_until_next, state.position + k.shape[2] + cfg.write_protection, dtype=torch.long)
            protected_until_next = torch.where(wrote_mask, new_protection, protected_until_next.to(device=k.device))

        write_mass = mass.squeeze(-1).to(dtype=torch.float32)
        # 诊断统计：仅 return_aux 消费者需要。训练热路径关掉（cfg.compute_write_stats=False），
        # 省掉每 chunk ~12 个 detach().mean()/.max() reduction kernel + 张量驻留。
        # 注：cfg 是编译期常量，if 分支会被 Dynamo 折叠，不引入图断点。
        if cfg.compute_write_stats:
            self.last_write_stats = {
                    "token_gate_mean": token_gate.detach().mean().to(dtype=torch.float32),
                    "write_mass_mean": write_mass.detach().mean(),
                    "write_mass_max": write_mass.detach().max(),
                    "write_gate_mean": write_gate.detach().mean().to(dtype=torch.float32),
                    "write_gate_max": write_gate.detach().max().to(dtype=torch.float32),
                    "forget_gate_mean": forget.detach().mean().to(dtype=torch.float32),
                    "forget_gate_max": forget.detach().max().to(dtype=torch.float32),
                    "conflict_mean": conflict.detach().mean().to(dtype=torch.float32),
                    "novelty_mean": novelty.detach().mean().to(dtype=torch.float32),
                    "overwrite_gate_mean": overwrite_gate.detach().mean().to(dtype=torch.float32),
                    "write_drive_mean": write_drive.detach().mean().to(dtype=torch.float32),
                    "write_mass": write_mass.detach(),
                    "write_gate": write_gate.detach().squeeze(-1).to(dtype=torch.float32),
                    "forget_gate": forget.detach().squeeze(-1).to(dtype=torch.float32),
                    "read_mass": read_mass.detach().to(dtype=torch.float32),
                }

        if cfg.detach_state:
            slot_k_next = slot_k_next.detach()
            slot_v_next = slot_v_next.detach()
            age_next = age_next.detach()
            usage_next = usage_next.detach()
            conf_next = conf_next.detach()

        return MHDSRA2State(
            slot_k=slot_k_next.contiguous(),
            slot_v=slot_v_next.contiguous(),
            age=age_next.contiguous(),
            usage=usage_next.contiguous(),
            confidence=conf_next.contiguous(),
            local_k=state.local_k,
            local_v=state.local_v,
            position=state.position + k.shape[2],
            stage_dominance=None,
            slot_positions=slot_positions_next,
            protected_until=protected_until_next,
        )

    def _retrieval_valid_mask(
        self,
        retrieved_k: Optional[torch.Tensor],
        retrieved_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return a batch-aware boolean validity mask for retrieved tokens.

        中文说明:
        - 调用方 / Called by: `_retrieval_attention`, `_forward_from_projected`
        - 调用对象 / Calls: shape inspection and tensor allocation
        - 作用 / Purpose: 把外部分页记忆返回的 mask 统一成可广播格式，防止 batch 对齐
          padding token 进入 retrieval softmax 或 fusion gate
        - 返回 / Returns: `[B,R]` 或 `[B,T,R]` bool mask；无召回时返回空 mask
        - 错误处理 / Error handling: 非法 rank 由调用方维度校验继续暴露
        - 关键词 / Keywords:
          retrieval|mask|padding|batch|softmax|gate|paged_memory|mhdsra2|掩码
        """
        if retrieved_k is None:
            return torch.zeros(0, 0, dtype=torch.bool)
        if retrieved_mask is not None:
            valid = retrieved_mask.to(device=retrieved_k.device, dtype=torch.bool)
            if retrieved_k.dim() == 4 and valid.dim() == 1:
                return valid.unsqueeze(0)
            if retrieved_k.dim() == 5 and valid.dim() == 1:
                return valid.view(1, 1, -1)
            if retrieved_k.dim() == 5 and valid.dim() == 2:
                if valid.shape[0] == retrieved_k.shape[0] and valid.shape[1] == retrieved_k.shape[3]:
                    return valid
                return valid.unsqueeze(0)
            return valid
        if retrieved_k.dim() == 4:
            return torch.ones(
                retrieved_k.shape[0],
                retrieved_k.shape[2],
                device=retrieved_k.device,
                dtype=torch.bool,
            )
        if retrieved_k.dim() == 5:
            return torch.ones(
                retrieved_k.shape[0],
                retrieved_k.shape[2],
                retrieved_k.shape[3],
                device=retrieved_k.device,
                dtype=torch.bool,
            )
        return torch.zeros(0, 0, dtype=torch.bool)

    def _retrieval_token_count(
        self,
        retrieved_k: Optional[torch.Tensor],
        retrieved_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return per-sample retrieved token counts.

        中文说明:
        - 调用方 / Called by: `_forward_from_projected`
        - 调用对象 / Calls: 张量 shape 读取
        - 作用 / Purpose: 为 fusion gate 诊断和可选 retrieval 质量偏置提供 batch-aware 信号
        - 参数 / Parameters: `retrieved_k` 是外部分页记忆返回的 K 张量或 None,
          `retrieved_mask` 是有效 token mask
        - 返回 / Returns: `[B]` float32 tensor；无召回时为长度 0
        - 错误处理 / Error handling: 未知 rank 返回 0，实际 attention 维度校验仍由 `_retrieval_attention` 负责
        - 关键词 / Keywords:
          retrieval|token_count|gate|diagnostic|quality|paged_memory|mhdsra2|召回

        English documentation:
        Function name:
            _retrieval_token_count
        Purpose:
            Provide a small retrieval-availability signal for diagnostics and
            optional gate biasing without changing the retrieval attention API.
        """
        valid = self._retrieval_valid_mask(retrieved_k, retrieved_mask)
        if valid.numel() == 0:
            return torch.zeros(0, dtype=torch.float32)
        if valid.dim() == 2:
            return valid.to(dtype=torch.float32).sum(dim=1)
        if valid.dim() == 3:
            return valid.to(dtype=torch.float32).any(dim=1).sum(dim=1)
        return torch.zeros(0, dtype=torch.float32)

    def _retrieval_quality_features(
        self,
        q: torch.Tensor,
        retrieved_k: Optional[torch.Tensor],
        retrieved_mask: Optional[torch.Tensor],
    ) -> torch.Tensor | None:
        """Build small per-sample quality features for the optional gate adapter.

        中文说明:
        - 调用方 / Called by: `_forward_from_projected`
        - 调用对象 / Calls: `_retrieval_valid_mask`, `F.normalize`, `torch.einsum`
        - 作用 / Purpose: 把 external retrieval 的可用性、最高分、分数差距和召回数量压缩成
          `[B,4]` 特征，供零初始化的轻量 gate adapter 学习何时信任 retrieval 分支。
        - 返回 / Returns: `[B,4]` float tensor；没有有效召回时返回 None。
        - 错误处理 / Error handling: 非法 retrieval rank 继续由 `_retrieval_attention` 暴露。
        - 关键词 / Keywords:
          retrieval|quality|features|gate_adapter|max_score|score_margin|mhdsra2|检索质量

        English documentation:
        Function name:
            _retrieval_quality_features
        Purpose:
            Convert retrieval candidates into compact per-sample quality features
            for an optional zero-initialized retrieval gate adapter.
        """
        if retrieved_k is None:
            return None
        valid = self._retrieval_valid_mask(retrieved_k, retrieved_mask).to(device=q.device)
        if valid.numel() == 0:
            return None
        if valid.shape[-1] == 0:
            return None

        scale = self.d_head**-0.5
        if retrieved_k.dim() == 4:
            logits = torch.einsum("bhtd,bhrd->bhtr", q, retrieved_k.to(dtype=q.dtype)) * scale
            token_scores = logits.mean(dim=(1, 2))
            valid_tokens = valid
        elif retrieved_k.dim() == 5:
            logits = torch.einsum("bhtd,bhtrd->bhtr", q, retrieved_k.to(dtype=q.dtype)) * scale
            token_scores = logits.mean(dim=1).amax(dim=1)
            valid_tokens = valid if valid.dim() == 2 else valid.any(dim=1)
        else:
            return None

        if valid_tokens.dim() != 2:
            return None
        valid_float = valid_tokens.to(dtype=torch.float32)
        valid_counts = valid_float.sum(dim=1)
        available = (valid_counts > 0).to(dtype=torch.float32)
        safe_scores = token_scores.to(dtype=torch.float32).masked_fill(
            ~valid_tokens,
            torch.finfo(torch.float32).min,
        )
        top_values = torch.topk(
            safe_scores,
            k=min(2, safe_scores.shape[1]),
            dim=1,
        ).values
        max_score = torch.where(
            available > 0,
            top_values[:, 0],
            torch.zeros_like(available),
        )
        if top_values.shape[1] > 1:
            second_score = top_values[:, 1]
        else:
            second_score = torch.zeros_like(max_score)
        has_second = valid_counts > 1
        margin = torch.where(
            has_second,
            (max_score - second_score).clamp(min=0.0, max=50.0),
            torch.zeros_like(max_score),
        )
        max_score = torch.where(available > 0, max_score.clamp(min=-50.0, max=50.0), max_score)
        count_feature = torch.log1p(valid_counts)
        return torch.stack((available, max_score, margin, count_feature), dim=1).to(
            device=q.device,
            dtype=q.dtype,
        )

    def _forward_from_projected(
        self,
        x: torch.Tensor,
        q_proj: torch.Tensor,
        k_proj: torch.Tensor,
        v_proj: torch.Tensor,
        *,
        state: Optional[MHDSRA2State] = None,
        retrieved_k: Optional[torch.Tensor] = None,
        retrieved_v: Optional[torch.Tensor] = None,
        retrieved_mask: Optional[torch.Tensor] = None,
        return_aux: bool = False,
        return_projection_aux: bool = False,
        stage_id: int | None = None,
        context_id: int | None = None,
    ):
        """Run forward after the caller has already computed fast Q/K/V.

        中文说明:
        - 调用方 / Called by: `forward`, `DSRA_Chunk_Layer.forward_step`.
        - 调用对象 / Calls: `_to_heads`, `_slot_read`, `_local_attention`,
          `_retrieval_attention`, `_slot_write`.
        - 作用 / Purpose: 复用已计算的 fast Q/K/V，避免逐 token 解码路径重复调用
          `self.qkv(x)`；保持门控融合和 slot 写入逻辑集中在一个实现里。
        - 参数 / Parameters: `q_proj/k_proj/v_proj` 是 `self.qkv(x)` 拆分后的
          token-space 投影；其余参数与 `forward` 一致。
        - 返回 / Returns: 与 `forward` 相同，按 `return_aux` 返回二元组或三元组。
        - 错误处理 / Error handling: 维度错误由上游 `forward` 或 PyTorch 操作抛出。

        English documentation:
        Function name:
            _forward_from_projected
        Purpose:
            Share the projected-QKV execution path between full forward and
            autoregressive compatibility code without changing the public API.
        """
        cfg = self.cfg
        b, t, d = x.shape
        if d != self.dim:
            raise ValueError(f"expected dim={self.dim}, got {d}")

        # CCFM: Update active context if provided
        if context_id is not None and cfg.use_context_film:
            self._active_context_id = int(context_id)
        if state is None:
            state = self.init_state(b, device=x.device, dtype=x.dtype)

        q, k, v = q_proj, k_proj, v_proj

        # CCFM: Context-Conditioned Feature Modulation (per-feature FiLM)
        if cfg.use_context_film and self.film_net is not None:
            ctx = self.context_embed(torch.tensor(self._active_context_id, device=x.device)).to(dtype=x.dtype)
            film = self.film_net(ctx)  # [6 * dim]
            sq, bq, sk, bk, sv, bv = film.chunk(6, dim=-1)
            q = q * (1.0 + torch.tanh(sq)) + bq
            k = k * (1.0 + torch.tanh(sk)) + bk
            v = v * (1.0 + torch.tanh(sv)) + bv

        # Momentum-QKV: slow q for slot read, fast q/k/v for write and local
        q = self._to_heads(q)
        k = self._to_heads(k)
        v = self._to_heads(v)

        if cfg.momentum_qkv and self.qkv_slow is not None:
            q_slow = self._to_heads(self.qkv_slow(x).chunk(3, dim=-1)[0])
            slot_out, aux_read = self._slot_read(q_slow, state)
        else:
            slot_out, aux_read = self._slot_read(q, state)
        local_out, new_local_k, new_local_v = self._local_attention(q, k, v, state)
        retrieval_scores = None
        if return_aux:
            retrieval_out, retrieval_weights, retrieval_scores = self._retrieval_attention(
                q,
                retrieved_k,
                retrieved_v,
                retrieved_mask,
                return_weights=True,
                return_scores=True,
            )
        else:
            retrieval_out = self._retrieval_attention(q, retrieved_k, retrieved_v, retrieved_mask)
            retrieval_weights = None

        gate_logits = self.fuse_gate(q)
        retrieved_counts = self._retrieval_token_count(retrieved_k, retrieved_mask).to(device=q.device)
        retrieval_available_by_sample = (
            cfg.use_retrieval
            and retrieved_v is not None
            and retrieved_counts.numel() > 0
            and bool((retrieved_counts > 0).any().item())
        )
        if cfg.retrieval_quality_gate_bias != 0.0 and retrieval_available_by_sample:
            gate_logits = gate_logits.clone()
            gate_logits[..., 2] = gate_logits[..., 2] + (
                (retrieved_counts > 0)
                .to(device=q.device, dtype=gate_logits.dtype)
                .view(-1, 1, 1)
                * float(cfg.retrieval_quality_gate_bias)
            )
        retrieval_quality_features = self._retrieval_quality_features(q, retrieved_k, retrieved_mask)
        retrieval_quality_delta = None
        if (
            self.retrieval_quality_adapter is not None
            and retrieval_available_by_sample
            and retrieval_quality_features is not None
        ):
            gate_logits = gate_logits.clone()
            retrieval_quality_delta = self.retrieval_quality_adapter(
                retrieval_quality_features
            ).view(-1, 1, 1)
            gate_logits[..., 2] = gate_logits[..., 2] + retrieval_quality_delta.to(
                dtype=gate_logits.dtype
            )
        gates = torch.sigmoid(gate_logits)
        gate_mask = torch.ones_like(gates)
        if not cfg.use_local or cfg.local_window <= 0:
            gate_mask[..., 1] = 0.0
        if not retrieval_available_by_sample:
            gate_mask[..., 2] = 0.0
        else:
            retrieval_sample_mask = (retrieved_counts > 0).to(
                device=q.device,
                dtype=gates.dtype,
            )
            gate_mask[..., 2] = gate_mask[..., 2] * retrieval_sample_mask.view(-1, 1, 1)
        gates = gates * gate_mask
        gates = gates / gates.sum(dim=-1, keepdim=True).clamp_min(cfg.eps)

        y_heads = (
            gates[..., 0:1] * slot_out
            + gates[..., 1:2] * local_out
            + gates[..., 2:3] * retrieval_out
        )
        y = self.out_proj(self._from_heads(y_heads))

        next_state = self._slot_write(
            k, v, state, aux_read["read_mass"], stage_id=stage_id,
        )
        next_state = replace(next_state, local_k=new_local_k, local_v=new_local_v)

        retrieval_score_by_sample = None
        retrieval_score_by_sample_for_loss = None
        retrieval_score_by_token = None
        retrieval_score_by_token_for_loss = None
        if retrieval_scores is not None and retrieved_k is not None:
            score_valid = self._retrieval_valid_mask(retrieved_k, retrieved_mask).to(
                device=q.device,
            )
            if score_valid.dim() == 2:
                score_valid_view = score_valid.view(score_valid.shape[0], 1, 1, score_valid.shape[1])
            elif score_valid.dim() == 3:
                score_valid_view = score_valid.unsqueeze(1)
            else:
                score_valid_view = None
            if score_valid_view is not None:
                safe_scores = torch.where(
                    score_valid_view,
                    retrieval_scores,
                    torch.zeros_like(retrieval_scores),
                )
                retrieval_score_by_sample = safe_scores.detach().mean(dim=(1, 2)).to(
                    dtype=torch.float32,
                )
                retrieval_score_by_sample_for_loss = safe_scores.mean(dim=(1, 2))
                retrieval_score_by_token = safe_scores.detach().mean(dim=1).to(
                    dtype=torch.float32,
                )
                retrieval_score_by_token_for_loss = safe_scores.mean(dim=1)

        if return_aux:
            aux = {
                "gates_mean": gates.detach().mean(dim=(0, 2)),
                "gate_slot_mean": gates[..., 0].detach().mean().to(dtype=torch.float32),
                "gate_local_mean": gates[..., 1].detach().mean().to(dtype=torch.float32),
                "gate_retrieval_mean": gates[..., 2].detach().mean().to(dtype=torch.float32),
                "retrieval_available": retrieval_available_by_sample,
                "retrieval_available_ratio": (
                    (retrieved_counts > 0).detach().to(dtype=torch.float32).mean()
                    if retrieved_counts.numel() > 0
                    else torch.tensor(0.0)
                ),
                "retrieved_token_count": int(retrieved_counts.max().item())
                if retrieved_counts.numel() > 0
                else 0,
                "retrieved_token_count_mean": retrieved_counts.detach().mean().to(dtype=torch.float32)
                if retrieved_counts.numel() > 0
                else torch.tensor(0.0),
                "retrieved_token_count_max": retrieved_counts.detach().max().to(dtype=torch.float32)
                if retrieved_counts.numel() > 0
                else torch.tensor(0.0),
                "retrieval_quality_features": retrieval_quality_features.detach()
                if retrieval_quality_features is not None
                else None,
                "retrieval_quality_adapter_delta": retrieval_quality_delta.detach()
                .squeeze(-1)
                .squeeze(-1)
                .to(dtype=torch.float32)
                if retrieval_quality_delta is not None
                else None,
                "retrieval_token_weight_by_sample": retrieval_weights.detach()
                .mean(dim=(1, 2))
                .to(dtype=torch.float32)
                if retrieval_weights is not None
                else None,
                "retrieval_token_weight_by_sample_for_loss": retrieval_weights.mean(dim=(1, 2))
                if retrieval_weights is not None
                else None,
                "retrieval_token_weight_by_token": retrieval_weights.detach()
                .mean(dim=1)
                .to(dtype=torch.float32)
                if retrieval_weights is not None
                else None,
                "retrieval_token_weight_by_token_for_loss": retrieval_weights.mean(dim=1)
                if retrieval_weights is not None
                else None,
                "retrieval_token_score_by_sample": retrieval_score_by_sample,
                "retrieval_token_score_by_sample_for_loss": retrieval_score_by_sample_for_loss,
                "retrieval_token_score_by_token": retrieval_score_by_token,
                "retrieval_token_score_by_token_for_loss": retrieval_score_by_token_for_loss,
                "gate_retrieval_by_sample": gates[..., 2]
                .detach()
                .mean(dim=(1, 2))
                .to(dtype=torch.float32),
                # Training-only callers can opt into this non-detached view for auxiliary
                # supervision; report writers should continue using the detached metric above.
                "gate_retrieval_by_sample_for_loss": gates[..., 2].mean(dim=(1, 2)),
                "gate_retrieval_by_token": gates[..., 2]
                .detach()
                .mean(dim=1)
                .to(dtype=torch.float32),
                "gate_retrieval_by_token_for_loss": gates[..., 2].mean(dim=1),
                "slot_confidence_mean": next_state.confidence.detach().mean().to(dtype=torch.float32),
                "read_mass": aux_read["read_mass"].detach(),
                "slot_usage": next_state.usage.detach(),
                "slot_confidence": next_state.confidence.detach(),
                "write_stats": getattr(self, "last_write_stats", None),
            }
            if return_projection_aux:
                retrieval_key_projection_by_sample = None
                retrieval_key_projection_by_sample_for_loss = None
                retrieval_key_projection_by_token = None
                retrieval_key_projection_by_token_for_loss = None
                if retrieved_k is not None:
                    if retrieved_k.dim() == 4:
                        retrieval_key_projection_by_sample = retrieved_k.detach()
                        retrieval_key_projection_by_sample_for_loss = retrieved_k.to(dtype=q.dtype)
                    elif retrieved_k.dim() == 5:
                        retrieval_key_projection_by_token = retrieved_k.detach()
                        retrieval_key_projection_by_token_for_loss = retrieved_k.to(dtype=q.dtype)
                aux.update(
                    {
                        "retrieval_query_projection_by_token": q.detach(),
                        "retrieval_query_projection_by_token_for_loss": q,
                        "retrieval_key_projection_by_sample": retrieval_key_projection_by_sample,
                        "retrieval_key_projection_by_sample_for_loss": (
                            retrieval_key_projection_by_sample_for_loss
                        ),
                        "retrieval_key_projection_by_token": retrieval_key_projection_by_token,
                        "retrieval_key_projection_by_token_for_loss": (
                            retrieval_key_projection_by_token_for_loss
                        ),
                    }
                )
            return y, next_state, aux
        return y, next_state

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[MHDSRA2State] = None,
        retrieved_k: Optional[torch.Tensor] = None,
        retrieved_v: Optional[torch.Tensor] = None,
        retrieved_mask: Optional[torch.Tensor] = None,
        return_aux: bool = False,
        return_projection_aux: bool = False,
        stage_id: int | None = None,
        context_id: int | None = None,
    ):
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        return self._forward_from_projected(
            x,
            q,
            k,
            v,
            state=state,
            retrieved_k=retrieved_k,
            retrieved_v=retrieved_v,
            retrieved_mask=retrieved_mask,
            return_aux=return_aux,
            return_projection_aux=return_projection_aux,
            stage_id=stage_id,
            context_id=context_id,
        )

    def forward_step(
        self,
        x_t: torch.Tensor,
        S_prev: Optional[MHDSRA2State] = None,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        retrieved_k: Optional[torch.Tensor] = None,
        retrieved_v: Optional[torch.Tensor] = None,
        retrieved_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, MHDSRA2State, Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]]:
        """Run one autoregressive decoding step with DSRA-compatible outputs.

        中文说明:
        - 调用方 / Called by: `MHDSRA2CompatChunkLayer.forward_step`,
          `scripts.json_retrieval_test.greedy_generate_answer`,
          `scripts.json_retrieval_test.rollout_generation_logits`
        - 调用对象 / Calls: `init_state`, `forward`
        - 作用 / Purpose: 为 MHDSRA2 提供逐 token 解码接口，兼容现有 DSRA generation 评测口径
        - 变量 / Variables:
          `x_t` 当前步输入 `[B, 1, D]`, `S_prev` 上一步状态,
          `kv_cache` 为兼容旧接口传入的局部缓存 `(local_k, local_v)`,
          `retrieved_k/retrieved_v` 为可选外部检索记忆
        - 接入 / Integration: 现有按 `model.dsra.forward_step(...)` 调用的脚本可直接接入 MHDSRA2
        - 错误处理 / Error handling:
          当输入不是三维或步长不为 `1` 时抛出 `ValueError`；缺失状态时自动初始化
        - 关键词 / Keywords:
          forward_step|generation|autoregressive|decode|streaming|kv_cache|state|mhdsra2|compat|逐步解码
        """
        if x_t.dim() != 3:
            raise ValueError(f"expected x_t rank=3, got shape={tuple(x_t.shape)}")
        if x_t.shape[1] != 1:
            raise ValueError(
                f"forward_step expects one token, got sequence length={x_t.shape[1]}"
            )

        state = S_prev
        if state is None:
            state = self.init_state(x_t.shape[0], device=x_t.device, dtype=x_t.dtype)

        if (
            kv_cache is not None
            and state.local_k is None
            and state.local_v is None
        ):
            cached_k, cached_v = kv_cache
            state = replace(state, local_k=cached_k, local_v=cached_v)

        out_t, next_state = self.forward(
            x_t,
            state=state,
            retrieved_k=retrieved_k,
            retrieved_v=retrieved_v,
            retrieved_mask=retrieved_mask,
            return_aux=False,
        )
        next_kv_cache = (next_state.local_k, next_state.local_v)
        return out_t, next_state, next_kv_cache

    def diversity_loss(self, state: MHDSRA2State) -> torch.Tensor:
        """Optional training regularizer; not needed for inference.

        Penalizes slot collapse within each head without using an O(K^3) inverse.
        Complexity is O(B * H * K^2 * d), so use only during training and small K.
        """
        sk = F.normalize(state.slot_k, dim=-1)
        gram = torch.einsum("bhkd,bhld->bhkl", sk, sk)
        eye = torch.eye(gram.shape[-1], device=gram.device, dtype=gram.dtype)
        off = gram - eye.view(1, 1, gram.shape[-1], gram.shape[-1])
        return (off**2).mean()


def estimate_attention_memory_bytes(
    seq_len: int,
    batch_size: int,
    dim: int,
    heads: int,
    chunk_size: int,
    slots: int,
    read_topk: int,
    write_topk: int,
    local_window: int,
    retrieval_tokens: int,
    dtype_bytes: int = 2,
    page_size: int = 1024,
    landmark_dim: Optional[int] = None,
    keep_full_input_output_on_gpu: bool = False,
) -> Dict[str, int]:
    """Estimate GPU memory for a single MHDSRA2 layer attention working set.

    This excludes model parameters and optimizer states. For ultra-long contexts,
    x must be streamed by chunks; if keep_full_input_output_on_gpu=True, the full
    sequence tensor is also counted and can dominate memory.
    """
    d_head = dim // heads
    landmark_dim = landmark_dim or min(64, d_head)
    pages = (seq_len + page_size - 1) // page_size

    mem = {}
    mem["chunk_input"] = batch_size * chunk_size * dim * dtype_bytes
    mem["qkv_chunk"] = 3 * batch_size * heads * chunk_size * d_head * dtype_bytes
    mem["chunk_output"] = batch_size * chunk_size * dim * dtype_bytes
    mem["slots_key_value"] = 2 * batch_size * heads * slots * d_head * dtype_bytes
    mem["slot_metadata_fp32"] = 3 * batch_size * heads * slots * 4
    mem["slot_logits"] = batch_size * heads * chunk_size * slots * dtype_bytes
    mem["topk_read_write"] = batch_size * heads * chunk_size * (read_topk + write_topk) * (
        dtype_bytes + 4
    )
    mem["local_kv_cache"] = 2 * batch_size * heads * local_window * d_head * dtype_bytes
    mem["retrieval_kv"] = 2 * batch_size * heads * retrieval_tokens * d_head * dtype_bytes
    mem["retrieval_logits"] = batch_size * heads * chunk_size * retrieval_tokens * dtype_bytes
    mem["page_landmarks_gpu"] = batch_size * heads * pages * landmark_dim * dtype_bytes
    if keep_full_input_output_on_gpu:
        mem["full_input_output_gpu"] = 2 * batch_size * seq_len * dim * dtype_bytes
    mem["estimated_total"] = sum(mem.values())
    return mem


def format_bytes(n: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    x = float(n)
    for u in units:
        if x < 1024.0 or u == units[-1]:
            return f"{x:.2f} {u}"
        x /= 1024.0
    return f"{x:.2f} TB"
