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

from dataclasses import dataclass
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
    read_tau_max: float = 64.0
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
    age_write_bias: float = 0.02
    conf_read_bias: float = 0.50
    age_read_penalty: float = 0.005
    detach_state: bool = True
    eps: float = 1e-6
    # CCFM: Context-Conditioned Feature Modulation
    use_context_film: bool = False
    max_contexts: int = 8
    # Momentum-QKV: slow-moving QKV for stable slot reading
    momentum_qkv: bool = False
    momentum_decay: float = 0.9999
    # RoPE position encoding for slot read
    slot_pe: str = "rope"  # "none" or "rope" — rope 默认启用，提升长序列位置感知
    hard_write: bool = False  # 使用硬写入路由
    hard_read: bool = False  # 使用硬读取路由
    # 精确模式：完全覆盖写入，不混合
    exact_write: bool = False  # 精确写入模式：完全覆盖 slot
    exact_read: bool = False   # 精确读取模式：只读 top-1 slot
    exact_write_gate: float = 1.0  # 精确写入时的 write_gate 值（默认完全覆盖）
    conflict_protection_coef: float = 0.3  # 冲突保护系数：write_gate * (1.0 - coef * confidence)
    # Write Strategy Parameters (方案 3)
    write_frequency: int = 1  # 写入频率：每 N 个 token 写入一次
    novelty_threshold: float = 0.0  # 写入阈值：novelty > threshold 才写入
    write_protection: int = 0  # 写入保护：写入后 N 步内不允许覆盖
    write_gate_min: float = 0.2  # token_write_gate 下限，防止梯度将写入概率压到过低
    forget_max: float = 0.95  # 遗忘门上限
    write_tau_max: float = 64.0  # 写入tau上限
    max_v_norm: float = 10.0
    write_gate_bias_init: float = 3.0
    write_gate_weight_std: float = 0.01
    init_confidence: float = 0.01
    age_reset_threshold: float = 0.1
    write_mass_threshold_multiplier: float = 10.0

    def __post_init__(self) -> None:
        if self.dim % self.heads != 0:
            raise ValueError("dim must be divisible by heads")
        if self.read_topk < 1 or self.write_topk < 1:
            raise ValueError("top-k values must be positive")
        if self.slots < 1:
            raise ValueError("slots must be positive")


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
    slot_positions: Optional[torch.Tensor] = None
    protected_until: Optional[torch.Tensor] = None

    def __post_init__(self):
        if self.slot_k is not None and self.position != 0:
            assert self.slot_k.device is not None, "state tensors must be on a valid device"

    def detach(self) -> "MHDSRA2State":
        def maybe_detach(value: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
            return value.detach() if value is not None else None

        return MHDSRA2State(
            slot_k=self.slot_k.detach(),
            slot_v=self.slot_v.detach(),
            age=self.age.detach(),
            usage=self.usage.detach(),
            confidence=self.confidence.detach(),
            local_k=maybe_detach(self.local_k),
            local_v=maybe_detach(self.local_v),
            position=self.position,
            stage_dominance=maybe_detach(self.stage_dominance),
            slot_positions=maybe_detach(self.slot_positions),
            protected_until=maybe_detach(self.protected_until),
        )


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
        # 初始化 token_write_gate 的 bias 为较大的正值，提高初始写入概率
        # sigmoid(3.0) ≈ 0.95，确保训练初期 MH 分支能正常写入 slot
        nn.init.constant_(self.token_write_gate.bias, cfg.write_gate_bias_init)
        nn.init.normal_(self.token_write_gate.weight, mean=0.0, std=cfg.write_gate_weight_std)
        self.fuse_gate = nn.Linear(self.d_head, 3)

        # CCFM: Context-Conditioned Feature Modulation (per-feature FiLM)
        if cfg.use_context_film:
            self.context_embed = nn.Embedding(cfg.max_contexts, cfg.dim)
            # FiLM network outputs 6*dim values: scale_q, bias_q, scale_k, bias_k, scale_v, bias_v
            # Each is a per-feature vector (not scalar), applied as: q_i * (1+tanh(s_i)) + b_i
            self.film_net = nn.Sequential(
                nn.Linear(cfg.dim, 8),
                nn.ReLU(),
                nn.Linear(8, 6 * cfg.dim),
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
        """Initialize bounded MHDSRA2 state with low-confidence empty slots.

        中文说明:
        - 调用方 / Called by: `forward`, `forward_step`
        - 调用对象 / Calls: `F.normalize`, `torch.zeros`, `torch.full_like`
        - 作用 / Purpose: 创建初始 slot 状态，使用低置信度 0.1 表示 slot 为空，
          便于后续写入时 confidence 明确区分"空"和"已写"
        - 变量 / Variables:
          `k/v` 是初始 slot 键值, `age/usage` 为零, `confidence=0.1` 表示未写入
        - 接入 / Integration: 每次新序列处理开始时自动调用
        - 错误处理 / Error handling: 维度/设备参数错误由 PyTorch 抛出
        """
        cfg = self.cfg
        k = F.normalize(self.slot_k_init, dim=-1).unsqueeze(0).expand(batch_size, -1, -1, -1)
        v = F.normalize(self.slot_v_init, dim=-1).unsqueeze(0).expand(batch_size, -1, -1, -1)
        k = k.to(device=device, dtype=dtype)
        v = v.to(device=device, dtype=dtype)
        zeros = torch.zeros(batch_size, cfg.heads, cfg.slots, device=device, dtype=torch.float32)
        conf = torch.full_like(zeros, 0.01)
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
        src = src.to(dtype=torch.float32)
        out = torch.zeros(b * h, slots, d, device=values.device, dtype=torch.float32)
        out.scatter_add_(1, idx_flat.unsqueeze(-1).expand(-1, -1, d), src)
        out = out.to(dtype=values.dtype)

        mass_src = weights.reshape(b * h, t * r, 1).to(dtype=torch.float32)
        mass = torch.zeros(b * h, slots, 1, device=values.device, dtype=torch.float32)
        mass.scatter_add_(1, idx_flat.unsqueeze(-1), mass_src)
        return out.view(b, h, slots, d), mass.view(b, h, slots, 1)

    @staticmethod
    def _causal_prefix_mask(
        t_q: int, t_k: int, prefix_len: int, device, dtype
    ) -> torch.Tensor:
        q_pos = torch.arange(t_q, device=device).unsqueeze(1) + prefix_len
        k_pos = torch.arange(t_k, device=device).unsqueeze(0)
        mask = torch.zeros(t_q, t_k, device=device, dtype=dtype)
        mask = mask.masked_fill(k_pos > q_pos, float("-inf"))
        return mask.unsqueeze(0).unsqueeze(0)

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

        tau = self.log_tau_read.exp().float()
        tau = tau.clamp(min=1.0, max=cfg.read_tau_max)
        logits = torch.einsum("bhtd,bhkd->bhtk", qn.float(), sk.float()) * tau
        logits = logits + cfg.conf_read_bias * state.confidence.float().unsqueeze(2)
        logits = logits - cfg.age_read_penalty * torch.log1p(state.age).float().unsqueeze(2)

        r = min(cfg.read_topk, cfg.slots)
        top_logits, top_idx = torch.topk(logits, r, dim=-1)
        probs = F.softmax(top_logits, dim=-1)
        
        # 硬读取路由
        if cfg.hard_read:
            # 选择 top-1 的 slot，使用直通估计器
            hard_probs = torch.zeros_like(probs)
            hard_probs[:, :, :, 0:1] = 1.0  # 第一个位置就是 top-1
            # 直通估计器
            probs = hard_probs + probs - probs.detach()
        
        selected_v = self._gather_slots(slot_v, top_idx)
        
        # 精确读取模式：只使用 top-1 slot
        if cfg.exact_read:
            out = selected_v[:, :, :, 0, :]  # [B, H, T, d_head]
        else:
            out = (probs.unsqueeze(-1) * selected_v).sum(dim=3)
        
        read_mass = self._scatter_mass(top_idx, probs, cfg.slots)
        return out, {
            "read_idx": top_idx,
            "read_probs": probs,
            "read_mass": read_mass,
            "read_logits_top": top_logits,
        }

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

    def _retrieval_attention(
        self,
        q: torch.Tensor,
        retrieved_k: Optional[torch.Tensor],
        retrieved_v: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Read exact external K/V memory with sharp score-normalized attention.

        中文说明:
        - 调用方 / Called by: `forward`
        - 调用对象 / Calls: `torch.einsum`, `F.softmax`
        - 作用 / Purpose: 对外部 paged recall 返回的 K/V 执行 retrieval 分支读出；
          使用带温度的 softmax 让最高相似 token 胜出，避免多个相近 distractor 用数量压过 exact match
        - 变量 / Variables:
          `q` 为当前 query heads, `retrieved_k/retrieved_v` 为外部记忆召回的键和值,
          `logits` 为缩放点积分数, `tau` 为 retrieval softmax 温度, `weights` 为 token 权重
        - 接入 / Integration: 调用 `forward(..., retrieved_k=..., retrieved_v=...)` 时自动启用；
          可通过 `MHDSRA2Config.retrieval_tau` 调整 retrieval 分支锐度
        - 错误处理 / Error handling: 关闭 retrieval 或缺少 K/V 时返回零张量；非法 K/V 维度抛出 `ValueError`
        - 关键词 / Keywords:
          retrieval_attention|paged_recall|softmax|retrieval_tau|exact_match|distractor|external_memory|mhdsra2|recall|检索注意力
        """
        cfg = self.cfg
        if (not cfg.use_retrieval) or retrieved_k is None or retrieved_v is None:
            return torch.zeros_like(q)
        scale = self.d_head**-0.5
        tau = torch.tensor(float(cfg.retrieval_tau), device=q.device, dtype=torch.float32)
        if retrieved_k.dim() == 4:
            logits = torch.einsum("bhtd,bhrd->bhtr", q.float(), retrieved_k.float()) * scale
            weights = F.softmax(tau * logits, dim=-1)
            return torch.einsum("bhtr,bhrd->bhtd", weights, retrieved_v.to(dtype=q.dtype))
        if retrieved_k.dim() == 5:
            logits = torch.einsum("bhtd,bhtrd->bhtr", q.float(), retrieved_k.float()) * scale
            weights = F.softmax(tau * logits, dim=-1)
            return torch.einsum("bhtr,bhtrd->bhtd", weights, retrieved_v.to(dtype=q.dtype))
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
          当 correction token 先读到旧槽位时，用 `read_mass` 强化同槽写回并遗忘旧值
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

        if cfg.slot_pe == "rope" and self.rotary is not None and state.slot_positions is not None:
            B, H, T, D = k.shape
            q_pos = torch.arange(state.position, state.position + T, device=k.device, dtype=torch.float32)
            q_pos = q_pos.view(1, 1, T, 1).expand(B, H, T, 1)
            k_pos = state.slot_positions.to(device=k.device, dtype=torch.float32).unsqueeze(-1)
            kn = F.normalize(self.rotary.apply(k, q_pos), dim=-1)
            sk = F.normalize(self.rotary.apply(slot_k, k_pos), dim=-1)
        else:
            kn = F.normalize(k, dim=-1)
            sk = F.normalize(slot_k, dim=-1)

        sim = torch.einsum("bhtd,bhkd->bhtk", kn, sk)
        max_sim = sim.max(dim=-1).values
        novelty = (1.0 - max_sim).clamp(0.0, 1.0)

        # 方案 3: 写入策略 - 写入频率控制
        # 如果 write_frequency > 1，只在每隔 N 个 token 时写入
        if cfg.write_frequency > 1:
            t = k.shape[2]
            freq_mask = torch.zeros(t, device=k.device, dtype=k.dtype)
            freq_mask[::cfg.write_frequency] = 1.0
            novelty = novelty * freq_mask.view(1, 1, -1)
        
        # 方案 3: 写入策略 - 写入阈值
        # 只有 novelty > threshold 的 token 才能写入
        if cfg.novelty_threshold > 0:
            novelty = novelty * (novelty > cfg.novelty_threshold).to(dtype=novelty.dtype)

        tau = self.log_tau_write.exp().float()
        tau = tau.clamp(min=1.0, max=cfg.write_tau_max)
        read_hint = read_mass.float().unsqueeze(2).clamp(0.0, 1.0)
        usage_penalty = cfg.usage_prior * torch.log1p(state.usage).float().unsqueeze(2)
        write_logits = sim.float() * tau
        write_logits = write_logits - usage_penalty * (1.0 - read_hint)
        write_logits = write_logits + cfg.age_write_bias * torch.log1p(state.age).float().unsqueeze(2)
        write_logits = write_logits + tau * read_hint

        w_top = min(cfg.write_topk, cfg.slots)
        top_logits, top_idx = torch.topk(write_logits, w_top, dim=-1)
        route = F.softmax(top_logits, dim=-1)
        
        # 硬写入路由
        if cfg.hard_write:
            # 选择 top-1 的 slot，使用直通估计器保持梯度
            hard_top_idx = top_idx[:, :, :, 0:1]
            # 创建 one-hot 路由
            hard_route = torch.zeros_like(route)
            hard_route[:, :, :, 0:1] = 1.0  # 第一个位置就是 top-1
            # 直通估计器：前向用 hard_route，反向用 route 的梯度
            route = hard_route + route - route.detach()
        
        raw_gate = torch.sigmoid(self.token_write_gate(k)).squeeze(-1).clamp(min=cfg.write_gate_min)
        token_gate = (raw_gate * novelty).clamp(min=1e-6)
        weights = route * token_gate.unsqueeze(-1)

        # Apply write protection: prevent writing to slots that are still protected
        if cfg.write_protection > 0 and state.protected_until is not None:
            # For each token in the chunk, calculate its position
            batch_size, heads, seq_len, write_topk = top_idx.shape
            token_positions = torch.arange(state.position, state.position + seq_len, 
                                            device=k.device, dtype=torch.long)
            # Shape: (seq_len,)
            
            # Expand protected_until to (batch_size, heads, seq_len, slots)
            protected_expanded = state.protected_until.unsqueeze(2).expand(-1, -1, seq_len, -1)
            # Expand token_positions to (batch_size, heads, seq_len, slots)
            token_pos_expanded = token_positions.view(1, 1, seq_len, 1).expand_as(protected_expanded)
            # Check which slots are protected for each token position
            protected_mask = (protected_expanded > token_pos_expanded).to(dtype=k.dtype)
            # Gather protection status for each selected slot
            selected_protected = protected_mask.gather(3, top_idx)
            # Zero out weights for protected slots
            weights = weights * (1.0 - selected_protected)

        agg_k, mass = self._scatter_values(top_idx, weights, k, cfg.slots)
        agg_v, _ = self._scatter_values(top_idx, weights, v, cfg.slots)
        safe_mask = (mass > cfg.eps).to(dtype=k.dtype)
        mass_safe = mass.clamp_min(cfg.eps).to(dtype=k.dtype)
        new_k = (agg_k / mass_safe) * safe_mask
        new_v = (agg_v / mass_safe) * safe_mask
        has_write = safe_mask

        conflict = (
            (1.0 - F.cosine_similarity(new_k, slot_k, dim=-1, eps=cfg.eps))
            .clamp(0.0, 2.0)
            .unsqueeze(-1)
        )
        read_mass_boost = read_mass.float().unsqueeze(-1).clamp(0.0, 1.0)
        reinforced_mass = mass + read_mass_boost * has_write
        write_gate = (1.0 - torch.exp(-cfg.eta * reinforced_mass.float())).clamp(
            0.0, cfg.max_update
        ).to(dtype=k.dtype) * has_write

        conflict_protection = state.confidence.float().unsqueeze(-1)
        write_gate = write_gate * (1.0 - cfg.conflict_protection_coef * conflict_protection).clamp(min=0.0).to(dtype=k.dtype)

        age_term = cfg.forget_age * torch.log1p(state.age).float().unsqueeze(-1)
        forget = cfg.forget_base + age_term + cfg.forget_conflict * write_gate.float() * conflict
        forget = torch.maximum(forget, write_gate * has_write)
        forget = forget.clamp(0.0, cfg.forget_max)

        # 精确写入模式：完全覆盖
        if cfg.exact_write:
            write_gate_exact = torch.full_like(write_gate, cfg.exact_write_gate) * has_write
            forget_exact = torch.maximum(forget, write_gate_exact)
            slot_k_next = (1.0 - forget_exact) * slot_k + write_gate_exact * new_k
            slot_k_next = F.normalize(slot_k_next, dim=-1)
            slot_v_next = (1.0 - forget_exact) * slot_v + write_gate_exact * new_v
            effective_write_gate = write_gate_exact
            effective_forget = forget_exact
        else:
            slot_k_next = (1.0 - forget) * slot_k + write_gate * new_k
            slot_k_next = F.normalize(slot_k_next, dim=-1)
            slot_v_next = (1.0 - forget) * slot_v + write_gate * new_v
            effective_write_gate = write_gate
            effective_forget = forget

        max_v_norm = cfg.max_v_norm
        v_norm = slot_v_next.norm(dim=-1, keepdim=True)
        scale = torch.clamp(max_v_norm / (v_norm + 1e-6), max=1.0)
        slot_v_next = slot_v_next * scale

        wg32 = effective_write_gate.squeeze(-1).to(dtype=torch.float32)
        fg32 = effective_forget.squeeze(-1).to(dtype=torch.float32)
        age_next = (state.age + k.shape[2]).to(dtype=torch.float32)
        age_reset_mask = (wg32 > cfg.age_reset_threshold).to(dtype=torch.float32)
        age_next = age_next * (1.0 - age_reset_mask)
        usage_next = cfg.usage_decay * state.usage + read_mass + mass.squeeze(-1).to(
            dtype=torch.float32
        )
        conf_new = (1.0 - conflict.squeeze(-1).clamp(0.0, 1.0)).to(dtype=torch.float32)
        conf_next = cfg.conf_decay * state.confidence * (1.0 - fg32) + wg32 * conf_new
        conf_next = conf_next.clamp(0.0, 1.0)

        # RoPE: update slot positions for written slots
        slot_positions_next = state.slot_positions
        if cfg.slot_pe == "rope" and state.slot_positions is not None:
            wrote_mask = (mass.squeeze(-1) > cfg.eps * cfg.write_mass_threshold_multiplier).to(device=k.device)
            new_pos = torch.full_like(state.slot_positions, state.position + k.shape[2], dtype=torch.long)
            slot_positions_next = torch.where(wrote_mask, new_pos, state.slot_positions.to(device=k.device))

        # 方案 3: 写入保护 - 更新 protected_until
        protected_until_next = state.protected_until
        if cfg.write_protection > 0:
            wrote_mask = (mass.squeeze(-1) > cfg.eps * cfg.write_mass_threshold_multiplier).to(device=k.device)
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
        self.last_write_stats = {
            "token_gate_mean": token_gate.detach().mean().to(dtype=torch.float32),
            "write_mass_mean": write_mass.detach().mean(),
            "write_mass_max": write_mass.detach().max(),
            "write_gate_mean": effective_write_gate.detach().mean().to(dtype=torch.float32),
            "write_gate_max": effective_write_gate.detach().max().to(dtype=torch.float32),
            "forget_gate_mean": effective_forget.detach().mean().to(dtype=torch.float32),
            "forget_gate_max": effective_forget.detach().max().to(dtype=torch.float32),
            "conflict_mean": conflict.detach().mean().to(dtype=torch.float32),
            "novelty_mean": novelty.detach().mean().to(dtype=torch.float32),
            "write_mass": write_mass.detach(),
            "write_gate": write_gate.detach().squeeze(-1).to(dtype=torch.float32),
            "forget_gate": forget.detach().squeeze(-1).to(dtype=torch.float32),
            "read_mass": read_mass.detach().to(dtype=torch.float32),
        }

        if cfg.detach_state:
            if getattr(self, "_capture_slot_k_before_detach", False):
                self._slot_k_before_detach = slot_k_next
            elif hasattr(self, "_slot_k_before_detach"):
                del self._slot_k_before_detach
            slot_k_next = slot_k_next.detach()
            slot_v_next = slot_v_next.detach()
            age_next = age_next.detach()
            usage_next = usage_next.detach()
            conf_next = conf_next.detach()
        else:
            if hasattr(self, '_slot_k_before_detach'):
                del self._slot_k_before_detach

        return MHDSRA2State(
            slot_k=slot_k_next.contiguous(),
            slot_v=slot_v_next.contiguous(),
            age=age_next.contiguous(),
            usage=usage_next.contiguous(),
            confidence=conf_next.contiguous(),
            local_k=state.local_k,
            local_v=state.local_v,
            position=min(state.position + k.shape[2], (1 << 24) - 1),
            stage_dominance=None,
            slot_positions=slot_positions_next,
            protected_until=protected_until_next,
        )

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[MHDSRA2State] = None,
        retrieved_k: Optional[torch.Tensor] = None,
        retrieved_v: Optional[torch.Tensor] = None,
        return_aux: bool = False,
        stage_id: int | None = None,
        context_id: int | None = None,
        _precomputed_qkv: Optional[torch.Tensor] = None,
    ):
        cfg = self.cfg
        b, t, d = x.shape
        if d != self.dim:
            raise ValueError(f"expected dim={self.dim}, got {d}")

        if context_id is not None and cfg.use_context_film:
            self._active_context_id = int(context_id)
        if state is None:
            state = self.init_state(b, device=x.device, dtype=x.dtype)
        elif cfg.detach_state:
            state = state.detach()

        if _precomputed_qkv is not None:
            qkv = _precomputed_qkv
        else:
            qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

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
        retrieval_out = self._retrieval_attention(q, retrieved_k, retrieved_v)

        gate_logits = self.fuse_gate(q)
        gates = torch.sigmoid(gate_logits)
        gate_mask = torch.ones_like(gates)
        if not cfg.use_local or cfg.local_window <= 0:
            gate_mask[..., 1] = 0.0
        if (not cfg.use_retrieval) or retrieved_k is None or retrieved_v is None:
            gate_mask[..., 2] = 0.0
        gates = gates * gate_mask
        gates = gates / gates.sum(dim=-1, keepdim=True).clamp_min(cfg.eps)

        y_heads = (
            gates[..., 0:1] * slot_out
            + gates[..., 1:2] * local_out
            + gates[..., 2:3] * retrieval_out
        )
        y = self.out_proj(self._from_heads(y_heads))

        next_state = self._slot_write(k, v, state, aux_read["read_mass"], stage_id=stage_id)
        next_state.local_k = new_local_k
        next_state.local_v = new_local_v

        if return_aux:
            aux = {
                "gates_mean": gates.detach().mean(dim=(0, 2)),
                "read_mass": aux_read["read_mass"].detach(),
                "slot_usage": next_state.usage.detach(),
                "slot_confidence": next_state.confidence.detach(),
                "write_stats": getattr(self, "last_write_stats", None),
            }
            return y, next_state, aux
        return y, next_state

    def forward_step(
        self,
        x_t: torch.Tensor,
        S_prev: Optional[MHDSRA2State] = None,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        retrieved_k: Optional[torch.Tensor] = None,
        retrieved_v: Optional[torch.Tensor] = None,
        _precomputed_qkv: Optional[torch.Tensor] = None,
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
            if self.cfg.detach_state:
                cached_k = cached_k.detach()
                cached_v = cached_v.detach()
            state.local_k = cached_k
            state.local_v = cached_v

        out_t, next_state = self.forward(
            x_t,
            state=state,
            retrieved_k=retrieved_k,
            retrieved_v=retrieved_v,
            return_aux=False,
            _precomputed_qkv=_precomputed_qkv,
        )
        next_kv_cache = (next_state.local_k, next_state.local_v)
        return out_t, next_state, next_kv_cache

    def diversity_loss(self, state: MHDSRA2State) -> torch.Tensor:
        """Optional training regularizer; not needed for inference.

        Penalizes slot collapse within each head without using an O(K^3) inverse.
        Complexity is O(B * H * K^2 * d), so use only during training and small K.
        """
        sk_source = getattr(self, '_slot_k_before_detach', state.slot_k)
        try:
            sk = F.normalize(sk_source, dim=-1)
            gram = torch.einsum("bhkd,bhld->bhkl", sk.float(), sk.float())
            eye = torch.eye(gram.shape[-1], device=gram.device, dtype=gram.dtype)
            off = gram - eye.view(1, 1, gram.shape[-1], gram.shape[-1])
            return (off**2).mean()
        finally:
            if hasattr(self, '_slot_k_before_detach'):
                del self._slot_k_before_detach


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
