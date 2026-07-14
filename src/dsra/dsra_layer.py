"""DSRA-compatible attention layer backed by MHDSRA2 and paged memory."""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn

from .domain import AttentionLayerSpec
from .dsra_model import select_mhdsra2_heads
from .mhdsra2.improved_dsra_mha import MHDSRA2Config, MHDSRA2State, MultiHeadDSRA2
from .mhdsra2.paged_exact_memory import PagedExactMemory

try:
    from ..config import DSRAArchitectureConfig
except ImportError:
    from config import DSRAArchitectureConfig


class PagedMemoryRepository:
    def __init__(
        self,
        enabled: bool = True,
        dtype: torch.dtype = torch.float32,
        page_size: Optional[int] = None,
        max_pages: Optional[int] = None,
        dsra_arch_config: DSRAArchitectureConfig | None = None,
    ) -> None:
        self.enabled = enabled
        self.memory = PagedExactMemory(
            page_size=page_size,
            dtype=dtype,
            dsra_arch_config=dsra_arch_config,
            max_pages=max_pages,
        )

    def clear(self, *, reset_position: bool = True) -> None:
        self.memory.clear(reset_position=reset_position)

    def append(self, key: torch.Tensor, value: torch.Tensor) -> None:
        if not self.enabled:
            return
        try:
            self.memory.append(key, value)
        except ValueError:
            return

    def retrieve(self, query: torch.Tensor, device: torch.device):
        if not self.enabled or len(self.memory) == 0:
            return None, None
        key, value, _ = self.memory.retrieve(query, device=device)
        return key, value


class StreamingAttentionUnitOfWork:
    def __init__(self, state, kv_cache, time_state, memory_repository) -> None:
        self.state = state
        self.kv_cache = kv_cache
        self.time_state = time_state
        self.memory_repository = memory_repository

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def retrieve(self, query: torch.Tensor, device: torch.device):
        return self.memory_repository.retrieve(query, device)

    def commit_forward(self, state, kv_cache, time_state) -> None:
        self.state = state
        self.kv_cache = kv_cache
        self.time_state = time_state


def apply_rotary_pos_emb(x: torch.Tensor, offset: int = 0) -> torch.Tensor:
    """Apply legacy RoPE rotation for callers that still import this helper.

    中文说明:
    - 调用方 / Called by: legacy tests and external compatibility code
    - 调用对象 / Calls: `torch.arange`, `torch.exp`, `torch.sin`, `torch.cos`
    - 作用 / Purpose: 保留旧 DSRA 模块公开的 RoPE 工具函数，便于渐进迁移
    - 变量 / Variables:
      `x` 是 `[B,T,D]` 输入张量, `offset` 是序列起始位置, `positions/freqs/angles`
      是旋转位置编码中间量
    - 接入 / Integration: 新代码优先通过 MHDSRA2 层接入；旧代码可继续直接调用本函数
    - 错误处理 / Error handling: 奇数维度会抛出 `ValueError`
    - 关键词 / Keywords:
      rope|rotary|position|embedding|legacy|dsra|tensor|compat|attention|旋转编码
    """
    _, token_count, dim = x.shape
    if dim % 2 != 0:
        raise ValueError("Dimension must be even for RoPE")
    positions = torch.arange(
        offset,
        offset + token_count,
        dtype=torch.float32,
        device=x.device,
    ).unsqueeze(1)
    exponent = torch.arange(0, dim, 2, dtype=torch.float32, device=x.device)
    freqs = torch.exp(exponent * -(torch.log(torch.tensor(10000.0, device=x.device)) / dim))
    angles = positions * freqs
    sin = torch.sin(angles).unsqueeze(0)
    cos = torch.cos(angles).unsqueeze(0)
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    x_rot = torch.zeros_like(x)
    x_rot[..., 0::2] = x1 * cos - x2 * sin
    x_rot[..., 1::2] = x1 * sin + x2 * cos
    return x_rot


def get_alibi_mask(
    seq_len_q: int,
    seq_len_k: int,
    is_causal: bool = True,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """Build the legacy ALiBi mask shape expected by old DSRA tests.

    中文说明:
    - 调用方 / Called by: legacy tests and callers using `pe_mode="alibi"`
    - 调用对象 / Calls: `torch.arange`, `torch.Tensor.masked_fill`
    - 作用 / Purpose: 保留旧公开辅助函数，返回 `[1,1,Q,K]` 可广播掩码
    - 变量 / Variables:
      `seq_len_q/seq_len_k` 是 query/key 长度, `is_causal` 控制未来位置屏蔽,
      `q_idx/k_idx/dist/mask` 是位置差与偏置张量
    - 接入 / Integration: 旧脚本可继续导入；核心注意力已迁移到 MHDSRA2
    - 错误处理 / Error handling: 依赖 PyTorch 对非法设备或 dtype 抛出异常
    - 关键词 / Keywords:
      alibi|mask|causal|legacy|attention|bias|broadcast|shape|dsra|位置偏置
    """
    slope = 0.125
    mask_dtype = dtype or torch.float32
    q_idx = torch.arange(seq_len_q, device=device).unsqueeze(1)
    k_idx = torch.arange(seq_len_k, device=device).unsqueeze(0)
    dist = k_idx - q_idx
    mask = dist.to(mask_dtype) * slope
    if is_causal:
        mask = mask.masked_fill(k_idx > q_idx, float("-inf"))
    return mask.unsqueeze(0).unsqueeze(0)


def get_chunk_causal_mask(
    seq_len_q: int,
    seq_len_k: int,
    prefix_len: int = 0,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """Build the legacy chunk causal mask for compatibility callers.

    中文说明:
    - 调用方 / Called by: external legacy code that imported `get_chunk_causal_mask`
    - 调用对象 / Calls: `torch.arange`, `torch.zeros`, `torch.Tensor.masked_fill`
    - 作用 / Purpose: 保留旧 DSRA 分块因果掩码工具，辅助逐步迁移调用方
    - 变量 / Variables:
      `seq_len_q/seq_len_k` 是当前 query/key 长度, `prefix_len` 是历史缓存长度,
      `allowed_k` 是每个 query 可访问的最大 key 位置
    - 接入 / Integration: 新核心层不直接依赖该函数，旧测试或脚本可继续调用
    - 错误处理 / Error handling: 非法设备或 dtype 由 PyTorch 抛出
    - 关键词 / Keywords:
      chunk|causal|mask|prefix|legacy|attention|compat|sequence|dsra|分块掩码
    """
    mask_dtype = dtype or torch.float32
    q_idx = torch.arange(seq_len_q, device=device).unsqueeze(1)
    k_idx = torch.arange(seq_len_k, device=device).unsqueeze(0)
    allowed_k = prefix_len + q_idx
    mask = torch.zeros(seq_len_q, seq_len_k, device=device, dtype=mask_dtype)
    mask = mask.masked_fill(k_idx > allowed_k, float("-inf"))
    return mask.unsqueeze(0)


def _select_heads(dim: int) -> int:
    """Select a conservative MHDSRA2 head count for the compatibility layer.

    中文说明:
    - 调用方 / Called by: `DSRA_Chunk_Layer.__init__`
    - 调用对象 / Calls: 内置 `min`, `max`
    - 作用 / Purpose: 在保持维度可整除的前提下为旧 DSRA 入口选择多头数量
    - 变量 / Variables: `dim` 是隐藏维度, 返回值是可整除的 head 数
    - 接入 / Integration: 旧入口未显式提供 heads，因此由该函数集中选择
    - 错误处理 / Error handling: 找不到更大可整除 head 时退回 `1`
    - 关键词 / Keywords:
      heads|select|compat|mhdsra2|dim|divisible|multihead|adapter|dsra|头数
    """
    return select_mhdsra2_heads(dim)


class _QKVProjectionView(nn.Module):
    """Expose a qkv slice as an `nn.Linear`-like compatibility view.

    中文说明:
    - 调用方 / Called by: `DSRA_Chunk_Layer.__init__`, diagnostic initialization helpers
    - 调用对象 / Calls: `torch.nn.functional.linear`
    - 作用 / Purpose: 让旧代码仍可访问 `W_q/W_v.weight`，同时实际权重存储在 MHDSRA2 `qkv`
    - 变量 / Variables:
      `qkv` 是核心投影层, `start/end` 是当前视图对应的行区间,
      `in_features/out_features` 模拟 `nn.Linear` 属性
    - 接入 / Integration: 旧初始化函数可像操作线性层一样修改该视图权重
    - 错误处理 / Error handling: 维度不一致时由底层线性计算抛出异常
    - 关键词 / Keywords:
      qkv|projection|view|linear|compat|weight|slice|mhdsra2|adapter|投影视图
    """

    def __init__(self, qkv: nn.Linear, start: int, end: int) -> None:
        """Create a projection view over a qkv row slice.

        中文说明:
        - 调用方 / Called by: `DSRA_Chunk_Layer.__init__`
        - 调用对象 / Calls: `nn.Module.__init__`
        - 作用 / Purpose: 记录 qkv 共享权重切片范围
        - 变量 / Variables: `qkv/start/end` 分别是源线性层与切片边界
        - 接入 / Integration: 不单独注册参数，权重由 `qkv` 拥有
        - 错误处理 / Error handling: 非法切片会在访问或前向时暴露
        - 关键词 / Keywords:
          init|qkv|projection|view|linear|slice|compat|weight|module|初始化
        """
        super().__init__()
        self.qkv = qkv
        self.start = start
        self.end = end
        self.in_features = qkv.in_features
        self.out_features = end - start
        self.bias = None

    @property
    def weight(self) -> torch.Tensor:
        """Return the shared qkv weight slice.

        中文说明:
        - 调用方 / Called by: diagnostic helpers and `forward`
        - 调用对象 / Calls: Tensor slicing
        - 作用 / Purpose: 暴露可原地修改的 qkv 权重视图
        - 变量 / Variables: `start/end` 是当前投影视图的行边界
        - 接入 / Integration: 可执行 `layer.W_q.weight.copy_(...)` 修改核心权重
        - 错误处理 / Error handling: 切片越界由 PyTorch 暴露
        - 关键词 / Keywords:
          weight|qkv|slice|view|projection|compat|linear|tensor|mhdsra2|权重
        """
        return self.qkv.weight[self.start : self.end]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project input with the shared qkv slice.

        中文说明:
        - 调用方 / Called by: legacy code that invokes `W_q(x)` or `W_v(x)`
        - 调用对象 / Calls: `torch.nn.functional.linear`
        - 作用 / Purpose: 提供旧线性层调用语义
        - 变量 / Variables: `x` 是输入张量，`weight` 是共享 qkv 切片
        - 接入 / Integration: 仅用于兼容；核心前向直接调用 MHDSRA2
        - 错误处理 / Error handling: 形状不匹配时由 PyTorch 抛出
        - 关键词 / Keywords:
          forward|projection|linear|qkv|slice|compat|tensor|weight|mhdsra2|前向
        """
        return torch.nn.functional.linear(x, self.weight, self.bias)


class DSRA_Chunk_Layer(nn.Module):
    """Legacy DSRA layer API implemented with MHDSRA2 and paged recall."""

    def __init__(
        self,
        dim: int,
        K: int = None,
        kr: int = None,
        eta: float = None,
        decay_lambda: float = None,
        use_orthogonal_update: bool = True,
        use_bypass: bool = True,
        pe_mode: str = "none",
        dsra_arch_config: DSRAArchitectureConfig | None = None,
    ) -> None:
        """Create a DSRA-compatible layer backed by `MultiHeadDSRA2`.

        中文说明:
        - 调用方 / Called by: `MultiLayerDSRAModel`, scripts and legacy tests
        - 调用对象 / Calls:
          `AttentionLayerSpec`, `MHDSRA2Config`, `MultiHeadDSRA2`, `PagedMemoryRepository`
        - 作用 / Purpose: 保留旧 DSRA 构造参数，同时把实际注意力机制替换为 MHDSRA2
        - 变量 / Variables:
          `dim/K/kr` 控制维度与槽位, `eta/decay_lambda` 映射到 MHDSRA2 更新与遗忘参数,
          `use_bypass` 控制 local 分支, `pe_mode` 保留旧兼容字段
        - 接入 / Integration: 旧代码继续实例化 `DSRA_Chunk_Layer` 即可使用新机制
        - 错误处理 / Error handling: 非法领域规格或 MHDSRA2 配置会抛出 `ValueError`
        - 关键词 / Keywords:
          dsra|mhdsra2|compat|layer|paged_memory|slots|local|forward|adapter|替换
        """
        super().__init__()
        if K is None:
            K = dsra_arch_config.compat_K if dsra_arch_config else 512
        if kr is None:
            kr = dsra_arch_config.compat_kr if dsra_arch_config else 16
        if eta is None:
            eta = dsra_arch_config.compat_eta if dsra_arch_config else 0.1
        if decay_lambda is None:
            decay_lambda = dsra_arch_config.compat_decay_lambda if dsra_arch_config else 0.01
        time_decay_alpha = dsra_arch_config.compat_time_decay_alpha if dsra_arch_config else 0.01
        confidence_init = dsra_arch_config.init_confidence if dsra_arch_config else 0.01
        page_size = dsra_arch_config.paged_memory_page_size if dsra_arch_config else 1024
        max_pages = dsra_arch_config.paged_memory_max_pages if dsra_arch_config else None
        self.dim = dim
        self.K = K
        self.kr = kr
        self.eta = eta
        self.decay_lambda = decay_lambda
        self.use_orthogonal_update = use_orthogonal_update
        self.use_bypass = use_bypass
        self.pe_mode = pe_mode
        self.time_decay_alpha = time_decay_alpha
        self.read_temperature = nn.Parameter(torch.tensor(1.0))

        heads = select_mhdsra2_heads(dim)
        self.spec = AttentionLayerSpec(
            dim=dim,
            slots=K,
            read_topk=max(1, min(kr, K)),
            write_topk=max(1, min(kr, K)),
            local_window=max(1, K if use_bypass else 0),
            pe_mode=pe_mode,
        )
        cfg = MHDSRA2Config(
            dim=dim,
            heads=heads,
            slots=K,
            read_topk=self.spec.read_topk,
            write_topk=self.spec.write_topk,
            local_window=self.spec.local_window,
            use_local=use_bypass,
            use_retrieval=dsra_arch_config.use_retrieval if dsra_arch_config else True,
            eta=max(float(eta), 1e-6),
            forget_base=max(float(decay_lambda), 0.0),
            detach_state=bool(getattr(dsra_arch_config, "detach_state", True)),
        )
        self.core = MultiHeadDSRA2(cfg)
        self.S_init = nn.Parameter(torch.randn(K, dim) / (dim**0.5))
        self.W_q = _QKVProjectionView(self.core.qkv, 0, dim)
        self.W_v = _QKVProjectionView(self.core.qkv, 2 * dim, 3 * dim)
        self.W_n = nn.Linear(dim + 1, K)
        self.W_m = nn.Linear(dim, 1)
        self.memory_repository = PagedMemoryRepository(
            enabled=True,
            dtype=torch.float32,
            page_size=page_size,
            max_pages=max_pages,
            dsra_arch_config=dsra_arch_config,
        )
        self.last_V_orth = torch.empty(0)
        self._confidence_init = confidence_init

    def reset_memory(self):
        """Reset the paged memory repository, clearing cross-sample data."""
        self.memory_repository.clear(reset_position=True)

    def sparse_topk_distribution(self, logits: torch.Tensor) -> torch.Tensor:
        """Return the legacy sparse top-k distribution helper.

        中文说明:
        - 调用方 / Called by: legacy tests and external callers
        - 调用对象 / Calls: `torch.topk`, `torch.nn.functional.softmax`, `Tensor.scatter`
        - 作用 / Purpose: 保留旧公开工具函数，便于验证稀疏路由概率
        - 变量 / Variables:
          `logits` 是路由分数, `topk_logits/topk_idx` 是候选分数与索引,
          `topk_probs` 是归一化概率
        - 接入 / Integration: 新核心层内部有自己的 top-k gather/scatter，本函数仅兼容旧入口
        - 错误处理 / Error handling: 非法张量维度由 PyTorch 抛出
        - 关键词 / Keywords:
          sparse|topk|distribution|softmax|scatter|legacy|route|probability|dsra|稀疏
        """
        topk = min(self.kr, logits.size(-1))
        topk_logits, topk_idx = torch.topk(logits, topk, dim=-1)
        topk_probs = torch.nn.functional.softmax(topk_logits, dim=-1)
        return torch.zeros_like(logits).scatter(-1, topk_idx, topk_probs)

    def _initial_state(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> MHDSRA2State:
        """Build an MHDSRA2 state from the legacy `S_init` parameter.

        中文说明:
        - 调用方 / Called by: `_coerce_state`
        - 调用对象 / Calls: `torch.nn.functional.normalize`, tensor reshape helpers
        - 作用 / Purpose: 将旧 `[K,D]` 初始化槽位转换为 MHDSRA2 `[B,H,K,d]` 状态
        - 变量 / Variables:
          `batch_size/device/dtype` 控制状态批量、设备与精度, `slot_view` 是分头后的槽位
        - 接入 / Integration: 无状态前向或逐 token 解码首次调用时自动使用
        - 错误处理 / Error handling: 维度不可整除已由 MHDSRA2 配置提前校验
        - 关键词 / Keywords:
          initial_state|S_init|mhdsra2|state|slots|heads|compat|batch|device|初始状态
        """
        heads = self.core.heads
        d_head = self.core.d_head
        slot_view = self.S_init.to(device=device, dtype=dtype).view(self.K, heads, d_head)
        slot_view = slot_view.permute(1, 0, 2).contiguous()
        slot_k = torch.nn.functional.normalize(slot_view, dim=-1)
        slot_v = slot_view
        slot_k = slot_k.unsqueeze(0).expand(batch_size, -1, -1, -1).contiguous()
        slot_v = slot_v.unsqueeze(0).expand(batch_size, -1, -1, -1).contiguous()
        zeros = torch.zeros(batch_size, heads, self.K, device=device, dtype=torch.float32)
        confidence = torch.full_like(zeros, self._confidence_init)
        slot_positions = torch.zeros(batch_size, heads, self.K, device=device, dtype=torch.long)
        return MHDSRA2State(slot_k, slot_v, zeros, zeros.clone(), confidence, slot_positions=slot_positions)

    def _coerce_state(
        self,
        state: Optional[MHDSRA2State | torch.Tensor],
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> MHDSRA2State:
        """Convert legacy tensor state or empty state into MHDSRA2 state.

        中文说明:
        - 调用方 / Called by: `forward`, `forward_step`
        - 调用对象 / Calls: `_initial_state`, `torch.nn.functional.normalize`
        - 作用 / Purpose: 让旧 `[B,K,D]` 状态和新 `MHDSRA2State` 都能进入统一核心层
        - 变量 / Variables:
          `state` 是旧或新状态, `batch_size/device/dtype` 是目标运行上下文
        - 接入 / Integration: 上层无需区分迁移前后的状态类型
        - 错误处理 / Error handling: 非法 legacy 状态维度抛出 `ValueError`
        - 关键词 / Keywords:
          coerce|state|legacy|tensor|mhdsra2|compat|slots|heads|forward|状态转换
        """
        if state is None:
            return self._initial_state(batch_size, device, dtype)
        if isinstance(state, MHDSRA2State):
            return state
        if state.dim() != 3 or state.shape[1] != self.K or state.shape[2] != self.dim:
            raise ValueError(f"legacy state must be [B,{self.K},{self.dim}]")
        heads = self.core.heads
        d_head = self.core.d_head
        slot_view = state.to(device=device, dtype=dtype).view(batch_size, self.K, heads, d_head)
        slot_view = slot_view.permute(0, 2, 1, 3).contiguous()
        zeros = torch.zeros(batch_size, heads, self.K, device=device, dtype=torch.float32)
        confidence = torch.full_like(zeros, self._confidence_init)
        slot_positions = torch.zeros(batch_size, heads, self.K, device=device, dtype=torch.long)
        return MHDSRA2State(
            torch.nn.functional.normalize(slot_view, dim=-1),
            slot_view,
            zeros,
            zeros.clone(),
            confidence,
            slot_positions=slot_positions,
        )

    def _cache_to_heads(
        self, kv_cache: Optional[Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]]
    ) -> Optional[Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]]:
        """Convert legacy `[B,T,D]` local cache to MHDSRA2 head format.

        中文说明:
        - 调用方 / Called by: `forward`, `forward_step`
        - 调用对象 / Calls: `MultiHeadDSRA2._to_heads`
        - 作用 / Purpose: 将旧接口缓存接入新核心层的 `[B,H,T,d]` local cache
        - 变量 / Variables: `kv_cache` 是旧或新格式的 key/value 缓存二元组
        - 接入 / Integration: 上层继续传旧缓存即可，适配层内部转换
        - 错误处理 / Error handling: 不支持的缓存 rank 会抛出 `ValueError`
        - 关键词 / Keywords:
          cache|heads|kv|legacy|mhdsra2|local|compat|forward_step|state|缓存转换
        """
        if kv_cache is None:
            return None
        key_cache, value_cache = kv_cache
        if key_cache is None or value_cache is None:
            return None
        if key_cache.dim() == 4 and value_cache.dim() == 4:
            return key_cache, value_cache
        if key_cache.dim() == 3 and value_cache.dim() == 3:
            return self.core._to_heads(key_cache), self.core._to_heads(value_cache)
        raise ValueError("kv_cache must contain [B,T,D] or [B,H,T,d] tensors")

    def _cache_from_state(
        self, state: MHDSRA2State
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Convert MHDSRA2 local cache back to legacy `[B,T,D]` format.

        中文说明:
        - 调用方 / Called by: `forward`, `forward_step`
        - 调用对象 / Calls: `MultiHeadDSRA2._from_heads`
        - 作用 / Purpose: 保持旧 DSRA 返回值中的 `bypass_kv` 形状兼容
        - 变量 / Variables: `state.local_k/local_v` 是核心层局部 KV 缓存
        - 接入 / Integration: 旧上层逻辑可继续裁剪或传递该缓存
        - 错误处理 / Error handling: 空 local cache 返回 `(None, None)`
        - 关键词 / Keywords:
          cache|from_state|kv|legacy|local|mhdsra2|heads|compat|return|缓存返回
        """
        if state.local_k is None or state.local_v is None:
            return None, None
        return self.core._from_heads(state.local_k), self.core._from_heads(state.local_v)

    def _timestamp_state(
        self,
        state: MHDSRA2State,
        previous_time: Optional[torch.Tensor],
        chunk_idx: int,
        chunk_tokens: int,
    ) -> Optional[torch.Tensor]:
        """Maintain the old timestamps return value from MHDSRA2 age metadata.

        中文说明:
        - 调用方 / Called by: `forward`
        - 调用对象 / Calls: tensor `mean`, `to`
        - 作用 / Purpose: 为 `pe_mode="timestamps"` 的旧调用方保留 `[B,K]` 时间状态
        - 变量 / Variables:
          `state` 是新状态, `previous_time` 是旧时间状态, `chunk_idx/chunk_tokens`
          用于计算当前 chunk 结束位置
        - 接入 / Integration: 仅当旧入口选择 timestamps 时返回非空张量
        - 错误处理 / Error handling: 非 timestamps 模式直接返回 `None`
        - 关键词 / Keywords:
          timestamps|state|age|compat|time|chunk|legacy|mhdsra2|position|时间状态
        """
        if self.pe_mode != "timestamps":
            return None
        current_end = float((chunk_idx + 1) * chunk_tokens)
        age_mean = state.age.mean(dim=1).to(dtype=torch.float32)
        inferred_time = torch.full_like(age_mean, current_end) - age_mean
        if previous_time is None:
            return inferred_time
        return torch.maximum(previous_time.to(device=age_mean.device), inferred_time)

    def _record_update_proxy(self, previous_state: MHDSRA2State, next_state: MHDSRA2State) -> None:
        """Store a legacy `last_V_orth` proxy from MHDSRA2 slot-value delta.

        中文说明:
        - 调用方 / Called by: `forward`, `forward_step`
        - 调用对象 / Calls: tensor transpose, reshape and detach
        - 作用 / Purpose: 旧饱和度脚本读取 `last_V_orth`，迁移后用 slot 更新量作为诊断代理
        - 变量 / Variables:
          `previous_state/next_state` 是前后 MHDSRA2 状态, `delta` 是槽值变化量
        - 接入 / Integration: 仅用于报告和兼容测试，不参与核心计算
        - 错误处理 / Error handling: 张量形状由核心状态保证
        - 关键词 / Keywords:
          last_V_orth|proxy|delta|slot|state|diagnostic|compat|saturation|mhdsra2|更新量
        """
        delta = next_state.slot_v - previous_state.slot_v.to(
            device=next_state.slot_v.device,
            dtype=next_state.slot_v.dtype,
        )
        batch_size, heads, slots, d_head = delta.shape
        self.last_V_orth = (
            delta.permute(0, 2, 1, 3)
            .contiguous()
            .view(batch_size, slots, heads * d_head)
            .detach()
        )

    def forward(
        self,
        x: torch.Tensor,
        S_prev: Optional[MHDSRA2State | torch.Tensor] = None,
        bypass_kv: Optional[Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]] = None,
        S_time_prev: Optional[torch.Tensor] = None,
        chunk_idx: int = 0,
    ) -> Tuple[
        torch.Tensor,
        MHDSRA2State,
        Tuple[Optional[torch.Tensor], Optional[torch.Tensor]],
        Optional[torch.Tensor],
    ]:
        """Run one DSRA-compatible chunk through MHDSRA2 and paged recall.

        中文说明:
        - 调用方 / Called by: model stacks, benchmark scripts, generation prefill wrappers
        - 调用对象 / Calls:
          `StreamingAttentionUnitOfWork`, `MultiHeadDSRA2.forward`,
          `PagedMemoryRepository.append`
        - 作用 / Purpose: 使用 MHDSRA2 的 slot/local/retrieval 三路机制替换旧 DSRA 前向
        - 变量 / Variables:
          `x` 是 `[B,T,D]` chunk, `S_prev` 是旧或新状态, `bypass_kv` 是兼容局部缓存,
          `S_time_prev` 是时间状态, `chunk_idx` 是当前分块序号
        - 接入 / Integration: 保持旧返回四元组，方便上层逐步迁移到新状态类型
        - 错误处理 / Error handling: 维度错误、状态错误和核心层错误全部向上抛出
        - 关键词 / Keywords:
          forward|chunk|mhdsra2|paged_memory|unit_of_work|state|kv_cache|compat|dsra|前向
        """
        batch_size, chunk_tokens, dim = x.shape
        if dim != self.dim:
            raise ValueError(f"expected dim={self.dim}, got {dim}")
        if S_prev is None:
            self.reset_memory()
        state = self._coerce_state(S_prev, batch_size, x.device, x.dtype)
        head_cache = self._cache_to_heads(bypass_kv)
        if head_cache is not None:
            state.local_k, state.local_v = head_cache

        qkv = self.core.qkv(x)
        query, key, value = qkv.chunk(3, dim=-1)
        query_heads = self.core._to_heads(query)
        key_heads = self.core._to_heads(key)
        value_heads = self.core._to_heads(value)

        with StreamingAttentionUnitOfWork(
            state=state,
            kv_cache=bypass_kv,
            time_state=S_time_prev,
            memory_repository=self.memory_repository,
        ) as unit_of_work:
            retrieved_k, retrieved_v = unit_of_work.retrieve(query_heads, x.device)
            out, next_state = self.core(
                x,
                state=unit_of_work.state,
                retrieved_k=retrieved_k,
                retrieved_v=retrieved_v,
                _precomputed_qkv=qkv,
            )
            self._record_update_proxy(state, next_state)
            self.memory_repository.append(key_heads, value_heads)
            next_cache = self._cache_from_state(next_state)
            next_time = self._timestamp_state(next_state, S_time_prev, chunk_idx, chunk_tokens)
            unit_of_work.commit_forward(
                state=next_state,
                kv_cache=next_cache,
                time_state=next_time,
            )
            return out, unit_of_work.state, unit_of_work.kv_cache, unit_of_work.time_state

    def forward_step(
        self,
        x_t: torch.Tensor,
        S_prev: Optional[MHDSRA2State | torch.Tensor],
        kv_cache: Optional[Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]] = None,
    ) -> Tuple[torch.Tensor, MHDSRA2State, Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]]:
        """Run one autoregressive step through the MHDSRA2 compatibility core.

        中文说明:
        - 调用方 / Called by: LLM compatibility wrappers and generation rollout scripts
        - 调用对象 / Calls: `_coerce_state`, `_cache_to_heads`, `MultiHeadDSRA2.forward_step`
        - 作用 / Purpose: 保留旧逐 token 解码 API，同时使用 MHDSRA2 状态和局部窗口
        - 变量 / Variables:
          `x_t` 是 `[B,1,D]` 当前 token, `S_prev` 是旧或新状态,
          `kv_cache` 是旧接口局部缓存
        - 接入 / Integration: 现有 `model.dsra.forward_step(...)` 调用无需改签名
        - 错误处理 / Error handling: 非单 token 输入由核心层抛出 `ValueError`
        - 关键词 / Keywords:
          forward_step|autoregressive|decode|mhdsra2|compat|kv_cache|state|generation|dsra|解码
        """
        if x_t.dim() != 3:
            raise ValueError(f"expected x_t rank=3, got shape={tuple(x_t.shape)}")
        if S_prev is None:
            self.reset_memory()
        state = self._coerce_state(S_prev, x_t.shape[0], x_t.device, x_t.dtype)
        head_cache = self._cache_to_heads(kv_cache)
        if head_cache is not None:
            state.local_k, state.local_v = head_cache
        qkv = self.core.qkv(x_t)
        q, k, v = qkv.chunk(3, dim=-1)
        query = self.core._to_heads(q)
        key_heads = self.core._to_heads(k)
        value_heads = self.core._to_heads(v)
        retrieved_k, retrieved_v = self.memory_repository.retrieve(query, x_t.device)
        out_t, next_state, _ = self.core.forward_step(
            x_t,
            S_prev=state,
            kv_cache=head_cache,
            retrieved_k=retrieved_k,
            retrieved_v=retrieved_v,
            _precomputed_qkv=qkv,
        )
        self.memory_repository.append(key_heads, value_heads)
        self._record_update_proxy(state, next_state)
        return out_t, next_state, self._cache_from_state(next_state)
