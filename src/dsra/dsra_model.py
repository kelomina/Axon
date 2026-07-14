"""Multi-layer token models for the active MHDSRA2 architecture."""

from __future__ import annotations

import torch
import torch.nn as nn

from .domain import normalize_model_type
from .mhdsra2.improved_dsra_mha import MHDSRA2Config, MultiHeadDSRA2

try:
    from ..config import DSRAArchitectureConfig
except ImportError:
    from config import DSRAArchitectureConfig


def select_mhdsra2_heads(dim: int) -> int:
    """Select a valid MHDSRA2 head count for a hidden dimension.

    中文说明:
    - 调用方 / Called by: `MultiLayerMHDSRA2Model.__init__`
    - 调用对象 / Calls: 内置 `range`, `min`, `max`
    - 作用 / Purpose: 为多层 MHDSRA2 模型选择能整除隐藏维度的保守 head 数
    - 变量 / Variables: `dim` 是隐藏维度, `heads` 是候选头数
    - 接入 / Integration: 新模型构建时复用本函数避免重复 head 选择逻辑
    - 错误处理 / Error handling: 找不到更大可整除值时返回 `1`
    - 关键词 / Keywords:
      mhdsra2|heads|select|dim|divisible|model|multi_layer|attention|config|头数
    """
    for heads in range(min(8, max(1, dim // 16)), 0, -1):
        if dim % heads == 0:
            return heads
    return 1


class MultiLayerMHDSRA2Model(nn.Module):
    """Stacked token model backed exclusively by MHDSRA2 layers."""

    def __init__(
        self,
        vocab_size: int,
        dim: int,
        num_layers: int = None,
        K: int = None,
        kr: int = None,
        chunk_size: int = None,
        *,
        use_retrieval: bool = False,
        model_type: str = "mhdsra2",
        mhdsra2_config_override: dict | None = None,
        dsra_arch_config: DSRAArchitectureConfig | None = None,
    ) -> None:
        """Create a stacked MHDSRA2 token model.

        中文说明:
        - 调用方 / Called by: `scripts.needle_in_haystack_test.build_niah_model`,
          legacy `MultiLayerDSRAModel`
        - 调用对象 / Calls: `normalize_model_type`, `select_mhdsra2_heads`,
          `MHDSRA2Config`, `MultiHeadDSRA2`, PyTorch layers
        - 作用 / Purpose: 为长上下文 token 任务提供正式多层 MHDSRA2 架构
        - 变量 / Variables:
          `vocab_size/dim/num_layers` 是模型规模, `K/kr/chunk_size` 是记忆和分块配置,
          `use_retrieval` 控制外部召回分支, `model_type` 记录归一化后的架构名
        - 接入 / Integration: 通过 `build_niah_model(model_type="mhdsra2")` 或兼容别名构造
        - 错误处理 / Error handling: 非法架构名、维度或 MHDSRA2 配置会抛出 `ValueError`
        - 关键词 / Keywords:
          mhdsra2|multilayer|model|token|chunked|streaming|slots|retrieval|logits|模型
        - Note:
           detach_state=True is the memory-safe default (gradient is truncated
           across chunk boundaries). Set to False via mhdsra2_config_override
           for shorter sequences where full BPTT gradient flow is affordable.
        """
        super().__init__()
        active_model_type = normalize_model_type(model_type)
        if active_model_type != "mhdsra2":
            raise ValueError(f"Unsupported multi-layer architecture: {model_type}")

        if dsra_arch_config is not None:
            if num_layers is None:
                num_layers = dsra_arch_config.num_layers
            if K is None:
                K = dsra_arch_config.slots
            if kr is None:
                kr = dsra_arch_config.read_topk
            if chunk_size is None:
                chunk_size = dsra_arch_config.local_window

        if num_layers is None:
            num_layers = 2
        if K is None:
            K = 128
        if kr is None:
            kr = 16
        if chunk_size is None:
            chunk_size = 256

        heads = select_mhdsra2_heads(dim)
        self.architecture = active_model_type
        self.dim = dim
        self.num_layers = num_layers
        self.chunk_size = chunk_size
        self.embedding = nn.Embedding(vocab_size, dim)
        base_cfg = MHDSRA2Config(
            dim=dim,
            heads=heads,
            slots=K,
            read_topk=max(1, min(kr, K)),
            write_topk=max(1, min(kr, K)),
            local_window=max(1, int(chunk_size)),
            use_local=True,
            use_retrieval=use_retrieval,
            detach_state=True,
        )
        if mhdsra2_config_override:
            for key, value in mhdsra2_config_override.items():
                if hasattr(base_cfg, key):
                    setattr(base_cfg, key, value)
        self.layers = nn.ModuleList(
            [MultiHeadDSRA2(base_cfg) for _ in range(num_layers)]
        )
        self.norms = nn.ModuleList([nn.LayerNorm(dim) for _ in range(num_layers)])
        self.final_norm = nn.LayerNorm(dim)
        self.out_proj = nn.Linear(dim, vocab_size)

    def _normalize_selected_positions(
        self,
        positions: int | torch.Tensor,
        batch_size: int,
        seq_len: int,
    ) -> torch.Tensor:
        """Normalize per-batch token positions for memory-bounded logit selection.

        中文说明:
        - 调用方 / Called by: `forward_selected_logits`
        - 调用对象 / Calls: `torch.as_tensor`, `torch.full`, tensor shape/range checks
        - 作用 / Purpose: 将单个位置或逐 batch 位置标准化为 CPU long tensor，支持负索引
        - 参数 / Parameters:
          `positions` 是单个全局 token 位置或 `[B]` 逐样本位置；`batch_size/seq_len`
          来自输入 token 序列，必须为正整数
        - 返回 / Returns: `[B]` CPU long tensor；非法位置抛出 `ValueError`
        - 接入 / Integration: 仅由模型层内部调用，不涉及事务、文件、网络或外部服务
        - 错误处理 / Error handling: batch 数不匹配或位置越界时直接抛出 `ValueError`
        - 副作用 / Side effects: 无；只创建小型 CPU tensor
        - 并发与幂等 / Concurrency and idempotency: 纯函数式转换，可重复调用
        - 关键词 / Keywords:
          position|selected_logits|niah|memory|streaming|batch|negative_index|mhdsra2|token|位置

        English documentation:
        Function name:
            _normalize_selected_positions
        Purpose:
            Normalize scalar or per-batch selected token positions for streaming logits.
        Called by:
            `forward_selected_logits`.
        Calls:
            `torch.as_tensor`, `torch.full`, and tensor validation operations.
        Parameters:
            - positions: int or tensor, selected global token positions.
            - batch_size: int, number of input samples.
            - seq_len: int, input sequence length.
        Returns:
            CPU long tensor with one valid position per batch item.
        Error handling:
            Raises `ValueError` for unsupported shapes or out-of-range positions.
        Side effects:
            None.
        Transaction boundary:
            Not applicable.
        Concurrency and idempotency:
            Reentrant and idempotent for the same input.
        English keywords:
            position, selected_logits, niah, memory, streaming, batch, negative_index, mhdsra2, token, validation
        """
        if isinstance(positions, int):
            normalized = torch.full((batch_size,), positions, dtype=torch.long)
        else:
            normalized = torch.as_tensor(positions, dtype=torch.long).detach().cpu().flatten()
            if normalized.numel() == 1:
                normalized = normalized.expand(batch_size).clone()

        if normalized.numel() != batch_size:
            raise ValueError(
                f"expected one selected position per batch item, got {normalized.numel()} "
                f"positions for batch_size={batch_size}"
            )

        normalized = torch.where(normalized < 0, normalized + seq_len, normalized)
        if bool(((normalized < 0) | (normalized >= seq_len)).any().item()):
            raise ValueError(f"selected positions must be within [0, {seq_len})")
        return normalized

    def forward_selected_logits(
        self,
        x: torch.Tensor,
        positions: int | torch.Tensor,
        stage_id: int | None = None,
        context_id: int | None = None,
        *,
        return_hidden: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Run MHDSRA2 over a long sequence and return logits only at selected positions.

        When return_hidden=True, also returns the pre-out_proj normalized hidden
        states of shape [B, dim] for use in hidden-state MSE auxiliary loss.
        """
        if x.dim() != 2:
            raise ValueError(f"expected token ids with shape [B, SeqLen], got {tuple(x.shape)}")

        batch_size, seq_len = x.shape
        selected_positions = self._normalize_selected_positions(positions, batch_size, seq_len)
        model_device = self.embedding.weight.device
        state_list = [None] * self.num_layers
        logits_by_batch: list[torch.Tensor | None] = [None] * batch_size
        hidden_by_batch: list[torch.Tensor | None] = [None] * batch_size if return_hidden else None

        for start in range(0, seq_len, self.chunk_size):
            end = min(start + self.chunk_size, seq_len)
            token_chunk = x[:, start:end].to(device=model_device, non_blocking=True)
            chunk = self.embedding(token_chunk)

            for layer_idx, (layer, norm) in enumerate(zip(self.layers, self.norms)):
                residual = chunk
                chunk_normed = norm(chunk)
                out_chunk, next_state = layer(
                    chunk_normed,
                    state=state_list[layer_idx],
                    stage_id=stage_id,
                    context_id=context_id,
                )
                state_list[layer_idx] = next_state
                chunk = residual + out_chunk

            selected_mask = (selected_positions >= start) & (selected_positions < end)
            if bool(selected_mask.any().item()):
                batch_idx_cpu = selected_mask.nonzero(as_tuple=True)[0]
                local_idx_cpu = selected_positions[batch_idx_cpu] - start
                batch_idx = batch_idx_cpu.to(device=model_device)
                local_idx = local_idx_cpu.to(device=model_device)
                selected_hidden = chunk[batch_idx, local_idx, :]
                hidden_normed = self.final_norm(selected_hidden)
                selected_logits = self.out_proj(hidden_normed)
                for row_idx, batch_id in enumerate(batch_idx_cpu.tolist()):
                    logits_by_batch[batch_id] = selected_logits[row_idx : row_idx + 1]
                    if return_hidden:
                        hidden_by_batch[batch_id] = hidden_normed[row_idx : row_idx + 1]

        if any(item is None for item in logits_by_batch):
            raise RuntimeError("failed to collect logits for all selected positions")
        logits = torch.cat([item for item in logits_by_batch if item is not None], dim=0)
        if return_hidden:
            hidden = torch.cat([item for item in hidden_by_batch if item is not None], dim=0)
            return logits, hidden
        return logits

    def forward(self, x: torch.Tensor, stage_id: int | None = None,
                context_id: int | None = None) -> torch.Tensor:
        """Run a stacked MHDSRA2 token model over chunked long sequences.

        中文说明:
        - 调用方 / Called by: full-sequence compatibility tests and legacy evaluation scripts
        - 调用对象 / Calls: `nn.Embedding`, `nn.LayerNorm`, `MultiHeadDSRA2.forward`, `nn.Linear`
        - 作用 / Purpose: 对 token id 序列执行多层流式 MHDSRA2 前向并返回全序列词表 logits
        - 变量 / Variables:
          `x` 是 `[B,SeqLen]` token ids, `token_chunk` 是当前分块 token ids,
          `state_list` 保存每层流式状态, `out_list` 收集每个分块输出
        - 接入 / Integration: 输入 token ids，输出 `[B,SeqLen,vocab_size]` logits；只监督少量位置时优先用
          `forward_selected_logits` 降低显存
        - 错误处理 / Error handling: 张量维度和底层配置错误由 PyTorch/MHDSRA2 向上抛出
        - 关键词 / Keywords:
          forward|mhdsra2|multilayer|chunked|streaming|state|token|logits|compat|前向
        """
        _, seq_len = x.shape
        model_device = self.embedding.weight.device
        state_list = [None] * self.num_layers
        out_list = []

        for start in range(0, seq_len, self.chunk_size):
            token_chunk = x[:, start : start + self.chunk_size].to(
                device=model_device, non_blocking=True
            )
            chunk = self.embedding(token_chunk)
            for layer_idx, (layer, norm) in enumerate(zip(self.layers, self.norms)):
                residual = chunk
                chunk_normed = norm(chunk)
                out_chunk, next_state = layer(
                    chunk_normed, state=state_list[layer_idx], stage_id=stage_id,
                    context_id=context_id,
                )
                state_list[layer_idx] = next_state
                chunk = residual + out_chunk
            out_list.append(chunk)

        out = torch.cat(out_list, dim=1)
        out = self.final_norm(out)
        return self.out_proj(out)


class MultiLayerDSRAModel(MultiLayerMHDSRA2Model):
    """Archived DSRA name retained as an MHDSRA2 compatibility alias."""

    def __init__(
        self,
        vocab_size: int,
        dim: int,
        num_layers: int = 2,
        K: int = 128,
        kr: int = 16,
        chunk_size: int = 256,
        use_orthogonal_update: bool = True,
        use_bypass: bool = True,
        pe_mode: str = "none",
    ) -> None:
        """Create the archived DSRA alias using the active MHDSRA2 model.

        中文说明:
        - 调用方 / Called by: legacy `model_type="dsra"` code paths
        - 调用对象 / Calls: `MultiLayerMHDSRA2Model.__init__`
        - 作用 / Purpose: 将旧 DSRA 多层模型名归档为兼容入口，实际全面使用 MHDSRA2
        - 变量 / Variables:
          `use_orthogonal_update/use_bypass/pe_mode` 是旧参数，仅用于兼容签名；
          `vocab_size/dim/num_layers/K/kr/chunk_size` 传递给 MHDSRA2 架构
        - 接入 / Integration: 外部旧导入无需改名即可获得 MHDSRA2 行为
        - 错误处理 / Error handling: 底层 MHDSRA2 配置错误向上抛出，不吞异常
        - 关键词 / Keywords:
          archived|dsra|alias|mhdsra2|compat|multilayer|model|migration|legacy|归档
        """
        self.archived_dsra_options = {
            "use_orthogonal_update": bool(use_orthogonal_update),
            "use_bypass": bool(use_bypass),
            "pe_mode": pe_mode,
        }
        super().__init__(
            vocab_size=vocab_size,
            dim=dim,
            num_layers=num_layers,
            K=K,
            kr=kr,
            chunk_size=chunk_size,
            use_retrieval=False,
            model_type="mhdsra2",
        )
