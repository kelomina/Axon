"""CPU-side paged exact memory for MHDSRA2.

This is a minimal reference implementation. It keeps all token-level K/V on CPU
and only returns a small retrieved subset to the GPU-side attention layer.
For production, replace the exact page scoring with FAISS/ScaNN or a custom ANN
index and store token K/V in pinned CPU memory, mmap, or NVMe.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F

try:
    from ...config import DSRAArchitectureConfig
except ImportError:
    try:
        from ..config import DSRAArchitectureConfig
    except ImportError:
        DSRAArchitectureConfig = None


@dataclass
class PageRecord:
    key: torch.Tensor
    value: torch.Tensor
    summary: torch.Tensor
    start: int
    end: int
    valid: bool = True
    version: int = 0


class PagedExactMemory:
    """Append-only paged token memory with simple top-page retrieval.

    The module is intentionally CPU-side. It gives DSRA-v2 a precise recall path
    without keeping 2M-token KV on GPU.
    """

    def __init__(
        self,
        page_size: int = None,
        dtype: torch.dtype = torch.float16,
        dsra_arch_config=None,
        max_pages: Optional[int] = None,
    ):
        if page_size is None:
            page_size = getattr(dsra_arch_config, 'paged_memory_page_size', 1024) if dsra_arch_config else 1024
        self.page_size = int(page_size)
        self.dtype = dtype
        self._dsra_arch_config = dsra_arch_config
        if max_pages is None and dsra_arch_config is not None:
            max_pages = getattr(dsra_arch_config, 'paged_memory_max_pages', None)
        self.max_pages = int(max_pages) if max_pages is not None else None
        if self.max_pages is not None and self.max_pages < 1:
            raise ValueError("max_pages must be positive when provided")
        self.pages: List[PageRecord] = []
        self.next_position = 0

    def __len__(self) -> int:
        return len(self.pages)

    def append(self, key: torch.Tensor, value: torch.Tensor) -> None:
        """Append chunk K/V.

        key/value shape: [1,H,T,d] or [H,T,d]. T may be larger than page_size.
        Only batch_size=1 is supported in this reference implementation.
        """
        if key.dim() == 4:
            if key.shape[0] != 1:
                raise ValueError("PagedExactMemory reference code supports batch_size=1")
            key = key[0]
            value = value[0]
        if key.dim() != 3:
            raise ValueError("key/value must be [H,T,d] or [1,H,T,d]")
        _, t, _ = key.shape
        for s in range(0, t, self.page_size):
            e = min(s + self.page_size, t)
            k_page = key[:, s:e, :].detach().to("cpu", dtype=self.dtype).contiguous()
            v_page = value[:, s:e, :].detach().to("cpu", dtype=self.dtype).contiguous()
            summary = F.normalize(k_page.float().mean(dim=1), dim=-1).to(dtype=self.dtype)
            self.pages.append(PageRecord(k_page, v_page, summary, self.next_position + s, self.next_position + e))
        self.next_position += t
        self._enforce_capacity()

    def invalidate_before(self, position: int) -> None:
        """Forget obsolete pages and release their tensor storage."""
        self.pages = [p for p in self.pages if p.end > position]

    def clear(self, *, reset_position: bool = False) -> None:
        """Release all stored pages.

        ``reset_position`` is used at sample boundaries so a long-lived process
        does not carry monotonically growing positions across independent inputs.
        """
        self.pages.clear()
        if reset_position:
            self.next_position = 0

    def _enforce_capacity(self) -> None:
        if self.max_pages is None or len(self.pages) <= self.max_pages:
            return
        self.pages = self.pages[-self.max_pages:]

    @staticmethod
    def _rank_by_score_then_position(scores: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """Rank candidates by score first and recency second.

        中文说明:
        - 调用方 / Called by: `retrieve`
        - 调用对象 / Calls: `torch.argsort`
        - 作用 / Purpose: 按相似度降序排序，并在相似度并列时按位置降序实现 latest-wins
        - 变量 / Variables:
          `scores` 为候选页或 token 的相似度分数, `positions` 为对应结束位置或 token 位置
          `recency_order` 是按新到旧排列的次级顺序, `score_order` 是稳定的主排序顺序
        - 接入 / Integration: 外部记忆新增排序策略时复用该函数，保证同分候选优先返回最新写入
        - 错误处理 / Error handling: 依赖上游保证两个一维张量长度一致；形状错误由 PyTorch 抛出
        - 关键词 / Keywords:
          latest_wins|stable_sort|recency|score|position|retrieve|paged_memory|tie_break|memory|最新优先
        """
        position_order = torch.argsort(positions, descending=True, stable=True)
        score_order = torch.argsort(scores[position_order], descending=True, stable=True)
        return position_order[score_order]

    def retrieve(
        self,
        query: torch.Tensor,
        top_pages: int = None,
        max_tokens: int = None,
        device: Optional[torch.device] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Return retrieved_k, retrieved_v, positions with latest-wins tie breaks.

        query shape: [1,H,T,d] or [H,T,d]. Returned K/V shape: [1,H,R,d].
        Retrieval is chunk-level: mean query selects pages, then mean query selects
        top tokens inside those pages. Per-token retrieval can be added on top of
        this interface but costs more memory.

        中文说明:
        - 调用方 / Called by: diagnostic runners, isolation tests, MHDSRA2 paged recall callers
        - 调用对象 / Calls: `F.normalize`, `_rank_by_score_then_position`, `torch.einsum`, `torch.cat`
        - 作用 / Purpose: 从 CPU 分页精确记忆中召回最相似的 K/V，并在同分时返回最新写入的事实
        - 变量 / Variables:
          `query_h/q_summary` 为查询表示, `valid_pages` 为仍有效的页面,
          `page_scores/token_scores` 为页级与 token 级相似度, `pos` 为全局 token 位置
        - 接入 / Integration: 调用者传入当前 query head tensor，并把返回的 `retrieved_k/v` 接入 retrieval attention
        - 错误处理 / Error handling: 空记忆或无有效页返回 `(None, None, None)`；非 batch=1 或非法维度抛出 `ValueError`
        - 关键词 / Keywords:
          retrieve|paged_exact_memory|latest_wins|external_memory|top_tokens|page_scores|token_scores|position|recall|分页检索
        """
        if top_pages is None:
            top_pages = getattr(self, '_dsra_arch_config', None)
            top_pages = getattr(top_pages, 'paged_memory_top_pages', 4) if top_pages else 4
        if max_tokens is None:
            max_tokens = getattr(self, '_dsra_arch_config', None)
            max_tokens = getattr(max_tokens, 'paged_memory_max_tokens', 128) if max_tokens else 128
        if len(self.pages) == 0:
            return None, None, None
        if query.dim() == 4:
            if query.shape[0] != 1:
                raise ValueError("PagedExactMemory reference code supports batch_size=1")
            query_h = query[0]
        elif query.dim() == 3:
            query_h = query
        else:
            raise ValueError("query must be [H,T,d] or [1,H,T,d]")

        device = device or query.device
        q_summary = F.normalize(query_h.float().mean(dim=1), dim=-1).cpu()
        valid_pages = [p for p in self.pages if p.valid]
        if not valid_pages:
            return None, None, None
        summaries = torch.stack([p.summary.float() for p in valid_pages], dim=1)
        page_scores = torch.einsum("hd,hpd->hp", q_summary, summaries).mean(dim=0)
        n_pages = min(top_pages, page_scores.numel())
        page_positions = torch.tensor([p.end for p in valid_pages], dtype=torch.long)
        top_page_idx = self._rank_by_score_then_position(page_scores, page_positions)[:n_pages]
        chosen = [valid_pages[int(i)] for i in top_page_idx.tolist()]

        k_cat = torch.cat([p.key.float() for p in chosen], dim=1)
        v_cat = torch.cat([p.value.float() for p in chosen], dim=1)
        pos = torch.cat([torch.arange(p.start, p.end) for p in chosen], dim=0)
        token_scores = torch.einsum("hd,hrd->hr", q_summary, F.normalize(k_cat, dim=-1)).mean(dim=0)
        r = min(max_tokens, token_scores.numel())
        tok_idx = self._rank_by_score_then_position(token_scores, pos)[:r]
        k_out = k_cat[:, tok_idx, :].unsqueeze(0).to(device=device, dtype=query.dtype)
        v_out = v_cat[:, tok_idx, :].unsqueeze(0).to(device=device, dtype=query.dtype)
        pos_out = pos[tok_idx].to(device=device)
        return k_out, v_out, pos_out


if __name__ == "__main__":
    torch.manual_seed(0)
    mem = PagedExactMemory(page_size=16)
    k = torch.randn(1, 4, 64, 32)
    v = torch.randn(1, 4, 64, 32)
    mem.append(k, v)
    q = k[:, :, 30:34, :] + 0.01 * torch.randn(1, 4, 4, 32)
    rk, rv, pos = mem.retrieve(q, top_pages=2, max_tokens=8)
    assert rk is not None and rv is not None and pos is not None
    print("[OK] PagedExactMemory smoke test passed")
    print("retrieved positions:", pos.tolist())
