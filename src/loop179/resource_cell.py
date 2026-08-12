"""Loop179 资源门框架（Phase 0：只定义接口与阈值，不绑定真实训练）。

本模块定义：
1. ResourceBudget：从 contracts.PHASE_A_GATE 派生的预算实例。
2. ResourceSample：单次采样点（GPU bytes、RSS bytes、wall seconds）。
3. ResourceCell：资源门控制器，记录采样、检测超限、生成 receipt。
4. IntegrityGate：完整性门（silent drop、OOM、timeout、nonfinite、determinism）。

Phase 0 阶段：
- 不导入 torch.cuda 或 psutil（避免环境依赖）
- 不启动真实监控线程
- 只定义接口和阈值，用合成样本测试逻辑

Phase A 授权后：实现 _sample_gpu_bytes / _sample_rss_bytes / _sample_wall_seconds。
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field

from .contracts import PHASE_A_GATE, PhaseAResourceGate


@dataclass(frozen=True)
class ResourceSample:
    """单次资源采样点。"""

    wall_seconds: float
    gpu_allocated_bytes: int = 0
    rss_bytes: int = 0
    epoch: int = 0
    step: int = 0
    note: str = ""


@dataclass(frozen=True)
class ResourceViolation:
    """单条资源违规记录。"""

    kind: str  # "gpu_over" | "rss_over" | "wall_over" | "silent_drop" | "oom" | "timeout" | "nonfinite" | "nondeterministic"
    sample: ResourceSample
    threshold: float
    actual: float
    detail: str


@dataclass
class ResourceCell:
    """资源门控制器，记录采样并检测超限。

    Phase 0：用 inject_sample 注入合成样本测试逻辑。
    Phase A：实现 start/stop 和真实采样。
    """

    budget: PhaseAResourceGate = field(default_factory=lambda: PHASE_A_GATE)
    samples: list[ResourceSample] = field(default_factory=list)
    violations: list[ResourceViolation] = field(default_factory=list)
    _start_wall: float | None = field(default=None, repr=False)

    def start(self) -> None:
        """启动资源门计时。Phase A 授权后绑定真实监控。"""

        self._start_wall = time.monotonic()

    def inject_sample(self, sample: ResourceSample) -> None:
        """注入一个采样点并检查预算（Phase 0 测试用）。"""

        self.samples.append(sample)
        self._check_sample(sample)

    def _check_sample(self, sample: ResourceSample) -> None:
        """检查单个采样点是否超限。"""

        if sample.gpu_allocated_bytes > self.budget.gpu_allocated_bytes:
            self.violations.append(ResourceViolation(
                kind="gpu_over",
                sample=sample,
                threshold=float(self.budget.gpu_allocated_bytes),
                actual=float(sample.gpu_allocated_bytes),
                detail=f"GPU {sample.gpu_allocated_bytes} > budget {self.budget.gpu_allocated_bytes}",
            ))
        if sample.rss_bytes > self.budget.rss_bytes:
            self.violations.append(ResourceViolation(
                kind="rss_over",
                sample=sample,
                threshold=float(self.budget.rss_bytes),
                actual=float(sample.rss_bytes),
                detail=f"RSS {sample.rss_bytes} > budget {self.budget.rss_bytes}",
            ))
        if sample.wall_seconds > self.budget.wall_seconds:
            self.violations.append(ResourceViolation(
                kind="wall_over",
                sample=sample,
                threshold=float(self.budget.wall_seconds),
                actual=float(sample.wall_seconds),
                detail=f"wall {sample.wall_seconds} > budget {self.budget.wall_seconds}",
            ))

    def record_integrity(
        self,
        *,
        kind: str,
        detail: str,
        sample: ResourceSample | None = None,
    ) -> None:
        """记录完整性门违规。"""

        if sample is None:
            sample = ResourceSample(wall_seconds=0.0)
        self.violations.append(ResourceViolation(
            kind=kind,
            sample=sample,
            threshold=0.0,
            actual=1.0,
            detail=detail,
        ))

    def passed(self) -> bool:
        """资源门是否通过（无违规）。"""

        return len(self.violations) == 0

    def build_receipt(self) -> dict[str, object]:
        """生成资源门 receipt，用于 Phase A 报告。"""

        return {
            "loop_id": "Loop179",
            "phase": "A",
            "budget": {
                "gpu_allocated_bytes": self.budget.gpu_allocated_bytes,
                "rss_bytes": self.budget.rss_bytes,
                "wall_seconds": self.budget.wall_seconds,
                "max_epochs": self.budget.max_epochs,
                "fit_rows": self.budget.fit_rows,
                "selection_rows": self.budget.selection_rows,
            },
            "sample_count": len(self.samples),
            "violation_count": len(self.violations),
            "violations": [
                {
                    "kind": v.kind,
                    "epoch": v.sample.epoch,
                    "step": v.sample.step,
                    "threshold": v.threshold,
                    "actual": v.actual,
                    "detail": v.detail,
                }
                for v in self.violations
            ],
            "passed": self.passed(),
        }

    # Phase A 真实采样实现
    def _sample_gpu_bytes(self) -> int:
        """采样当前 GPU 已分配字节（peak max_memory_allocated）。"""

        try:
            import torch

            if torch.cuda.is_available():
                return int(torch.cuda.max_memory_allocated())
        except ImportError:
            pass
        return 0

    def _sample_rss_bytes(self) -> int:
        """采样当前进程 RSS 字节，优先 psutil，回退 resource。"""

        try:
            import psutil

            return int(psutil.Process(os.getpid()).memory_info().rss)
        except ImportError:
            pass
        try:
            import resource

            value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
            # Linux: KB, macOS: bytes, Windows: 无 resource 模块
            return value if os.name == "posix" and value > 1024 * 1024 else value * 1024
        except (ImportError, OSError):
            return 0

    def _sample_wall_seconds(self) -> float:
        """采样当前 wall clock 秒数。"""

        if self._start_wall is None:
            raise RuntimeError("ResourceCell.start() must be called before sampling")
        return time.monotonic() - self._start_wall

    def sample_and_inject(
        self,
        *,
        epoch: int = 0,
        step: int = 0,
        note: str = "",
    ) -> ResourceSample:
        """Phase A 便捷方法：采样真实 GPU/RSS/wall 并注入检查。"""

        sample = ResourceSample(
            wall_seconds=self._sample_wall_seconds(),
            gpu_allocated_bytes=self._sample_gpu_bytes(),
            rss_bytes=self._sample_rss_bytes(),
            epoch=epoch,
            step=step,
            note=note,
        )
        self.inject_sample(sample)
        return sample


# ---------------------------------------------------------------------------
# 完整性门
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class IntegrityGateResult:
    """完整性门结果。"""

    silent_drop_rows: int
    all_rows_accounted: bool
    oom: bool
    timeout: bool
    nonfinite: bool
    bitwise_deterministic_eval: bool

    def passed(self) -> bool:
        """完整性门是否通过。"""

        return (
            self.silent_drop_rows == 0
            and self.all_rows_accounted
            and not self.oom
            and not self.timeout
            and not self.nonfinite
            and self.bitwise_deterministic_eval
        )


def check_integrity(
    *,
    rows_input: int,
    rows_output: int,
    oom: bool = False,
    timeout: bool = False,
    nonfinite: bool = False,
    bitwise_deterministic_eval: bool = True,
) -> IntegrityGateResult:
    """检查 Phase A 完整性门。"""

    return IntegrityGateResult(
        silent_drop_rows=max(0, rows_input - rows_output),
        all_rows_accounted=(rows_input == rows_output),
        oom=oom,
        timeout=timeout,
        nonfinite=nonfinite,
        bitwise_deterministic_eval=bitwise_deterministic_eval,
    )


# ---------------------------------------------------------------------------
# 确定性验证（Phase 0：合成张量；Phase A：真实 eval logits）
# ---------------------------------------------------------------------------

def assert_bitwise_deterministic(
    first: object,
    second: object,
    *,
    label: str = "eval_logits",
) -> None:
    """验证两次 forward 输出 bitwise 一致。

    Phase 0：用合成张量测试。
    Phase A：用真实 eval logits 测试（eval 模式、固定 seed）。
    """

    # 鸭子类型，避免直接依赖 torch
    equal = getattr(first, "eq", None)
    if equal is None:
        raise ValueError(f"{label}: first tensor must expose .eq()")
    if not equal(second).all().item():
        raise AssertionError(f"{label}: bitwise determinism check failed")


# ---------------------------------------------------------------------------
# Phase 0 预算自检
# ---------------------------------------------------------------------------

def assert_budget_invariants() -> None:
    """验证 Phase A 预算的内部一致性。"""

    gate = PHASE_A_GATE
    assert gate.fit_rows > 0, "fit_rows must be positive"
    assert gate.selection_rows > 0, "selection_rows must be positive"
    assert gate.fit_rows + gate.selection_rows == 16_000, "must cover 16000 Train rows"
    assert gate.fold0_model_rows == 0, "Phase A must not train fold0"
    assert gate.max_epochs > 0, "max_epochs must be positive"
    assert gate.microbatch > 0, "microbatch must be positive"
    assert gate.accumulation > 0, "accumulation must be positive"
    assert gate.effective_batch == gate.microbatch * gate.accumulation, "batch contract"
    assert gate.gpu_allocated_bytes > 0, "gpu budget must be positive"
    assert gate.rss_bytes > 0, "rss budget must be positive"
    assert gate.wall_seconds > 0, "wall budget must be positive"
    assert gate.silent_drop_rows == 0, "silent drops must be zero"
    assert gate.all_rows_accounted, "all rows must be accounted"
    assert not gate.oom, "oom must be false"
    assert not gate.timeout, "timeout must be false"
    assert not gate.nonfinite, "nonfinite must be false"
    assert gate.bitwise_deterministic_eval, "determinism must be required"
