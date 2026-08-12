"""Loop184 资源门框架（与 Loop183 一致，仅修改 loop_id）。"""

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

    kind: str
    sample: ResourceSample
    threshold: float
    actual: float
    detail: str


@dataclass
class ResourceCell:
    """资源门控制器。"""

    budget: PhaseAResourceGate = field(default_factory=lambda: PHASE_A_GATE)
    samples: list[ResourceSample] = field(default_factory=list)
    violations: list[ResourceViolation] = field(default_factory=list)
    _start_wall: float | None = field(default=None, repr=False)

    def start(self) -> None:
        self._start_wall = time.monotonic()

    def inject_sample(self, sample: ResourceSample) -> None:
        self.samples.append(sample)
        self._check_sample(sample)

    def _check_sample(self, sample: ResourceSample) -> None:
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
        return len(self.violations) == 0

    def build_receipt(self) -> dict[str, object]:
        return {
            "loop_id": "Loop184",
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

    def _sample_gpu_bytes(self) -> int:
        try:
            import torch
            if torch.cuda.is_available():
                return int(torch.cuda.max_memory_allocated())
        except ImportError:
            pass
        return 0

    def _sample_rss_bytes(self) -> int:
        try:
            import psutil
            return int(psutil.Process(os.getpid()).memory_info().rss)
        except ImportError:
            pass
        try:
            import resource
            value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
            return value if os.name == "posix" and value > 1024 * 1024 else value * 1024
        except (ImportError, OSError):
            return 0

    def _sample_wall_seconds(self) -> float:
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
    return IntegrityGateResult(
        silent_drop_rows=max(0, rows_input - rows_output),
        all_rows_accounted=(rows_input == rows_output),
        oom=oom,
        timeout=timeout,
        nonfinite=nonfinite,
        bitwise_deterministic_eval=bitwise_deterministic_eval,
    )


def assert_bitwise_deterministic(
    first: object,
    second: object,
    *,
    label: str = "eval_logits",
) -> None:
    equal = getattr(first, "eq", None)
    if equal is None:
        raise ValueError(f"{label}: first tensor must expose .eq()")
    if not equal(second).all().item():
        raise AssertionError(f"{label}: bitwise determinism check failed")


def assert_budget_invariants() -> None:
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
