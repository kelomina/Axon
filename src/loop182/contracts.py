"""Loop182 冻结契约：Multi-Scale HGConv 架构改进。

Loop182 = Loop179 基础架构 + Multi-Scale HGConvBlock（并行多 filter_length）。
核心改进：在 HGConvBlock 中并行使用 filter_lengths=[8, 16, 32, 64]，
捕捉不同尺度的字节模式，通过可学习权重融合。

其余架构与训练超参与 Loop179 一致，以隔离多尺度卷积的因果效应。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

# ---------------------------------------------------------------------------
# 顶层身份与源码闭包
# ---------------------------------------------------------------------------

LOOP_ID: Final[str] = "Loop182"
LINEAGE: Final[str] = "multi_scale_hgconv_region"
PROPOSAL_VERSION: Final[str] = "2026-07-20-phase0-frozen"

# Phase 0 允许依赖的源码白名单（相对项目根）。
PHASE0_SOURCE_WHITELIST: Final[tuple[str, ...]] = (
    "src/loop182/__init__.py",
    "src/loop182/contracts.py",
    "src/loop182/hgconv.py",
    "src/loop182/model.py",
    "src/loop182/source_closure.py",
    "src/loop182/data_adapter.py",
    "src/loop182/resource_cell.py",
)

# Phase 0 不允许出现的导入符号（防止顺手引入真实数据路径）。
FORBIDDEN_IMPORT_PATTERNS: Final[tuple[str, ...]] = (
    "src.loop151",
    "src.loop164",
    "src.loop175",
    "src.loop166",
    "src.loop179",
    "pandas",
    "numpy.random.default_rng",
    "sklearn.",
    "xgboost",
    "lightgbm",
    "catboost",
)

# ---------------------------------------------------------------------------
# 输入 ABI（与 Loop175 region cache 对齐，但只读契约，不访问真实文件）
# ---------------------------------------------------------------------------

EXPECTED_REGIONS: Final[int] = 16
EXPECTED_REGION_BYTES: Final[int] = 8192
VOCABULARY_SIZE: Final[int] = 257
PADDING_TOKEN: Final[int] = 256

REGION_TYPE_COUNT: Final[int] = 6
BUCKET_COUNT: Final[int] = 64

B0_FEATURE_DIM: Final[int] = 571

# ---------------------------------------------------------------------------
# HGConv 核心冻结值（与 Loop179 一致）
# ---------------------------------------------------------------------------

MODEL_DIM: Final[int] = 192
HGCONV_BLOCKS: Final[int] = 1
HGCONV_FILTER_LENGTH: Final[int] = 32
PATCH_SIZE: Final[int] = 16

# Loop182 核心改进：Multi-Scale 并行 filter lengths
MULTI_SCALE_FILTER_LENGTHS: Final[tuple[int, ...]] = (8, 16, 32, 64)

# 单 region patch 序列长度 = 8192 / 16 = 512
PATCH_SEQUENCE_LENGTH: Final[int] = EXPECTED_REGION_BYTES // PATCH_SIZE

# ---------------------------------------------------------------------------
# Transformer region aggregator 冻结值（与 Loop179 一致）
# ---------------------------------------------------------------------------

TRANSFORMER_LAYERS: Final[int] = 2
TRANSFORMER_HEADS: Final[int] = 6
TRANSFORMER_FFN_DIM: Final[int] = 768
DROPOUT: Final[float] = 0.1

# 输出 ABI
REGION_FEATURE_DIM: Final[int] = MODEL_DIM
REGION_LOGIT_DIM: Final[int] = 2
FUSION_LOGIT_DIM: Final[int] = 2
B0_PROJECTED_DIM: Final[int] = 128

# ---------------------------------------------------------------------------
# Phase A 资源门阈值（与 Loop179 一致）
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PhaseAResourceGate:
    """Phase A 资源门阈值，任何超限必须 fail-fast。"""

    fit_rows: int = 12_000
    selection_rows: int = 4_000
    fold0_model_rows: int = 0

    max_epochs: int = 12
    microbatch: int = 2
    accumulation: int = 16
    effective_batch: int = 32
    learning_rate: float = 3.0e-4
    weight_decay: float = 1.0e-2
    warmup_steps: int = 1
    grad_clip: float = 1.0
    ema_decay: float = 0.999

    autocast_dtype: str = "bfloat16"
    master_dtype: str = "float32"
    fft_master_dtype: str = "float32"

    gpu_allocated_bytes: int = 6_500_000_000
    rss_bytes: int = 11_000_000_000
    wall_seconds: int = 21_600

    silent_drop_rows: int = 0
    all_rows_accounted: bool = True
    oom: bool = False
    timeout: bool = False
    nonfinite: bool = False
    bitwise_deterministic_eval: bool = True


@dataclass(frozen=True)
class PhaseBGate:
    """Phase B J 臂晋级门（与 Loop179 一致）。"""

    j_net_fewer_errors_vs_a: int = 30
    j_repairs_vs_a: int = 50
    j_override_precision: float = 0.80
    j_net_positive_folds: int = 4
    j_bootstrap_lcb_vs_a: float = 0.0

    fp_relative_worsening: float = 0.05
    fn_relative_worsening: float = 0.05

    k_more_errors_vs_j: int = 30
    k_bootstrap_lcb_vs_j: float = 0.0


@dataclass(frozen=True)
class NumericTolerance:
    """Phase 0 数值验证容差。"""

    float64_rtol: float = 1.0e-12
    float64_atol: float = 1.0e-12
    float32_rtol: float = 1.0e-5
    float32_atol: float = 1.0e-6
    fft_norm_eps: float = 1.0e-7


PHASE_A_GATE: Final[PhaseAResourceGate] = PhaseAResourceGate()
PHASE_B_GATE: Final[PhaseBGate] = PhaseBGate()
NUMERIC_TOLERANCE: Final[NumericTolerance] = NumericTolerance()


def assert_contract_invariants() -> None:
    """启动时自检冻结常量的一致性。"""

    assert PATCH_SEQUENCE_LENGTH > HGCONV_FILTER_LENGTH, (
        "patch sequence must be longer than HGConv filter"
    )
    # 多尺度 filter 中最长的不能超过 patch 序列
    assert max(MULTI_SCALE_FILTER_LENGTHS) <= PATCH_SEQUENCE_LENGTH, (
        "max multi-scale filter length must not exceed patch sequence"
    )
    assert MODEL_DIM % TRANSFORMER_HEADS == 0, "model_dim must be divisible by heads"
    assert EXPECTED_REGION_BYTES % PATCH_SIZE == 0, "region bytes must be divisible by patch size"
    assert PADDING_TOKEN == VOCABULARY_SIZE - 1, "padding token must be last vocab item"
    assert (
        PHASE_A_GATE.effective_batch
        == PHASE_A_GATE.microbatch * PHASE_A_GATE.accumulation
    ), "effective batch must equal microbatch * accumulation"
    assert (
        PHASE_A_GATE.fit_rows + PHASE_A_GATE.selection_rows == 16_000
    ), "fit + selection must cover 16000 Train rows"
    assert PHASE_A_GATE.fold0_model_rows == 0, "Phase A must not train fold0 model"
