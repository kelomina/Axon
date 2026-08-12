"""Loop183 冻结契约：Multi-Scale HGConv + 强正则化 + 数据增强。

Loop183 = Loop182 多尺度 HGConv 架构 + 强正则化组合：
- weight_decay: 1e-2 → 3e-2（3x）
- dropout: 0.1 → 0.2（2x）
- max_epochs: 12 → 20（给正则化时间）
- 训练时数据增强（在训练脚本实现，不在契约）：
  - Mixup α=0.4（region tokens 层面混合）
  - Region dropout p=0.3（随机 mask 1 个 region）
  - Label smoothing ε=0.05
- 模型选择标准：loss-based → F1-based（避免过早停止）

架构与 Loop182 完全一致，以隔离正则化的因果效应。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

# ---------------------------------------------------------------------------
# 顶层身份与源码闭包
# ---------------------------------------------------------------------------

LOOP_ID: Final[str] = "Loop183"
LINEAGE: Final[str] = "multi_scale_hgconv_strong_reg"
PROPOSAL_VERSION: Final[str] = "2026-07-20-phase0-frozen"

# Phase 0 允许依赖的源码白名单（相对项目根）。
PHASE0_SOURCE_WHITELIST: Final[tuple[str, ...]] = (
    "src/loop183/__init__.py",
    "src/loop183/contracts.py",
    "src/loop183/hgconv.py",
    "src/loop183/model.py",
    "src/loop183/source_closure.py",
    "src/loop183/data_adapter.py",
    "src/loop183/resource_cell.py",
)

# Phase 0 不允许出现的导入符号（防止顺手引入真实数据路径或其他 loop）。
FORBIDDEN_IMPORT_PATTERNS: Final[tuple[str, ...]] = (
    "src.loop151",
    "src.loop164",
    "src.loop175",
    "src.loop166",
    "src.loop179",
    "src.loop182",
    "pandas",
    "numpy.random.default_rng",
    "sklearn.",
    "xgboost",
    "lightgbm",
    "catboost",
)

# ---------------------------------------------------------------------------
# 输入 ABI（与 Loop182/Loop179 一致，复用 Loop175 region cache）
# ---------------------------------------------------------------------------

EXPECTED_REGIONS: Final[int] = 16
EXPECTED_REGION_BYTES: Final[int] = 8192
VOCABULARY_SIZE: Final[int] = 257
PADDING_TOKEN: Final[int] = 256

REGION_TYPE_COUNT: Final[int] = 6
BUCKET_COUNT: Final[int] = 64

B0_FEATURE_DIM: Final[int] = 571

# ---------------------------------------------------------------------------
# HGConv 核心冻结值（与 Loop182 一致）
# ---------------------------------------------------------------------------

MODEL_DIM: Final[int] = 192
HGCONV_BLOCKS: Final[int] = 1
HGCONV_FILTER_LENGTH: Final[int] = 32
PATCH_SIZE: Final[int] = 16

# 多尺度并行 filter lengths（与 Loop182 一致）
MULTI_SCALE_FILTER_LENGTHS: Final[tuple[int, ...]] = (8, 16, 32, 64)

# 单 region patch 序列长度 = 8192 / 16 = 512
PATCH_SEQUENCE_LENGTH: Final[int] = EXPECTED_REGION_BYTES // PATCH_SIZE

# ---------------------------------------------------------------------------
# Transformer region aggregator 冻结值
# ---------------------------------------------------------------------------

TRANSFORMER_LAYERS: Final[int] = 2
TRANSFORMER_HEADS: Final[int] = 6
TRANSFORMER_FFN_DIM: Final[int] = 768
# Loop183 改进：dropout 0.1 → 0.2
DROPOUT: Final[float] = 0.2

# 输出 ABI
REGION_FEATURE_DIM: Final[int] = MODEL_DIM
REGION_LOGIT_DIM: Final[int] = 2
FUSION_LOGIT_DIM: Final[int] = 2
B0_PROJECTED_DIM: Final[int] = 128

# ---------------------------------------------------------------------------
# Loop183 数据增强超参（在训练脚本中读取，冻结在契约中以保证可复现）
# ---------------------------------------------------------------------------

MIXUP_ALPHA: Final[float] = 0.4
REGION_DROPOUT_PROB: Final[float] = 0.3
LABEL_SMOOTHING_EPS: Final[float] = 0.05

# ---------------------------------------------------------------------------
# Phase A 资源门阈值
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PhaseAResourceGate:
    """Phase A 资源门阈值，任何超限必须 fail-fast。

    Loop183 改动：
    - weight_decay: 1e-2 → 3e-2
    - max_epochs: 12 → 20
    """

    fit_rows: int = 12_000
    selection_rows: int = 4_000
    fold0_model_rows: int = 0

    max_epochs: int = 20
    microbatch: int = 2
    accumulation: int = 16
    effective_batch: int = 32
    learning_rate: float = 3.0e-4
    weight_decay: float = 3.0e-2  # Loop183: 1e-2 → 3e-2
    warmup_steps: int = 1
    grad_clip: float = 1.0
    ema_decay: float = 0.999

    autocast_dtype: str = "bfloat16"
    master_dtype: str = "float32"
    fft_master_dtype: str = "float32"

    # wall_seconds 提升以容纳 20 epochs（原 6h = 21600s，Loop183 每 epoch ~380s × 20 = 7600s）
    # 保持 21600s（6h）足够
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
    # Loop183 数据增强超参验证
    assert MIXUP_ALPHA > 0.0, "MIXUP_ALPHA must be positive"
    assert 0.0 <= REGION_DROPOUT_PROB < 1.0, "REGION_DROPOUT_PROB must be in [0, 1)"
    assert 0.0 <= LABEL_SMOOTHING_EPS < 0.5, "LABEL_SMOOTHING_EPS must be in [0, 0.5)"
