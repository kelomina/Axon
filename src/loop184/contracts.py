"""Loop184 冻结契约：架构扩展 + SAM + SWA。

Loop184 = Loop183 数据增强基础 + 4 项激进改进：
1. 架构扩展：transformer_layers 2→4, hgconv_blocks 1→2（参数量 1.62M → 2.61M, +60.9%）
2. SAM 优化器（rho=0.05）：找平坦极小值，每个 effective batch 应用
3. SWA（70% epoch 启动）：平均最后 4 个 epoch 权重，替代 EMA
4. 保持 Loop183 数据增强：Mixup α=0.4, Region dropout p=0.3, Label smoothing ε=0.05

资源估算（实测）：
- 参数量 2,610,573（仍在 6.5 GiB GPU 预算内）
- SAM 2x forward-backward：每 effective batch 3.03s
- 每 epoch 375 effective batches × 3.03s = 1136s = 18.9 min
- 10 epochs × 19 min = 190 min = 3.2h（在 6h 预算内）
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

# ---------------------------------------------------------------------------
# 顶层身份与源码闭包
# ---------------------------------------------------------------------------

LOOP_ID: Final[str] = "Loop184"
LINEAGE: Final[str] = "arch_expand_sam_swa"
PROPOSAL_VERSION: Final[str] = "2026-07-20-phase0-frozen"

# Phase 0 允许依赖的源码白名单（相对项目根）。
PHASE0_SOURCE_WHITELIST: Final[tuple[str, ...]] = (
    "src/loop184/__init__.py",
    "src/loop184/contracts.py",
    "src/loop184/hgconv.py",
    "src/loop184/model.py",
    "src/loop184/source_closure.py",
    "src/loop184/data_adapter.py",
    "src/loop184/resource_cell.py",
)

# Phase 0 不允许出现的导入符号（防止顺手引入真实数据路径或其他 loop）。
FORBIDDEN_IMPORT_PATTERNS: Final[tuple[str, ...]] = (
    "src.loop151",
    "src.loop164",
    "src.loop175",
    "src.loop166",
    "src.loop179",
    "src.loop182",
    "src.loop183",
    "pandas",
    "numpy.random.default_rng",
    "sklearn.",
    "xgboost",
    "lightgbm",
    "catboost",
)

# ---------------------------------------------------------------------------
# 输入 ABI（与 Loop183 一致，复用 Loop175 region cache）
# ---------------------------------------------------------------------------

EXPECTED_REGIONS: Final[int] = 16
EXPECTED_REGION_BYTES: Final[int] = 8192
VOCABULARY_SIZE: Final[int] = 257
PADDING_TOKEN: Final[int] = 256

REGION_TYPE_COUNT: Final[int] = 6
BUCKET_COUNT: Final[int] = 64

B0_FEATURE_DIM: Final[int] = 571

# ---------------------------------------------------------------------------
# HGConv 核心冻结值
# ---------------------------------------------------------------------------

MODEL_DIM: Final[int] = 192
# Loop184 改进：hgconv_blocks 1 → 2
HGCONV_BLOCKS: Final[int] = 2
HGCONV_FILTER_LENGTH: Final[int] = 32
PATCH_SIZE: Final[int] = 16

# 多尺度并行 filter lengths（与 Loop183 一致）
MULTI_SCALE_FILTER_LENGTHS: Final[tuple[int, ...]] = (8, 16, 32, 64)

# 单 region patch 序列长度 = 8192 / 16 = 512
PATCH_SEQUENCE_LENGTH: Final[int] = EXPECTED_REGION_BYTES // PATCH_SIZE

# ---------------------------------------------------------------------------
# Transformer region aggregator 冻结值
# ---------------------------------------------------------------------------

# Loop184 改进：transformer_layers 2 → 4
TRANSFORMER_LAYERS: Final[int] = 4
TRANSFORMER_HEADS: Final[int] = 6
TRANSFORMER_FFN_DIM: Final[int] = 768
DROPOUT: Final[float] = 0.2

# 输出 ABI
REGION_FEATURE_DIM: Final[int] = MODEL_DIM
REGION_LOGIT_DIM: Final[int] = 2
FUSION_LOGIT_DIM: Final[int] = 2
B0_PROJECTED_DIM: Final[int] = 128

# ---------------------------------------------------------------------------
# Loop184 数据增强超参（与 Loop183 一致）
# ---------------------------------------------------------------------------

MIXUP_ALPHA: Final[float] = 0.4
REGION_DROPOUT_PROB: Final[float] = 0.3
LABEL_SMOOTHING_EPS: Final[float] = 0.05

# ---------------------------------------------------------------------------
# Loop184 SAM + SWA 超参
# ---------------------------------------------------------------------------

# SAM rho：扰动半径（CIFAR 标准值 0.05）
SAM_RHO: Final[float] = 0.05

# SWA：在 70% epoch 启动（10 epochs → epoch 7 开始）
SWA_START_EPOCH: Final[int] = 7
# SWA 学习率：base lr 的 10%
SWA_LR: Final[float] = 3.0e-5
# SWA 退火 epochs（从 base lr 退火到 swa_lr）
SWA_ANNEAL_EPOCHS: Final[int] = 2

# ---------------------------------------------------------------------------
# Phase A 资源门阈值
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PhaseAResourceGate:
    """Phase A 资源门阈值，任何超限必须 fail-fast。

    Loop184 改动：
    - weight_decay 保持 3e-2（与 Loop183 一致）
    - max_epochs 20 → 10（SAM 2x 导致每 epoch ~19min，10 epochs ≈ 3.2h）
    - ema_decay 保留字段但不再使用（SWA 替代）
    - 新增 sam_rho, swa_start_epoch, swa_lr 字段
    """

    fit_rows: int = 12_000
    selection_rows: int = 4_000
    fold0_model_rows: int = 0

    max_epochs: int = 10
    microbatch: int = 2
    accumulation: int = 16
    effective_batch: int = 32
    learning_rate: float = 3.0e-4
    weight_decay: float = 3.0e-2
    warmup_steps: int = 1
    grad_clip: float = 1.0
    ema_decay: float = 0.999  # 保留字段以兼容 Loop183 接口，实际不使用

    # Loop184 新增
    sam_rho: float = 0.05
    swa_start_epoch: int = 7
    swa_lr: float = 3.0e-5
    swa_anneal_epochs: int = 2

    autocast_dtype: str = "bfloat16"
    master_dtype: str = "float32"
    fft_master_dtype: str = "float32"

    # wall_seconds 保持 21600（6h），SAM 2x + 10 epochs ≈ 3.2h（含评估开销）
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
    """Phase B J 臂晋级门（与 Loop183 一致）。"""

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
    # 数据增强超参验证
    assert MIXUP_ALPHA > 0.0, "MIXUP_ALPHA must be positive"
    assert 0.0 <= REGION_DROPOUT_PROB < 1.0, "REGION_DROPOUT_PROB must be in [0, 1)"
    assert 0.0 <= LABEL_SMOOTHING_EPS < 0.5, "LABEL_SMOOTHING_EPS must be in [0, 0.5)"
    # Loop184 SAM/SWA 超参验证
    assert SAM_RHO > 0.0, "SAM_RHO must be positive"
    assert SWA_START_EPOCH > 0, "SWA_START_EPOCH must be positive"
    assert SWA_START_EPOCH < PHASE_A_GATE.max_epochs, "SWA_START_EPOCH must be < max_epochs"
    assert SWA_LR > 0.0, "SWA_LR must be positive"
    assert SWA_ANNEAL_EPOCHS > 0, "SWA_ANNEAL_EPOCHS must be positive"
    # 架构扩展验证
    assert HGCONV_BLOCKS >= 1, "HGCONV_BLOCKS must be at least 1"
    assert TRANSFORMER_LAYERS >= 1, "TRANSFORMER_LAYERS must be at least 1"
