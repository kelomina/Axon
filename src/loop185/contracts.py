"""Loop185 冻结契约：4-fold OOF 集成（Loop184 架构, 无 SAM, SWA, logit average）。

Loop185 = Loop184 架构 + 4-fold OOF 集成：
1. 复用 Loop184 架构（2 hgconv blocks, 4 transformer layers, 2.61M params）
2. 关闭 SAM（2x 加速，使 4-fold OOF 在 6h 预算内可行）
3. 保持 SWA（start_epoch=6, 平均最后 3 个 epoch 权重）
4. 保持 Loop183 数据增强：Mixup α=0.4, Region dropout p=0.3, Label smoothing ε=0.05
5. 4-fold OOF：fold 1,2,3,4 互相 OOF，fold 0 保持 forbidden
6. 集成方法：logit average（4 个模型的 softmax 概率平均）

资源估算（基于 Loop184 实测，无 SAM 应减半）：
- 参数量 2,610,573（与 Loop184 一致）
- 无 SAM：每 effective batch ~1.5s（Loop184 SAM 3.03s / 2）
- 每 epoch 375 effective batches × 1.5s = 562s = 9.4 min
- 8 epochs × 9.4 min = 75 min = 1.25h/fold
- 4 fold × 1.25h = 5h（含评估开销，在 6h 预算内）
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

# ---------------------------------------------------------------------------
# 顶层身份与源码闭包
# ---------------------------------------------------------------------------

LOOP_ID: Final[str] = "Loop185"
LINEAGE: Final[str] = "oof_ensemble_logit_avg"
PROPOSAL_VERSION: Final[str] = "2026-07-21-phase0-frozen"

# Phase 0 允许依赖的源码白名单（相对项目根）。
PHASE0_SOURCE_WHITELIST: Final[tuple[str, ...]] = (
    "src/loop185/__init__.py",
    "src/loop185/contracts.py",
    "src/loop185/hgconv.py",
    "src/loop185/model.py",
    "src/loop185/source_closure.py",
    "src/loop185/data_adapter.py",
    "src/loop185/resource_cell.py",
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
    "src.loop184",
    "pandas",
    "numpy.random.default_rng",
    "sklearn.",
    "xgboost",
    "lightgbm",
    "catboost",
)

# ---------------------------------------------------------------------------
# 输入 ABI（与 Loop184 一致，复用 Loop175 region cache）
# ---------------------------------------------------------------------------

EXPECTED_REGIONS: Final[int] = 16
EXPECTED_REGION_BYTES: Final[int] = 8192
VOCABULARY_SIZE: Final[int] = 257
PADDING_TOKEN: Final[int] = 256

REGION_TYPE_COUNT: Final[int] = 6
BUCKET_COUNT: Final[int] = 64

B0_FEATURE_DIM: Final[int] = 571

# ---------------------------------------------------------------------------
# HGConv 核心冻结值（与 Loop184 一致）
# ---------------------------------------------------------------------------

MODEL_DIM: Final[int] = 192
HGCONV_BLOCKS: Final[int] = 2
HGCONV_FILTER_LENGTH: Final[int] = 32
PATCH_SIZE: Final[int] = 16

MULTI_SCALE_FILTER_LENGTHS: Final[tuple[int, ...]] = (8, 16, 32, 64)

PATCH_SEQUENCE_LENGTH: Final[int] = EXPECTED_REGION_BYTES // PATCH_SIZE

# ---------------------------------------------------------------------------
# Transformer region aggregator 冻结值（与 Loop184 一致）
# ---------------------------------------------------------------------------

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
# Loop185 数据增强超参（与 Loop183/184 一致）
# ---------------------------------------------------------------------------

MIXUP_ALPHA: Final[float] = 0.4
REGION_DROPOUT_PROB: Final[float] = 0.3
LABEL_SMOOTHING_EPS: Final[float] = 0.05

# ---------------------------------------------------------------------------
# Loop185 SWA 超参（调整：8 epochs → start at 6）
# ---------------------------------------------------------------------------

# SWA：在 75% epoch 启动（8 epochs → epoch 6 开始，平均最后 3 个 epoch）
SWA_START_EPOCH: Final[int] = 6
SWA_LR: Final[float] = 3.0e-5
SWA_ANNEAL_EPOCHS: Final[int] = 2

# ---------------------------------------------------------------------------
# 4-fold OOF 配置
# ---------------------------------------------------------------------------

# 4-fold OOF：fold 1,2,3,4 互相 OOF，fold 0 保持 forbidden（holdout）
OOF_FOLDS: Final[tuple[int, ...]] = (1, 2, 3, 4)
OOF_FORBIDDEN_FOLD: Final[int] = 0

# 每个 OOF fold 的 (fit_folds, selection_fold) 配置
OOF_FOLD_CONFIGS: Final[tuple[tuple[tuple[int, ...], int], ...]] = (
    ((2, 3, 4), 1),  # fold 1 as selection
    ((1, 3, 4), 2),  # fold 2 as selection
    ((1, 2, 4), 3),  # fold 3 as selection
    ((1, 2, 3), 4),  # fold 4 as selection
)

# ---------------------------------------------------------------------------
# Phase A 资源门阈值
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PhaseAResourceGate:
    """Phase A 资源门阈值（4-fold OOF 版本）。

    Loop185 改动（vs Loop184）：
    - max_epochs: 10 → 8（无 SAM 加速，但仍保持足够训练）
    - swa_start_epoch: 7 → 6
    - 移除 sam_rho 字段（关闭 SAM）
    - wall_seconds 保持 21600（6h），4 fold × 1.25h = 5h + 评估开销
    """

    fit_rows: int = 12_000  # 每 fold 12000 fit
    selection_rows: int = 4_000  # 每 fold 4000 selection
    fold0_model_rows: int = 0  # fold 0 forbidden

    max_epochs: int = 8  # Loop184: 10 → Loop185: 8
    microbatch: int = 2
    accumulation: int = 16
    effective_batch: int = 32
    learning_rate: float = 3.0e-4
    weight_decay: float = 3.0e-2
    warmup_steps: int = 1
    grad_clip: float = 1.0
    ema_decay: float = 0.999  # 保留字段以兼容接口，实际不使用

    # SWA 配置
    swa_start_epoch: int = 6
    swa_lr: float = 3.0e-5
    swa_anneal_epochs: int = 2

    autocast_dtype: str = "bfloat16"
    master_dtype: str = "float32"
    fft_master_dtype: str = "float32"

    # 4-fold OOF 总预算：6h × 4 fold = 24h？不，是单 fold 6h 预算
    # 但 4 fold 串行需要在 6h 内完成，所以 wall_seconds 是总预算
    gpu_allocated_bytes: int = 6_500_000_000
    rss_bytes: int = 11_000_000_000
    wall_seconds: int = 21_600  # 6h 总预算（4 fold 串行）

    silent_drop_rows: int = 0
    all_rows_accounted: bool = True
    oom: bool = False
    timeout: bool = False
    nonfinite: bool = False
    bitwise_deterministic_eval: bool = True


@dataclass(frozen=True)
class PhaseBGate:
    """Phase B J 臂晋级门（与 Loop184 一致）。"""

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
    # SWA 超参验证
    assert SWA_START_EPOCH > 0, "SWA_START_EPOCH must be positive"
    assert SWA_START_EPOCH < PHASE_A_GATE.max_epochs, "SWA_START_EPOCH must be < max_epochs"
    assert SWA_LR > 0.0, "SWA_LR must be positive"
    assert SWA_ANNEAL_EPOCHS > 0, "SWA_ANNEAL_EPOCHS must be positive"
    # 4-fold OOF 配置验证
    assert len(OOF_FOLDS) == 4, "OOF_FOLDS must have 4 folds"
    assert OOF_FORBIDDEN_FOLD not in OOF_FOLDS, "forbidden fold must not be in OOF folds"
    assert len(OOF_FOLD_CONFIGS) == 4, "OOF_FOLD_CONFIGS must have 4 entries"
    for fit_folds, selection_fold in OOF_FOLD_CONFIGS:
        assert selection_fold in OOF_FOLDS, "selection fold must be in OOF_FOLDS"
        assert selection_fold not in fit_folds, "selection fold must not be in fit folds"
        assert len(fit_folds) == 3, "each OOF config must have 3 fit folds"
        assert OOF_FORBIDDEN_FOLD not in fit_folds, "forbidden fold must not be in fit folds"
    # 架构验证
    assert HGCONV_BLOCKS >= 1, "HGCONV_BLOCKS must be at least 1"
    assert TRANSFORMER_LAYERS >= 1, "TRANSFORMER_LAYERS must be at least 1"
