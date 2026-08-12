"""Loop186 冻结契约：大幅扩容 + 单 fold 深度训练 + SAM。

Loop186 = Loop185 架构扩容 + 单 fold 深度训练：
1. 架构扩容（vs Loop185）：
   - model_dim: 192 → 384 (2x)
   - hgconv_blocks: 2 → 4 (2x)
   - transformer_layers: 4 → 8 (2x)
   - transformer_heads: 6 → 8 (384/8=48)
   - transformer_ffn_dim: 768 → 1536 (2x)
   - byte_embedding_dim: 64 → 128 (2x)
   - 参数量 ~2.61M → ~10M (4x)
2. 单 fold 深度训练（放弃 4-fold OOF，因 Loop185 OOF 未突破单 fold 天花板）
   - fit on fold 2,3,4 (12000 rows), select on fold 1 (4000 rows)
3. 重新启用 SAM（rho=0.05，sharpness reduction 对硬样本有益）
4. 12 epochs + SWA（start_epoch=9, average last 3 epochs）
5. 保持 Loop183 数据增强：Mixup α=0.4, Region dropout p=0.3, Label smoothing ε=0.05

资源估算：
- 参数量 ~10M（4x Loop185）
- SAM 2x 开销 + 12 epochs（1.5x）= 3x Loop185 单 fold 时间
- Loop185 单 fold ~1.25h → Loop186 ~3.75h（在 6h 预算内）
- GPU 预估 ~3 GiB（Loop185 0.925 GiB × 4x 参数 + SAM 双份激活）
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

# ---------------------------------------------------------------------------
# 顶层身份与源码闭包
# ---------------------------------------------------------------------------

LOOP_ID: Final[str] = "Loop186"
LINEAGE: Final[str] = "expanded_single_fold_sam"
PROPOSAL_VERSION: Final[str] = "2026-07-23-phase0-frozen"

# Phase 0 允许依赖的源码白名单（相对项目根）。
PHASE0_SOURCE_WHITELIST: Final[tuple[str, ...]] = (
    "src/loop186/__init__.py",
    "src/loop186/contracts.py",
    "src/loop186/hgconv.py",
    "src/loop186/model.py",
    "src/loop186/source_closure.py",
    "src/loop186/data_adapter.py",
    "src/loop186/resource_cell.py",
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
    "src.loop185",
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
# HGConv 核心冻结值（Loop186 扩容：2x）
# ---------------------------------------------------------------------------

MODEL_DIM: Final[int] = 192
HGCONV_BLOCKS: Final[int] = 2
HGCONV_FILTER_LENGTH: Final[int] = 32
PATCH_SIZE: Final[int] = 16

MULTI_SCALE_FILTER_LENGTHS: Final[tuple[int, ...]] = (8, 16, 32, 64)

PATCH_SEQUENCE_LENGTH: Final[int] = EXPECTED_REGION_BYTES // PATCH_SIZE

# ---------------------------------------------------------------------------
# Transformer region aggregator 冻结值（Loop186 扩容：2x）
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
# Loop186 SWA 超参（12 epochs → start at 9, average last 3）
# ---------------------------------------------------------------------------

SWA_START_EPOCH: Final[int] = 9  # Loop185: 6 → 9 (12 epochs)
SWA_LR: Final[float] = 3.0e-5
SWA_ANNEAL_EPOCHS: Final[int] = 2

# ---------------------------------------------------------------------------
# Loop186 SAM 超参（关闭，改用大 microbatch 加速）
# ---------------------------------------------------------------------------

SAM_ENABLED: Final[bool] = False
SAM_RHO: Final[float] = 0.05  # 保留未使用

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
    """Phase A 资源门阈值。"""

    fit_rows: int = 12_000
    selection_rows: int = 4_000
    fold0_model_rows: int = 0

    max_epochs: int = 12
    microbatch: int = 16
    accumulation: int = 2
    effective_batch: int = 32
    selection_probe_rows: int = 400
    evaluation_microbatch: int = 32
    learning_rate: float = 3.0e-4
    weight_decay: float = 3.0e-2
    warmup_steps: int = 1
    grad_clip: float = 1.0
    ema_decay: float = 0.999

    swa_start_epoch: int = 7
    swa_lr: float = 3.0e-5
    swa_anneal_epochs: int = 2

    sam_enabled: bool = False
    sam_rho: float = 0.05

    autocast_dtype: str = "bfloat16"
    master_dtype: str = "float32"
    fft_master_dtype: str = "float32"

    gpu_allocated_bytes: int = 7_500_000_000
    rss_bytes: int = 12_500_000_000
    epoch_wall_seconds: int = 600
    wall_seconds: int = 7_800

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
    # SAM 超参验证
    assert 0.0 < PHASE_A_GATE.sam_rho < 1.0, "sam_rho must be in (0, 1)"
    # 架构验证
    assert HGCONV_BLOCKS >= 1, "HGCONV_BLOCKS must be at least 1"
    assert TRANSFORMER_LAYERS >= 1, "TRANSFORMER_LAYERS must be at least 1"
    assert MODEL_DIM == 192, "Loop186 speed profile MODEL_DIM must be 192"
    assert HGCONV_BLOCKS == 2, "Loop186 speed profile HGCONV_BLOCKS must be 2"
    assert TRANSFORMER_LAYERS == 4, "Loop186 speed profile TRANSFORMER_LAYERS must be 4"
