"""Loop179 冻结契约：ABI 形状、资源门阈值、训练超参、四臂门。

本模块只定义常量与 dataclass，不导入 torch、不访问真实数据。
任何对冻结值的偏离必须在 Phase 0 静态检查或 Phase A preflight 中被拒绝。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

# ---------------------------------------------------------------------------
# 顶层身份与源码闭包
# ---------------------------------------------------------------------------

LOOP_ID: Final[str] = "Loop179"
LINEAGE: Final[str] = "hgconv_region"
PROPOSAL_VERSION: Final[str] = "2026-07-20-phase0-frozen"

# Phase 0 允许依赖的源码白名单（相对项目根）。
# 任何白名单外的源码引入都会被 source_closure 拒绝。
PHASE0_SOURCE_WHITELIST: Final[tuple[str, ...]] = (
    "src/loop179/__init__.py",
    "src/loop179/contracts.py",
    "src/loop179/hgconv.py",
    "src/loop179/model.py",
    "src/loop179/source_closure.py",
    "src/loop179/data_adapter.py",
    "src/loop179/resource_cell.py",
)

# Phase 0 不允许出现的导入符号（防止顺手引入真实数据路径）。
FORBIDDEN_IMPORT_PATTERNS: Final[tuple[str, ...]] = (
    "src.loop151",
    "src.loop164",
    "src.loop175",
    "src.loop166",
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
# HGConv 核心冻结值
# ---------------------------------------------------------------------------

MODEL_DIM: Final[int] = 192
HGCONV_BLOCKS: Final[int] = 1
HGCONV_FILTER_LENGTH: Final[int] = 32
PATCH_SIZE: Final[int] = 16

# 单 region patch 序列长度 = 8192 / 16 = 512
PATCH_SEQUENCE_LENGTH: Final[int] = EXPECTED_REGION_BYTES // PATCH_SIZE

# ---------------------------------------------------------------------------
# Transformer region aggregator 冻结值
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
# Phase A 资源门阈值
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PhaseAResourceGate:
    """Phase A 资源门阈值，任何超限必须 fail-fast。"""

    # 数据口径：仅 Train outer-fit folds 2/3/4，不触碰 Val/Test-10k/full-test
    fit_rows: int = 12_000
    selection_rows: int = 4_000
    fold0_model_rows: int = 0  # Phase A 不训练 fold0 模型

    # 训练超参
    max_epochs: int = 12
    microbatch: int = 2
    accumulation: int = 16
    effective_batch: int = 32  # = microbatch * accumulation
    learning_rate: float = 3.0e-4
    weight_decay: float = 1.0e-2
    warmup_steps: int = 1
    grad_clip: float = 1.0
    ema_decay: float = 0.999

    # 精度合同：BF16 autocast + FP32 optimizer/norm/loss/FFT
    autocast_dtype: str = "bfloat16"
    master_dtype: str = "float32"
    fft_master_dtype: str = "float32"

    # 资源上限（与 Loop175 seed41 合同对齐，留 5% 安全边际）
    gpu_allocated_bytes: int = 6_500_000_000  # 6.5 GiB（Loop175 上限 6.98 GiB）
    rss_bytes: int = 11_000_000_000  # 11 GiB（Loop175 上限 11.81 GiB）
    wall_seconds: int = 21_600  # 6h（Loop175 上限 6h）

    # 完整性门
    silent_drop_rows: int = 0
    all_rows_accounted: bool = True
    oom: bool = False
    timeout: bool = False
    nonfinite: bool = False
    bitwise_deterministic_eval: bool = True


# ---------------------------------------------------------------------------
# Phase B 四臂门
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PhaseBGate:
    """Phase B J 臂晋级门，任一失败关闭 HGConv-Region 路线。"""

    # J 相对 A 的因果收益门
    j_net_fewer_errors_vs_a: int = 30
    j_repairs_vs_a: int = 50
    j_override_precision: float = 0.80
    j_net_positive_folds: int = 4  # / 5
    j_bootstrap_lcb_vs_a: float = 0.0  # one-sided 95% LCB > 0

    # FP/FN 恶化门
    fp_relative_worsening: float = 0.05
    fn_relative_worsening: float = 0.05

    # K 臂反事实门（证明收益来自区域归属）
    k_more_errors_vs_j: int = 30
    k_bootstrap_lcb_vs_j: float = 0.0


# ---------------------------------------------------------------------------
# 浮点容差
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class NumericTolerance:
    """Phase 0 数值验证容差。"""

    float64_rtol: float = 1.0e-12
    float64_atol: float = 1.0e-12
    float32_rtol: float = 1.0e-5
    float32_atol: float = 1.0e-6
    fft_norm_eps: float = 1.0e-7


# ---------------------------------------------------------------------------
# 导出冻结实例
# ---------------------------------------------------------------------------

PHASE_A_GATE: Final[PhaseAResourceGate] = PhaseAResourceGate()
PHASE_B_GATE: Final[PhaseBGate] = PhaseBGate()
NUMERIC_TOLERANCE: Final[NumericTolerance] = NumericTolerance()


# ---------------------------------------------------------------------------
# 契约自检
# ---------------------------------------------------------------------------

def assert_contract_invariants() -> None:
    """启动时自检冻结常量的一致性，防止手误修改。"""

    # patch 序列必须长于 HGConv filter
    assert PATCH_SEQUENCE_LENGTH > HGCONV_FILTER_LENGTH, (
        "patch sequence must be longer than HGConv filter"
    )
    # model_dim 必须能被 transformer heads 整除
    assert MODEL_DIM % TRANSFORMER_HEADS == 0, "model_dim must be divisible by heads"
    # region bytes 必须能被 patch size 整除
    assert EXPECTED_REGION_BYTES % PATCH_SIZE == 0, "region bytes must be divisible by patch size"
    # padding token 必须是词表最后一项
    assert PADDING_TOKEN == VOCABULARY_SIZE - 1, "padding token must be last vocab item"
    # effective batch 必须 = microbatch * accumulation
    assert (
        PHASE_A_GATE.effective_batch
        == PHASE_A_GATE.microbatch * PHASE_A_GATE.accumulation
    ), "effective batch must equal microbatch * accumulation"
    # fit + selection 必须 = 16000（Loop175 region cache 全量 Train 行）
    assert (
        PHASE_A_GATE.fit_rows + PHASE_A_GATE.selection_rows == 16_000
    ), "fit + selection must cover 16000 Train rows"
    # Phase A 不允许触碰 fold0
    assert PHASE_A_GATE.fold0_model_rows == 0, "Phase A must not train fold0 model"
