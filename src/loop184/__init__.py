"""Loop184 包：架构扩展 + SAM + SWA。

Loop184 = Loop183 数据增强基础 + 4 项激进改进：
1. 架构扩展：transformer_layers 2→4, hgconv_blocks 1→2（参数量 +33%）
2. SAM 优化器（rho=0.05）：找平坦极小值，每个 effective batch 应用
3. SWA（75% epoch 启动）：平均最后 5 个 epoch 权重，替代 EMA
4. 保持 Loop183 数据增强：Mixup α=0.4, Region dropout p=0.3, Label smoothing ε=0.05
"""

from __future__ import annotations

from .contracts import (
    B0_FEATURE_DIM,
    B0_PROJECTED_DIM,
    BUCKET_COUNT,
    DROPOUT,
    EXPECTED_REGION_BYTES,
    EXPECTED_REGIONS,
    FORBIDDEN_IMPORT_PATTERNS,
    HGCONV_BLOCKS,
    HGCONV_FILTER_LENGTH,
    LABEL_SMOOTHING_EPS,
    LINEAGE,
    LOOP_ID,
    MIXUP_ALPHA,
    MODEL_DIM,
    MULTI_SCALE_FILTER_LENGTHS,
    NUMERIC_TOLERANCE,
    PADDING_TOKEN,
    PATCH_SEQUENCE_LENGTH,
    PATCH_SIZE,
    PHASE0_SOURCE_WHITELIST,
    PHASE_A_GATE,
    PHASE_B_GATE,
    PROPOSAL_VERSION,
    REGION_DROPOUT_PROB,
    REGION_TYPE_COUNT,
    SAM_RHO,
    SWA_ANNEAL_EPOCHS,
    SWA_LR,
    SWA_START_EPOCH,
    TRANSFORMER_FFN_DIM,
    TRANSFORMER_HEADS,
    TRANSFORMER_LAYERS,
    VOCABULARY_SIZE,
    NumericTolerance,
    PhaseAResourceGate,
    PhaseBGate,
    assert_contract_invariants,
)
from .data_adapter import (
    CANONICAL_B0_CACHE_RELATIVE_PATH,
    CANONICAL_FOLD_MANIFEST_RELATIVE_PATH,
    CANONICAL_FOLD_MANIFEST_SHA256,
    CANONICAL_REGION_CACHE_RELATIVE_PATH,
    CANONICAL_REGION_CACHE_SHA256,
    FULL_TRAIN_ROWS,
    PHASE_A_FIT_FOLDS,
    PHASE_A_FOLD_SPLIT,
    PHASE_A_FORBIDDEN_FOLDS,
    PHASE_A_SELECTION_FOLD,
    PhaseADataLoader,
    RegionBatch,
    ROWS_PER_FOLD,
    make_synthetic_batch,
)
from .hgconv import (
    HGConvBlock,
    HGConvConfig,
    MultiScaleHGConvBlock,
    MultiScaleHGConvConfig,
    approximate_inverse,
    circular_convolution,
    malware_kernel_precondition,
)
from .model import HGConvRegionConfig, HGConvRegionNet, parameter_count
from .resource_cell import (
    IntegrityGateResult,
    ResourceCell,
    ResourceSample,
    ResourceViolation,
    assert_bitwise_deterministic,
    assert_budget_invariants,
    check_integrity,
)
from .source_closure import (
    ClosureReport,
    ClosureViolation,
    assert_phase0_closure,
    build_current_manifest,
    scan_source_closure,
)

__all__ = [
    # 身份
    "LOOP_ID",
    "LINEAGE",
    "PROPOSAL_VERSION",
    # 源码闭包
    "PHASE0_SOURCE_WHITELIST",
    "FORBIDDEN_IMPORT_PATTERNS",
    # 输入 ABI
    "EXPECTED_REGIONS",
    "EXPECTED_REGION_BYTES",
    "VOCABULARY_SIZE",
    "PADDING_TOKEN",
    "REGION_TYPE_COUNT",
    "BUCKET_COUNT",
    "B0_FEATURE_DIM",
    # HGConv 核心冻结值
    "MODEL_DIM",
    "HGCONV_BLOCKS",
    "HGCONV_FILTER_LENGTH",
    "PATCH_SIZE",
    "MULTI_SCALE_FILTER_LENGTHS",
    "PATCH_SEQUENCE_LENGTH",
    # Transformer region aggregator 冻结值
    "TRANSFORMER_LAYERS",
    "TRANSFORMER_HEADS",
    "TRANSFORMER_FFN_DIM",
    "DROPOUT",
    # 输出 ABI
    "REGION_FEATURE_DIM",
    "REGION_LOGIT_DIM",
    "FUSION_LOGIT_DIM",
    "B0_PROJECTED_DIM",
    # 数据增强超参
    "MIXUP_ALPHA",
    "REGION_DROPOUT_PROB",
    "LABEL_SMOOTHING_EPS",
    # SAM + SWA 超参
    "SAM_RHO",
    "SWA_START_EPOCH",
    "SWA_LR",
    "SWA_ANNEAL_EPOCHS",
    # Phase gates
    "PHASE_A_GATE",
    "PHASE_B_GATE",
    "NUMERIC_TOLERANCE",
    "PhaseAResourceGate",
    "PhaseBGate",
    "NumericTolerance",
    "assert_contract_invariants",
    # 数据加载器
    "PhaseADataLoader",
    "RegionBatch",
    "make_synthetic_batch",
    "PHASE_A_FOLD_SPLIT",
    "PHASE_A_FIT_FOLDS",
    "PHASE_A_SELECTION_FOLD",
    "PHASE_A_FORBIDDEN_FOLDS",
    "ROWS_PER_FOLD",
    "FULL_TRAIN_ROWS",
    "CANONICAL_REGION_CACHE_RELATIVE_PATH",
    "CANONICAL_REGION_CACHE_SHA256",
    "CANONICAL_FOLD_MANIFEST_RELATIVE_PATH",
    "CANONICAL_FOLD_MANIFEST_SHA256",
    "CANONICAL_B0_CACHE_RELATIVE_PATH",
    # HGConv
    "HGConvConfig",
    "HGConvBlock",
    "MultiScaleHGConvConfig",
    "MultiScaleHGConvBlock",
    "circular_convolution",
    "approximate_inverse",
    "malware_kernel_precondition",
    # 模型
    "HGConvRegionConfig",
    "HGConvRegionNet",
    "parameter_count",
    # 资源门
    "ResourceCell",
    "ResourceSample",
    "ResourceViolation",
    "IntegrityGateResult",
    "check_integrity",
    "assert_bitwise_deterministic",
    "assert_budget_invariants",
    # 源码闭包
    "ClosureReport",
    "ClosureViolation",
    "scan_source_closure",
    "build_current_manifest",
    "assert_phase0_closure",
]
