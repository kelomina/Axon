"""Axon v2.6 实验配置模块。

定义模型、数据集和训练的配置文件。
所有可调参数统一在此管理，避免各模块硬编码魔法数字。
"""

from dataclasses import dataclass, field
from typing import Optional, List, Tuple
from pathlib import Path
import platform


@dataclass
class AxonExperimentConfig:
    """Axon v2.6 实验配置"""

    # ==================== 数据配置 ====================
    max_byte_length: int = 65536
    pe_feature_dim: int = 1500
    stat_feature_dim: int = 49
    batch_size: int = 16  # backward compat: TrainingConfig.batch_size takes precedence

    # ==================== 模型架构配置 ====================
    byte_embedding_dim: int = 128
    use_byte_embedding: bool = True
    vocab_size: int = 256
    max_pos_len: int = 65536

    # DSRA 配置
    dsra_dim: int = 128
    dsra_heads: int = 4
    dsra_slots: int = 128
    dsra_read_topk: int = 8
    dsra_write_topk: int = 4
    dsra_local_window: int = 256
    dsra_chunk_size: int = 512
    byte_chunk_pooling: str = "last"
    dsra_arch_config: Optional['DSRAArchitectureConfig'] = None

    # PE 特征处理
    pe_projection_dim: int = 128
    pe_projector_hidden_dim: int = 256
    pe_projector_num_hidden_layers: int = 0

    # 分类器
    classifier_hidden_dim: int = 64

    # 融合配置
    fusion_type: str = "concat"
    fusion_num_heads: int = 4
    dropout: float = 0.1

    # 位置编码
    pos_encoding_mode: str = "sinusoidal"
    use_pos_encoding: bool = True

    # ==================== 分类配置 ====================
    num_classes: int = 2

    # ==================== 数据验证配置 ====================
    max_file_size: int = 1 * 1024 * 1024 * 1024  # 1GB

    # ==================== 标签推断配置 ====================
    malicious_keywords: List[str] = field(default_factory=lambda: ["malware", "malicious", "virus", "trojan", "ransomware", "spyware", "adware", "worm", "backdoor", "rootkit", "keylogger", "botnet", "exploit", "dropper", "loader", "miner"])
    benign_keywords: List[str] = field(default_factory=lambda: ["benign", "clean", "safe", "legitimate", "good", "normal", "harmless"])
    malicious_dir_names: List[str] = field(default_factory=lambda: ["malware", "malicious", "virus", "trojan", "ransomware", "spyware", "adware", "worm", "backdoor", "rootkit", "samples", "dirty"])
    benign_dir_names: List[str] = field(default_factory=lambda: ["benign", "clean", "safe", "legitimate", "good", "normal", "harmless", "white"])

    # ==================== 特征提取配置 ====================
    extraction_max_file_size: int = 65536
    byte_histogram_bins: int = 256
    stat_chunk_count: int = 10
    entropy_block_size: int = 4096
    entropy_sample_size: int = 4096
    size_norm_max: int = 100 * 1024 * 1024
    timestamp_year_base: int = 1970
    timestamp_year_max: int = 2099
    large_trailing_data_size: int = 1024 * 1024
    entropy_high_threshold: float = 0.8
    section_entropy_min_size: int = 256
    overlay_entropy_min_size: int = 256
    lightweight_feature_dim: int = 256

    # PE 特征提取详细配置
    pe_schema_version: str = "legacy_dynamic"
    pe_fixed_section_slots: int = 32
    common_section_names: List[str] = field(default_factory=lambda: [".text", ".data", ".rdata", ".rsrc", ".idata", ".edata", ".bss", ".reloc", ".tls", ".gfids", ".00cfg"])
    packer_keywords: List[str] = field(default_factory=lambda: ["upx", "themida", "vmprotect", "aspack", "mpress", "pecompact", "obsidium", "enigma", "packed"])
    system_dlls: List[str] = field(default_factory=lambda: ["kernel32.dll", "user32.dll", "advapi32.dll", "shell32.dll", "ole32.dll", "oleaut32.dll", "msvcrt.dll", "ntdll.dll", "ws2_32.dll", "wininet.dll", "urlmon.dll", "crypt32.dll", "secur32.dll", "netapi32.dll", "dnsapi.dll", "iphlpapi.dll", "gdi32.dll", "comdlg32.dll", "comctl32.dll", "shlwapi.dll", "version.dll", "setupapi.dll", "imm32.dll", "midimap.dll", "msacm32.dll", "ddraw.dll", "dinput.dll", "dsound.dll"])
    api_categories: dict = field(default_factory=lambda: {
        "network": ["internet", "http", "socket", "connect", "recv", "send", "url", "download", "upload", "proxy", "wsa", "ftp", "smtp"],
        "process": ["createprocess", "openprocess", "virtualalloc", "virtualprotect", "writeprocessmemory", "readprocessmemory", "createremotethread", "shellexecute", "winexec", "loadlibrary", "getprocaddress"],
        "filesystem": ["createfile", "readfile", "writefile", "deletefile", "movefile", "copyfile", "getfilesize", "setfilepointer", "findfirstfile", "findnextfile", "gettemppath"],
        "registry": ["regopenkey", "regsetvalue", "regcreatekey", "regdeletekey", "regqueryvalue", "regclosekey", "savekey", "restorekey"],
        "crypto": ["cryptencrypt", "cryptdecrypt", "cryptderivekey", "cryptgenkey", "cryptcreatehash", "crypthashdata", "cryptsignhash", "cryptverify"],
        "injection": ["createremotethread", "virtualallocex", "writeprocessmemory", "readprocessmemory", "queueuserapc", "setwindowshookex", "rtlcreateuserthread", "ntcreatethreadex"],
    })
    fallback_max_read_size: int = 1 * 1024 * 1024
    section_data_max_size: int = 10 * 1024 * 1024
    section_size_ratio_threshold: float = 2.0
    section_size_half_ratio: float = 0.5
    ascii_printable_min: int = 32
    ascii_printable_max: int = 127
    stat_segment_count: int = 3
    lightweight_hash_mod: int = 128
    lightweight_hash_offset: int = 128
    lightweight_section_hash_mod: int = 32
    lightweight_section_hash_offset: int = 224
    lightweight_read_size: int = 65536
    lightweight_dll_patterns: List[str] = field(default_factory=lambda: ["kernel32.dll", "user32.dll", "ntdll.dll", "advapi32.dll", "ws2_32.dll", "wininet.dll", "ole32.dll", "shell32.dll", "msvcrt.dll", "msvcrtd.dll", "vcruntime.dll", "ucrtbase.dll"])
    lightweight_api_patterns: List[str] = field(default_factory=lambda: ["VirtualAlloc", "VirtualProtect", "CreateRemoteThread", "WriteProcessMemory", "ReadProcessMemory", "WinExec", "ShellExecute", "LoadLibrary", "GetProcAddress", "CreateProcess", "InternetOpen", "InternetReadFile", "URLDownloadToFile", "CreateFile", "RegOpenKey"])
    lightweight_section_patterns: List[str] = field(default_factory=lambda: [".text", ".data", ".rdata", ".rsrc", ".reloc", ".code", ".idata", ".edata", ".tls", ".bss"])
    api_prefix_only_keywords: List[str] = field(default_factory=lambda: ["connect", "send", "recv"])
    strict_pe_parsing: bool = True
    allow_pe_fallback: bool = False

    # ==================== 路径配置 ====================
    data_dir: Optional[str] = None
    cache_dir: Optional[str] = None
    model_save_dir: str = "models"
    log_dir: str = "reports/logs"
    benign_dir_names_fs: List[str] = field(default_factory=lambda: ["benign", "待加入白名单"])
    malicious_dir_names_fs: List[str] = field(default_factory=lambda: ["malicious", "待拉黑"])
    follow_symlinks: bool = False
    allowed_symlink_roots: List[str] = field(default_factory=list)
    strict_unknown_labels: bool = True

    # ==================== 实验配置 ====================
    experiment_name: str = "axon_v2.6_exp"
    seed: int = 42
    use_wandb: bool = False
    device: str = "cuda"

    # ==================== 快速训练模式 ====================
    fast_mode: bool = False
    fast_mode_samples: int = 5000
    fast_mode_epochs: int = 8
    fast_mode_byte_length: int = 512

    # ==================== 数据增强配置 ====================
    augmentation: Optional['DataAugmentationConfig'] = None

    # ==================== 数据拆分配置 ====================
    val_ratio: float = 0.04
    test_ratio: float = 0.8

    # ==================== 评估配置 ====================
    eval_interval: int = 1  # backward compat: TrainingConfig.eval_interval takes precedence
    save_best_only: bool = True

    # ==================== 辅助函数 ====================
    def __post_init__(self):
        """配置后处理"""
        if self.model_save_dir:
            self.model_save_dir = Path(self.model_save_dir)
        if self.log_dir:
            self.log_dir = Path(self.log_dir)

        if self.dsra_dim % self.dsra_heads != 0:
            raise ValueError(f"dsra_dim ({self.dsra_dim}) must be divisible by dsra_heads ({self.dsra_heads})")

        if self.fusion_type not in ["concat", "add", "attention", "gated", "residual_stat_gate", "residual_channel_gate"]:
            raise ValueError(f"Unknown fusion_type: {self.fusion_type}")
        if self.byte_chunk_pooling not in {"last", "mean", "active_mean", "active_mean_detached"}:
            raise ValueError(f"Unknown byte_chunk_pooling: {self.byte_chunk_pooling}")

        expected_stat_feature_dim = self.expected_stat_feature_dim()
        if self.stat_feature_dim != expected_stat_feature_dim:
            raise ValueError(
                f"stat_feature_dim ({self.stat_feature_dim}) must match stat_segment_count "
                f"and stat_chunk_count derived dimension ({expected_stat_feature_dim})"
            )
        if self.val_ratio < 0 or self.test_ratio < 0:
            raise ValueError("val_ratio and test_ratio must be non-negative")
        if self.val_ratio + self.test_ratio >= 1:
            raise ValueError("val_ratio + test_ratio must be less than 1")
        if self.pe_schema_version not in ["legacy_dynamic", "fixed_v2"]:
            raise ValueError(f"Unknown pe_schema_version: {self.pe_schema_version}")
        if self.pe_fixed_section_slots <= 0:
            raise ValueError("pe_fixed_section_slots must be positive")
        if self.pe_schema_version == "fixed_v2" and self.pe_feature_dim < self.fixed_pe_schema_used_dim():
            raise ValueError(
                f"pe_feature_dim ({self.pe_feature_dim}) must be at least "
                f"{self.fixed_pe_schema_used_dim()} for fixed_v2 PE schema"
            )

    def expected_stat_feature_dim(self):
        return 7 + 4 + 1 + 3 * self.stat_segment_count + 2 * self.stat_chunk_count + 8

    def fixed_pe_schema_used_dim(self):
        return 18 + 3 * self.pe_fixed_section_slots + 29

    def get_device(self):
        """获取计算设备"""
        import torch
        if self.device == "cuda" and torch.cuda.is_available():
            return torch.device("cuda:0")
        return torch.device("cpu")

    def to_dict(self):
        """转换为字典"""
        result = {}
        for key, value in self.__dict__.items():
            if isinstance(value, Path):
                result[key] = str(value)
            elif hasattr(value, '__dataclass_fields__'):
                result[key] = value.to_dict() if hasattr(value, 'to_dict') else str(value)
            else:
                result[key] = value
        return result

    @classmethod
    def from_dict(cls, config_dict):
        """从字典创建配置"""
        if 'model_save_dir' in config_dict:
            config_dict['model_save_dir'] = Path(config_dict['model_save_dir'])
        if 'log_dir' in config_dict:
            config_dict['log_dir'] = Path(config_dict['log_dir'])
        if 'dsra_arch_config' in config_dict and isinstance(config_dict['dsra_arch_config'], dict):
            config_dict['dsra_arch_config'] = DSRAArchitectureConfig.from_dict(config_dict['dsra_arch_config'])
        if 'augmentation' in config_dict and isinstance(config_dict['augmentation'], dict):
            config_dict['augmentation'] = DataAugmentationConfig(**config_dict['augmentation'])
        return cls(**{k: v for k, v in config_dict.items() if k in cls.__dataclass_fields__})


@dataclass
class DSRAArchitectureConfig:
    """DSRA 架构特定配置"""

    # 基础配置
    dim: int = 128
    heads: int = 4
    slots: int = 128
    read_topk: int = 8
    write_topk: int = 4
    local_window: int = 256

    # 位置编码
    pe_mode: str = "rope"

    # 可选机制
    use_local: bool = True
    use_retrieval: bool = False
    use_context_film: bool = False
    momentum_qkv: bool = False
    slot_pe: str = "rope"
    max_contexts: int = 8
    momentum_decay: float = 0.9999

    # 路由策略
    hard_write: bool = False
    hard_read: bool = False
    exact_write: bool = False
    exact_read: bool = False

    # 温度参数
    tau_init: float = 8.0
    read_tau_max: float = 64.0
    tau_write_init: float = 4.0
    retrieval_tau: float = 8.0

    # 遗忘机制
    forget_base: float = 0.001
    forget_conflict: float = 0.20
    forget_age: float = 0.0002

    # 使用 decay
    usage_decay: float = 0.995
    conf_decay: float = 0.999
    usage_prior: float = 0.25

    # 写入策略
    write_frequency: int = 1
    novelty_threshold: float = 0.0
    write_protection: int = 0
    write_gate_min: float = 0.2
    eta: float = 0.25
    max_update: float = 0.50
    exact_write_gate: float = 1.0
    eps: float = 1e-6

    # 偏差参数
    age_write_bias: float = 0.02
    conf_read_bias: float = 0.50
    age_read_penalty: float = 0.005
    conflict_protection_coef: float = 0.3

    # 初始化参数
    write_gate_bias_init: float = 3.0
    write_gate_weight_std: float = 0.01
    init_confidence: float = 0.01

    # 约束参数
    max_v_norm: float = 10.0
    age_reset_threshold: float = 0.1
    write_mass_threshold_multiplier: float = 10.0
    forget_max: float = 0.95
    write_tau_max: float = 64.0

    # 多层模型参数
    num_layers: int = 2
    detach_state: bool = True

    # 兼容层参数
    compat_K: int = 512
    compat_kr: int = 16
    compat_eta: float = 0.1
    compat_decay_lambda: float = 0.01
    compat_time_decay_alpha: float = 0.01
    compat_confidence_init: float = 0.5

    # 分页精确记忆参数
    paged_memory_page_size: int = 1024
    paged_memory_top_pages: int = 4
    paged_memory_max_tokens: int = 128
    paged_memory_max_pages: Optional[int] = None

    def to_dict(self):
        from dataclasses import asdict
        return asdict(self)

    @classmethod
    def from_dict(cls, d):
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def __post_init__(self):
        if self.slot_pe == "rope" and self.pe_mode in {"none", "rope"}:
            self.slot_pe = self.pe_mode


@dataclass
class TrainingConfig:
    """训练特定配置"""

    # 优化器
    optimizer: str = "adamw"
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    betas: Tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    sgd_momentum: float = 0.9

    # 学习率调度
    lr_scheduler: str = "cosine"
    warmup_epochs: int = 3
    warmup_start_lr: float = 1e-6
    min_lr: float = 1e-6
    step_lr_size: int = 10
    step_lr_gamma: float = 0.1

    # 梯度
    gradient_clip: float = 1.0
    mixed_precision: bool = False

    # 训练策略
    max_epochs: int = 50
    early_stopping_patience: int = 5
    early_stopping_min_delta: float = 0.0
    eval_interval: int = 1

    # 批次
    batch_size: int = 16
    num_workers: Optional[int] = None
    pin_memory: bool = True

    # 损失函数
    label_smoothing: float = 0.0
    focal_gamma: float = 0.0
    focal_alpha: float = 0.25
    diversity_loss_weight: float = 0.05

    # 小族群加权训练。只影响训练 loss，验证/测试仍按普通指标计算。
    rare_group_weighting: bool = False
    singleton_group_weight: float = 1.8
    rare_group_weight: float = 1.5
    medium_group_weight: float = 1.2

    # 二分类判定阈值。低于 0.5 会更偏向召回恶意样本，但可能增加误报。
    decision_threshold: float = 0.5

    # 日志
    log_interval: int = 10

    # 检查点
    best_model_filename: str = "best_model.pt"
    final_model_filename: str = "final_model.pt"
    swanlab_project: str = "Axon-v2.6"
    enable_swanlab: bool = False

    # SWA (Stochastic Weight Averaging)
    use_swa: bool = False
    swa_start_epoch: int = -10      # 负数表示从 max_epochs 倒数
    swa_lr: float = 1e-5

    # EMA (Exponential Moving Average)
    use_ema: bool = False
    ema_decay: float = 0.999

    # 近阈值样本加权
    near_threshold_weight: float = 1.0   # >1.0 时启用
    near_threshold_low: float = 0.35
    near_threshold_high: float = 0.65

    def __post_init__(self):
        self.optimizer = self.optimizer.lower()
        self.lr_scheduler = self.lr_scheduler.lower()
        if self.num_workers is None:
            self.num_workers = 0 if platform.system().lower() == "windows" else 4

        if self.optimizer not in {"adam", "adamw", "sgd"}:
            raise ValueError(f"Unknown optimizer: {self.optimizer}")
        if self.lr_scheduler not in {"none", "cosine", "step"}:
            raise ValueError(f"Unknown lr_scheduler: {self.lr_scheduler}")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.weight_decay < 0:
            raise ValueError("weight_decay must be non-negative")
        if self.eps <= 0:
            raise ValueError("eps must be positive")
        if len(self.betas) != 2 or not (0 <= self.betas[0] < 1) or not (0 <= self.betas[1] < 1):
            raise ValueError("betas must contain two values in [0, 1)")
        if self.sgd_momentum < 0:
            raise ValueError("sgd_momentum must be non-negative")
        if self.max_epochs <= 0:
            raise ValueError("max_epochs must be positive")
        if self.warmup_epochs < 0:
            raise ValueError("warmup_epochs must be non-negative")
        if self.lr_scheduler == "cosine" and self.warmup_epochs >= self.max_epochs:
            raise ValueError("warmup_epochs must be smaller than max_epochs when using cosine scheduler")
        if self.warmup_start_lr <= 0:
            raise ValueError("warmup_start_lr must be positive")
        if self.min_lr < 0:
            raise ValueError("min_lr must be non-negative")
        if self.step_lr_size <= 0:
            raise ValueError("step_lr_size must be positive")
        if not (0 < self.step_lr_gamma <= 1):
            raise ValueError("step_lr_gamma must be in (0, 1]")
        if self.gradient_clip < 0:
            raise ValueError("gradient_clip must be non-negative")
        if self.early_stopping_patience <= 0:
            raise ValueError("early_stopping_patience must be positive")
        if self.early_stopping_min_delta < 0:
            raise ValueError("early_stopping_min_delta must be non-negative")
        if self.eval_interval <= 0:
            raise ValueError("eval_interval must be positive")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.num_workers < 0:
            raise ValueError("num_workers must be non-negative")
        if not (0 <= self.label_smoothing < 1):
            raise ValueError("label_smoothing must be in [0, 1)")
        if self.focal_gamma < 0:
            raise ValueError("focal_gamma must be non-negative")
        if not (0 <= self.focal_alpha <= 1):
            raise ValueError("focal_alpha must be in [0, 1]")
        if self.diversity_loss_weight < 0:
            raise ValueError("diversity_loss_weight must be non-negative")
        if self.singleton_group_weight <= 0:
            raise ValueError("singleton_group_weight must be positive")
        if self.rare_group_weight <= 0:
            raise ValueError("rare_group_weight must be positive")
        if self.medium_group_weight <= 0:
            raise ValueError("medium_group_weight must be positive")
        if not (0 < self.decision_threshold < 1):
            raise ValueError("decision_threshold must be in (0, 1)")
        if self.log_interval <= 0:
            raise ValueError("log_interval must be positive")
        if self.use_swa and (self.swa_start_epoch == 0 or self.swa_start_epoch < -self.max_epochs):
            raise ValueError("swa_start_epoch must be non-zero and >= -max_epochs")
        if self.swa_lr <= 0:
            raise ValueError("swa_lr must be positive")
        if not (0 < self.ema_decay < 1):
            raise ValueError("ema_decay must be in (0, 1)")
        if self.near_threshold_weight < 1.0:
            raise ValueError("near_threshold_weight must be >= 1.0")
        if not (0 <= self.near_threshold_low < self.near_threshold_high <= 1):
            raise ValueError("near_threshold_low and near_threshold_high must satisfy 0 <= low < high <= 1")


@dataclass
class DataAugmentationConfig:
    """数据增强配置"""

    enable: bool = False

    # 字节级增强
    byte_dropout: float = 0.0
    byte_swap: float = 0.0
    byte_noise: float = 0.0

    # 特征级增强
    feature_noise: float = 0.0
    feature_mask: float = 0.0

    # Mixup
    use_mixup: bool = False
    mixup_alpha: float = 0.2

    # CutMix
    use_cutmix: bool = False
    cutmix_alpha: float = 1.0
