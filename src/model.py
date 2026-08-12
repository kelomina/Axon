"""Axon v2.6 融合模型模块。

结合 KVD 特征提取和 DSRA 流式注意力的恶意软件检测模型。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict
from dataclasses import dataclass
import dataclasses

try:
    from .config import AxonExperimentConfig, DSRAArchitectureConfig
    from .dsra.mhdsra2 import MultiHeadDSRA2, MHDSRA2Config
except ImportError:
    from config import AxonExperimentConfig, DSRAArchitectureConfig
    from dsra.mhdsra2 import MultiHeadDSRA2, MHDSRA2Config


class PositionalEncoding(nn.Module):
    """位置编码模块
    
    支持：
    - 可学习的位置编码
    - 正弦位置编码（默认）
    """

    def __init__(self, d_model: int, max_len: int = 65536, mode: str = "sinusoidal"):
        super().__init__()
        self.d_model = d_model
        self.mode = mode
        
        if mode == "learnable":
            self.pos_embedding = nn.Embedding(max_len, d_model)
        elif mode == "sinusoidal":
            self.register_buffer('pe', self._create_sinusoidal_encoding(max_len, d_model))
        else:
            self.pe = None
    
    def _create_sinusoidal_encoding(self, max_len: int, d_model: int) -> torch.Tensor:
        """创建正弦位置编码"""
        position = torch.arange(max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model)
        )
        
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[:d_model // 2])
        
        return pe
    
    def forward(self, x: torch.Tensor, offset: int = 0) -> torch.Tensor:
        """
        Args:
            x: [B, seq_len, d_model]
            offset: 该 chunk 在完整序列中的起始位置
        Returns:
            [B, seq_len, d_model]
        """
        seq_len = x.shape[1]
        
        if self.mode == "learnable":
            positions = torch.arange(offset, offset + seq_len, device=x.device).unsqueeze(0)
            return x + self.pos_embedding(positions)
        elif self.mode == "sinusoidal":
            return x + self.pe[offset:offset + seq_len].unsqueeze(0)
        else:
            return x


class ByteEmbedding(nn.Module):
    """字节嵌入层
    
    将字节值 (0-255) 映射到嵌入向量。
    """
    
    def __init__(self, vocab_size: int = 256, embedding_dim: int = 128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        # OOB 诊断开关：clamp 后 oob.any() 恒为 False（死代码），关闭可省掉每 chunk
        # 一次 GPU->CPU 同步（训练每 step 8 次）以及 compile 图断点。
        self.debug_oob = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, seq_len] 字节值 (0-255)
        Returns:
            [B, seq_len, embedding_dim]
        """
        # Clamp 确保在有效范围内
        x = torch.clamp(x, 0, self.vocab_size - 1)
        if x.dtype != torch.long:
            x = x.long()
        if self.training and self.debug_oob:
            oob = (x < 0) | (x >= self.vocab_size)
            if oob.any():
                import warnings
                warnings.warn(f"ByteEmbedding: {(oob).sum().item()} values out of [0, {self.vocab_size}) range, clamped")
        return self.embedding(x) * (self.embedding_dim ** 0.5)


class PEFeatureProjector(nn.Module):
    """PE 特征投影器
    
    将高维 PE 结构特征投影到低维空间。
    """
    
    def __init__(
        self,
        input_dim: int = 1500,
        hidden_dim: int = 256,
        output_dim: int = 128,
        num_hidden_layers: int = 0,
        dropout: float = 0.1
    ):
        super().__init__()
        
        layers = []
        
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.LayerNorm(hidden_dim))
        layers.append(nn.GELU())
        layers.append(nn.Dropout(dropout))
        
        for _ in range(num_hidden_layers):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
        
        layers.append(nn.Linear(hidden_dim, output_dim))
        
        self.projector = nn.Sequential(*layers)
        self.output_dim = output_dim
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, input_dim] PE 特征
        Returns:
            [B, output_dim]
        """
        return self.projector(x)


class DSRAEncoder(nn.Module):
    """DSRA 序列编码器
    
    使用 MHDSRA2 流式注意力处理字节序列。
    """
    
    def __init__(self, config: DSRAArchitectureConfig):
        super().__init__()
        
        # 创建 MHDSRA2 配置
        mhdsra_fields = {f.name for f in dataclasses.fields(MHDSRA2Config)}
        config_dict = {
            k: getattr(config, k)
            for k in mhdsra_fields
            if hasattr(config, k)
        }
        mhdsra_config = MHDSRA2Config(**config_dict)

        
        num_layers = config.num_layers if hasattr(config, 'num_layers') else 1
        if num_layers > 1:
            self.dsra_layers = nn.ModuleList([MultiHeadDSRA2(mhdsra_config) for _ in range(num_layers)])
            self.dsra = self.dsra_layers[0]
        else:
            self.dsra = MultiHeadDSRA2(mhdsra_config)
            self.dsra_layers = None
        self.config = config
    
    def forward(
        self,
        x: torch.Tensor,
        state=None,
        return_aux: bool = False
    ) -> Tuple[torch.Tensor, Optional[Dict], Optional[any]]:
        if self.dsra_layers is not None:
            if state is None:
                states = [layer.init_state(x.shape[0], device=x.device, dtype=x.dtype) for layer in self.dsra_layers]
            elif isinstance(state, (list, tuple)):
                states = list(state)
            else:
                states = [state] * len(self.dsra_layers)
            out = x
            aux = None
            for i, layer in enumerate(self.dsra_layers):
                if return_aux:
                    out, states[i], layer_aux = layer(out, states[i], return_aux=True)
                    if layer_aux is not None:
                        aux = layer_aux
                else:
                    out, states[i] = layer(out, states[i])
            return out, states, aux
        else:
            if state is None:
                state = self.dsra.init_state(x.shape[0], device=x.device, dtype=x.dtype)
            if return_aux:
                out, next_state, aux = self.dsra(x, state, return_aux=True)
                return out, next_state, aux
            else:
                out, next_state = self.dsra(x, state)
                return out, next_state, None
    
    def init_state(self, batch_size: int, device=None, dtype=None):
        if self.dsra_layers is not None:
            return [layer.init_state(batch_size, device=device, dtype=dtype) for layer in self.dsra_layers]
        return self.dsra.init_state(batch_size, device=device, dtype=dtype)


class MalwareDSRAEncoder(nn.Module):
    """恶意软件 DSRA 编码器
    
    结合字节嵌入、位置编码和 DSRA 流式注意力的编码器。
    """
    
    def __init__(
        self,
        byte_embedding_dim: int = 128,
        max_byte_length: int = 65536,
        dsra_config: Optional[DSRAArchitectureConfig] = None,
        pe_feature_dim: int = 1500,
        pe_projection_dim: int = 128,
        pe_projector_hidden_dim: int = 256,
        use_pos_encoding: bool = True,
        pos_encoding_mode: str = "sinusoidal",
        dropout: float = 0.1,
        chunk_size: int = 512,
        vocab_size: int = 256,
        byte_chunk_pooling: str = "last",
    ):
        super().__init__()
        
        self.byte_embedding_dim = byte_embedding_dim
        self.max_byte_length = max_byte_length
        self.chunk_size = chunk_size
        if byte_chunk_pooling not in {"last", "mean", "active_mean", "active_mean_detached"}:
            raise ValueError(f"Unknown byte_chunk_pooling: {byte_chunk_pooling}")
        self.byte_chunk_pooling = byte_chunk_pooling
        
        # 字节嵌入
        self.byte_embedding = ByteEmbedding(
            vocab_size=vocab_size,
            embedding_dim=byte_embedding_dim
        )
        
        # 位置编码
        self.use_pos_encoding = use_pos_encoding
        if use_pos_encoding:
            self.pos_encoding = PositionalEncoding(
                d_model=byte_embedding_dim,
                max_len=max_byte_length,
                mode=pos_encoding_mode
            )
        
        # 输入投影
        self.input_proj = nn.Sequential(
            nn.Linear(byte_embedding_dim, dsra_config.dim if dsra_config else byte_embedding_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # DSRA 配置
        if dsra_config is None:
            dsra_config = DSRAArchitectureConfig(dim=byte_embedding_dim)
        
        # DSRA 编码器
        self.dsra_encoder = DSRAEncoder(dsra_config)
        
        # PE 特征投影
        self.pe_projector = PEFeatureProjector(
            input_dim=pe_feature_dim,
            hidden_dim=pe_projector_hidden_dim,
            output_dim=pe_projection_dim,
            dropout=dropout
        )
        
        # 输出维度
        self.output_dim = dsra_config.dim + pe_projection_dim
        self._last_diversity_loss = None

    def _primary_dsra(self):
        return self.dsra_encoder.dsra

    def _set_diversity_capture(self, enabled: bool) -> None:
        primary = self._primary_dsra()
        primary._capture_slot_k_before_detach = bool(enabled)
        if not enabled and hasattr(primary, '_slot_k_before_detach'):
            del primary._slot_k_before_detach
    
    def forward(
        self,
        byte_seq: torch.Tensor,
        pe_features: torch.Tensor,
        state=None,
        return_aux: bool = False,
        compute_diversity_loss: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[any]]:
        """
        Args:
            byte_seq: [B, max_byte_length] 字节序列
            pe_features: [B, pe_feature_dim] PE 结构特征
            state: 可选的 DSRA 状态
            return_aux: 是否返回辅助信息
            compute_diversity_loss: 是否计算 DSRA slot 多样性辅助损失
        Returns:
            Tuple of (byte_repr, pe_repr, state)
            - byte_repr: [B, dsra_dim] 字节序列表示
            - pe_repr: [B, pe_projection_dim] PE 特征表示
            - state: DSRA 状态
        """
        self._last_diversity_loss = None
        self._set_diversity_capture(compute_diversity_loss)
        try:
            # 分块处理（如果需要）
            chunk_size = self.chunk_size
            seq_len = byte_seq.shape[1]

            if seq_len <= chunk_size:
                # 短序列直接处理；长序列走下面的逐块路径，避免构建完整激活图。
                byte_emb = self.byte_embedding(byte_seq)  # [B, L, byte_emb_dim]
                if self.use_pos_encoding:
                    byte_emb = self.pos_encoding(byte_emb)
                byte_emb = self.input_proj(byte_emb)  # [B, L, dsra_dim]
                byte_out, next_state, aux = self.dsra_encoder(byte_emb, state, return_aux=return_aux)
                state = next_state
                if compute_diversity_loss and hasattr(self.dsra_encoder.dsra, 'diversity_loss'):
                    if aux is None:
                        aux = {}
                    if isinstance(aux, dict) and 'diversity_loss' not in aux:
                        dsra_state_for_div = state[0] if isinstance(state, (list, tuple)) else state
                        aux['diversity_loss'] = self.dsra_encoder.dsra.diversity_loss(dsra_state_for_div)
            else:
                last_byte_out = None
                chunk_reprs = []
                chunk_weights = []
                aux_all = []

                _float32_max_exact_int = (1 << 24) - chunk_size - 1

                for i in range(0, seq_len, chunk_size):
                    chunk_seq = byte_seq[:, i:i+chunk_size]
                    chunk = self.byte_embedding(chunk_seq)
                    if self.use_pos_encoding:
                        chunk = self.pos_encoding(chunk, offset=i)
                    chunk = self.input_proj(chunk)
                    if state is not None and hasattr(state, 'position') and state.position > _float32_max_exact_int:
                        state = self.dsra_encoder.init_state(chunk.shape[0], device=chunk.device, dtype=chunk.dtype)
                    chunk_out, state, chunk_aux = self.dsra_encoder(chunk, state, return_aux=return_aux)
                    last_byte_out = chunk_out
                    if self.byte_chunk_pooling != "last":
                        chunk_repr = chunk_out.mean(dim=1)
                        if self.byte_chunk_pooling == "active_mean_detached" and i + chunk_size < seq_len:
                            chunk_repr = chunk_repr.detach()
                        chunk_reprs.append(chunk_repr)
                        if self.byte_chunk_pooling in {"active_mean", "active_mean_detached"}:
                            chunk_weights.append(chunk_seq.ne(0).any(dim=1).to(dtype=chunk_out.dtype))
                    if chunk_aux:
                        aux_all.append(chunk_aux)

                byte_out = last_byte_out
                if self.byte_chunk_pooling == "mean":
                    byte_repr = torch.stack(chunk_reprs, dim=0).mean(dim=0)
                elif self.byte_chunk_pooling in {"active_mean", "active_mean_detached"}:
                    reprs = torch.stack(chunk_reprs, dim=1)
                    weights = torch.stack(chunk_weights, dim=1).unsqueeze(-1)
                    denom = weights.sum(dim=1).clamp_min(1.0)
                    byte_repr = (reprs * weights).sum(dim=1) / denom
                aux = aux_all if aux_all else None
                if compute_diversity_loss and hasattr(self.dsra_encoder.dsra, 'diversity_loss'):
                    dsra_state_for_div = state[0] if isinstance(state, (list, tuple)) else state
                    div_loss = self.dsra_encoder.dsra.diversity_loss(dsra_state_for_div)
                    if aux is None:
                        aux = {}
                    if isinstance(aux, list):
                        aux = {'chunk_aux': aux}
                    aux['diversity_loss'] = div_loss
        finally:
            self._set_diversity_capture(False)

        # 序列表示：默认保持历史行为，只取最后一块；长上下文实验可选择跨块聚合。
        if seq_len <= chunk_size or self.byte_chunk_pooling == "last":
            byte_repr = byte_out.mean(dim=1)  # [B, dsra_dim]

        # PE 特征投影
        pe_repr = self.pe_projector(pe_features)  # [B, pe_projection_dim]

        if compute_diversity_loss and isinstance(aux, dict) and 'diversity_loss' in aux:
            self._last_diversity_loss = aux['diversity_loss']

        return byte_repr, pe_repr, state


class AxonMalwareModel(nn.Module):
    """Axon 恶意软件检测模型
    
    融合 DSRA 字节序列编码和 PE 结构特征。
    
    架构：
        字节序列 -> ByteEmbedding -> DSRA 流式编码 -> 序列特征
            ↓
        融合层 <- PE 特征投影
            ↓
        分类头 -> 二分类输出
    """
    
    def __init__(self, config: AxonExperimentConfig):
        super().__init__()
        self.config = config
        
        # DSRA 编码器
        if config.dsra_arch_config is not None:
            dsra_config = config.dsra_arch_config
        else:
            dsra_config = DSRAArchitectureConfig(
                dim=config.dsra_dim,
                heads=config.dsra_heads,
                slots=config.dsra_slots,
                read_topk=config.dsra_read_topk,
                write_topk=config.dsra_write_topk,
                local_window=config.dsra_local_window,
            )
        
        self.dsra_encoder = MalwareDSRAEncoder(
            byte_embedding_dim=config.byte_embedding_dim,
            max_byte_length=config.max_byte_length,
            dsra_config=dsra_config,
            pe_feature_dim=config.pe_feature_dim,
            pe_projection_dim=config.pe_projection_dim,
            pe_projector_hidden_dim=config.pe_projector_hidden_dim,
            use_pos_encoding=config.use_pos_encoding,
            pos_encoding_mode=config.pos_encoding_mode,
            dropout=config.dropout,
            chunk_size=config.dsra_chunk_size,
            vocab_size=config.vocab_size,
            byte_chunk_pooling=config.byte_chunk_pooling,
        )

        actual_dsra_dim = dsra_config.dim

        stat_projection_dim = config.pe_projection_dim
        self.stat_projector = nn.Sequential(
            nn.Linear(config.stat_feature_dim, stat_projection_dim),
            nn.LayerNorm(stat_projection_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
        )

        fusion_dim = actual_dsra_dim + config.pe_projection_dim + stat_projection_dim
        
        if config.fusion_type == "concat":
            self.fusion = nn.Identity()
            classifier_input_dim = fusion_dim
        elif config.fusion_type == "add":
            self.fusion = nn.Linear(actual_dsra_dim, config.pe_projection_dim)
            classifier_input_dim = config.pe_projection_dim
        elif config.fusion_type == "gated":
            self.byte_to_pe = nn.Linear(actual_dsra_dim, config.pe_projection_dim)
            self.fusion_gate = nn.Sequential(
                nn.Linear(config.pe_projection_dim * 3, config.pe_projection_dim),
                nn.GELU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.pe_projection_dim, 3),
            )
            classifier_input_dim = config.pe_projection_dim
        elif config.fusion_type == "residual_stat_gate":
            self.stat_gate = nn.Sequential(
                nn.Linear(fusion_dim, config.pe_projection_dim),
                nn.GELU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.pe_projection_dim, stat_projection_dim),
                nn.Sigmoid(),
            )
            classifier_input_dim = fusion_dim
        elif config.fusion_type == "residual_channel_gate":
            self.channel_gate = nn.Sequential(
                nn.LayerNorm(fusion_dim),
                nn.Linear(fusion_dim, config.classifier_hidden_dim),
                nn.GELU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.classifier_hidden_dim, fusion_dim),
                nn.Sigmoid(),
            )
            classifier_input_dim = fusion_dim
        else:
            self.fusion = nn.MultiheadAttention(
                embed_dim=config.pe_projection_dim,
                num_heads=config.fusion_num_heads,
                dropout=config.dropout
            )
            self.byte_to_pe = nn.Linear(actual_dsra_dim, config.pe_projection_dim)
            classifier_input_dim = config.pe_projection_dim
        
        self.fusion_type = config.fusion_type
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.LayerNorm(classifier_input_dim),
            nn.Dropout(config.dropout),
            nn.Linear(classifier_input_dim, config.classifier_hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.classifier_hidden_dim, config.num_classes)
        )
        
        # 辅助任务头（可选）
        self.aux_head = None
        if hasattr(config, 'use_aux_task') and config.use_aux_task:
            self.aux_head = nn.Linear(classifier_input_dim, 1)
    
    def forward(
        self,
        byte_seq: torch.Tensor,
        pe_features: torch.Tensor,
        stat_features: Optional[torch.Tensor] = None,
        return_features: bool = False,
        return_state: bool = False,
        compute_diversity_loss: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            byte_seq: [B, max_byte_length] 字节序列
            pe_features: [B, pe_feature_dim] PE 结构特征
            stat_features: [stat_feature_dim] 统计特征（可选）
            return_features: 是否返回中间特征
            return_state: 是否返回 DSRA 状态
            compute_diversity_loss: 是否计算 DSRA slot 多样性辅助损失
        Returns:
            Dict containing:
            - logits: [B, num_classes] 分类 logits
            - features: (可选) 融合特征
            - byte_repr: (可选) 字节序列表示
            - pe_repr: (可选) PE 特征表示
            - dsra_state: (可选) DSRA 状态
        """
        # DSRA 编码
        byte_repr, pe_repr, dsra_state = self.dsra_encoder(
            byte_seq,
            pe_features,
            compute_diversity_loss=compute_diversity_loss,
        )

        div_loss = None
        if compute_diversity_loss:
            div_loss = getattr(self.dsra_encoder, '_last_diversity_loss', None)
            self.dsra_encoder._last_diversity_loss = None

        # 统计特征投影
        if stat_features is not None:
            stat_repr = self.stat_projector(stat_features)
        else:
            stat_repr = torch.zeros(byte_repr.shape[0], self.stat_projector[0].out_features, device=byte_repr.device, dtype=byte_repr.dtype)

        # 特征融合
        if self.fusion_type == "concat":
            fused_features = torch.cat([byte_repr, pe_repr, stat_repr], dim=-1)
        elif self.fusion_type == "add":
            projected_byte = self.fusion(byte_repr)
            fused_features = projected_byte + pe_repr + stat_repr
        elif self.fusion_type == "gated":
            byte_repr_pe = self.byte_to_pe(byte_repr)
            gate_input = torch.cat([byte_repr_pe, pe_repr, stat_repr], dim=-1)
            fusion_gate_weights = torch.softmax(self.fusion_gate(gate_input), dim=-1)
            fusion_inputs = torch.stack([byte_repr_pe, pe_repr, stat_repr], dim=1)
            fused_features = (fusion_inputs * fusion_gate_weights.unsqueeze(-1)).sum(dim=1)
        elif self.fusion_type == "residual_stat_gate":
            gate_input = torch.cat([byte_repr, pe_repr, stat_repr], dim=-1)
            stat_gate_weights = self.stat_gate(gate_input)
            gated_stat_repr = stat_repr * stat_gate_weights
            fused_features = torch.cat([byte_repr, pe_repr, gated_stat_repr], dim=-1)
        elif self.fusion_type == "residual_channel_gate":
            concat_features = torch.cat([byte_repr, pe_repr, stat_repr], dim=-1)
            channel_gate_weights = 0.5 + self.channel_gate(concat_features)
            fused_features = concat_features * channel_gate_weights
        else:
            byte_repr_pe = self.byte_to_pe(byte_repr)
            byte_repr_expanded = byte_repr_pe.unsqueeze(0)
            pe_repr_expanded = pe_repr.unsqueeze(0)
            attn_out, _ = self.fusion(byte_repr_expanded, pe_repr_expanded, pe_repr_expanded)
            fused_features = attn_out.squeeze(0) + stat_repr
        
        # 分类
        logits = self.classifier(fused_features)
        
        if return_features:
            result = {
                'logits': logits,
                'features': fused_features,
                'byte_repr': byte_repr,
                'pe_repr': pe_repr
            }
            if self.fusion_type == "gated":
                result['fusion_gate_weights'] = fusion_gate_weights
            elif self.fusion_type == "residual_stat_gate":
                result['stat_gate_weights'] = stat_gate_weights
            elif self.fusion_type == "residual_channel_gate":
                result['channel_gate_weights'] = channel_gate_weights
        else:
            result = {'logits': logits}

        if return_state:
            result['dsra_state'] = dsra_state

        if div_loss is not None:
            result['diversity_loss'] = div_loss

        return result

    @property
    def dsra(self):
        return self.dsra_encoder.dsra_encoder.dsra

    def get_logits(self, byte_seq: torch.Tensor, pe_features: torch.Tensor) -> torch.Tensor:
        """获取分类 logits"""
        return self.forward(byte_seq, pe_features)['logits']
    
    def predict_proba(self, byte_seq: torch.Tensor, pe_features: torch.Tensor) -> torch.Tensor:
        """预测类别概率"""
        with torch.inference_mode():
            logits = self.get_logits(byte_seq, pe_features)
            return F.softmax(logits, dim=-1)
    
    def predict(self, byte_seq: torch.Tensor, pe_features: torch.Tensor) -> torch.Tensor:
        """预测类别"""
        with torch.inference_mode():
            proba = self.predict_proba(byte_seq, pe_features)
            return torch.argmax(proba, dim=-1)


class HybridLightGBMModel(nn.Module):
    """混合 LightGBM + DSRA 模型
    
    将预训练的 LightGBM 特征与 DSRA 特征融合。
    """
    
    def __init__(
        self,
        lgb_feature_dim: int = 1500,
        dsra_dim: int = 128,
        num_classes: int = 2,
        dropout: float = 0.1,
        lgb_proj_hidden_dim: int = 256,
        lgb_proj_dim: int = 128,
        fusion_hidden_dim: int = 128,
        config: Optional[AxonExperimentConfig] = None,
    ):
        super().__init__()
        
        if config is not None:
            lgb_feature_dim = config.pe_feature_dim
            dsra_dim = config.dsra_dim
            num_classes = config.num_classes
            dropout = config.dropout
            lgb_proj_hidden_dim = config.pe_projector_hidden_dim
            lgb_proj_dim = config.pe_projection_dim
            fusion_hidden_dim = config.classifier_hidden_dim

        # LightGBM 特征处理
        self.lgb_proj = nn.Sequential(
            nn.Linear(lgb_feature_dim, lgb_proj_hidden_dim),
            nn.LayerNorm(lgb_proj_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(lgb_proj_hidden_dim, lgb_proj_dim)
        )

        # DSRA 编码器
        if config is None:
            dsra_arch_config = DSRAArchitectureConfig(dim=dsra_dim)
        elif config.dsra_arch_config is not None:
            dsra_arch_config = config.dsra_arch_config
        else:
            dsra_arch_config = DSRAArchitectureConfig(
                dim=config.dsra_dim,
                heads=config.dsra_heads,
                slots=config.dsra_slots,
                read_topk=config.dsra_read_topk,
                write_topk=config.dsra_write_topk,
                local_window=config.dsra_local_window,
            )
        self.dsra_encoder = MalwareDSRAEncoder(
            byte_embedding_dim=config.byte_embedding_dim if config else dsra_dim,
            max_byte_length=config.max_byte_length if config else 65536,
            dsra_config=dsra_arch_config,
            pe_feature_dim=lgb_feature_dim,
            pe_projection_dim=lgb_proj_dim,
            pe_projector_hidden_dim=config.pe_projector_hidden_dim if config else 256,
            use_pos_encoding=config.use_pos_encoding if config else True,
            pos_encoding_mode=config.pos_encoding_mode if config else "sinusoidal",
            dropout=dropout,
            chunk_size=config.dsra_chunk_size if config else 512,
            vocab_size=config.vocab_size if config else 256,
        )

        actual_dsra_dim = dsra_arch_config.dim

        self.fusion = nn.Sequential(
            nn.Linear(lgb_proj_dim + actual_dsra_dim + lgb_proj_dim, fusion_hidden_dim),
            nn.LayerNorm(fusion_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # 分类头
        self.classifier = nn.Linear(fusion_hidden_dim, num_classes)
    
    def forward(
        self,
        lgb_features: torch.Tensor,
        byte_seq: torch.Tensor,
        pe_features: torch.Tensor
    ) -> torch.Tensor:
        """前向传播
        
        Args:
            lgb_features: [B, lgb_feature_dim] LightGBM 特征
            byte_seq: [B, max_byte_length] 字节序列
            pe_features: [B, pe_feature_dim] PE 特征
        """
        # LightGBM 特征处理
        lgb_repr = self.lgb_proj(lgb_features)
        
        # DSRA 编码
        byte_repr, pe_repr, _ = self.dsra_encoder(byte_seq, pe_features)
        
        # 融合
        fused = torch.cat([lgb_repr, byte_repr, pe_repr], dim=-1)
        fused = self.fusion(fused)
        
        return self.classifier(fused)
