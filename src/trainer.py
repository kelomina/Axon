"""Axon v2.6 训练器模块。

提供模型训练、验证和测试的完整流程。
集成 SwanLab 实验跟踪和 AMP 混合精度训练。
"""

import os
import time
import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, List
from dataclasses import dataclass, asdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR, LinearLR, SequentialLR

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)


def autocast_for_device(device_type: str, enabled: bool = True):
    return torch.amp.autocast(device_type=device_type, enabled=enabled)


def create_grad_scaler(device_type: str, enabled: bool = True):
    return torch.amp.GradScaler(device_type, enabled=enabled)


try:
    from .config import AxonExperimentConfig, TrainingConfig
    from .model import AxonMalwareModel
    from .security import load_safe_checkpoint
except ImportError:
    from config import AxonExperimentConfig, TrainingConfig
    from model import AxonMalwareModel
    from security import load_safe_checkpoint


@dataclass
class TrainingMetrics:
    """训练指标"""
    epoch: int
    phase: str
    loss: float
    accuracy: float
    precision: float
    recall: float
    f1: float
    auc: Optional[float] = None
    learning_rate: float = 0.0
    batch_time: float = 0.0
    true_positive: int = 0
    true_negative: int = 0
    false_positive: int = 0
    false_negative: int = 0
    false_positive_rate: float = 0.0
    false_negative_rate: float = 0.0


class EarlyStopping:
    """早停机制"""
    
    def __init__(
        self,
        patience: int = 5,
        min_delta: float = 0.0,
        mode: str = "max"
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
    
    def __call__(self, score: float) -> bool:
        if self.best_score is None:
            self.best_score = score
            return False
        
        if self.mode == "max":
            improved = score > self.best_score + self.min_delta
        else:
            improved = score < self.best_score - self.min_delta
        
        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        
        return self.early_stop


class MetricsTracker:
    """指标追踪器"""
    
    def __init__(self):
        self.history: List[TrainingMetrics] = []
        self.best_metrics: Dict[str, float] = {}
    
    def update(self, metrics: TrainingMetrics):
        self.history.append(metrics)
    
    def get_best(self, metric_name: str, mode: str = "max") -> Optional[float]:
        if not self.history:
            return None
        
        values = [getattr(m, metric_name) for m in self.history if hasattr(m, metric_name) and getattr(m, metric_name) is not None]
        
        if not values:
            return None
        
        if mode == "max":
            return max(values)
        else:
            return min(values)
    
    def save(self, path: Path):
        data = [
            {
                'epoch': m.epoch,
                'phase': m.phase,
                'loss': m.loss,
                'accuracy': m.accuracy,
                'precision': m.precision,
                'recall': m.recall,
                'f1': m.f1,
                'auc': m.auc,
                'learning_rate': m.learning_rate,
                'true_positive': m.true_positive,
                'true_negative': m.true_negative,
                'false_positive': m.false_positive,
                'false_negative': m.false_negative,
                'false_positive_rate': m.false_positive_rate,
                'false_negative_rate': m.false_negative_rate,
            }
            for m in self.history
        ]
        
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load(self, path: Path):
        with open(path, 'r') as f:
            data = json.load(f)
        
        self.history = [
            TrainingMetrics(**d) for d in data
        ]


class AxonTrainer:
    """Axon 模型训练器
    
    支持：
    - AMP 混合精度训练
    - SwanLab 实验跟踪
    - 早停机制
    - 梯度裁剪
    """
    
    def __init__(
        self,
        model: AxonMalwareModel,
        config: AxonExperimentConfig,
        train_config: Optional[TrainingConfig] = None,
        device: Optional[torch.device] = None
    ):
        self.model = model
        self.config = config
        self.train_config = train_config or TrainingConfig()
        
        if device is None:
            device = config.get_device()
        self.device = device
        self.model.to(device)
        
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        self.criterion = self._create_criterion()
        
# AMP 混合精度
        self.use_amp = self.train_config.mixed_precision and self.device.type == "cuda"
        self.scaler = create_grad_scaler(self.device.type, enabled=self.use_amp)
        
        # SWA (Stochastic Weight Averaging)
        self.swa_model = None
        self.swa_scheduler = None
        self.swa_start_epoch = None
        self.swa_active = False
        if self.train_config.use_swa:
            self.swa_start_epoch = self.train_config.max_epochs + self.train_config.swa_start_epoch
            if self.train_config.swa_start_epoch > 0:
                self.swa_start_epoch = self.train_config.swa_start_epoch
            print(f"[SWA] Will start at epoch {self.swa_start_epoch}")
        
        # EMA (Exponential Moving Average)
        self.ema_model = None
        if self.train_config.use_ema:
            self.ema_model = self._state_dict_to_cpu(self.model.state_dict())
            self.ema_decay = self.train_config.ema_decay
            print(f"[EMA] Enabled with decay={self.ema_decay}")
        
        # 指标追踪
        self.metrics_tracker = MetricsTracker()
        self.best_f1 = 0.0
        self.best_epoch = 0
        
        # 早停
        self.early_stopping = EarlyStopping(
            patience=self.train_config.early_stopping_patience,
            min_delta=self.train_config.early_stopping_min_delta,
            mode="max"
        )
        
        # 输出目录
        self.output_dir = Path(config.model_save_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 日志目录
        self.log_dir = Path(config.log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # SwanLab
        self.swanlab_run = None
    
    def _create_optimizer(self):
        if self.train_config.optimizer == "adam":
            optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=self.train_config.learning_rate,
                weight_decay=self.train_config.weight_decay,
                betas=self.train_config.betas,
                eps=self.train_config.eps
            )
        elif self.train_config.optimizer == "sgd":
            optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=self.train_config.learning_rate,
                weight_decay=self.train_config.weight_decay,
                momentum=self.train_config.sgd_momentum
            )
        elif self.train_config.optimizer == "adamw":
            optimizer = AdamW(
                self.model.parameters(),
                lr=self.train_config.learning_rate,
                weight_decay=self.train_config.weight_decay,
                betas=self.train_config.betas,
                eps=self.train_config.eps
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.train_config.optimizer}")

        return optimizer
    
    def _create_scheduler(self):
        if self.train_config.lr_scheduler == "none":
            return None
        
        warmup_scheduler = LinearLR(
            self.optimizer,
            start_factor=self.train_config.warmup_start_lr / self.train_config.learning_rate,
            end_factor=1.0,
            total_iters=self.train_config.warmup_epochs
        )
        
        if self.train_config.lr_scheduler == "cosine":
            main_scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=max(1, self.train_config.max_epochs - self.train_config.warmup_epochs),
                eta_min=self.train_config.min_lr
            )
        elif self.train_config.lr_scheduler == "step":
            main_scheduler = StepLR(
                self.optimizer,
                step_size=self.train_config.step_lr_size,
                gamma=self.train_config.step_lr_gamma
            )
        else:
            raise ValueError(f"Unknown lr_scheduler: {self.train_config.lr_scheduler}")
        
        scheduler = SequentialLR(
            self.optimizer,
            schedulers=[warmup_scheduler, main_scheduler],
            milestones=[self.train_config.warmup_epochs]
        )
        
        return scheduler
    
    def _create_criterion(self):
        if self.train_config.focal_gamma > 0:
            def focal_loss(logits, targets, gamma=self.train_config.focal_gamma, alpha=self.train_config.focal_alpha, reduction='mean'):
                if self.train_config.label_smoothing > 0:
                    ce_loss = F.cross_entropy(logits, targets, reduction='none',
                                              label_smoothing=self.train_config.label_smoothing)
                else:
                    ce_loss = F.cross_entropy(logits, targets, reduction='none')
                pt = torch.exp(-ce_loss)
                focal_weight = (1 - pt) ** gamma
                alpha_t = alpha * (targets == 1).float() + (1 - alpha) * (targets == 0).float()
                per_sample = alpha_t * focal_weight * ce_loss
                if reduction == 'none':
                    return per_sample
                return per_sample.mean()
            return focal_loss
        else:
            return nn.CrossEntropyLoss(
                label_smoothing=self.train_config.label_smoothing
            )

    def _criterion_per_sample(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if self.train_config.focal_gamma > 0:
            return self.criterion(logits, labels, reduction='none')
        return F.cross_entropy(
            logits,
            labels,
            reduction='none',
            label_smoothing=self.train_config.label_smoothing,
        )

    def _training_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        sample_weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Near-threshold weighting: upweight samples with predictions near the decision boundary
        if self.train_config.near_threshold_weight > 1.0:
            per_sample_loss = self._criterion_per_sample(logits, labels)
            with torch.no_grad():
                probs = torch.softmax(logits, dim=-1)[:, 1]
                near_mask = (probs > self.train_config.near_threshold_low) & (probs < self.train_config.near_threshold_high)
                weight = torch.where(near_mask, 
                                   torch.tensor(self.train_config.near_threshold_weight, device=probs.device), 
                                   torch.tensor(1.0, device=probs.device))
            # Combine with existing sample_weights if present
            if sample_weights is not None:
                weights = sample_weights.to(per_sample_loss.device, dtype=per_sample_loss.dtype)
                weight = weight * weights
            return (per_sample_loss * weight).sum() / weight.sum().clamp_min(1e-8)
        
        if sample_weights is None:
            return self.criterion(logits, labels)

        per_sample_loss = self._criterion_per_sample(logits, labels)
        weights = sample_weights.to(per_sample_loss.device, dtype=per_sample_loss.dtype)
        return (per_sample_loss * weights).sum() / weights.sum().clamp_min(1e-8)
    
    def _init_swanlab(self, fast_mode: bool = False):
        if not self.train_config.enable_swanlab:
            return
        try:
            import swanlab
            config_dict = self.config.to_dict()
            config_dict["fast_mode"] = fast_mode
            config_dict["use_amp"] = self.use_amp
            
            self.swanlab_run = swanlab.init(
                project=self.train_config.swanlab_project,
                experiment_name=self.config.experiment_name,
                config=config_dict,
            )
            print(f"[SwanLab] Initialized: project={self.train_config.swanlab_project}, experiment={self.config.experiment_name}")
        except ImportError:
            print("[SwanLab] swanlab not installed, skipping experiment tracking")
        except Exception as e:
            print(f"[SwanLab] Failed to initialize: {e}")
    
    def _log_metrics(self, metrics: TrainingMetrics, step: int):
        if self.swanlab_run is None:
            return
        try:
            import swanlab
            prefix = metrics.phase
            log_dict = {
                f"{prefix}/loss": metrics.loss,
                f"{prefix}/accuracy": metrics.accuracy,
                f"{prefix}/precision": metrics.precision,
                f"{prefix}/recall": metrics.recall,
                f"{prefix}/f1": metrics.f1,
                f"{prefix}/learning_rate": metrics.learning_rate,
                "epoch": metrics.epoch,
            }
            if metrics.auc is not None:
                log_dict[f"{prefix}/auc"] = metrics.auc
            if metrics.batch_time > 0:
                log_dict[f"{prefix}/batch_time"] = metrics.batch_time
            
            swanlab.log(log_dict, step=step)
        except Exception:
            pass

    def _finish_swanlab(self):
        if self.swanlab_run is None:
            return
        try:
            import swanlab
            swanlab.finish()
            print("[SwanLab] Run finished")
        except Exception:
            pass
        finally:
            self.swanlab_run = None
    
    def train_epoch(
        self,
        train_loader: DataLoader,
        epoch: int
    ) -> TrainingMetrics:
        self.model.train()
        
        all_preds = []
        all_targets = []
        all_probs = []
        total_loss = 0.0
        num_batches = 0
        
        start_time = time.time()
        
        for batch_idx, batch in enumerate(train_loader):
            if len(batch) == 5:
                byte_seq, pe_features, stat_features, labels, sample_weights = batch
            else:
                byte_seq, pe_features, stat_features, labels = batch
                sample_weights = None
            byte_seq = byte_seq.to(self.device, non_blocking=True)
            pe_features = pe_features.to(self.device, non_blocking=True)
            stat_features = stat_features.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            if sample_weights is not None:
                sample_weights = sample_weights.to(self.device, non_blocking=True)
            
            self.optimizer.zero_grad(set_to_none=True)
            
            # AMP 混合精度前向传播
            with autocast_for_device(self.device.type, enabled=self.use_amp):
                compute_diversity = self.train_config.diversity_loss_weight > 0
                outputs = self.model(
                    byte_seq,
                    pe_features,
                    stat_features=stat_features,
                    return_state=compute_diversity,
                    compute_diversity_loss=compute_diversity,
                )
                logits = outputs['logits']
                loss = self._training_loss(logits, labels, sample_weights)
                
                dsra_state = outputs.get('dsra_state') if compute_diversity else None
                if dsra_state is not None:
                    div_loss = outputs.get('diversity_loss')
                    if div_loss is None:
                        dsra_state_for_div = dsra_state[0] if isinstance(dsra_state, (list, tuple)) else dsra_state
                        div_loss = self.model.dsra.diversity_loss(dsra_state_for_div)
                    loss = loss + self.train_config.diversity_loss_weight * div_loss

                if not torch.isfinite(loss):
                    logits_finite = torch.isfinite(logits).all().item()
                    raise FloatingPointError(
                        "Non-finite training loss detected "
                        f"at epoch={epoch}, batch={batch_idx + 1}. "
                        f"logits_finite={logits_finite}, "
                        f"mixed_precision={self.use_amp}"
                    )
            
            # AMP 反向传播
            self.scaler.scale(loss).backward()
            
            if self.train_config.gradient_clip > 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.train_config.gradient_clip
                )
            
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            # EMA: update EMA model after each optimizer step
            self._update_ema()
            
            total_loss += loss.item()
            num_batches += 1
            
            with torch.no_grad():
                probs = torch.softmax(logits.detach(), dim=1)[:, 1]
                preds = (probs >= self.train_config.decision_threshold).long()
            
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())
            all_probs.extend(probs.detach().cpu().numpy())
            
            # 每10个batch打印一次进度
            if (batch_idx + 1) % self.train_config.log_interval == 0 or (batch_idx + 1) == len(train_loader):
                elapsed = time.time() - start_time
                print(f"  Batch {batch_idx+1}/{len(train_loader)} | "
                      f"Loss: {loss.item():.4f} | "
                      f"Elapsed: {elapsed:.1f}s", end="\r")
        
        batch_time = time.time() - start_time
        print()
        if num_batches == 0:
            raise ValueError("train_loader is empty")
        
        metrics = self._compute_metrics(
            epoch, "train",
            np.array(all_targets),
            np.array(all_preds),
            np.array(all_probs),
            total_loss / num_batches,
            self.optimizer.param_groups[0]['lr'],
            batch_time
        )
        
        return metrics
    
    @torch.no_grad()
    def evaluate(
        self,
        eval_loader: DataLoader,
        epoch: int,
        phase: str = "val"
    ) -> TrainingMetrics:
        self.model.eval()
        
        all_preds = []
        all_targets = []
        all_probs = []
        total_loss = 0.0
        num_batches = 0
        
        for byte_seq, pe_features, stat_features, labels in eval_loader:
            byte_seq = byte_seq.to(self.device, non_blocking=True)
            pe_features = pe_features.to(self.device, non_blocking=True)
            stat_features = stat_features.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            
            with autocast_for_device(self.device.type, enabled=self.use_amp):
                outputs = self.model(byte_seq, pe_features, stat_features=stat_features)
                logits = outputs['logits']
                loss = self.criterion(logits, labels)
            
            total_loss += loss.item()
            num_batches += 1
            
            probs = torch.softmax(logits, dim=1)[:, 1]
            preds = (probs >= self.train_config.decision_threshold).long()
            
            all_preds.extend(preds.detach().cpu().numpy())
            all_targets.extend(labels.detach().cpu().numpy())
            all_probs.extend(probs.detach().cpu().numpy())
        
        if num_batches == 0:
            raise ValueError(f"{phase}_loader is empty")

        metrics = self._compute_metrics(
            epoch, phase,
            np.array(all_targets),
            np.array(all_preds),
            np.array(all_probs),
            total_loss / num_batches,
            self.optimizer.param_groups[0]['lr'],
            0.0
        )
        
        return metrics

    @torch.no_grad()
    def threshold_sweep(
        self,
        eval_loader: DataLoader,
        thresholds: List[float],
        epoch: int = 0,
        phase: str = "val"
    ) -> List[Dict[str, float]]:
        """在同一批模型输出上扫描多个判定阈值。"""
        self.model.eval()

        all_targets = []
        all_probs = []
        total_loss = 0.0
        num_batches = 0

        for byte_seq, pe_features, stat_features, labels in eval_loader:
            byte_seq = byte_seq.to(self.device, non_blocking=True)
            pe_features = pe_features.to(self.device, non_blocking=True)
            stat_features = stat_features.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            with torch.inference_mode(), autocast_for_device(self.device.type, enabled=self.use_amp):
                outputs = self.model(byte_seq, pe_features, stat_features=stat_features)
                logits = outputs['logits']
                loss = self.criterion(logits, labels)

            total_loss += loss.item()
            num_batches += 1

            probs = torch.softmax(logits, dim=1)[:, 1]
            all_targets.extend(labels.detach().cpu().numpy())
            all_probs.extend(probs.detach().cpu().numpy())

        if num_batches == 0:
            raise ValueError(f"{phase}_loader is empty")

        targets = np.array(all_targets)
        probs = np.array(all_probs)
        avg_loss = total_loss / num_batches
        rows = []
        for threshold in thresholds:
            preds = (probs >= threshold).astype(np.int64)
            metrics = self._compute_metrics(
                epoch,
                phase,
                targets,
                preds,
                probs,
                avg_loss,
                self.optimizer.param_groups[0]['lr'],
                0.0
            )
            rows.append({
                'threshold': float(threshold),
                'loss': float(metrics.loss),
                'accuracy': float(metrics.accuracy),
                'precision': float(metrics.precision),
                'recall': float(metrics.recall),
                'f1': float(metrics.f1),
                'auc': float(metrics.auc) if metrics.auc is not None else None,
                'true_positive': int(metrics.true_positive),
                'true_negative': int(metrics.true_negative),
                'false_positive': int(metrics.false_positive),
                'false_negative': int(metrics.false_negative),
                'false_positive_rate': float(metrics.false_positive_rate),
                'false_negative_rate': float(metrics.false_negative_rate),
            })

        return rows
    
    def _compute_metrics(
        self,
        epoch: int,
        phase: str,
        targets: np.ndarray,
        preds: np.ndarray,
        probs: np.ndarray,
        loss: float,
        lr: float,
        batch_time: float
    ) -> TrainingMetrics:
        accuracy = accuracy_score(targets, preds)
        precision = precision_score(targets, preds, zero_division=0)
        recall = recall_score(targets, preds, zero_division=0)
        f1 = f1_score(targets, preds, zero_division=0)
        
        try:
            auc = roc_auc_score(targets, probs)
        except ValueError:
            auc = None

        labels = np.array([0, 1])
        tn, fp, fn, tp = confusion_matrix(targets, preds, labels=labels).ravel()
        false_positive_rate = fp / max(1, fp + tn)
        false_negative_rate = fn / max(1, fn + tp)
        
        metrics = TrainingMetrics(
            epoch=epoch,
            phase=phase,
            loss=loss,
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1=f1,
            auc=auc,
            learning_rate=lr,
            batch_time=batch_time,
            true_positive=int(tp),
            true_negative=int(tn),
            false_positive=int(fp),
            false_negative=int(fn),
            false_positive_rate=float(false_positive_rate),
            false_negative_rate=float(false_negative_rate),
        )
        
        return metrics
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        test_loader: Optional[DataLoader] = None,
        fast_mode: bool = False
    ) -> Dict[str, List[TrainingMetrics]]:
        try:
            return self._train_impl(train_loader, val_loader, test_loader, fast_mode)
        finally:
            self._finish_swanlab()

    def _train_impl(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        test_loader: Optional[DataLoader] = None,
        fast_mode: bool = False
    ) -> Dict[str, List[TrainingMetrics]]:
        
        results = {
            'train': [],
            'val': [],
            'test': []
        }
        
        num_epochs = self.train_config.max_epochs
        
        # 初始化 SwanLab
        self._init_swanlab(fast_mode)
        
        print(f"\n{'='*60}")
        print(f"Starting training: {self.config.experiment_name}")
        if fast_mode:
            print("[FAST MODE] Training with reduced samples and epochs")
        if self.use_amp:
            print("[AMP] Mixed precision training enabled")
        print(f"Device: {self.device}")
        print(f"Epochs: {num_epochs}")
        print(f"{'='*60}\n")
        
        start_epoch = getattr(self, '_resumed_epoch', 0) + 1 if hasattr(self, '_resumed_epoch') else 1
        
        last_epoch_ran = start_epoch - 1
        for epoch in range(start_epoch, num_epochs + 1):
            last_epoch_ran = epoch
            epoch_start = time.time()
            
            # SWA: check if we should start SWA
            if self.train_config.use_swa and epoch >= self.swa_start_epoch:
                if self.swa_model is None:
                    from torch.optim.swa_utils import AveragedModel
                    self.swa_model = AveragedModel(self.model)
                    self.swa_active = True
                if self.swa_scheduler is None:
                    from torch.optim.swa_utils import SWALR
                    self.swa_scheduler = SWALR(self.optimizer, swa_lr=self.train_config.swa_lr)
                    print(f"[SWA] Started at epoch {epoch} with lr={self.train_config.swa_lr}")
            
            train_metrics = self.train_epoch(train_loader, epoch)
            results['train'].append(train_metrics)
            self.metrics_tracker.update(train_metrics)
            
            self._print_metrics(train_metrics, prefix="Train")
            self._log_metrics(train_metrics, step=epoch)
            
            # SWA: update SWA model
            if self.swa_model is not None and epoch >= self.swa_start_epoch:
                self.swa_model.update_parameters(self.model)
                self.swa_scheduler.step()
            
            if val_loader is not None and epoch % self.train_config.eval_interval == 0:
                # Use EMA model for evaluation if available
                eval_model = self.model
                if self.ema_model is not None:
                    self._apply_ema_to_model()
                    eval_model = self.model
                try:
                    val_metrics = self.evaluate(val_loader, epoch, "val")
                    results['val'].append(val_metrics)
                    self.metrics_tracker.update(val_metrics)
                    self._print_metrics(val_metrics, prefix="Val")
                    self._log_metrics(val_metrics, step=epoch)
                finally:
                    # Restore original model weights even if validation is interrupted.
                    if self.ema_model is not None:
                        self._restore_model_from_ema_backup()
                
                if val_metrics.f1 > self.best_f1:
                    self.best_f1 = val_metrics.f1
                    self.best_epoch = epoch
                    # Save EMA model if available, otherwise save regular model
                    if self.ema_model is not None:
                        self._apply_ema_to_model()
                        try:
                            self.save_checkpoint(self.train_config.best_model_filename, last_epoch=epoch)
                        finally:
                            self._restore_model_from_ema_backup()
                    else:
                        self.save_checkpoint(self.train_config.best_model_filename, last_epoch=epoch)
                    print(f"  [Best model saved] F1: {val_metrics.f1:.4f}")
                
                if self.early_stopping(val_metrics.f1):
                    print(f"\nEarly stopping triggered at epoch {epoch}")
                    break
            
            if self.scheduler is not None and (not self.train_config.use_swa or epoch < self.swa_start_epoch):
                self.scheduler.step()
            
            epoch_time = time.time() - epoch_start
            print(f"  Epoch {epoch} total time: {epoch_time:.1f}s")
        
        # Apply SWA weights to model before saving final checkpoint
        if self.swa_model is not None:
            from torch.optim.swa_utils import update_bn
            print("[SWA] Updating batch normalization statistics...")
            update_bn(train_loader, self.swa_model, device=self.device)
            # Copy SWA weights back to model
            self.model.load_state_dict(self.swa_model.module.state_dict())
            self.swa_model.to("cpu")
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
            print("[SWA] Weights applied to model")
        
        self.save_checkpoint(self.train_config.final_model_filename, last_epoch=last_epoch_ran)

        if test_loader is not None:
            print(f"\n{'='*60}")
            print("Evaluating on test set...")
            if self.swa_model is not None:
                # SWA 启用时：final_model.pt 已包含 SWA 平均权重，用它做测试评估
                final_path = self.output_dir / self.train_config.final_model_filename
                if final_path.exists():
                    checkpoint = load_safe_checkpoint(final_path, map_location="cpu")
                    self.model.load_state_dict(checkpoint['model_state_dict'])
                    del checkpoint
                    if self.device.type == "cuda":
                        torch.cuda.empty_cache()
                    print(f"[SWA] Loaded final SWA model for test evaluation")
                test_epoch = last_epoch_ran
            else:
                self._load_best_model_for_test()
                test_epoch = self.best_epoch
            test_metrics = self.evaluate(test_loader, test_epoch, "test")
            results['test'].append(test_metrics)
            self.metrics_tracker.update(test_metrics)
            self._print_metrics(test_metrics, prefix="Test")
            self._log_metrics(test_metrics, step=self.best_epoch)
            print(f"{'='*60}\n")
        
        self.metrics_tracker.save(self.log_dir / "training_history.json")
        
        self._finish_swanlab()
        
        return results

    def _load_best_model_for_test(self) -> bool:
        """测试集评估前加载验证集最佳 checkpoint，保证 test 指标口径正确。"""
        best_path = self.output_dir / self.train_config.best_model_filename
        if self.best_epoch <= 0 or not best_path.exists():
            return False

        checkpoint = load_safe_checkpoint(best_path, map_location="cpu")
        self.model.load_state_dict(checkpoint['model_state_dict'])
        del checkpoint
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        print(f"  [Best model loaded for test] Epoch: {self.best_epoch}")
        return True
    
    def _print_metrics(self, metrics: TrainingMetrics, prefix: str = ""):
        auc_str = f", AUC: {metrics.auc:.4f}" if metrics.auc is not None else ""
        time_str = f", Time: {metrics.batch_time:.1f}s" if metrics.batch_time > 0 else ""
        print(
            f"{prefix} | Epoch: {metrics.epoch:3d} | "
            f"Loss: {metrics.loss:.4f} | "
            f"Acc: {metrics.accuracy:.4f} | "
            f"Prec: {metrics.precision:.4f} | "
            f"Rec: {metrics.recall:.4f} | "
            f"F1: {metrics.f1:.4f}{auc_str} | "
            f"FP: {metrics.false_positive} | "
            f"FN: {metrics.false_negative} | "
            f"FPR: {metrics.false_positive_rate:.4f} | "
            f"FNR: {metrics.false_negative_rate:.4f} | "
            f"LR: {metrics.learning_rate:.2e}{time_str}"
        )
    
    def _update_ema(self):
        """Update EMA model weights."""
        if self.ema_model is None:
            return
        with torch.no_grad():
            for k, v in self.model.state_dict().items():
                value = v.detach().to("cpu")
                if torch.is_floating_point(value):
                    self.ema_model[k].mul_(self.ema_decay).add_(value, alpha=1 - self.ema_decay)
                else:
                    self.ema_model[k].copy_(value)
    
    def _apply_ema_to_model(self):
        """Apply EMA weights to model (backup original weights first)."""
        if self.ema_model is None:
            return
        self._ema_backup = self._state_dict_to_cpu(self.model.state_dict())
        self.model.load_state_dict(self.ema_model)
    
    def _restore_model_from_ema_backup(self):
        """Restore original model weights from EMA backup."""
        if not hasattr(self, '_ema_backup') or self._ema_backup is None:
            return
        self.model.load_state_dict(self._ema_backup)
        self._ema_backup = None
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

    def _state_dict_to_cpu(self, state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return {
            k: v.detach().cpu().clone()
            for k, v in state_dict.items()
        }

    def _move_checkpoint_value_to_cpu(self, value: Any) -> Any:
        if torch.is_tensor(value):
            return value.detach().cpu()
        if isinstance(value, dict):
            return {k: self._move_checkpoint_value_to_cpu(v) for k, v in value.items()}
        if isinstance(value, list):
            return [self._move_checkpoint_value_to_cpu(v) for v in value]
        if isinstance(value, tuple):
            return tuple(self._move_checkpoint_value_to_cpu(v) for v in value)
        return value
    
    def save_checkpoint(self, filename: str, last_epoch: int = 0):
        checkpoint = {
            'epoch': self.best_epoch,
            'last_epoch': last_epoch,
            'model_state_dict': self._state_dict_to_cpu(self.model.state_dict()),
            'optimizer_state_dict': self._move_checkpoint_value_to_cpu(self.optimizer.state_dict()),
            'best_f1': self.best_f1,
            'config': self.config.to_dict(),
            'train_config': asdict(self.train_config),
        }
        
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self._move_checkpoint_value_to_cpu(self.scheduler.state_dict())
        
        if self.early_stopping is not None:
            checkpoint['early_stopping_state'] = {
                'best_score': self.early_stopping.best_score,
                'counter': self.early_stopping.counter,
            }
        
        if self.use_amp:
            checkpoint['scaler_state_dict'] = self._move_checkpoint_value_to_cpu(self.scaler.state_dict())
        
        try:
            torch.save(checkpoint, self.output_dir / filename)
        finally:
            del checkpoint
    
    def load_checkpoint(self, checkpoint_path: Path):
        checkpoint = load_safe_checkpoint(
            checkpoint_path,
            map_location="cpu",
            required_keys={"optimizer_state_dict", "epoch", "best_f1"},
        )
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.best_epoch = checkpoint['epoch']
        self.best_f1 = checkpoint['best_f1']
        self._resumed_epoch = checkpoint.get('last_epoch', self.best_epoch)
        
        if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        if self.early_stopping is not None and 'early_stopping_state' in checkpoint:
            es_state = checkpoint['early_stopping_state']
            self.early_stopping.best_score = es_state['best_score']
            self.early_stopping.counter = es_state['counter']
        
        if self.use_amp and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        summary = {
            'epoch': self.best_epoch,
            'last_epoch': self._resumed_epoch,
            'best_f1': self.best_f1,
        }
        del checkpoint
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        return summary
    
    def predict(
        self,
        byte_seq: torch.Tensor,
        pe_features: torch.Tensor
    ) -> Tuple[np.ndarray, np.ndarray]:
        self.model.eval()
        
        with torch.no_grad():
            byte_seq = byte_seq.to(self.device)
            pe_features = pe_features.to(self.device)
            
            logits = self.model(byte_seq, pe_features)['logits']
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)
        
        return preds.cpu().numpy(), probs[:, 1].cpu().numpy()


def train_model(
    model: AxonMalwareModel,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader] = None,
    config: Optional[AxonExperimentConfig] = None,
    train_config: Optional[TrainingConfig] = None,
    fast_mode: bool = False,
) -> Tuple[AxonMalwareModel, Dict]:
    
    if config is None:
        config = AxonExperimentConfig()
    if train_config is None:
        train_config = TrainingConfig()
    
    trainer = AxonTrainer(model, config, train_config)
    results = trainer.train(train_loader, val_loader, fast_mode=fast_mode)
    
    return model, results
