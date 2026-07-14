"""Pro 分支的策略梯度训练器。"""

from __future__ import annotations

from typing import Dict, Iterable, Tuple

import torch
from torch.optim import AdamW

from .agent import AxonPolicyAgent
from .config import RLTrainingConfig
from .environment import MalwareBanditEnv


Batch = Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]


class PolicyGradientTrainer:
    """最小可用的策略梯度训练循环。

    默认使用期望奖励版本，适合当前这种“一次判断一个样本”的场景；
    如果后续接入真正多步环境，可以把 use_expected_reward 关掉，
    改成采样动作后的 REINFORCE 训练。
    """

    def __init__(
        self,
        agent: AxonPolicyAgent,
        env: MalwareBanditEnv,
        config: RLTrainingConfig | None = None,
    ):
        self.agent = agent
        self.env = env
        self.config = config or RLTrainingConfig()
        self.device = torch.device(
            "cuda:0" if self.config.device == "cuda" and torch.cuda.is_available() else "cpu"
        )
        self.agent.to(self.device)
        self.optimizer = AdamW(
            self.agent.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        self.reward_baseline = 0.0

    def _move_batch(self, batch: Batch) -> Batch:
        byte_seq, pe_features, stat_features, labels = batch
        return (
            byte_seq.to(self.device),
            pe_features.to(self.device),
            stat_features.to(self.device),
            labels.to(self.device),
        )

    def _policy_loss(
        self,
        byte_seq: torch.Tensor,
        pe_features: torch.Tensor,
        stat_features: torch.Tensor,
        labels: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        dist = self.agent.distribution(byte_seq, pe_features, stat_features)
        entropy = dist.entropy().mean()

        if self.config.use_expected_reward:
            reward_table = self.env.reward_table(labels).to(self.device)
            expected_reward = (dist.probs * reward_table).sum(dim=-1)
            policy_loss = -expected_reward.mean()
            actions = (dist.probs[:, 1] >= self.config.decision_threshold).long()
            sampled_reward = self.env.reward_for_actions(actions, labels).to(self.device)
        else:
            actions = dist.sample()
            log_probs = dist.log_prob(actions)
            sampled_reward = self.env.reward_for_actions(actions, labels).to(self.device)
            batch_reward = sampled_reward.mean().item()
            self.reward_baseline = (
                self.config.moving_baseline_beta * self.reward_baseline
                + (1.0 - self.config.moving_baseline_beta) * batch_reward
            )
            advantage = sampled_reward - self.reward_baseline
            policy_loss = -(log_probs * advantage.detach()).mean()

        # 熵可以理解成“鼓励模型别过早只押一个答案”，smoke 阶段更稳定。
        loss = policy_loss - self.config.entropy_coef * entropy
        metrics = self.env.metrics(actions.detach(), labels.detach())
        metrics.update(
            {
                "loss": float(loss.detach().item()),
                "policy_loss": float(policy_loss.detach().item()),
                "entropy": float(entropy.detach().item()),
                "reward": float(sampled_reward.detach().mean().item()),
            }
        )
        return loss, metrics

    def train_epoch(self, loader: Iterable[Batch], epoch: int) -> Dict[str, float]:
        self.agent.train()
        totals: Dict[str, float] = {}
        batches = 0

        for batch_idx, batch in enumerate(loader, start=1):
            byte_seq, pe_features, stat_features, labels = self._move_batch(batch)
            self.optimizer.zero_grad(set_to_none=True)
            loss, metrics = self._policy_loss(byte_seq, pe_features, stat_features, labels)

            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite RL loss at epoch={epoch}, batch={batch_idx}")

            loss.backward()
            if self.config.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.agent.parameters(), self.config.gradient_clip)
            self.optimizer.step()

            for key, value in metrics.items():
                totals[key] = totals.get(key, 0.0) + float(value)
            batches += 1

            if batch_idx % self.config.log_interval == 0:
                print(
                    f"  RL epoch {epoch} batch {batch_idx} | "
                    f"loss={metrics['loss']:.4f} reward={metrics['reward']:.4f} "
                    f"acc={metrics['accuracy']:.4f}"
                )

        return {key: value / max(1, batches) for key, value in totals.items()}

    @torch.no_grad()
    def evaluate(self, loader: Iterable[Batch], decision_threshold: float | None = None) -> Dict[str, float]:
        self.agent.eval()
        threshold = self.config.decision_threshold if decision_threshold is None else decision_threshold
        confusion = {
            "tp": 0,
            "tn": 0,
            "fp": 0,
            "fn": 0,
        }
        total_reward = 0.0
        total_samples = 0

        for batch in loader:
            byte_seq, pe_features, stat_features, labels = self._move_batch(batch)
            actions = self.agent.threshold_actions(
                byte_seq,
                pe_features,
                stat_features,
                decision_threshold=threshold,
            )
            rewards = self.env.reward_for_actions(actions, labels)

            confusion["tp"] += int(((actions == 1) & (labels == 1)).sum().item())
            confusion["tn"] += int(((actions == 0) & (labels == 0)).sum().item())
            confusion["fp"] += int(((actions == 1) & (labels == 0)).sum().item())
            confusion["fn"] += int(((actions == 0) & (labels == 1)).sum().item())
            total_reward += float(rewards.sum().item())
            total_samples += int(labels.numel())

        tp = confusion["tp"]
        tn = confusion["tn"]
        fp = confusion["fp"]
        fn = confusion["fn"]
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        f1 = 2 * precision * recall / max(1e-12, precision + recall)

        return {
            "reward": total_reward / max(1, total_samples),
            "accuracy": (tp + tn) / max(1, total_samples),
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "false_positive_rate": fp / max(1, fp + tn),
            "false_negative_rate": fn / max(1, fn + tp),
            "decision_threshold": threshold,
            "samples": float(total_samples),
        }

    @torch.no_grad()
    def threshold_sweep(self, loader: Iterable[Batch], thresholds: Iterable[float]) -> Dict[str, Dict[str, float]]:
        """用同一个策略模型评估多个判黑阈值。"""
        return {
            f"{threshold:.3f}": self.evaluate(loader, decision_threshold=float(threshold))
            for threshold in thresholds
        }

    def train(self, train_loader: Iterable[Batch], val_loader: Iterable[Batch] | None = None) -> Dict[str, Dict[str, float]]:
        history: Dict[str, Dict[str, float]] = {}
        for epoch in range(1, self.config.max_epochs + 1):
            train_metrics = self.train_epoch(train_loader, epoch)
            history[f"train_epoch_{epoch}"] = train_metrics
            print(
                f"Train RL | Epoch {epoch} | loss={train_metrics['loss']:.4f} "
                f"reward={train_metrics['reward']:.4f} acc={train_metrics['accuracy']:.4f} "
                f"precision={train_metrics['precision']:.4f} recall={train_metrics['recall']:.4f}"
            )

            if val_loader is not None:
                val_metrics = self.evaluate(val_loader)
                history[f"val_epoch_{epoch}"] = val_metrics
                print(
                    f"Val RL   | Epoch {epoch} | reward={val_metrics['reward']:.4f} "
                    f"acc={val_metrics['accuracy']:.4f} precision={val_metrics['precision']:.4f} "
                    f"recall={val_metrics['recall']:.4f} fp_rate={val_metrics['false_positive_rate']:.4f} "
                    f"threshold={val_metrics['decision_threshold']:.2f}"
                )
        return history
