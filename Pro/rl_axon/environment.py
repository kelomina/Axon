"""把恶意软件分类任务包装成一阶强化学习环境。"""

from __future__ import annotations

from dataclasses import asdict
from typing import Dict, Tuple

import torch
from torch.utils.data import Dataset

from .config import RewardConfig


class MalwareBanditEnv:
    """一阶恶意软件判定环境。

    这里的“环境”不是游戏地图，而是奖励规则。模型每看到一个样本就选择：
    0=白文件，1=黑文件。环境根据真实标签返回奖励。
    """

    def __init__(self, reward_config: RewardConfig | None = None):
        self.reward_config = reward_config or RewardConfig()

    def reward_for_actions(self, actions: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """根据动作和真实标签返回每个样本的奖励。"""
        actions = actions.long()
        labels = labels.long()

        true_negative = (actions == 0) & (labels == 0)
        true_positive = (actions == 1) & (labels == 1)
        false_positive = (actions == 1) & (labels == 0)
        false_negative = (actions == 0) & (labels == 1)

        rewards = torch.zeros_like(labels, dtype=torch.float32)
        rewards = torch.where(
            true_negative,
            torch.full_like(rewards, self.reward_config.true_negative_reward),
            rewards,
        )
        rewards = torch.where(
            true_positive,
            torch.full_like(rewards, self.reward_config.true_positive_reward),
            rewards,
        )
        rewards = torch.where(
            false_positive,
            torch.full_like(rewards, self.reward_config.false_positive_penalty),
            rewards,
        )
        rewards = torch.where(
            false_negative,
            torch.full_like(rewards, self.reward_config.false_negative_penalty),
            rewards,
        )
        return rewards

    def reward_table(self, labels: torch.Tensor) -> torch.Tensor:
        """返回每个样本对 action=0/action=1 的奖励表。

        形状是 [batch, 2]。它让 smoke 测试能使用稳定的期望奖励训练，
        不必依赖大量随机采样才能看到梯度。
        """
        action_zero = torch.zeros_like(labels)
        action_one = torch.ones_like(labels)
        return torch.stack(
            [
                self.reward_for_actions(action_zero, labels),
                self.reward_for_actions(action_one, labels),
            ],
            dim=-1,
        )

    def metrics(self, actions: torch.Tensor, labels: torch.Tensor) -> Dict[str, float]:
        """计算 RL 分支关心的基础结果。"""
        actions = actions.long()
        labels = labels.long()
        rewards = self.reward_for_actions(actions, labels)

        true_positive = ((actions == 1) & (labels == 1)).sum().item()
        true_negative = ((actions == 0) & (labels == 0)).sum().item()
        false_positive = ((actions == 1) & (labels == 0)).sum().item()
        false_negative = ((actions == 0) & (labels == 1)).sum().item()
        total = max(1, labels.numel())

        precision = true_positive / max(1, true_positive + false_positive)
        recall = true_positive / max(1, true_positive + false_negative)
        f1 = 2 * precision * recall / max(1e-12, precision + recall)

        return {
            "reward": float(rewards.mean().item()),
            "accuracy": float((actions == labels).float().mean().item()),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "false_positive_rate": float(false_positive / max(1, false_positive + true_negative)),
            "false_negative_rate": float(false_negative / max(1, false_negative + true_positive)),
            "samples": float(total),
        }

    def to_dict(self) -> Dict[str, float]:
        return asdict(self.reward_config)


class SyntheticMalwareDataset(Dataset):
    """smoke 测试用合成数据集。

    这个数据集只用于验证 RL 分支是否能完整跑通。它不会模拟真实 PE 文件，
    只是在 byte/PE/stat 三类输入中注入一点可学习信号。
    """

    def __init__(
        self,
        num_samples: int,
        max_byte_length: int,
        pe_feature_dim: int,
        stat_feature_dim: int,
        seed: int = 42,
    ):
        self.num_samples = num_samples
        self.max_byte_length = max_byte_length
        self.pe_feature_dim = pe_feature_dim
        self.stat_feature_dim = stat_feature_dim

        generator = torch.Generator().manual_seed(seed)
        labels = torch.arange(num_samples, dtype=torch.long) % 2
        labels = labels[torch.randperm(num_samples, generator=generator)]
        self.labels = labels

        self.byte_sequences = torch.empty(num_samples, max_byte_length, dtype=torch.long)
        self.pe_features = torch.randn(num_samples, pe_feature_dim, generator=generator) * 0.05
        self.stat_features = torch.randn(num_samples, stat_feature_dim, generator=generator) * 0.05

        for idx, label in enumerate(labels.tolist()):
            if label == 1:
                # 黑样本：高字节值更多，PE/stat 前几列也偏高。
                self.byte_sequences[idx] = torch.randint(
                    96, 256, (max_byte_length,), generator=generator, dtype=torch.long
                )
                self.pe_features[idx, 0] += 0.85
                self.pe_features[idx, 5] += 0.70
                self.pe_features[idx, 6] += 0.65
                self.stat_features[idx, 0] += 0.80
                self.stat_features[idx, 10] += 0.70
            else:
                # 白样本：低字节值更多，PE/stat 前几列偏低。
                self.byte_sequences[idx] = torch.randint(
                    0, 160, (max_byte_length,), generator=generator, dtype=torch.long
                )
                self.pe_features[idx, 0] -= 0.55
                self.pe_features[idx, 5] -= 0.45
                self.pe_features[idx, 6] -= 0.35
                self.stat_features[idx, 0] -= 0.50
                self.stat_features[idx, 10] -= 0.40

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            self.byte_sequences[idx],
            self.pe_features[idx].float(),
            self.stat_features[idx].float(),
            self.labels[idx],
        )
