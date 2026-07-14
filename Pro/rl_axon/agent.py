"""把现有 Axon 深度学习模型包装成 RL 策略网络。"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
from torch.distributions import Categorical


class AxonPolicyAgent(nn.Module):
    """策略智能体。

    复用主项目的 AxonMalwareModel。原模型输出二分类 logits；
    在 RL 分支中，这两个 logits 被解释成 action=0/1 的动作偏好。
    """

    def __init__(self, base_model: nn.Module):
        super().__init__()
        self.base_model = base_model

    def forward(
        self,
        byte_seq: torch.Tensor,
        pe_features: torch.Tensor,
        stat_features: torch.Tensor,
    ) -> torch.Tensor:
        output = self.base_model(byte_seq, pe_features, stat_features)
        return output["logits"]

    def distribution(
        self,
        byte_seq: torch.Tensor,
        pe_features: torch.Tensor,
        stat_features: torch.Tensor,
    ) -> Categorical:
        logits = self.forward(byte_seq, pe_features, stat_features)
        return Categorical(logits=logits)

    def act(
        self,
        byte_seq: torch.Tensor,
        pe_features: torch.Tensor,
        stat_features: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        dist = self.distribution(byte_seq, pe_features, stat_features)
        actions = dist.sample()
        return {
            "actions": actions,
            "log_probs": dist.log_prob(actions),
            "entropy": dist.entropy(),
            "probs": dist.probs,
        }

    def probabilities(
        self,
        byte_seq: torch.Tensor,
        pe_features: torch.Tensor,
        stat_features: torch.Tensor,
    ) -> torch.Tensor:
        """返回 action=0/1 的概率。"""
        logits = self.forward(byte_seq, pe_features, stat_features)
        return torch.softmax(logits, dim=-1)

    def threshold_actions(
        self,
        byte_seq: torch.Tensor,
        pe_features: torch.Tensor,
        stat_features: torch.Tensor,
        decision_threshold: float = 0.5,
    ) -> torch.Tensor:
        """按判黑阈值输出动作。

        action=1 是“判为黑文件”。阈值越高，模型越保守，误报通常越少。
        """
        probs = self.probabilities(byte_seq, pe_features, stat_features)
        return (probs[:, 1] >= decision_threshold).long()

    def greedy_actions(
        self,
        byte_seq: torch.Tensor,
        pe_features: torch.Tensor,
        stat_features: torch.Tensor,
    ) -> torch.Tensor:
        logits = self.forward(byte_seq, pe_features, stat_features)
        return logits.argmax(dim=-1)
