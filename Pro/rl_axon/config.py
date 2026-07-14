"""强化学习实验配置。

这个分支不改动主项目的 TrainingConfig，因为 RL 训练需要的是奖励、熵奖励、
策略梯度等参数。单独放在 Pro 里，避免影响当前 fixed PE256 主配置。
"""

from dataclasses import dataclass


@dataclass
class RewardConfig:
    """恶意软件判定动作的奖励表。

    action=0 表示判为白文件，action=1 表示判为黑文件。
    用户当前更在意减少误报，因此 false_positive_penalty 默认更重。
    """

    true_negative_reward: float = 1.0
    true_positive_reward: float = 1.0
    false_positive_penalty: float = -2.0
    false_negative_penalty: float = -1.2


@dataclass
class RLTrainingConfig:
    """Pro 分支的最小 RL 训练配置。"""

    max_epochs: int = 6
    batch_size: int = 8
    learning_rate: float = 5e-4
    weight_decay: float = 1e-5
    gradient_clip: float = 1.0
    entropy_coef: float = 0.005
    use_expected_reward: bool = True
    moving_baseline_beta: float = 0.9
    decision_threshold: float = 0.65
    log_interval: int = 5
    device: str = "cpu"
