"""训练与验证循环。"""

from .evaluator import evaluate
from .trainer import train_one_epoch

__all__ = ["evaluate", "train_one_epoch"]

