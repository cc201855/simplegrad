from core.model import Model
from core.tensor import Tensor


class _Loss(Model):
    """
    损失函数基类
    """
    reduction: str  # none | mean | sum

    def __init__(self, str: str = "mean") -> None:
        """
        :param str:损失函数聚合方式
        """
        self.reduction = str
