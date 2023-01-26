import inspect

import numpy as np
from typing import List
from core.parameter import Parameter
from core.tensor import Tensor


class Model:
    """
    所有模型的基类
    """

    def parameters(self) -> List[Parameter]:
        """
        :return:返回模型所有参数
        """
        parameters = []
        for name, value in inspect.getmembers(self):
            if isinstance(value, Parameter):
                parameters.append(value)
            elif isinstance(value, Model):
                parameters.extend(value.parameters())
        return parameters

    def zero_grad(self) -> None:
        """
        清空模型参数的梯度
        """
        for p in self.parameters():
            p.zero_grad()

    def __call__(self, *args, **kwargs) -> Tensor:
        """
        进行模型前向运算
        :return:返回模型前向运算结果
        """
        return self.forward(*args, **kwargs)

    def forward(self, *args, **kwargs) -> Tensor:
        """
        进行模型前向运算
        :return:返回模型前向运算结果
        """
        raise NotImplemented("This model not implement forward pass!")
