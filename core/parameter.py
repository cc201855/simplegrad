from typing import Union
from core.tensor import Tensor, ndarrayble


class Parameter(Tensor):
    def __init__(self, data: Union[ndarrayble, Tensor]) -> None:
        if isinstance(data, Tensor):
            data = data.data
        # 模型参数都需要计算梯度
        super().__init__(data, requires_grad=True)
