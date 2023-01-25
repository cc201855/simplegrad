from typing import Union, Tuple
from numbers import Number

import numpy as np

# 指定Tensor数据类型为float32
default_type = np.float32
# 可转换为ndarray类型的数据
ndarrayble = Union[Number, list, np.ndarray]


def ensure_ndarray(data: ndarrayble) -> np.ndarray:
    """
    将data转换为ndarray数据类型
    :param data:待转换数据
    :return:转换后的ndarray类型数据
    """
    if isinstance(data, np.ndarray):
        # 如果本身是ndarray，则直接返回
        return data
    # 其他情况则转换为Numpy数组
    return np.array(data, dtype=default_type)


# 可以转换为Tensor类型的数据
tensorable = Union[Number, "Tensor", np.ndarray]


def ensure_tensor(data: tensorable) -> "Tensor":
    """
    将data转换为tensor数据类型
    :param data:待转换数据
    :return:转换后的tensor类型数据
    """
    if isinstance(data, Tensor):
        # 如果本身是Tensor，则直接返回
        return data
    # 其他情况则转换为Tensor对象
    return Tensor(data)


class Tensor:
    def __init__(self, data: ndarrayble, requires_grad: bool = False) -> None:
        """
        初始化Tensor对象
        :param data:待转换数据
        :param requires_grad:是否需要计算梯度
        """
        # tensor内真实数据，为numpy的ndarray数组
        self._data = ensure_ndarray(data)
        # tensor内对应梯度
        self._grad = None
        self.requires_grad = requires_grad

        if self.requires_grad:
            # 如果需要计算梯度，先将梯度清零
            self.zero_grad()

        # 反向传播时计算图所保存的所需变量
        self._ctx = None

    def zero_grad(self) -> None:
        """
        将梯度初始化为0
        """
        # 直接将梯度置为零矩阵即可，但是需注意，梯度也是Tensor类型
        self._grad = Tensor(np.zeros_like(self._data, dtype=default_type))

    @property
    def grad(self):
        """
        :return:tensor对应梯度
        """
        return self._grad

    @property
    def data(self) -> np.ndarray:
        """
        :return: tensor内Numpy数据
        """
        return self._data

    @data.setter
    def data(self, value) -> None:
        """
        将value中的数据赋值到本tensor中的ndarray数组
        :param value:新数据
        """
        value = ensure_ndarray(value)
        # 检查维度是否匹配
        assert value.shape == self.shape
        self._data = value
        # 重新赋值后梯度清零
        self._grad = None

    @property
    def shape(self) -> Tuple:
        """
        :return: tensor形状
        """
        return self._data.shape

    @property
    def ndim(self) -> int:
        """
        :return: tensor内维度个数
        """
        return self._data.ndim

    @property
    def size(self) -> int:
        """
        :return: Tensor中元素的个数 等同于np.prod(a.shape)
        """
        return self._data.size

    @property
    def dtype(self) -> np.dtype:
        """
        :return: tensor内数据类型
        """
        return self._data.dtype

    def __repr__(self):
        return f"Tensor({self._data}, requires_grad={self.requires_grad})"

    def __len__(self):
        return len(self._data)

    def assign(self, value) -> "Tensor":
        """
        将value中的数据赋值到本tensor中
        :param value:新赋值数据
        :return:拥有新数据的tensor对象
        """
        value = ensure_tensor(value)
        # 检查维度是否匹配
        assert value.shape == self.shape
        # 这里不能写self._data = new_data._data
        self.data = value.data
        return self

    def numpy(self) -> np.ndarray:
        """
        将Tensor转换为Numpy数组
        :return:Numpy数组
        """
        return self._data
