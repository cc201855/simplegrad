import numpy as np

from core.tensor import Tensor
import torch
import random

TEST_NUMBER = 500
DIMENSION_LOWER = 2
DIMENSION_UPPER = 1000


def generate_dimension_number(number: int, lower: int = DIMENSION_LOWER, upper: int = DIMENSION_UPPER):
    """
    随机生成维度在[DIMENSION_LOWER, DIMENSION_UPPER]之间的number个数
    :param number:dimension数量
    :param lower:dimension下界
    :param upper:dimension上界
    """
    return [random.randint(lower, upper) for i in range(number)]


def judge_data(a: Tensor, b: torch.tensor, rtol=1.e-4, atol=1.e-4):
    """
    比较自己的Tensor与PyTorch的实现算数值差距
    :param a:自己实现的Tensor
    :param b:PyTorch的Tensor
    :param rtol:允许的差值
    :param atol:绝对值差值
    :return:
    """
    return np.allclose(a.data, b.data, rtol=rtol, atol=atol)


def judge_grad(a: Tensor, b: torch.tensor, rtol=1.e-4, atol=1.e-4):
    """
    比较自己的Tensor与PyTorch的实现差梯度值距
    :param a:自己实现的Tensor
    :param b:PyTorch的Tensor
    :param rtol:允许的差值
    :param atol:绝对值差值
    :return:
    """
    return np.allclose(a.grad.data, b.grad.data, rtol=rtol, atol=atol)
