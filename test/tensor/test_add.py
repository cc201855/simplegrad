import random
import torch
import numpy as np

from core.tensor import Tensor
from test.tensor.util import judge_data, judge_grad, TEST_NUMBER, generate_dimension_number


def test_simple_add():
    for k in range(TEST_NUMBER):
        tx = torch.randn(1, dtype=torch.float32, requires_grad=True)
        x = Tensor(tx.item(), requires_grad=True)
        y = random.randint(1, 1000)

        z = x + y
        tz = tx + y
        assert judge_data(z, tz)

        z.backward()
        tz.backward()
        assert judge_grad(x, tx)


def test_array_add():
    for k in range(TEST_NUMBER):
        a = generate_dimension_number(1)
        tx = torch.randn(a, dtype=torch.float32, requires_grad=True)
        ty = torch.randn(a, dtype=torch.float32, requires_grad=True)
        x = Tensor(tx.data, requires_grad=True)
        y = Tensor(ty.data, requires_grad=True)

        z = x + y
        tz = tx + ty
        assert judge_data(z, tz)

        # 如果
        grad = np.ones_like(z.data)
        z.backward(grad)
        tz.backward(torch.tensor(grad))

        assert judge_grad(x, tx)
        assert judge_grad(y, ty)

        original_data = x.data
        x += 1
        assert x.grad is None
        assert judge_data(x.data, Tensor(original_data + 1))


def test_broadcast_add():
    """
    测试当发生广播时，我们的代码还能表现正常吗。
    对于 z = x + y
    如果x.shape == y.shape，那么就像上面的例子一样，没什么问题。
    如果x.shape == (2,3)  y.shape == (3,) 那么，根据广播，先会在y左边插入一个维度1，变成 -> y.shape == (1,3)
        接着，在第0个维度上进行复制，使得新的维度 y.shape == (2,3)
    这样的话，对x求梯度时，梯度要和x的shape保持一致；对y求梯度时，也要和y的shape保持一致。
    """
    for k in range(TEST_NUMBER):
        a, b = generate_dimension_number(2)
        tx = torch.randn((a, b), dtype=torch.float32, requires_grad=True)
        ty = torch.randn(b, dtype=torch.float32, requires_grad=True)
        x = Tensor(tx.data, requires_grad=True)  # (a, b)
        y = Tensor(ty.data, requires_grad=True)  # (b,)

        z = x + y  # (a, b)
        tz = tx + ty

        assert z.shape == (a, b)
        assert judge_data(z, tz)

        grad = np.ones_like(z.data)
        z.backward(grad)  # grad.shape == z.shape
        tz.backward(torch.tensor(grad))

        assert judge_grad(x, tx)
        assert judge_grad(y, ty)


def test_broadcast_add2():
    for k in range(TEST_NUMBER):
        a, b = generate_dimension_number(2)
        tx = torch.randn((a, b), dtype=torch.float32, requires_grad=True)
        ty = torch.randn((1, b), dtype=torch.float32, requires_grad=True)
        x = Tensor(tx.data, requires_grad=True)  # (a, b)
        y = Tensor(ty.data, requires_grad=True)  # (1, b)

        z = x + y  # (2,3)
        tz = tx + ty
        assert judge_data(z, tz)

        grad = np.ones_like(z.data)
        z.backward(grad)  # grad.shape == z.shape
        tz.backward(torch.tensor(grad))

        assert judge_grad(x, tx)
        assert judge_grad(y, ty)
