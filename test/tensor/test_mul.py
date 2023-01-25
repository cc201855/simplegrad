import random
import numpy as np
import torch

from core.tensor import Tensor
from test.tensor.util import judge_data, judge_grad, TEST_NUMBER, generate_dimension_number


def test_simple_mul():
    for k in range(TEST_NUMBER):
        tx = torch.randn(1, dtype=torch.float32, requires_grad=True)
        x = Tensor(tx.item(), requires_grad=True)
        y = random.random()
        z = x * y
        tz = tx * y
        assert judge_data(z, tz)
        z.backward()
        tz.backward()
        assert judge_grad(x, tx)


def test_array_mul():
    for k in range(TEST_NUMBER):
        a = generate_dimension_number(1)
        tx = torch.randn(a, dtype=torch.float32, requires_grad=True)
        ty = torch.randn(a, dtype=torch.float32, requires_grad=True)
        x = Tensor(tx.data, requires_grad=True)
        y = Tensor(ty.data, requires_grad=True)

        z = x * y
        tz = tx * ty
        assert judge_data(z, tz)

        grad = np.ones_like(z.data)
        z.backward(grad)
        tz.backward(torch.tensor(grad))

        assert judge_grad(x, tx)
        assert judge_grad(y, ty)

        original_data = x.data
        x *= 0.1
        assert x.grad is None
        assert judge_data(x.data, Tensor(original_data * 0.1))


def test_broadcast_mul():
    for k in range(TEST_NUMBER):
        a, b = generate_dimension_number(2)
        tx = torch.randn((a, b), dtype=torch.float32, requires_grad=True)
        ty = torch.randn(b, dtype=torch.float32, requires_grad=True)
        x = Tensor(tx.data, requires_grad=True)  # (a, b)
        y = Tensor(ty.data, requires_grad=True)  # (b,)

        z = x * y  # (a,b) * (b,) => (a,b) * (a,b) -> (a,b)
        tz = tx * ty

        assert judge_data(z, tz)

        grad = np.ones_like(z.data)
        z.backward(grad)  # grad.shape == z.shape
        tz.backward(torch.tensor(grad))

        assert judge_grad(x, tx)
        assert judge_grad(y, ty)


def test_broadcast_mul2():
    for k in range(TEST_NUMBER):
        a, b = generate_dimension_number(2)
        tx = torch.randn((a, b), dtype=torch.float32, requires_grad=True)
        ty = torch.randn((1, b), dtype=torch.float32, requires_grad=True)
        x = Tensor(tx.data, requires_grad=True)  # (a, b)
        y = Tensor(ty.data, requires_grad=True)  # (1, b)

        z = x * y  # (a,b) * (1, b) => (a,b) * (a,b) -> (a,b)
        tz = tx * ty

        assert judge_data(z, tz)

        grad = np.ones_like(z.data)
        z.backward(grad)  # grad.shape == z.shape
        tz.backward(torch.tensor(grad))

        assert judge_grad(x, tx)
        assert judge_grad(y, ty)
