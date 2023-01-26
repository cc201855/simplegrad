import random

import numpy as np
import torch

from core.tensor import Tensor
from test.tensor.util import judge_data, judge_grad, TEST_NUMBER, generate_dimension_number


def test_simple_matmul():
    for k in range(TEST_NUMBER):
        a, b, c = generate_dimension_number(3)

        tx = torch.randn((a, b), dtype=torch.float32, requires_grad=True)
        ty = torch.randn((b, c), dtype=torch.float32, requires_grad=True)

        x = Tensor(tx.data, requires_grad=True)  # (a, b)
        y = Tensor(ty.data, requires_grad=True)  # (b, c)

        z = x @ y  # (a, b) @ (b, c) -> (a, c)

        tz = tx @ ty

        assert judge_data(z, tz)

        grad = np.ones_like(z.data)
        z.backward(grad)
        tz.backward(torch.tensor(grad))

        assert judge_grad(x, tx)
        assert judge_grad(y, ty)


def test_broadcast_matmul():
    for k in range(TEST_NUMBER):
        N, a, b, c = generate_dimension_number(4, 2, 100)

        tx = torch.randn((N, a, b), dtype=torch.float32, requires_grad=True)  # (N, a, b)
        ty = torch.randn((b, c), dtype=torch.float32, requires_grad=True)  # (b, c)

        x = Tensor(tx.data, requires_grad=True)  # (N, a, b)
        y = Tensor(ty.data, requires_grad=True)  # (b, c)

        z = x @ y  # (N,a,b) @ (b,c) -> (N,a,b) @ (1,b,c) => (N,a,b) @ (N,b,c)  -> (N,a,c)
        tz = tx @ ty
        assert z.shape == (N, a, c)
        assert judge_data(z, tz, rtol=1.e-3, atol=1.e-3)

        z.backward(np.ones_like(z.data))
        tz.backward(torch.tensor(np.ones_like(z.data)))

        # 和老大哥 pytorch保持一致就行了
        assert judge_grad(x, tx, rtol=1.e-3, atol=1.e-3)
        assert judge_grad(y, ty, rtol=1.e-3, atol=1.e-3)
