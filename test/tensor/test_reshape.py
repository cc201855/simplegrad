import torch
import numpy as np

from core.tensor import Tensor
from test.tensor.util import judge_data, judge_grad, TEST_NUMBER, generate_dimension_number


def test_reshape():
    for k in range(TEST_NUMBER):
        a, b = generate_dimension_number(2)
        tx = torch.randn(a * b, dtype=torch.float32, requires_grad=True)
        x = Tensor(tx.data, requires_grad=True)

        z = x.reshape((a, b))
        tz = tx.reshape(a, b)
        assert judge_data(z, tz)

        grad = np.ones_like(z.data)
        z.backward(grad)
        tz.backward(torch.tensor(grad))
        assert judge_grad(x, tx)


def test_matrix_reshape():
    for k in range(TEST_NUMBER):
        a, b, c, d = generate_dimension_number(4, 1, 100)
        tx = torch.randn((a * b, c * d), dtype=torch.float32, requires_grad=True)
        x = Tensor(tx.data, requires_grad=True)

        z = x.reshape((a*c, b*d))
        tz = tx.reshape(a*c, b*d)
        assert judge_data(z, tz)

        grad = np.ones_like(z.data)
        z.backward(grad)
        tz.backward(torch.tensor(grad))
        assert judge_grad(x, tx)
