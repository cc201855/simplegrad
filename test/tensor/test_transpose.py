import torch
import numpy as np

from core.tensor import Tensor
from test.tensor.util import judge_data, judge_grad, TEST_NUMBER, generate_dimension_number


def test_transpose():
    for k in range(TEST_NUMBER):
        a, b = generate_dimension_number(2)
        tx = torch.randn((a, b), dtype=torch.float32, requires_grad=True)
        x = Tensor(tx.data, requires_grad=True)

        z = x.T
        tz = tx.T
        assert z.shape == tz.shape
        assert judge_data(z, tz)

        grad = np.ones_like(z.data)
        z.backward(grad)
        tz.backward(torch.tensor(grad))
        assert judge_grad(x, tx)

def test_matrix_transpose():
    for k in range(TEST_NUMBER):
        a, b, c = generate_dimension_number(3, 1, 100)
        tx = torch.randn((a, b, c), dtype=torch.float32, requires_grad=True)
        x = Tensor(tx.data, requires_grad=True)

        z = x.transpose((0, 1, 2))
        tz = tx.permute(0, 1, 2)
        assert z.shape == tz.shape
        assert judge_data(z, tz)

        z = z.transpose((0, 2, 1))
        tz = tz.permute(0, 2, 1)
        assert z.shape == tz.shape
        assert judge_data(z, tz)
        z = z.transpose((1, 0, 2))
        tz = tz.permute(1, 0, 2)
        assert z.shape == tz.shape
        assert judge_data(z, tz)

        z = z.transpose((1, 2, 0))
        tz = tz.permute(1, 2, 0)
        assert z.shape == tz.shape
        assert judge_data(z, tz)

        z = z.transpose((2, 0, 1))
        tz = tz.permute(2, 0, 1)
        assert z.shape == tz.shape
        assert judge_data(z, tz)

        z = z.transpose((2, 1, 0))
        tz = tz.permute(2, 1, 0)
        assert z.shape == tz.shape
        assert judge_data(z, tz)

        grad = np.ones_like(z.data)
        z.backward(grad)
        tz.backward(torch.tensor(grad))
        assert judge_grad(x, tx)
