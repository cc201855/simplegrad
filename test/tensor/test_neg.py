import torch
import numpy as np

from core.tensor import Tensor
from test.tensor.util import judge_data, judge_grad, TEST_NUMBER, generate_dimension_number


def test_simple_neg():
    for k in range(TEST_NUMBER):
        a = np.random.randn()
        tx = torch.tensor(a, dtype=torch.float32, requires_grad=True)
        x = Tensor(tx.item(), requires_grad=True)

        z = -x
        tz = -tx

        assert judge_data(z, tz)

        z.backward()
        tz.backward()
        assert judge_grad(x, tx)

def test_array_neg():
    for k in range(TEST_NUMBER):
        a = generate_dimension_number(1)
        tx = torch.randn(a, dtype=torch.float32, requires_grad=True)
        x = Tensor(tx.data, requires_grad=True)

        z = -x
        tz = -tx
        assert judge_data(z, tz)

        grad = np.ones_like(z.data)
        z.backward(grad)
        tz.backward(torch.tensor(grad))
        assert judge_grad(x, tx)
