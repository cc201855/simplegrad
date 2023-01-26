import torch
import numpy as np

from core.tensor import Tensor
from test.tensor.util import judge_data, judge_grad, TEST_NUMBER, generate_dimension_number


def test_simple_pow():
    for k in range(TEST_NUMBER):
        a = np.random.randint(1, 10)
        b = np.random.randint(1, 10)
        tx = torch.tensor(a, dtype=torch.float32, requires_grad=True)
        x = Tensor(tx.item(), requires_grad=True)

        z = x.pow(b)  # x**b
        tz = tx.pow(b)  # tx**b

        assert judge_data(z, tz)

        z.backward()
        tz.backward()
        assert judge_grad(x, tx)


def test_array_pow():
    for k in range(TEST_NUMBER):
        a = generate_dimension_number(1)
        b = np.random.randint(1, 10)
        tx = torch.randn(a, dtype=torch.float32, requires_grad=True)
        x = Tensor(tx.data, requires_grad=True)

        z = x ** b
        tz = tx ** b
        assert judge_data(z, tz)

        grad = np.ones_like(z.data)
        z.backward(grad)
        tz.backward(torch.tensor(grad))
        assert judge_grad(x, tx)
