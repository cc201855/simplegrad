import random
import torch
import numpy as np

from core.tensor import Tensor
from test.tensor.util import judge_data, judge_grad, TEST_NUMBER, generate_dimension_number


def test_get_by_index():
    for k in range(TEST_NUMBER):
        a, = generate_dimension_number(1)
        b = int(random.uniform(0, a - 1))
        tx = torch.randn(a, dtype=torch.float32, requires_grad=True)
        x = Tensor(tx.data, requires_grad=True)

        z = x[b]
        tz = tx[b]
        assert judge_data(z, tz)

        z.backward()
        tz.backward()
        assert judge_grad(x, tx)


def test_slice():
    for k in range(TEST_NUMBER):
        a, = generate_dimension_number(1)
        b = int(random.uniform(0, a / 2))
        c = int(random.uniform(a / 2, a - 1))
        tx = torch.randn(a, dtype=torch.float32, requires_grad=True)
        x = Tensor(tx.data, requires_grad=True)

        z = x[b:c]
        tz = tx[b:c]
        assert judge_data(z, tz)

        grad = np.ones_like(z.data)
        z.backward(grad)
        tz.backward(torch.tensor(grad))
        assert judge_grad(x, tx)


def test_matrix_slice():
    for k in range(TEST_NUMBER):
        a, b = generate_dimension_number(2)
        c, d = int(random.uniform(0, a / 2)), int(random.uniform(a / 2, a - 1))
        e, f = int(random.uniform(0, b / 2)), int(random.uniform(b / 2, b - 1))
        tx = torch.randn((a, b), dtype=torch.float32, requires_grad=True)
        x = Tensor(tx.data, requires_grad=True)

        z = x[c:d, e:f]
        tz = tx[c:d, e:f]
        assert judge_data(z, tz)

        grad = np.ones_like(z.data)
        z.backward(grad)
        tz.backward(torch.tensor(grad))
        assert judge_grad(x, tx)
