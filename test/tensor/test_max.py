import torch
import numpy as np

from core.tensor import Tensor
from test.tensor.util import judge_data, judge_grad, TEST_NUMBER, generate_dimension_number


def test_simple_max():
    for k in range(TEST_NUMBER):
        a = generate_dimension_number(1)
        tx = torch.randn(a, dtype=torch.float32, requires_grad=True)
        x = Tensor(tx.data, requires_grad=True)

        z = x.max()
        tz = tx.max()
        assert judge_data(z, tz)

        z.backward()
        tz.backward()
        assert judge_grad(x, tx)


def test_simple_max2():
    tx = torch.tensor([1, 2, 9, 9, 7, 9, 9], dtype=torch.float32, requires_grad=True)
    x = Tensor(tx.data, requires_grad=True)

    z = x.max()
    tz = tx.max()
    assert z.data == [9]  # 最大值还是9
    assert judge_data(z, tz)

    z.backward()
    tz.backward()

    # 但是有两个最大值，所以梯度被均分了
    assert x.grad.data.tolist() == [0, 0, 0.25, 0.25, 0, 0.25, 0.25]
    assert judge_grad(x, tx)


def test_matrix_max():
    a = np.array([[1., 1., 8., 9., 1.],
                  [4., 5., 9., 9., 8.],
                  [8., 6., 9., 7., 9.],
                  [8., 6., 1., 9., 8.]])

    tx = torch.tensor(a, dtype=torch.float32, requires_grad=True)
    x = Tensor(tx.data, requires_grad=True)

    z = x.max()
    tz = tx.max()

    assert z.data == [9]  # 最大值是9
    assert judge_data(z, tz)

    z.backward()
    tz.backward()

    # 总共有6个9
    np.testing.assert_array_almost_equal(x.grad.data, [[0, 0, 0, 1 / 6, 0],
                                                       [0, 0, 1 / 6, 1 / 6, 0],
                                                       [0, 0, 1 / 6, 0, 1 / 6],
                                                       [0, 0, 0, 1 / 6, 0]])
    assert judge_grad(x, tx)


def test_matrix_max2():
    for k in range(TEST_NUMBER):
        a, b = generate_dimension_number(2)
        tx = torch.randn((a, b), dtype=torch.float32, requires_grad=True)
        x = Tensor(tx.data, requires_grad=True)  # (a, b)

        y = x.max(axis=0)
        ty, idx = tx.max(dim=0)

        assert y.shape == ty.shape
        assert judge_data(y, ty, rtol=1.e-2, atol=1.e-2)

        grad = np.ones_like(y.data)
        y.backward(grad)
        ty.backward(torch.tensor(grad))
        assert judge_grad(x, tx, rtol=1.e-2, atol=1.e-2)
