import numpy as np
import torch

from core.tensor import Tensor
from test.tensor.util import judge_data, judge_grad, TEST_NUMBER, generate_dimension_number


def test_simple_sum():
    for k in range(TEST_NUMBER):
        a = generate_dimension_number(1)
        tx = torch.randn(a, dtype=torch.float32, requires_grad=True)
        x = Tensor(tx.data, requires_grad=True)
        y = x.sum()
        ty = tx.sum()

        assert judge_data(y, ty, rtol=1.e-3, atol=1.e-3)

        y.backward()
        ty.backward()

        assert judge_grad(x, tx, rtol=1.e-3, atol=1.e-3)


def test_sum_with_grad():
    for k in range(TEST_NUMBER):
        a = generate_dimension_number(1)
        tx = torch.randn(a, dtype=torch.float32, requires_grad=True)
        x = Tensor(tx.data, requires_grad=True)
        y = x.sum()
        ty = tx.sum()

        assert judge_data(y, ty, rtol=1.e-3, atol=1.e-3)

        grad = np.array(3)
        y.backward(grad)
        ty.backward(torch.tensor(grad))

        assert judge_grad(x, tx, rtol=1.e-3, atol=1.e-3)


def test_matrix_sum():
    for k in range(TEST_NUMBER):
        a, b = generate_dimension_number(2)
        tx = torch.randn((a, b), dtype=torch.float32, requires_grad=True)
        x = Tensor(tx.data, requires_grad=True)  # (a, b)
        y = x.sum()
        ty = tx.sum()

        assert judge_data(y, ty, rtol=1.e-3, atol=1.e-3)

        y.backward()
        ty.backward()
        assert judge_grad(x, tx, rtol=1.e-3, atol=1.e-3)


def test_matrix_with_axis():
    for k in range(TEST_NUMBER):
        a, b = generate_dimension_number(2)
        tx = torch.randn((a, b), dtype=torch.float32, requires_grad=True)
        x = Tensor(tx.data, requires_grad=True)  # (a, b)
        y = x.sum(axis=0)# keepdims = False
        ty = tx.sum(dim=0)

        assert y.shape == ty.shape
        assert judge_data(y, ty, rtol=1.e-3, atol=1.e-3)

        grad = np.ones_like(y.data)
        y.backward(grad)
        ty.backward(torch.tensor(grad))
        assert judge_grad(x, tx, rtol=1.e-3, atol=1.e-3)


def test_matrix_with_keepdims():
    for k in range(TEST_NUMBER):
        a, b = generate_dimension_number(2)
        tx = torch.randn((a, b), dtype=torch.float32, requires_grad=True)
        x = Tensor(tx.data, requires_grad=True)  # (a, b)
        y = x.sum(axis=0, keepdims=True)# keepdims = True
        ty = tx.sum(dim=0, keepdims=True)

        assert y.shape == ty.shape
        assert judge_data(y, ty, rtol=1.e-3, atol=1.e-3)

        grad = np.ones_like(y.data)
        y.backward(grad)
        ty.backward(torch.tensor(grad))
        assert judge_grad(x, tx, rtol=1.e-3, atol=1.e-3)
