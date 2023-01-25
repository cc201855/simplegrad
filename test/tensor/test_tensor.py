from core.tensor import Tensor
import numpy as np


def test_creat_tensor():
    t = Tensor(range(10))
    print(t)
    assert t.shape == (10,)
    assert t.size == 10
    assert t.dtype == np.float32


def test_tensor_graph():
    a, b = Tensor(2, requires_grad=True), Tensor(1, requires_grad=True)
    e = (a + b) * (b + 1)
    res = list(e._rev_topo_sort())
    print(res)


def test_tensor_grad():
    a, b = Tensor(2, requires_grad=True), Tensor(1, requires_grad=True)
    e = (a + b) * (b + 1)
    e.backward()
    assert a.grad.data == 2
    assert b.grad.data == 5
