from core.tensor import Tensor
import numpy as np


def test_creat_tensor():
    t = Tensor(range(10))
    print(t)
    assert t.shape == (10,)
    assert t.size == 10
    assert t.dtype == np.float32
