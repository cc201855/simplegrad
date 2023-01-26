from typing import Any, Tuple
from core.tensor import Tensor
from numpy import ndarray
import numpy as np


class _Function:
    def __init__(self, *tensors: "Tensor") -> None:
        """
        :param tensors:该操作所依赖的所有输入
        """
        self.depends_on = [t for t in tensors]
        # 保存需要在backward()中使用的Tensor或其他信息（如shape）
        self.saved_tensors = []

    def save_for_backward(ctx, *x: Any) -> None:
        """
        :param x:待保存Tensor
        """
        ctx.saved_tensors.extend(x)

    def __new__(cls, *args, **kwargs):
        """
        ___new___为静态方法，当该类被实例化时调用
        """
        # 将forward和backward转换为静态方法，可以通过类名直接调用
        cls.forward = staticmethod(cls.forward)
        cls.backward = staticmethod(cls.backward)
        cls.apply = staticmethod(cls.apply)
        return super().__new__(cls)

    def forward(ctx, *args: Any, **kwargs: Any) -> ndarray:
        """
        进行前向传播，真正运算的地方
        :return: 返回运算结果
        """
        return NotImplemented("You must implement the forward function for custom Function.")

    def backward(ctx, grad: ndarray) -> Any:
        """
        实现反向传播，计算梯度
        :param grad:前一个运算梯度
        :return:返回计算后的梯度
        """
        return NotImplemented("You must implement the backward method for your custom Function" +
                              "to use it with backward mode AD.")

    def apply(fxn, *xs: "Tensor", **kwargs) -> "Tensor":
        """
        与PyTorch相同，我们也不直接调用forward，而是调用此方法
        :param ctx:
        :param xs:
        :param kwargs:
        :return:
        """
        # 先调用构造函数，传入运算依赖的Tensor
        ctx = fxn(*xs)  # 调用到了_Function的__init__方法，传入该运算所依赖的输入。

        # [t.data for t in xs] 遍历xs中的所有tensor，并取出data(np.ndarray)
        # 即实际参与运算的都是ndarray数组
        ret = Tensor(ctx.forward(ctx, *[t.data for t in xs], **kwargs),
                     requires_grad=any([t.requires_grad for t in xs]))

        if ret.requires_grad:
            # 如果需要求梯度，那么保存上下文信息
            ret._ctx = ctx

        return ret


def unbroadcast(grad: ndarray, in_shape: Tuple) -> ndarray:
    """
    广播操作的逆操作，确保grad转换成in_shape的形状
    :param grad:梯度
    :param in_shape:梯度要转换的形状
    :return:转换后的梯度
    """
    # 首先计算维度个数之差
    ndims_added = grad.ndim - len(in_shape)
    # 由于广播时，先从左边插入，再进行复制，所以逆操作时，也从左边开始，进行复制的逆操作（求和）
    for _ in range(ndims_added):
        # 在axis=0上进行求和，去掉第0个维度，如果ndims_added > 1，就需要不停的在第0个维度上面求和
        grad = np.sum(grad, axis=0)

    # 处理 (2,3) + (1,3) => (2,3) grad的情况
    # 看in_shape中有没有维度=1的情况
    for i, dim in enumerate(in_shape):
        if dim == 1:
            # 那么需要在该axis上求和，并且保持维度 这里(2,3) => (1,3) grad 就和输入维度保持一致了
            grad = np.sum(grad, axis=i, keepdims=True)
    return grad


# **********二元运算**********
class Add(_Function):
    def forward(ctx, x: ndarray, y: ndarray) -> ndarray:
        """
        实现 z = x+y,这里的x，y都是numpy数组，因此可能发生广播操作
        反向传播时需要注意
        :param x: tensorA
        :param y: tensorB
        :return: A+B
        """
        # 只需要保存输入各自的形状即可
        ctx.save_for_backward(x.shape, y.shape)
        # 进行真正运算
        return x + y

    def backward(ctx, grad: ndarray) -> Tuple[ndarray, ndarray]:
        """
        z = x+y
        输入有两个，因此需要计算的梯度也是两个，因此输出也是两个
        ∂l/∂x = (∂l/∂z) * (∂z/∂x) = ∂l/∂z = grad
        ∂l/∂y = (∂l/∂z) * (∂z/∂y) = ∂l/∂z = grad
        :param grad: 上层节点的梯度
        :return:加法算子计算出的梯度
        """
        shape_x, shape_y = ctx.saved_tensors
        # 如有广播，进行还原
        return unbroadcast(grad, shape_x), unbroadcast(grad, shape_y)


class Sub(_Function):
    def forward(ctx, x: ndarray, y: ndarray) -> ndarray:
        """
        实现 z = x-y,这里的x，y都是numpy数组，因此可能发生广播操作
        反向传播时需要注意
        :param x: tensorA
        :param y: tensorB
        :return: A-B
        """
        # 只需要保存输入各自的形状即可
        ctx.save_for_backward(x.shape, y.shape)
        # 进行真正运算
        return x - y

    def backward(ctx, grad: ndarray) -> Tuple[ndarray, ndarray]:
        """
        输入有两个，因此需要计算的梯度也是两个，因此输出也是两个
        z = x-y
        ∂l/∂x = (∂l/∂z) * (∂z/∂x) = ∂l/∂z  = grad
        ∂l/∂y = (∂l/∂z) * (∂z/∂y) = ∂l/∂z * -1 = grad * -1
        :param grad: 上层节点的梯度
        :return:减法算子计算出的梯度
        """
        shape_x, shape_y = ctx.saved_tensors
        # 如有广播，进行还原
        return unbroadcast(grad, shape_x), unbroadcast(-grad, shape_y)


class Mul(_Function):
    def forward(ctx, x: ndarray, y: ndarray) -> ndarray:
        """
        实现 z = x*y,这里的x，y都是numpy数组，因此可能发生广播操作
        反向传播时需要注意
        :param x: tensorA
        :param y: tensorB
        :return: A*B
        """
        # 求梯度时需要x, y，因此进行保存操作
        ctx.save_for_backward(x, y)
        # 进行真正运算
        return x * y

    def backward(ctx, grad: ndarray) -> Tuple[ndarray, ndarray]:
        """
        z = x*y
        输入有两个，因此需要计算的梯度也是两个，因此输出也是两个
        ∂l/∂x = (∂l/∂z) * (∂z/∂x) = ∂l/∂z * y = grad * y
        ∂l/∂y = (∂l/∂z) * (∂z/∂y) = ∂l/∂z * x = grad * x
        :param grad: 上层节点的梯度
        :return:乘法算子计算出的梯度
        """
        x, y = ctx.saved_tensors
        # 如有广播，进行还原
        return unbroadcast(grad * y, x.shape), unbroadcast(grad * x, y.shape)

# __truediv__ 相关魔法方法实现了/，因此名字设置为这样方便注入
class TrueDiv(_Function):
    def forward(ctx, x: ndarray, y: ndarray) -> ndarray:
        """
        实现 z = x/y,这里的x，y都是numpy数组，因此可能发生广播操作
        反向传播时需要注意
        :param x: tensorA
        :param y: tensorB
        :return: A/B
        """
        # 求梯度时需要x, y，因此进行保存操作
        ctx.save_for_backward(x, y)
        # 进行真正运算
        return x / y

    def backward(ctx, grad: ndarray) -> Tuple[ndarray, ndarray]:
        """
        z = x/y
        输入有两个，因此需要计算的梯度也是两个，因此输出也是两个
        ∂l/∂x = (∂l/∂z) * (∂z/∂x) = ∂l/∂z / y = grad / y
        ∂l/∂y = (∂l/∂z) * (∂z/∂y) = ∂l/∂z * (-x/y^2) = grad * (-x/y^2)
        :param grad: 上层节点的梯度
        :return:除法算子计算出的梯度
        """
        x, y = ctx.saved_tensors
        # 如有广播，进行还原
        return unbroadcast(grad / y, x.shape), unbroadcast(grad * (-x / np.power(y, 2)), y.shape)

# **********矩阵运算**********
class Matmul(_Function):
    def forward(ctx, x: ndarray, y: ndarray) -> ndarray:
        """
        实现 z = X@Y, 这里的x，y都是numpy数组
        :param x: tensorA
        :param y: tensorB
        :return: A @ B
        """
        assert x.ndim > 1 and y.ndim > 1, f"the dim number of x or y must >= 2, actual x:{x.ndim}, y:{y.ndim}"
        # 求梯度时需要x, y，因此进行保存操作
        ctx.save_for_backward(x, y)
        # 进行真正运算
        return np.matmul(x, y)

    def backward(ctx, grad: ndarray) -> Tuple[ndarray, ndarray]:
        """
        z = x@y
        输入有两个，因此需要计算的梯度也是两个，因此输出也是两个
        ∂l/∂x = (∂l/∂z) * (∂z/∂x) = ∂l/∂z @ yT = grad @ y^T
        ∂l/∂y = (∂l/∂z) * (∂z/∂y) = xT @ ∂l/∂z = xT @ grad
        :param grad: 上层节点的梯度
        :return:矩阵乘法算子计算出的梯度
        """
        x, y = ctx.saved_tensors
        # 如有广播，进行还原
        return unbroadcast(np.matmul(grad, y.swapaxes(-2, -1)), x.shape), unbroadcast(
            np.matmul(x.swapaxes(-2, -1), grad), y.shape)
