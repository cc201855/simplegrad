from typing import Union, Tuple, Any
from numbers import Number

import numpy as np

# 指定Tensor数据类型为float32
default_type = np.float32
# 可转换为ndarray类型的数据
ndarrayble = Union[Number, list, np.ndarray]


def ensure_ndarray(data: ndarrayble) -> np.ndarray:
    """
    将data转换为ndarray数据类型
    :param data:待转换数据
    :return:转换后的ndarray类型数据
    """
    if isinstance(data, np.ndarray):
        # 如果本身是ndarray，则直接返回
        return data
    # 其他情况则转换为Numpy数组
    return np.array(data, dtype=default_type)


# 可以转换为Tensor类型的数据
tensorable = Union[Number, "Tensor", np.ndarray]


def ensure_tensor(data: tensorable) -> "Tensor":
    """
    将data转换为tensor数据类型
    :param data:待转换数据
    :return:转换后的tensor类型数据
    """
    if isinstance(data, Tensor):
        # 如果本身是Tensor，则直接返回
        return data
    # 其他情况则转换为Tensor对象
    return Tensor(data)


class Tensor:
    def __init__(self, data: ndarrayble, requires_grad: bool = False) -> None:
        """
        初始化Tensor对象
        :param data:待转换数据
        :param requires_grad:是否需要计算梯度
        """
        # tensor内真实数据，为numpy的ndarray数组
        self._data = ensure_ndarray(data)
        # tensor内对应梯度
        self._grad = None
        self.requires_grad = requires_grad

        if self.requires_grad:
            # 如果需要计算梯度，先将梯度清零
            self.zero_grad()

        # 反向传播时计算图所保存的所需变量
        self._ctx = None

    def zero_grad(self) -> None:
        """
        将梯度初始化为0
        """
        # 直接将梯度置为零矩阵即可，但是需注意，梯度也是Tensor类型
        self._grad = Tensor(np.zeros_like(self._data, dtype=default_type))

    @property
    def grad(self):
        """
        :return:tensor对应梯度
        """
        return self._grad

    @property
    def data(self) -> np.ndarray:
        """
        :return: tensor内Numpy数据
        """
        return self._data

    @data.setter
    def data(self, value) -> None:
        """
        将value中的数据赋值到本tensor中的ndarray数组
        :param value:新数据
        """
        value = ensure_ndarray(value)
        # 检查维度是否匹配
        assert value.shape == self.shape
        self._data = value
        # 重新赋值后梯度清零
        self._grad = None

    @property
    def shape(self) -> Tuple:
        """
        :return: tensor形状
        """
        return self._data.shape

    @property
    def ndim(self) -> int:
        """
        :return: tensor内维度个数
        """
        return self._data.ndim

    @property
    def size(self) -> int:
        """
        :return: Tensor中元素的个数 等同于np.prod(a.shape)
        """
        return self._data.size

    @property
    def dtype(self) -> np.dtype:
        """
        :return: tensor内数据类型
        """
        return self._data.dtype

    def __repr__(self):
        return f"Tensor({self._data}, requires_grad={self.requires_grad})"

    def __len__(self):
        return len(self._data)

    def assign(self, value) -> "Tensor":
        """
        将value中的数据赋值到本tensor中
        :param value:新赋值数据
        :return:拥有新数据的tensor对象
        """
        value = ensure_tensor(value)
        # 检查维度是否匹配
        assert value.shape == self.shape
        # 这里不能写self._data = new_data._data
        self.data = value.data
        return self

    def numpy(self) -> np.ndarray:
        """
        将Tensor转换为Numpy数组
        :return:Numpy数组
        """
        return self._data

    """
        backward函数现在应该从当前节点(Tensor)回溯到所有依赖节点(depends_on)，计算路径上的偏导
            # 我们分为两部分
            # a) 遍历计算图
            #    如果c是a经过某个函数的结果( c=f(a) )，我们无法知道a的梯度，直到我们得到了c的梯度(链式法则)
            #    所以我们需要逆序计算图中的拓扑结构(reverse mode)，相当沿着有向图的←方向(从指向节点到起始节点)进行计算
            # b) 应用梯度
            #    现在我们能访问到每个node,我们用它的backward函数将梯度传递给它们的depends_on
        """

    def _rev_topo_sort(self):
        """
        a) 遍历计算图，逆序计算图中的拓扑结构
        """

        def visit(node, visited, nodes):
            # 标记为已访问
            visited.add(node)
            if node._ctx:
                # 遍历所有依赖节点，递归调用visit
                [visit(nd, visited, nodes) for nd in node._ctx.depends_on if nd not in visited]
                # 递归调用后将node加入nodes
                nodes.append(node)
            return nodes

        return reversed(visit(self, set(), []))

    def backward(self, grad: "Tensor" = None) -> None:
        """
        实现Tensor的反向传播
        :param grad:如果该Tensor不是标量，则需要传递梯度进来
        :return:
        """
        # 只能在requires_grad=True的Tensor上调用此方法
        assert self.requires_grad, "called backward on tensor do not require grad"

        # 如果传递过来的grad为空
        if grad is None:
            if self.shape == ():
                # 设置梯度值为1，grad本身不需要计算梯度
                self._grad = Tensor(1.)
            else:
                # 如果当前Tensor得到不是标量，那么grad必须指定
                raise RuntimeError("grad must be specified for non scalar")
        else:
            self._grad = ensure_tensor(grad)

        for t in self._rev_topo_sort():
            assert t.grad is not None
            # 以逆序计算梯度，调用t相关运算操作的backward静态方法
            # 计算流向其依赖节点上的梯度(流向其下游)
            grads = t._ctx.backward(t._ctx, t.grad.data)
            # 如果只依赖一个输入，我们也通过列表来封装，防止zip将其继续拆分
            if len(t._ctx.depends_on) == 1:
                grads = [grads]

            for t, g in zip(t._ctx.depends_on, grads):
                # 计算其下游节点上的累积梯度，因为可能有多条边
                if t.requires_grad and g is not None:
                    # t.shape要和grad.shape保持一致
                    assert t.shape == g.shape, f"grad shape must match tensor shape in {self._ctx!r}, {g.shape!r} != {t.shape!r}"
                    # grad Tensor
                    gt = Tensor(g)
                    t._grad = gt if t.grad is None else t.grad + gt

    def __add__(self, other):
        ctx = Add(self, ensure_tensor(other))
        return ctx.apply(ctx, self, ensure_tensor(other))

    def __mul__(self, other):
        ctx = Mul(self, ensure_tensor(other))
        return ctx.apply(ctx, self, ensure_tensor(other))


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
        return super().__new__(cls)

    def forward(ctx, *args: Any, **kwargs: Any) -> np.ndarray:
        """
        进行前向传播，真正运算的地方
        :return: 返回运算结果
        """
        return NotImplemented("You must implement the forward function for custom Function.")

    def backward(ctx, grad: np.ndarray) -> Any:
        """
        实现反向传播，计算梯度
        :param grad:前一个运算梯度
        :return:返回计算后的梯度
        """
        return NotImplemented("You must implement the backward method for your custom Function" +
                              "to use it with backward mode AD.")

    def apply(self, ctx, *xs: "Tensor", **kwargs) -> "Tensor":
        """
        与PyTorch相同，我们也不直接调用forward，而是调用此方法
        :param ctx:
        :param xs:
        :param kwargs:
        :return:
        """
        # [t.data for t in xs] 遍历xs中的所有tensor，并取出data(np.ndarray)
        # 即实际参与运算的都是ndarray数组
        ret = Tensor(ctx.forward(ctx, *[t.data for t in xs], **kwargs),
                     requires_grad=any([t.requires_grad for t in xs]))

        if ret.requires_grad:
            # 如果需要求梯度，那么保存上下文信息
            ret._ctx = ctx

        return ret


def unbroadcast(grad: np.ndarray, in_shape: Tuple) -> np.ndarray:
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
    def forward(ctx, x: np.ndarray, y: np.ndarray) -> np.ndarray:
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

    def backward(ctx, grad: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
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


class Mul(_Function):
    def forward(ctx, x: np.ndarray, y: np.ndarray) -> np.ndarray:
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

    def backward(ctx, grad: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
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
