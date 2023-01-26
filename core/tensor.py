import importlib
import inspect
from typing import Union, Tuple
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
    if isinstance(data, (np.ndarray, slice, tuple)):
        # 如果本身是ndarray、slice、tuple（里面全为slice），则直接返回
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

    def __getitem__(self, idxs) -> "Tensor":
        return self.slice(idxs)

    @property
    def T(self) -> "Tensor":
        return self.transpose(axes=None)

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


# 采用自动注入来实现相应的魔法方法。
def register(name, fxn):
    """
    将name中的所有函数都进注入进tensor中
    :param name:函数名
    :param fxn:
    :return:
    """

    def dispatch(*xs, **kwargs):
        # 把所有的输入都转换为Tensor
        xs = [ensure_tensor(x) for x in xs]
        # 调用apply方法
        return fxn.apply(fxn, *xs, **kwargs)

    if name in ['pow', 'neg']:
        setattr(Tensor, f'__{name}__', dispatch)
    # 为Tensor添加属性，名为name，值为dispatch函数引用
    setattr(Tensor, name, dispatch)

    # 这几个方法都有__xx__, __ixx__, __rxx__ 魔法方法
    # 比如对于add，这段代码会把__add__、__iadd__、__radd__和add绑定到其内部的dispatch方法。
    if name in ["add", "sub", "mul", "truediv", "matmul"]:
        setattr(Tensor, f"__{name}__", dispatch)
        # __i*__ 代表原地操作，即 tensor += x
        setattr(Tensor, f"__i{name}__", lambda self, x: self.assign(dispatch(self, x)))
        # __r*__ 代表 other在操作符前, self在操作符后，即 x + tensor
        setattr(Tensor, f"__r{name}__", lambda self, x: dispatch(x, self))


def _register_ops(namespace):
    for name, cls in inspect.getmembers(namespace, inspect.isclass):
        if name[0] != "_" and name != 'Tensor':
            # 注册所有_Function的子类
            register(name.lower(), cls)


try:
    _register_ops(importlib.import_module("core.ops"))
except ImportError as e:
    print(e)
