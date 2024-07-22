import math


class Scalar():
    def __init__(self, value, prev=(), grad=0) -> None:
        self.value = value
        self.grad = grad

        # self._prev 是计算图的前驱节点，当然是对于前向计算过程而言的
        # 对于反向传播过程而言，self._prev 就是后继节点了
        self._prev = prev

    def __repr__(self) -> str:
        return f'Scalar(value={self.value})'

    def __add__(self, other: "Scalar"):
        output = Scalar(self.value + other.value, prev=(self, other))

        def _backward() -> None:
            # 很容易想到的代码实现是： self.grad= output.grad; other.grad = output.grad
            # If self & other 是两个独立变量，这很 OK. But:
            # 1. If other == self，此时实际上是 output = 2 * self，那么 set 两次 self.grad = output.grad 显然是不对的。
            # 2. If other != self，但 other 由 self 计算得到（比如 other = self ** 2），
            #    （假设 output.grad=1）此时手算一下正确的 self.grad 应为 (self.value+1)，而不是 1 或者self.value （透过 other 传播）

            # 也就是说对一个因变量分开求了两次导数 —— 那么该怎么解决呢？backward 的时候不是直接 set 而是 add？

            self.grad = self.grad + output.grad
            other.grad = other.grad + output.grad

        output._backward = _backward
        return output

    def __mul__(self, other: "Scalar"):
        output = Scalar(self.value * other.value, prev=(self, other))

        def _backward() -> None:
            self.grad = self.grad + other.value * output.grad
            other.grad = other.grad + self.value * output.grad

        output._backward = _backward
        return output

    def _backward(self) -> None:
        # default backward, do nothing
        return 1

    def backward(self) -> None:
        sorted_queue = []
        visited_node = set()

        def _topo_sort(node: Scalar) -> None:
            # 1. 一般来说，拓扑排序的实现是依次把入度（或者出度）为 0 的节点添加到列表中
            # 2. 在 Scalar 的实现中，每个节点记录且仅记录了 self.prev（从反向传播过程视角来说是后继节点），即出度信息很好获取
            #
            # 所以此处拓扑排序的实现，是将出度为 0 的节点逐步添加到队列
            for successor in node._prev:
                if successor not in visited_node:
                    _topo_sort(successor)

            sorted_queue.append(node)
            visited_node.add(node)

        _topo_sort(self)
        # 需要注意，由于是依据出度得来的拓扑序，所以需要反转一下
        sorted_queue.reverse()
        print(sorted_queue)

        # 反向传播
        self.grad = 1
        for node in sorted_queue:
            print(node)
            node._backward()


def tanh(input: Scalar) -> Scalar:
    def _backward() -> None:
        # backward to input
        pass

    output = Scalar(
        (1 - math.exp(-2 * input.value)) / (1 + math.exp(-2 * input.value))
    )
    output._backward = _backward

    return output


if __name__ == '__main__':
    # test basic arithmetic operation
    a = Scalar(4)
    b = Scalar(3)

    print(f'a + b = {a + b}')
    print(f'a * b = {a * b}')

    # test tanh function
    x = Scalar(100)
    print(f'tanh({x}) = {tanh(x)}')

    x.value = 0
    print(f'tanh({x}) = {tanh(x)}')

    x.value = -10
    print(f'tanh({x}) = {tanh(x)}')

    # test topological sort, 暂时还没实现 reset grad
    # simple test 1
    a = Scalar(3.0)
    b = a + a
    b.backward()
    assert a.grad == 2

    # simple test 2
    a = Scalar(3.0)
    b = Scalar(4.0)
    c = a * b
    d = c + c  # 还没实现 int * scalar，暂时用加法
    d.backward()
    assert a.grad == 8
    assert c.grad == 2
    assert b.grad == 6

    # operation between Scalar and Python data type (int/float).
