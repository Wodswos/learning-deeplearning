import math


class Scalar():
    def __init__(self, value) -> None:
        self.value = value
    
    def __repr__(self) -> str:
        return f'Scalar(value={self.value})'
    
    def __add__(self, other):
        return Scalar(self.value + other.value)

    def __mul__(self, other):
        return Scalar(self.value * other.value)


def tanh(input: Scalar) -> Scalar:
    def _backward() -> None:
        # backward to input
        pass

    output = Scalar()
    output._backward = _backward
    
    return output


if __name__ == '__main__':
    # test basic arithmetic operation
    a = Scalar(4)
    b = Scalar(3)

    print(f'a + b = {a + b}')
    print(f'a * b = {a * b}')

    # print(scalar)
