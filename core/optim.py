from typing import List

from core.parameter import Parameter


class Optimizer:
    def __init__(self, parameters: List[Parameter]) -> None:
        self.parameters = parameters

    def zero_grad(self) -> None:
        for p in self.parameters:
            p.zero_grad()

    def step(self) -> None:
        raise NotImplemented("This optimizer not implement the parameter update ops")
