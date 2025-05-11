
import torch
import torch.nn as nn
from slurp.src.module import GnamModule


class Linear(GnamModule):
    def __init__(self, bias: bool = True):
        super(Linear, self).__init__()

        self._bias = bias

        self.alpha = nn.Parameter(torch.randn(1), requires_grad=True)

        if bias:
            self.beta = nn.Parameter(torch.randn(1), requires_grad=True)

    @property
    def bias(self):
        return self._bias

    def forward(self, x):
        output = self.alpha * x
        if self._bias:
            output += self.beta
