
import torch
import torch.nn as nn
from slurp.src.module import GnamModule

class Scalar(GnamModule):
    def __init__(self):
        super(Scalar, self).__init__()
        self.alpha = nn.Parameter(torch.randn(1), requires_grad=True)


    def forward(self, x):
        ones_like_x = torch.ones_like(x)
        output = ones_like_x * self.alpha
        return output