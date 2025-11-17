
import torch.nn as nn
import polars as pl
from typing import List
import torch

class GnAMModule(nn.Module):
    def __init__(self, *args, **kwargs):
        super(GnAMModule, self).__init__(*args, **kwargs)

    def __add__(self, other):
        return AddGnAMModule(self, other)
    
    def __mul__(self, other):
        return ProdGnAMModule(self, other)
    
    def __truediv__(self, other):
        return DivGnAMModule(self, other)
    
    def add_index(self, y, X, index: List = None):
        y = pl.concat([X.select(index), y])
        return y


    def regularisation(self, x):
        '''
        Return |f''(x)|^2
        '''
        pass
    

class OperationModule(GnAMModule):
    def __init__(self, *args, **kwargs):
        super(OperationModule, self).__init__()

    def forward(self, x):
        raise NotImplementedError("Subclasses should implement this method.")

    def to_latex(self, compact: bool = False):
        raise NotImplementedError("Subclasses should implement this method.")

class AddGnAMModule(OperationModule):
    def __init__(self, a: GnAMModule, b: GnAMModule):
        super(AddGnAMModule, self).__init__()
        self.a = a
        self.b = b
    
    def forward(self, x):
        return self.a(x) + self.b(x)
    
    def to_latex(self, compact: bool = False):
        return fr"\left({self.a.to_latex(compact=compact)} + {self.b.to_latex(compact=compact)} \right)"
    
    def grad(self, x):
        return self.a.grad(x) + self.b.grad(x)
        
class ProdGnAMModule(OperationModule):
    def __init__(self, a: GnAMModule, b: GnAMModule):
        super(ProdGnAMModule, self).__init__()
        self.a = a
        self.b = b
    
    def forward(self, x):
        return self.a(x) * self.b(x)
    
    def to_latex(self, compact: bool = False):
        return fr"{self.a.to_latex(compact)} \cdot {self.b.to_latex(compact)}"
    
    def grad(self, x):
        return self.a.grad(x) * self.b(x) + self.b.grad(x) * self.a(x)
    
class DivGnAMModule(OperationModule):
    def __init__(self, a: GnAMModule, b: GnAMModule, eps: float = 1e-8):
        super(DivGnAMModule, self).__init__()
        self.a = a
        self.b = b
        self.eps = eps
    
    def forward(self, x):
        return self.a(x) / (self.b(x) + self.eps)

    def to_latex(self, compact: bool = False):
        return fr"\frac{{{self.a.to_latex(compact=compact)}}}{{{self.b.to_latex(compact=compact)}}}"
    
    def grad(self, x):
        b_val = self.b(x) + self.eps
        return (self.a.grad(x) * b_val - self.b.grad(x) * self.a(x)) / (b_val ** 2)