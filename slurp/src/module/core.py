
import torch.nn as nn
import polars as pl

class GnamModule(nn.Module):
    def __init__(self, *args, **kwargs):
        super(GnamModule, self).__init__(*args, **kwargs)

    def __add__(self, other):
        return AddGnamModule(self, other)
    
    def __mul__(self, other):
        return ProdGnamModule(self, other)
    
    def __truediv__(self, other):
        return DivGnamModule(self, other)


class OperationModule(GnamModule):
    def __init__(self, *args, **kwargs):
        super(OperationModule, self).__init__()

    def forward(self, x):
        raise NotImplementedError("Subclasses should implement this method.")

    def to_latex(self, compact: bool = False):
        raise NotImplementedError("Subclasses should implement this method.")

class AddGnamModule(OperationModule):
    def __init__(self, a: GnamModule, b: GnamModule):
        super(AddGnamModule, self).__init__()
        self.a = a
        self.b = b
    
    def forward(self, x):
        return self.a(x) + self.b(x)
    
    def to_latex(self, compact: bool = False):
        return fr"\left({self.a.to_latex(compact=compact)} + {self.b.to_latex(compact=compact)} \right)"
    
    def regularisation(self, x):
        return self.a.regularisation(x) + self.b.regularisation(x)
    
class ProdGnamModule(OperationModule):
    def __init__(self, a: GnamModule, b: GnamModule):
        super(ProdGnamModule, self).__init__()
        self.a = a
        self.b = b
    
    def forward(self, x):
        return self.a(x) * self.b(x)
    
    def to_latex(self, compact: bool = False):
        return fr"{self.a.to_latex(compact)} \cdot {self.b.to_latex(compact)}"
    
    def regularisation(self, x):
        return self.a.regularisation(x) + self.b.regularisation(x)
    
class DivGnamModule(OperationModule):
    def __init__(self, a: GnamModule, b: GnamModule, eps: float = 1e-8):
        super(DivGnamModule, self).__init__()
        self.a = a
        self.b = b
        self.eps = eps
    
    def forward(self, x):
        return self.a(x) / (self.b(x) + self.eps)

    def to_latex(self, compact: bool = False):
        return fr"\frac{{{self.a.to_latex(compact=compact)}}}{{{self.b.to_latex(compact=compact)}}}"
    
    def regularisation(self, x):
        return self.a.regularisation(x) + self.b.regularisation(x)