
import torch
import torch.nn as nn
import polars as pl
from typing import List
from slurp.src.module.core import GnAMModule


class lModule(GnAMModule):
    def __init__(self, bias: bool = True):
        super().__init__()

        self._bias = bias
        self.linear = torch.nn.Linear(1, 1, bias = bias)

    @property
    def bias(self):
        return self._bias

    def forward(self, x):
        return self.linear(x.to(torch.float32))

    def grad(self, x):
        x = x.clone().detach().requires_grad_(True)
        x_grad = torch.autograd.grad(self(x), x, create_graph=True)[0]
        return x_grad


class l(lModule):
    def __init__(self, term: str, bias: bool = True, tag: str = None):
        super().__init__(bias=bias)
        
        self._term = term

        self._tag = tag
        if tag is None:
            self._tag = f'l({term})'


    @property
    def tag(self):
        return self._tag

    @property
    def term(self):
        return self._term
    
    def forward(self, x: pl.DataFrame):
        x = x.select(pl.col(self.term)).to_torch()
        out = super(l, self).forward(x)
        return out

    def to_latex(self, compact: bool = False):
        """
        Return the latex representation of the spline
        """
        if compact:
            return fr"l \left( {self.term} \right)"
        else:
            return fr"l \left( {self.term}, bias={self._bias} \right)"
            
    
    def predict(self, X: pl.DataFrame, index: List = None):
        """
        Predict the spline value for a given input
        """
        y_pred = self.forward(X)
        y_pred = pl.DataFrame({self.tag: y_pred.detach().numpy().flatten()})

        if index:
            y_pred = pl.concat([X.select(index), y_pred])
        return y_pred
    

    def grad(self, x):
        x = x.select(pl.col(self.term)).to_torch()
        return super(l, self).grad(x)
