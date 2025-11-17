

import torch
import torch.nn as nn
import numpy as np
import polars as pl
from typing import List

from slurp.src.module.core import GnAMModule

class csModule(GnAMModule):
    def __init__(self, period: float, order: int = 3, bias: bool = True):
        super().__init__()
        
        self._period = period
        self._order = order
        self._bias = bias

        self.linear = nn.Linear(2*order, 1, bias=bias)

    @property
    def bias(self):
        return self._bias
    
    @property
    def period(self):
        return self._period
    
    @property
    def order(self):
        return self._order
    
    def _normalize_period(self, x, period: float):
        x = x / period
        x = x * 2 * np.pi
        return x
    
    def _build_sincos(self, x, order: int):
        '''
        Build matric x_length tims 2*order that contains the sin(n*x) and cos(n*x) for n in [1, order]
        x: torch.Tensor of shape (x_length, 1)
        order: int, the order of the spline
        '''

        assert x.shape[1] == 1, f"Input x should be of shape (x_length, 1), but got {x.shape}"
        
        x_sin_cos = x.repeat(1, 2 * order)
        for i in range(order):
            x_sin_cos[:, 2 * i] = torch.sin((i + 1) * x_sin_cos[:, 2 * i])
            x_sin_cos[:, 2 * i + 1] = torch.cos((i + 1) * x_sin_cos[:, 2 * i + 1])
        return x_sin_cos

    def forward(self, x):
        x = self._normalize_period(x, period=self.period)
        x = self._build_sincos(x, order=self.order)
        out = self.linear(x.to(torch.float32))    
        return out
    
    def grad(self, x):
        x = x.clone().detach().requires_grad_(True)
        x_grad = torch.autograd.grad(self(x), x, create_graph=True)[0]
        return x_grad


class cs(csModule):
    def __init__(self, term: str, period: float, order: int = 3, bias: bool = False, tag: str = None):
        super(cs, self).__init__(period=period, order=order, bias=bias)
        
        self._term = term
        self._period = period
        self._order = order
        self._bias = bias
        
        self._tag = tag
        if tag is None:
            self._tag = f'cs({term})'

    @property
    def tag(self):
        return self._tag
    
    @property
    def term(self):
        return self._term
    
    @property
    def period(self):
        return self._period
    
    @property
    def order(self):
        return self._order
    
    @property
    def bias(self):
        return self._bias
    
    def forward(self, x: pl.DataFrame):
        x = x.select(pl.col(self.term)).to_torch()
        out = super(cs, self).forward(x)
        return out
    
    def to_latex(self, compact: bool = False):
        """
        Return the latex representation of the spline
        """
        if compact:
            return fr"cs\left( {self.term} \right)"
        else:
            return fr"cs\left( {self.term}, period={self.period}, order={self.order} \right)"
    
    def predict(self, X: pl.DataFrame, index: List = None):
        """
        Predict the spline value for a given input
        """
        y_pred = self.forward(X)
        y_pred = pl.DataFrame({self.tag: y_pred.detach().numpy().flatten()})

        if index:
            y_pred = pl.concat([X.select(index), y_pred])
        return y_pred
    
    def predict_basis(self, X: pl.DataFrame, index: List = None, scaled: bool = True):
        """
        Return the basis functions values
        """

        x = X.select(pl.col(self.term)).to_torch()
        x = self._normalize_period(x, period=self.period)
        x_sin_cos = self._build_sincos(x, order=self.order)

        if scaled:
            x_sin_cos = self.linear.weight * x_sin_cos

        basis_functions = []
        for i in range(self.order):
            sin_col = pl.DataFrame({f'{self.tag}_sin_{i+1}': x_sin_cos[:, 2 * i].detach().numpy().flatten()})
            cos_col = pl.DataFrame({f'{self.tag}_cos_{i+1}': x_sin_cos[:, 2 * i + 1].detach().numpy().flatten()})
            basis_functions.append(sin_col)
            basis_functions.append(cos_col)

        y_basis = pl.concat(basis_functions, how='horizontal')

        if index:
            y_basis = self.add_index(y=y_basis, X=X, index=index)
        
        return y_basis
    
    def grad(self, x):
        x = x.select(pl.col(self.term)).to_torch()
        return super(cs, self).grad(x)


