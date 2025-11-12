

import torch
import torch.nn as nn
import numpy as np
import polars as pl
from typing import List

from slurp.src.module.core import GnamModule

class CyclicSplineModule(GnamModule):
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
            x_sin_cos[:, 2 * i + 1] = torch.cos((i + 1) * x_sin_cos[:, 2 * i])
        return x_sin_cos

    def forward(self, x):
        x = self._normalize_period(x, period=self.period)
        x = self._build_sincos(x, order=self.order)
        out = self.linear(x)    
        return out
    
        
    def regularisation(self, x):
        """
        Compute regularisation term |f''(x)|^2
        """
        return 0



class CyclicSpline(CyclicSplineModule):
    def __init__(self, term: str, period: float, order: int = 3, bias: bool = False, tag: str = None):
        super(CyclicSpline, self).__init__(period=period, order=order, bias=bias)
        
        self._term = term
        self._period = period
        self._order = order
        self._bias = bias
        
        self._tag = tag
        if tag is None:
            self._tag = f'CS({term})'

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
        out = super(CyclicSpline, self).forward(x)
        return out
    
    def regularisation(self, x: pl.DataFrame):
        x = x.select(pl.col(self.term)).to_torch()
        out = super(CyclicSpline, self).regularisation(x)
        return out
    
    def to_latex(self, compact: bool = False):
        """
        Return the latex representation of the spline
        """
        if compact:
            return fr"CS\left( {self.term} \right)"
        else:
            return fr"CS\left( {self.term}, period={self.period}, order={self.order} \right)"
    
    def predict(self, X: pl.DataFrame, index: List = None):
        """
        Predict the spline value for a given input
        """
        y_pred = self.forward(X)
        y_pred = pl.DataFrame({self.tag: y_pred.detach().numpy().flatten()})

        if index:
            y_pred = pl.concat([X.select(index), y_pred])
        return y_pred

if __name__ == '__main__':
    x = torch.tensor([0, 1, 2, 3, 4, 2, 2, 2, 2, 2, 2, 2]).reshape(-1, 1)
    period = 5
    order = 3
    cs = CyclicSplineModule(period=period, order=order)
    output = cs(x)