

import torch
import torch.nn as nn
from typing import Iterable, List
import scipy.interpolate as si
import numpy as np
import polars as pl
from torch.autograd.functional import hessian

from slurp.src.module.core import GnamModule

class SplineModule(GnamModule):
    def __init__(self, knots: Iterable, order: int = 3):
        super().__init__()
        
        self._knots = knots
        self._order = order

        self.linear = nn.Linear((len(knots)), 1, bias=False)
        self.batchnorm = nn.BatchNorm1d(len(knots), affine=False)
        self._build_bspline(knots=knots, order=order)

    @property
    def knots(self):
        return self._knots
    
    @property
    def order(self):
        return self._order

    def _build_bspline(self, knots, order: int):
        self.bs = si.BSpline(knots, np.eye(len(knots)), k=order, extrapolate=True)

    def forward(self, x):
        basis = self.bs(x)
        basis = torch.tensor(basis, dtype=torch.float).reshape(-1, len(self.knots))
        basis = self.batchnorm(basis)
        out = self.linear(basis)
        return out
    
    def regularisation(self, x):
        """
        Compute regularisation term |f''(x)|^2
        """
        deriv2 = self.bs(x, nu=2)
        deriv2 = torch.tensor(deriv2, dtype=torch.float)
        spline_deriv2 = torch.matmul(deriv2, self.linear.weight.T)
        return torch.mean(spline_deriv2 ** 2)
    



class Spline(SplineModule):
    def __init__(self, term: str, knots: Iterable, order: int = 3, tag: str = None):
        super().__init__(knots=knots, order=order)
        
        self._term = term
        self._knots = knots
        self._order = order

        self._tag = tag
        if tag is None:
            self._tag = f'S({term})'


    @property
    def tag(self):
        return self._tag

    @property
    def knots(self):
        return self._knots
    
    @property
    def order(self):
        return self._order
    
    @property
    def term(self):
        return self._term
    
    def _build_knots(self, x: pl.DataFrame):
        '''Build equispaced knots between min and max of x'''
        x_min = x.select(pl.col(self.term)).to_numpy().min()
        x_max = x.select(pl.col(self.term)).to_numpy().max()
        knots = np.linspace(x_min, x_max, len(self.knots))
        return knots

    def forward(self, x: pl.DataFrame):
        x = x.select(pl.col(self.term)).to_torch()
        out = super(Spline, self).forward(x)
        return out
    
    def regularisation(self, x: pl.DataFrame):
        x = x.select(pl.col(self.term)).to_torch()
        out = super(Spline, self).regularisation(x)
        return out

    def to_latex(self, compact: bool = False):
        """
        Return the latex representation of the spline
        """
        if compact:
            return fr"S \left( {self.term} \right)"
        else:
            return fr"S \left( {self.term}, order={self.order} \right)"
            
    
    def predict(self, X: pl.DataFrame, index: List = None):
        """
        Predict the spline value for a given input
        """
        y_pred = self.forward(X)
        y_pred = pl.DataFrame({self.tag: y_pred.detach().numpy().flatten()})

        if index:
            y_pred = pl.concat([X.select(index), y_pred])
        return y_pred
    

    def contraint(self, loss):
        return loss

    def _monotonic_inc_constraint_penalty(self):
        return 0

    def _monotonic_dec_constraint_penalty(self):
        return 0
    

    if __name__ == "__main__":
        import polars as pl
        import numpy as np
        import matplotlib.pyplot as plt

        # Example usage
        knots = [0, 1, 2, 3, 4]
        spline = Spline(term='x', knots=knots)
        x = pl.DataFrame({'x': np.linspace(0, 4, 100)})
        y = spline.predict(x)
        plt.plot(x['x'], y[spline.tag])
        plt.show()