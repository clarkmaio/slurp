


import sys
import os
sys.path.append('/home/clarkmaio/workspace/slurp/')


import torch
import torch.nn as nn
from typing import Iterable, List
import scipy.interpolate as si
import numpy as np
import polars as pl

from slurp.src.module.core import GnAMModule


class indicator(nn.Module):
    def __init__(self, start: float = 0.0, end: float = 1.0, smoothalpha: float = None):
        super().__init__()
        self.start = start
        self.end = end
        self.smoothalpha = smoothalpha

    def forward(self, x: torch.Tensor):
        if self.smoothalpha:
            return 0.5 * (torch.tanh(self.smoothalpha * (x - self.start)) + torch.tanh(- self.smoothalpha * (x - self.end)))
        else:
            return torch.where((x > self.start) & (x < self.end), torch.ones_like(x), torch.zeros_like(x))
        

class bbasis(nn.Module):
    """
    Comnpute B-spline basis functions using Cox-de Boor recursion formula

    Parameters
    ----------
    k: int
        Index of the basis function
    knots: torch.Tensor
        Knot vector
    degree: int
        Degree of the B-spline
    smoothalpha: float
        Smoothing parameter for the indicator function

    Returns
    -------
    torch.Tensor
        Value of the basis function at x.
        Output shape is the same as input x.
    """
    def __init__(self, k: int, knots: torch.Tensor, degree: int = 3, smoothalpha: float = None):
        super().__init__()
        self.k = k
        self.knots = knots
        self.degree = degree
        self.smoothalpha = smoothalpha

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        if self.degree == 0:
            return indicator(start=self.knots[self.k], end=self.knots[self.k + 1], smoothalpha=self.smoothalpha)(x)
        else:
            basis1 = bbasis(self.k, self.knots, self.degree - 1, self.smoothalpha)(x)
            basis2 = bbasis(self.k + 1, self.knots, self.degree - 1, self.smoothalpha)(x)

            term1 = (x-self.knots[self.k]) / (self.knots[self.k + self.degree] - self.knots[self.k]) * basis1
            term2 = (self.knots[self.k + self.degree + 1]-x) / (self.knots[self.k + self.degree + 1] - self.knots[self.k + 1]) * basis2 
            return term1 + term2

    def derivative(self, x: torch.Tensor, nu: int = 1) -> torch.Tensor:
        if nu == 0:
            return self.forward(x)
        else:
            coeff1 = self.degree / (self.knots[self.k + self.degree] - self.knots[self.k])
            coeff2 = self.degree / (self.knots[self.k + self.degree + 1] - self.knots[self.k + 1])

            basis1_deriv = bbasis(self.k, self.knots, self.degree - 1, self.smoothalpha).derivative(x, nu - 1)
            basis2_deriv = bbasis(self.k + 1, self.knots, self.degree - 1, self.smoothalpha).derivative(x, nu - 1)

            return coeff1 * basis1_deriv - coeff2 * basis2_deriv
        

class bspline(nn.Module):
    def __init__(self, knots: Iterable, degree: int = 3, smoothalpha: float = None):
        super().__init__()
        self.knots = torch.tensor(knots, dtype=torch.float)
        self.degree = degree
        self.smoothalpha = smoothalpha
        self.n_basis = len(knots) - degree - 1

        self.bases = nn.ModuleList([bbasis(k=i, knots=self.knots, degree=self.degree, smoothalpha=self.smoothalpha) for i in range(self.n_basis)])
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        basis_values = [b(x) for b in self.bases]
        return torch.concat(basis_values, dim=1)

    def derivative(self, x: torch.Tensor, nu: int =1) -> torch.Tensor:
        deriv_values = [b.derivative(x, nu=nu) for b in self.bases]
        return torch.concat(deriv_values, dim=1)



class sModule(GnAMModule):
    def __init__(self, knots: Iterable, degree: int = 3):
        super().__init__()
        
        self._knots = knots
        self._degree = degree
        self._n_splines = len(knots) - degree - 1

        self.linear = nn.Linear(self._n_splines, 1, bias=False)
        self.batchnorm = nn.BatchNorm1d(len(knots), affine=False)
        self._build_bspline(knots=knots, degree=degree)

    @property
    def n_splines(self):
        return self._n_splines

    @property
    def knots(self):
        return self._knots
    
    @property
    def degree(self):
        return self._degree

    def _build_bspline(self, knots, degree: int):
        self.bs = bspline(knots=knots, degree=degree, smoothalpha=None)

    def forward(self, x):
        basis = self.bs(x).to(torch.float32)
        out = self.linear(basis)
        return out

        
    def grad(self, x):
        x = x.clone().detach().requires_grad_(True)
        x_grad = torch.autograd.grad(outputs=self(x), 
                                     inputs=x, 
                                     grad_outputs=torch.ones_like(x),
                                     create_graph=True)[0]
        return x_grad
    
    def grad2(self, x):
        x = x.clone().detach().requires_grad_(True)
        x_grad = self.grad(x)
        x_grad2 = torch.autograd.grad(outputs=x_grad, 
                                    inputs=x, 
                                    grad_outputs=torch.ones_like(x),
                                    create_graph=True)[0]
        return x_grad2

    


class s(sModule):
    def __init__(self, term: str, knots: Iterable, degree: int = 3, tag: str = None):
        super().__init__(knots=knots, degree=degree)
        
        self._term = term
        self._knots = knots
        self._degree = degree

        self._tag = tag
        if tag is None:
            self._tag = f's({term})'


    @property
    def tag(self):
        return self._tag

    @property
    def knots(self):
        return self._knots
    
    @property
    def degree(self):
        return self._degree
    
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
        out = super(s, self).forward(x)
        return out

    def to_latex(self, compact: bool = False):
        """
        Return the latex representation of the spline
        """
        if compact:
            return fr"s \left( {self.term} \right)"
        else:
            return fr"s \left( {self.term}, degree={self.degree} \right)"
            
    
    def predict(self, X: pl.DataFrame, index: List = None):
        """
        Predict the spline value for a given input
        """
        y_pred = self.forward(X)
        y_pred = pl.DataFrame({self.tag: y_pred.detach().numpy().flatten()})

        if index:
            y_pred = self.add_index(y_pred=y_pred, X=X, index=index)
        return y_pred
    
    def predict_basis(self, X: pl.DataFrame, index: List = None, scaled: bool = False) -> pl.DataFrame:
        """
        Predict the basis function values for a given input
        """
        x = X.select(pl.col(self.term)).to_torch()
        basis_values = self.bs(x).to(torch.float32)

        if scaled:
            basis_values = self.linear.weight * basis_values

        basis_df = pl.DataFrame({f'basis_{i}': basis_values[:, i].detach().numpy() for i in range(basis_values.shape[1])})

        if index:
            basis_df = self.add_index(y=basis_df, X=X, index=index)

        return basis_df
    
    def grad(self, x):
        #x = x.select(self.term).to_torch()
        return super(s, self).grad(x)

    