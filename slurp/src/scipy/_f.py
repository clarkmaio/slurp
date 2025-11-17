




import polars as pl
import autograd.numpy as np
from slurp.src.scipy._spline import spline


class f(spline):
    def __init__(self, n_class: int):
        self._n_class = n_class
        self._n_params = n_class

    @property
    def n_class(self):
        return self._n_class
    
    @property
    def n_params(self):
        return self._n_params
    
    def __call__(self, params: np.array, x: np.array) -> np.array:
        self.check_params(params=params)

        onehot = np.zeros(shape=(x.size, self.n_class), dtype=int)
        onehot[np.arange(x.size), x] = 1
        return np.dot(onehot, params)


"""
# IMPLEMENTATION CLASS TO HANDLE POLARS 
class fpolars:
    def __init__(self, term: str, n_class: int):
        self._n_class = n_class
        self._n_params = n_class
        self._term = term

        self.params = np.random.randn(self.n_params)

    @property
    def n_class(self):
        return self._n_class
    
    @property
    def n_params(self):
        return self._n_params
    
    @property
    def term(self):
        return self._term

    def __call__(self, X: pl.DataFrame):
        x = X[self.term]
"""