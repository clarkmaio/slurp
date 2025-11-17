


"""
# TODO
- by
"""




import autograd.numpy as np
from slurp.src.scipy._spline import spline


def cyclicbasis(x: np.array, order: int, period: float, bias: bool = False) -> np.array:
    normx = x * 2 * np.pi / period
    basis = np.zeros(shape=(x.size, 2*order))
    for i in range(order):
        basis[:, 2*i] = np.sin(normx * (i+1))
        basis[:, 2*i + 1] = np.cos(normx * (i+1))
    
    if bias:
        return np.concatenate([np.ones((x.size, 1)), basis], axis=1)
    return basis


class cc(spline):
    def __init__(self, order: int,  period: float, bias: bool = False):
        self._order = order
        self._bias = bias
        self._period = period
        self._n_params = 2 * order + int(bias)

    @property
    def period(self):
        return self._period
    
    @property
    def order(self):
        return self._order
    
    @property
    def bias(self):
        return self._bias

    @property
    def n_params(self):
        return self._n_params
    

    def basis(self, x: np.array):
        return cyclicbasis(x=x, order=self.order, period=self.period, bias=self.bias)
    
    def components(self, params, x: np.array) -> np.array:
        """
        Return ndarray. Each column is a different basis term
        """
        self.check_params(params=params)
        
        basis = self.basis(x=x)
        out = basis * np.tile(params, (x.size, 1))
        return out
        
    
    def __call__(self, params: np.array, x: np.array) -> np.array:
        self.check_params(params=params)
        
        basis = self.basis(x=x)
        out = np.dot(basis, params)
        return out


