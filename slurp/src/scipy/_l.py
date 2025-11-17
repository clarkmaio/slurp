


import autograd.numpy as np
from slurp.src.scipy._spline import spline


class l(spline):
    def __init__(self, bias: bool = True):
        self._bias = bias
        self._n_params = 1 + int(bias)

    @property
    def bias(self):
        return self._bias
    
    @property
    def n_params(self):
        return self._n_params

    def __call__(self, params: np.array, x: np.array) -> np.array:
        self.check_params(params=params)

        out = x * params[0]
        if self.bias:
            out = out + params[1]
        return out


