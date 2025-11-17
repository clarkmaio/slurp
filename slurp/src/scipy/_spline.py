

import autograd.numpy as np
from abc import abstractmethod


class spline: 
    def __init__(self):
        pass
    
    
    def check_params(self, params: np.array):
        assert len(params) == self.n_params, f'params variable must have length {self.n_params}, it has length {len(params)} instead.'

    @abstractmethod
    def latex(self) -> str:
        raise NotImplementedError('Missing latex')