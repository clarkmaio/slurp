




"""
# TODO
- by
- 2d
"""



import autograd.numpy as np
from slurp.src.scipy._spline import spline

def bsbasis(x: np.array, knots: np.array, degree: int) -> np.array:
    """
    Return bs basis components using De Boor's recursive formula
    """
    if degree == 0:
        return np.where( (x.reshape(-1, 1) >= knots[:-1]) & (x.reshape(-1, 1) < knots[1:]), 1, 0)
    else:
        B_prev = bsbasis(x=x, knots=knots, degree=degree - 1)
        term1 = (x.reshape(-1, 1) - knots[:-degree-1]) / (knots[degree:-1] - knots[:-degree-1])
        term2 = (knots[degree+1:] - x.reshape(-1, 1)) / (knots[degree+1:] - knots[1:-degree])
        return term1 * B_prev[:, :-1] + term2 * B_prev[:, 1:]


class s(spline):
    def __init__(self, knots: np.array, degree: int):
        self._knots = knots
        self._degree = degree

        self._n_params = knots.size - degree - 1
        assert self._n_params > 0, 'Number of knots must be greater than degree + 1.'
        

    @property
    def knots(self):
        return self._knots
    
    @property
    def degree(self):
        return self._degree
    
    @property
    def n_params(self):
        return self._n_params
    

    def basis(self, x: np.array, degree: int):
        '''
        Build basis using De Boor's algorithm
        '''
        return bsbasis(x=x, knots=self.knots, degree=degree)
    
    def components(self, params, x: np.array) -> np.array:
        """
        Return ndarray. Each column is a different basis term
        """
        basis = self.basis(x, degree=self.degree)
        return basis * np.tile(params, (x.size, 1))
    
    def __call__(self, params: np.array, x: np.array) -> np.array:
        self.check_params(params=params)
        basis = self.basis(x=x, degree=self.degree)
        return np.dot(basis, params)

