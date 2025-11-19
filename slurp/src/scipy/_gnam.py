

from typing import Dict, Callable
import autograd.numpy as np
import polars as pl
import autograd
import scipy

from slurp.src.scipy._spline import spline



class GnAM:
    # TODO define design as input of init and deduce spline catalog from there
    def __init__(self):
        self.optres = None
        self._spline_catalog = self.init_splines()
        self._params_catalog = {}

    def split_params(self, params: np.array) -> Dict:
        """
        Build params catalog. 
        Return a dictionary of parameters using same keys of spline catalog.
        """
        
        pc = {}
        i = 0
        for k, s in self.spline_catalog.items():
            pc[k] = params[i:i+s.n_params]
            i += s.n_params

        return pc
    

    @property
    def n_params(self) -> int:
        return np.sum([s.n_params for _, s in self.spline_catalog.items()])

    @property
    def spline_catalog(self):
        return self._spline_catalog
     
    def get_spline(self, tag: str) -> spline:
        return self.spline_catalog[tag]
    
    def init_splines(self) -> Dict:
        """
        Collection of splines that will be used in design
        """

        # DEFINE HERE YOUR SPLINE CATALOG AND RETURN DICTIONARY
        raise NotImplementedError('Spline catalog is not implemented')
    


    def loss(self, params: np.array, y: np.array, *args):
        yhat = self.design(params, *args)
        return np.mean((y-yhat)**2)
    

    def design(self, params: np.array, *args) -> np.array:
        """
        Define output using splines in spline_catalog.
        Use self.split_params to get params catalog
        """
        
        # IMPLEMENT HERE YOUR MODEL DESIGN
        params_catalog = self.split_params(params)

        raise NotImplementedError('Missing design implementation')


    def fit(self, y: np.array, *args) -> None:
        
        self.optres = scipy.optimize.minimize(
            fun=self.loss,
            args=(y, *args),
            x0=np.random.randn(self.n_params),
            method='L-BFGS-B',
            jac=autograd.grad(self.loss, argnum=0)
        )
        print(self.optres)


    def predict(self, *args, **kwargs) -> np.array:
        return self.design(self.optres.x, *args, **kwargs)
    




