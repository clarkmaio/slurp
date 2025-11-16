



# import torch
# from slurp.src.module.core import GnAMModule
# import polars as pl
# from typing import List


# class fModule(GnAMModule):
#     def __init__(self, num_classes: int):
#         super().__init__()
        
#         self._num_classes = num_classes
#         self.alpha = torch.nn.Linear(num_classes, 1, bias=False)
    
#     @property
#     def num_classes(self):
#         return self._num_classes

#     def forward(self, x):
#         x_onehot = torch.nn.functional.one_hot(x, num_classes=self.num_classes)
#         x_onehot = x_onehot.float()
#         x_out = self.alpha(x_onehot)
#         return x_out
    
#     def grad(self, x):
#         x = x.clone().detach().requires_grad_(True)
#         x_grad = torch.autograd.grad(self(x), x, create_graph=True)[0]
#         return x_grad

    

# class f(fModule):
#     def __init__(self, term: str, num_classes: int, tag: str = None):
#         super().__init__(num_classes=num_classes)
        
#         self._term = term
#         self._num_classes = num_classes

#         self._tag = tag
#         if tag is None:
#             self._tag = f'f({term})'

#     @property
#     def tag(self):
#         return self._tag

#     @property
#     def num_classes(self):
#         return self._num_classes

#     @property
#     def term(self):
#         return self._term
    
#     def forward(self, x: pl.DataFrame):
#         x = x.select(pl.col(self.term)).to_torch()
#         x_out = super(f, self).forward(x)
#         return x_out

#     def to_latex(self, compact: bool = False):
#         if compact:
#             return fr"f\left({self.term}\right)"
#         else:
#             return fr"f\left({self.term}, num_classes={self.num_classes}\right)"


#     def predict(self, X: pl.DataFrame, index: List = None):
#         """
#         Predict the spline value for a given input
#         """
#         y_pred = self.forward(X)
#         y_pred = pl.DataFrame({self.tag: y_pred.detach().numpy().flatten()})

#         if index:
#             y_pred = pl.concat([X.select(index), y_pred])
#         return y_pred
    
#     def grad(self, x):
#         x = x.select(pl.col(self.term)).to_torch()
#         return super(f, self).grad(x)



import autograd as np
import polars as pl


class _f:
    def __init__(self, n: int):
        self._n = n
        self._n_params = n

    @property
    def n_params(self):
        return self._n_params

    @property
    def n(self):
        return self._n
    

    def __call__(self, params: np.array, x: np.array):
        return params[x]
    





class f:
    def __init__(self, term: str, n: int, x0: np.array = None, tag: str = None):
        self._n = n
        self._n_params = n
        self._term = term
        self._tag = tag
        self.x0 = x0
        self.init_params(x0)

    @property
    def term(self):
        return self._term
    
    @property
    def tag(self):
        return self._tag

    def init_params(self, x0: np.array) -> None:
        self.params = x0
        if not x0:
            self.params = np.random.randn(self.n_params)

    def __call__(self, X: pl.DataFrame):
        x = X[self.term].to_numpy()

    def eval(self, params, X):
        return



if __name__ == '__main__':

    X = pl.Dataframe({'x': np.randint(0, 2, 100), 'y': np.random.randn(100)})
    f(term='x',n=2)

    def loss(params, X, y):
        return




