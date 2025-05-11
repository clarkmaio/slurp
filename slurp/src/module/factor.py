



import torch
import torch.nn as nn
from slurp.src.module import GnamModule
import polars as pl


class FactorModule(GnamModule):
    def __init__(self, num_classes: int):
        super().__init__()
        
        self._num_classes = num_classes
        self.alpha = torch.nn.Linear(num_classes, 1, bias=False)
    
    @property
    def num_classes(self):
        return self._num_classes

    def forward(self, x):
        x_onehot = torch.nn.functional.one_hot(x, num_classes=self.num_classes)
        x_onehot = x_onehot.float()
        x_out = self.alpha(x_onehot)
        return x_out
    

class Factor(FactorModule):
    def __init__(self, term: str, num_classes: int, tag: str = None):
        super().__init__(num_classes=num_classes)
        
        self._term = term
        self._num_classes = num_classes

        self._tag = tag
        if tag is None:
            self._tag = f'F({term})'

    @property
    def tag(self):
        return self._tag

    @property
    def num_classes(self):
        return self._num_classes

    @property
    def term(self):
        return self._term
    
    def forward(self, x: pl.DataFrame):
        x = x.select(pl.col(self.term)).to_torch()
        x_out = super(Factor, self).forward(x)
        return x_out

    def to_latex(self, compact: bool = False):
        if compact:
            return fr"F\left({self.term}\right)"
        else:
            return fr"f\left({self.term}, num_classes={self.num_classes}\right)"

