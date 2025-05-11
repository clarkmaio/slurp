

import marimo

__generated_with = "0.13.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import sys
    import os
    sys.path.append('/home/clarkmaio/workspace/slurp/slurp')
    sys.path.append('/home/clarkmaio/workspace/slurp/')

    from slurp.src.module.spline import SplineModule
    import polars as pl
    import torch
    import numpy as np

    return SplineModule, np, torch


@app.cell
def _(SplineModule, np, torch):
    sl = SplineModule(knots=[0,1,2,3, 4, 5, 6], order=2)
    x = torch.tensor(np.linspace(0, 10, 100)).reshape(-1, 1)
    return sl, x


@app.cell
def _(sl, x):
    sl.regularisation(x)
    return


@app.cell
def _(sl, x):
    type(sl(x.detach()))
    return


@app.cell
def _(torch):
    from torch.autograd.functional import hessian

    class SimpleModule(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear1 = torch.nn.Linear(5, 1)
            self.relu = torch.nn.GELU()

        def forward(self, x):
            x = self.relu(self.linear1(x))
            return x


    xx = torch.randn(10, 5)
    sm = SimpleModule()


    hessian(sm, xx[0, :])

    return


if __name__ == "__main__":
    app.run()
