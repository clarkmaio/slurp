

import marimo

__generated_with = "0.13.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import sys
    import os
    sys.path.append('/home/clarkmaio/workspace/slurp/slurp')
    sys.path.append('/home/clarkmaio/workspace/slurp/')
    return


@app.cell
def _():
    import marimo as mo

    import torch
    import scipy.interpolate as si
    import numpy as np
    import matplotlib.pyplot as plt
    from tqdm import tqdm
    import altair as alt
    import polars as pl
    import pandas as pd

    alt.data_transformers.enable("vegafusion")

    from slurp.src.module.spline import SplineModule
    from slurp.src.module.cyclic_spline import CyclicSplineModule
    from slurp import Gnam, Spline, CyclicSpline
    return Gnam, Spline, mo, np, pl, plt


@app.cell
def _(mo):
    end_range = mo.ui.number(start=1, stop=10, value=5, label='domain')
    n_knots = mo.ui.number(start=3, stop=100, value=20, label='n knots')
    return end_range, n_knots


@app.cell
def _(end_range, mo, n_knots):
    mo.md(
    f"""
    {end_range.left()}
    {n_knots.left()}
    """
    )
    return


@app.cell
def _(end_range, n_knots, np):
    # Data generation
    N = 1000
    xx = np.linspace(0, end_range.value, N)
    noise = 0.3 * np.random.randn(N)
    yy = np.sin(2 * np.pi * xx) * np.exp(xx/5) + noise
    knots = np.linspace(0, end_range.value, n_knots.value)
    return knots, xx, yy


@app.cell
def _(np, pl, xx, yy):
    # Split X,y
    X, y = pl.DataFrame({'x': xx, 'random': np.random.randn(len(xx))}), pl.Series(values=yy, name='y')

    X = X.select(pl.all().cast(pl.Float32))
    y = y.cast(pl.Float32)
    return X, y


@app.cell
def _(Gnam, Spline, X, knots, y):
    gm = Gnam(
        #design = CyclicSpline(order=2, term='x', period=1., bias=False) * Spline(knots=knots, order=3, term='x')
        design = Spline(knots=knots, order=3, term='x')
    )

    gm.fit(X, y, epochs=5000, lr = 0.005, weight_decay=0.0, gamma=0.0)
    yout = gm.predict(X)
    return gm, yout


@app.cell
def _(gm, mo):
    mo.md(
        fr'''
        $$
        {gm.latex_design(compact=True)}
        $$
        '''
    )
    return


@app.cell
def _(gm, plt):
    plt.plot(gm._loss_history)
    return


@app.cell
def _(X, plt, xx, yout, yy):
    plt.scatter(xx, yy, facecolor='none', edgecolors='blue', alpha = 0.2)
    plt.plot(X['x'], yout, color='red')
    return


@app.cell
def _(X, gm):
    components = gm.predict_components(X, index=['x'])
    return (components,)


@app.cell
def _(components, mo):
    mo.ui.altair_chart(
        (components.plot.line(x='x', y='CS(x)') | components.plot.line(x='x', y='S(x)'))
    )
    return


@app.cell
def _(mo):
    mo.md(r"""# Viz""")
    return


if __name__ == "__main__":
    app.run()
