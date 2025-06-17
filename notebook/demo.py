

import marimo

__generated_with = "0.13.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import sys
    import os
    sys.path.append(os.getcwd())
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
    from slurp import Gnam, Spline, CyclicSpline
    return CyclicSpline, Gnam, Spline, mo, np, pl, plt


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
    return xx, yy


@app.cell
def _(np, pl, xx, yy):
    # Split X,y
    X, y = pl.DataFrame({'x': xx, 'random': np.random.randn(len(xx))}), pl.Series(values=yy, name='y')

    X = X.select(pl.all().cast(pl.Float32))
    y = y.cast(pl.Float32)
    return X, y


@app.cell
def _(CyclicSpline, Gnam, Spline, X, end_range, np, y):
    gm = Gnam(
        design = CyclicSpline(order=2, term='x', period=1., bias=False) * Spline(knots=np.linspace(0, end_range.value, 8), order=3, term='x')
    )

    gm.fit(X, y, epochs=5000, lr = 0.001, weight_decay=0.00, gamma=0.00)
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


if __name__ == "__main__":
    app.run()
