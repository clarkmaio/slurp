import scipy.interpolate as si









if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt

    # Plot b spline
    x = np.linspace(0, 1, 1000)
    N = 20
    knots = np.linspace(0, 1, N)
    bs = si.BSpline(knots, np.eye(N), 1, extrapolate=True)
    
    plt.plot(x, bs(x)[:, 4], label="B-spline")
    plt.plot(x, np.sum(bs(x), axis=1), label="Sum of B-spline", linestyle="--")
    plt.show()
