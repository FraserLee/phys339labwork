import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
# ---------------------- analysis part 1: polarised light ----------------------

filename = 'data/polarised_10000_2025_02_20-15_44_19.txt'
data = np.loadtxt(filename)
# we should have exactly 400 data points, corresponding to 1 full rotation
assert data.shape == (400,)

def fit_and_plot(data, f, plot=True):
    x = np.linspace(0, len(data), len(data))

    x = np.linspace(0, len(data), len(data))
    popt, _ = opt.curve_fit(f, x, data)
    if plot:
        plt.plot(x, data, label='data')
        plt.plot(x, f(x, *popt), label='fit')
        plt.legend()
        plt.show()

def f(x, a, b, theta):
    return a * np.cos(2 * np.pi * x / 400 + theta) ** 2 + b
fit_and_plot(data, f)
