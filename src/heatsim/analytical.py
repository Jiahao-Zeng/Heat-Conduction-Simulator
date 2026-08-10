import numpy as np


def gaussian_point_source(x, t, alpha, x0, total_heat = 1.0):
    if t <= 0.0:
        raise ValueError("t must be positive (the solution is singular at t=0)")
    return (total_heat / np.sqrt(4.0 * np.pi * alpha * t)
            * np.exp(-(x - x0) ** 2 / (4.0 * alpha * t)))
