import numpy as np


def gaussian_point_source(x, t, alpha, x0, total_heat=1.0):
    if t <= 0.0:
        raise ValueError("t must be positive (the solution is singular at t=0)")
    return (total_heat / np.sqrt(4.0 * np.pi * alpha * t)
            * np.exp(-(x - x0) ** 2 / (4.0 * alpha * t)))


def gaussian_point_source_2d(x, y, t, alpha, x0, y0, total_heat=1.0):
    if t <= 0.0:
        raise ValueError("t must be positive (the solution is singular at t=0)")
    r2 = (x - x0) ** 2 + (y - y0) ** 2
    return (total_heat / (4.0 * np.pi * alpha * t)
            * np.exp(-r2 / (4.0 * alpha * t)))


def gaussian_point_source_2d_anisotropic(x, y, t, alpha_x, alpha_y, x0, y0,
                                          total_heat=1.0):
    if t <= 0.0:
        raise ValueError("t must be positive (the solution is singular at t=0)")
    return (total_heat / (4.0 * np.pi * t * np.sqrt(alpha_x * alpha_y))
            * np.exp(-(x - x0) ** 2 / (4.0 * alpha_x * t)
                     - (y - y0) ** 2 / (4.0 * alpha_y * t)))