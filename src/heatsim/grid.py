import numpy as np


def face_diffusivity(alpha):
    a_left = alpha[:-1]
    a_right = alpha[1:]
    total = a_left + a_right
    out = np.zeros_like(total)
    np.divide(2.0 * a_left * a_right, total, out=out, where=total > 0.0)
    return out


class Grid1D:

    def __init__(self, length, n_points, alpha, initial_temperature=0.0):
        if n_points < 3:
            raise ValueError("n_points must be at least 3")
        if length <= 0.0:
            raise ValueError("length must be positive")

        self.length = float(length)
        self.n_points = int(n_points)
        self.dx = self.length / (self.n_points - 1)
        self.x = np.linspace(0.0, self.length, self.n_points)

        self.alpha = np.broadcast_to(
            np.asarray(alpha, dtype=float), (self.n_points,)
        ).copy()
        if np.any(self.alpha < 0.0):
            raise ValueError("diffusivity must be non-negative")

        self.u = np.broadcast_to(
            np.asarray(initial_temperature, dtype=float), (self.n_points,)
        ).copy()

        self._face_alpha = None
        self._solver_cache = {}

    @property
    def face_alpha(self):
        if self._face_alpha is None:
            self._face_alpha = face_diffusivity(self.alpha)
        return self._face_alpha

    def invalidate_material(self):
        self._face_alpha = None
        self._solver_cache.clear()

    def copy(self):
        return Grid1D(self.length, self.n_points, self.alpha, self.u)

    def total_heat(self):
        return float(np.sum(self.u) * self.dx)