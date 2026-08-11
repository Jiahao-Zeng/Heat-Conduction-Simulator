import numpy as np


def _slice_along(array, axis, part):
    index = [slice(None)] * array.ndim
    index[axis] = part
    return array[tuple(index)]


def harmonic_face_mean(field, axis=-1):
    left = _slice_along(field, axis, slice(0, -1))
    right = _slice_along(field, axis, slice(1, None))
    total = left + right
    out = np.zeros_like(total)
    np.divide(2.0 * left * right, total, out=out, where=total > 0.0)
    return out


face_diffusivity = harmonic_face_mean


def _material_fields(alpha, k, rho_c, shape):
    if (alpha is None) == (k is None):
        raise ValueError("provide either alpha, or both k and rho_c")
    if alpha is not None:
        if rho_c is not None:
            raise ValueError("rho_c cannot be combined with alpha; pass k instead")
        k_field = np.broadcast_to(np.asarray(alpha, dtype=float), shape).copy()
        rho_c_field = np.ones(shape)
    else:
        if rho_c is None:
            raise ValueError("k must be paired with rho_c")
        k_field = np.broadcast_to(np.asarray(k, dtype=float), shape).copy()
        rho_c_field = np.broadcast_to(np.asarray(rho_c, dtype=float), shape).copy()

    if np.any(k_field < 0.0):
        raise ValueError("conductivity must be non-negative")
    if np.any(rho_c_field <= 0.0):
        raise ValueError("volumetric heat capacity must be positive")
    return k_field, rho_c_field


class _MaterialCacheMixin:
    MAX_SOLVER_CACHE_ENTRIES = 4

    def _reset_material_cache(self):
        self._alpha = None
        self._inv_rho_c = None
        self._face_k = None
        self._face_k_x = None
        self._face_k_y = None
        self._solver_cache = {}

    def invalidate_material(self):
        self._reset_material_cache()

    def _cache_solver(self, key, value):
        cache = self._solver_cache
        if key not in cache and len(cache) >= self.MAX_SOLVER_CACHE_ENTRIES:
            del cache[next(iter(cache))]
        cache[key] = value
        return value

    @property
    def alpha(self):
        if self._alpha is None:
            self._alpha = self.k / self.rho_c
        return self._alpha

    @property
    def inv_rho_c(self):
        if self._inv_rho_c is None:
            self._inv_rho_c = 1.0 / self.rho_c
        return self._inv_rho_c


class Grid1D(_MaterialCacheMixin):
    def __init__(self, length, n_points, alpha=None, initial_temperature=0.0, k=None, rho_c=None):
        if n_points < 3:
            raise ValueError("n_points must be at least 3")
        if length <= 0.0:
            raise ValueError("length must be positive")

        self.length = float(length)
        self.n_points = int(n_points)
        self.dx = self.length / (self.n_points - 1)
        self.x = np.linspace(0.0, self.length, self.n_points)

        self.k, self.rho_c = _material_fields(
            alpha, k, rho_c, (self.n_points,))

        self.u = np.broadcast_to(
            np.asarray(initial_temperature, dtype=float), (self.n_points,)
        ).copy()

        self._reset_material_cache()

    @property
    def face_k(self):
        if self._face_k is None:
            self._face_k = harmonic_face_mean(self.k)
        return self._face_k

    def copy(self):
        return Grid1D(self.length, self.n_points, initial_temperature=self.u,
                      k=self.k, rho_c=self.rho_c)

    def total_heat(self):
        return float(np.sum(self.rho_c * self.u) * self.dx)


class Grid2D(_MaterialCacheMixin):
    def __init__(self, length_x, length_y, nx, ny, alpha=None,
                 initial_temperature=0.0, k=None, rho_c=None):
        if nx < 3 or ny < 3:
            raise ValueError("nx and ny must each be at least 3")
        if length_x <= 0.0 or length_y <= 0.0:
            raise ValueError("length_x and length_y must be positive")

        self.length_x = float(length_x)
        self.length_y = float(length_y)
        self.nx = int(nx)
        self.ny = int(ny)
        self.dx = self.length_x / (self.nx - 1)
        self.dy = self.length_y / (self.ny - 1)
        self.x = np.linspace(0.0, self.length_x, self.nx)
        self.y = np.linspace(0.0, self.length_y, self.ny)

        self.k, self.rho_c = _material_fields(
            alpha, k, rho_c, (self.nx, self.ny))

        self.u = np.broadcast_to(
            np.asarray(initial_temperature, dtype=float), (self.nx, self.ny)
        ).copy()

        self._reset_material_cache()

    @property
    def face_k_x(self):
        if self._face_k_x is None:
            self._face_k_x = harmonic_face_mean(self.k, axis=0)
        return self._face_k_x

    @property
    def face_k_y(self):
        if self._face_k_y is None:
            self._face_k_y = harmonic_face_mean(self.k, axis=1)
        return self._face_k_y

    def copy(self):
        return Grid2D(self.length_x, self.length_y, self.nx, self.ny,
                      initial_temperature=self.u, k=self.k, rho_c=self.rho_c)

    def total_heat(self):
        return float(np.sum(self.rho_c * self.u) * self.dx * self.dy)