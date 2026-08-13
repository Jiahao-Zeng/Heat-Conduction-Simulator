import math
from dataclasses import dataclass
from typing import Optional

from .grid import Grid1D, Grid2D


def _check_positive(value, label):
    if not math.isfinite(value):
        raise ValueError(f"{label} must be a finite number")
    if value <= 0.0:
        raise ValueError(f"{label} must be positive")


@dataclass(frozen=True)
class Material:
    name: str
    k: float
    rho_c: float
    k_y: Optional[float] = None

    def __post_init__(self):
        if not self.name:
            raise ValueError("name must be a non-empty string")
        _check_positive(self.k, "k")
        _check_positive(self.rho_c, "rho_c")
        if self.k_y is not None:
            _check_positive(self.k_y, "k_y")

    @property
    def is_anisotropic(self):
        return self.k_y is not None and self.k_y != self.k

    def kwargs_1d(self):
        return dict(k=self.k, rho_c=self.rho_c)

    def kwargs_2d(self):
        kwargs = dict(k=self.k, rho_c=self.rho_c)
        if self.is_anisotropic:
            kwargs["k_y"] = self.k_y
        return kwargs


MATERIALS = {
    "copper": Material("copper", k=401.0, rho_c=8960.0 * 385.0),
    "aluminum": Material("aluminum", k=237.0, rho_c=2700.0 * 897.0),
    "gold": Material("gold", k=317.0, rho_c=19300.0 * 129.0),
    "silver": Material("silver", k=429.0, rho_c=10490.0 * 235.0),
    "steel": Material("steel", k=50.0, rho_c=7850.0 * 490.0),
    "glass": Material("glass", k=1.0, rho_c=2500.0 * 840.0),
    "concrete": Material("concrete", k=1.7, rho_c=2300.0 * 880.0),
    "water": Material("water", k=0.6, rho_c=998.0 * 4182.0),
    "ice": Material("ice", k=2.18, rho_c=917.0 * 2050.0),
    "air": Material("air", k=0.026, rho_c=1.2 * 1005.0),
    "pine_wood": Material(
        "pine_wood",
        k=0.22,
        k_y=0.14,
        rho_c=500.0 * 2300.0,
    ),
    "pyrolytic_graphite": Material(
        "pyrolytic_graphite",
        k=1700.0,
        k_y=6.0,
        rho_c=2200.0 * 710.0,
    ),
}


def build_grid_1d(material, length, n_points, initial_temperature=0.0):
    return Grid1D(length, n_points, initial_temperature=initial_temperature,
                  **material.kwargs_1d())


def build_grid_2d(material, length_x, length_y, nx, ny, initial_temperature=0.0):
    return Grid2D(length_x, length_y, nx, ny,
                  initial_temperature=initial_temperature,
                  **material.kwargs_2d())