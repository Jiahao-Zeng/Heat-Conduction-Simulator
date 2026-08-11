import numpy as np
import pytest

from heatsim.grid import Grid1D
from heatsim.solvers import explicit_step, max_stable_dt, run_explicit
from heatsim.analytical import gaussian_point_source


LENGTH = 1.0
ALPHA = 0.01
X0 = LENGTH / 2.0
T_START = 0.05
T_END = 0.10


def _seeded_grid(n_points, alpha=ALPHA):
    x = np.linspace(0.0, LENGTH, n_points)
    u0 = gaussian_point_source(x, T_START, alpha, X0)
    return Grid1D(LENGTH, n_points, alpha, initial_temperature=u0)


def _relative_rmse(n_points, boundary="dirichlet"):
    grid = _seeded_grid(n_points)
    run_explicit(grid, T_START, T_END, boundary=boundary)
    exact = gaussian_point_source(grid.x, T_END, ALPHA, X0)
    rmse = np.sqrt(np.mean((grid.u - exact) ** 2))
    return rmse / exact.max()


def test_matches_gaussian_point_source():
    assert _relative_rmse(401) < 1e-3


def test_is_second_order_accurate_in_space():
    errors = [_relative_rmse(n) for n in (101, 201, 401, 801)]
    orders = [np.log2(a / b) for a, b in zip(errors, errors[1:])]

    assert all(o > 1.8 for o in orders), orders
    assert orders[-1] == pytest.approx(2.0, abs=0.15)


def test_insulated_boundaries_conserve_heat():
    grid = _seeded_grid(401)
    before = grid.total_heat()
    run_explicit(grid, T_START, T_END, boundary="neumann")

    assert grid.total_heat() == pytest.approx(before, rel=1e-12)


def test_uniform_material_has_uniform_face_conductivity():
    from heatsim.grid import harmonic_face_mean

    alpha = np.full(6, 0.01)
    assert np.allclose(harmonic_face_mean(alpha), 0.01)


@pytest.mark.parametrize("factor", [1.2, 1.5])
def test_diverges_above_cfl_limit(factor):
    grid = _seeded_grid(401)
    dt = max_stable_dt(grid, safety=1.0) * factor
    peak_before = np.abs(grid.u).max()

    for _ in range(400):
        explicit_step(grid, dt)

    assert np.abs(grid.u).max() > 1e6 * peak_before


def test_stays_bounded_below_cfl_limit():
    grid = _seeded_grid(401)
    dt = max_stable_dt(grid, safety=0.99)
    peak_before = np.abs(grid.u).max()

    for _ in range(400):
        explicit_step(grid, dt)

    assert np.abs(grid.u).max() < peak_before