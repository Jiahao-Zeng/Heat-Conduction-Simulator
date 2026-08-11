import numpy as np
import pytest

from heatsim.grid import Grid2D
from heatsim.solvers import (
    explicit_step_2d,
    run_explicit_2d,
    max_stable_dt_2d,
    crank_nicolson_step_2d,
    run_crank_nicolson_2d,
)
from heatsim.analytical import gaussian_point_source_2d


LENGTH = 1.0
ALPHA = 0.01
X0 = Y0 = LENGTH / 2.0
T_START = 0.02
T_END = 0.05


def _grid_xy(n):
    xs = np.linspace(0.0, LENGTH, n)
    ys = np.linspace(0.0, LENGTH, n)
    return np.meshgrid(xs, ys, indexing="ij")


def _seeded_grid(n, alpha=ALPHA):
    X, Y = _grid_xy(n)
    u0 = gaussian_point_source_2d(X, Y, T_START, alpha, X0, Y0)
    return Grid2D(LENGTH, LENGTH, n, n, alpha, u0)


def _relative_rmse(u_numerical, n):
    X, Y = _grid_xy(n)
    exact = gaussian_point_source_2d(X, Y, T_END, ALPHA, X0, Y0)
    rmse = np.sqrt(np.mean((u_numerical - exact) ** 2))
    return rmse / exact.max()


def _exact_fourier_mode(X, Y, t, alpha=ALPHA):
    decay_rate = alpha * np.pi ** 2 * 2.0
    return np.sin(np.pi * X) * np.sin(np.pi * Y) * np.exp(-decay_rate * t)


def test_explicit_matches_gaussian_point_source():
    grid = _seeded_grid(81)
    run_explicit_2d(grid, T_START, T_END)
    assert _relative_rmse(grid.u, 81) < 1e-3


def test_explicit_is_second_order_accurate_in_space():
    errors = []
    for n in (161, 321, 641):
        grid = _seeded_grid(n)
        run_explicit_2d(grid, T_START, T_END, safety=0.9)
        errors.append(_relative_rmse(grid.u, n))
    orders = [np.log2(a / b) for a, b in zip(errors, errors[1:])]
    assert all(o > 1.8 for o in orders), orders


def test_explicit_insulated_boundaries_conserve_heat():
    n = 81
    X, Y = _grid_xy(n)
    u0 = np.exp(-3.0 * (X + Y)) + 0.1 * (X ** 2 + Y ** 2)
    grid = Grid2D(LENGTH, LENGTH, n, n, ALPHA, u0)
    before = grid.total_heat()

    dt = max_stable_dt_2d(grid, safety=0.9)
    for _ in range(3000):
        explicit_step_2d(grid, dt, boundary="neumann")

    assert grid.total_heat() == pytest.approx(before, rel=1e-10)


def test_explicit_diverges_above_2d_cfl_limit():
    grid = _seeded_grid(81)
    dt = max_stable_dt_2d(grid, safety=1.0) * 3.0
    peak_before = np.abs(grid.u).max()

    for _ in range(300):
        explicit_step_2d(grid, dt)

    assert np.abs(grid.u).max() > 1e6 * peak_before


def test_crank_nicolson_matches_gaussian_point_source():
    grid = _seeded_grid(81)
    dt_scale = max_stable_dt_2d(grid, safety=0.9)
    n_steps = int(np.ceil((T_END - T_START) / dt_scale))

    run_crank_nicolson_2d(grid, T_START, T_END, n_steps=n_steps)

    assert _relative_rmse(grid.u, 81) < 1e-2


def test_crank_nicolson_matches_exact_fourier_mode():
    n = 41
    t0, t1 = 0.0, 0.5
    X, Y = _grid_xy(n)
    u0 = _exact_fourier_mode(X, Y, t0)
    exact = _exact_fourier_mode(X, Y, t1)

    grid = Grid2D(LENGTH, LENGTH, n, n, ALPHA, u0)
    run_crank_nicolson_2d(grid, t0, t1, n_steps=20)

    rel = np.sqrt(np.mean((grid.u - exact) ** 2)) / np.max(np.abs(exact))
    assert rel < 1e-4


def test_crank_nicolson_is_second_order_accurate_in_space():
    errors = []
    for n in (41, 81, 161, 321):
        grid = _seeded_grid(n)
        run_crank_nicolson_2d(grid, T_START, T_START + 0.01, n_steps=400)
        X, Y = _grid_xy(n)
        exact = gaussian_point_source_2d(X, Y, T_START + 0.01, ALPHA, X0, Y0)
        rmse = np.sqrt(np.mean((grid.u - exact) ** 2)) / exact.max()
        errors.append(rmse)
    orders = [np.log2(a / b) for a, b in zip(errors, errors[1:])]
    assert all(o > 1.8 for o in orders), orders


def test_crank_nicolson_stable_well_past_explicit_cfl_limit():
    grid = _seeded_grid(81)
    dt = max_stable_dt_2d(grid, safety=1.0) * 3.0
    peak_before = np.abs(grid.u).max()

    for _ in range(300):
        crank_nicolson_step_2d(grid, dt)

    assert np.all(np.isfinite(grid.u))
    assert np.abs(grid.u).max() < 10 * peak_before


def test_crank_nicolson_rejects_neumann():
    grid = _seeded_grid(21)
    with pytest.raises(ValueError):
        crank_nicolson_step_2d(grid, 1e-4, boundary="neumann")


def test_1d_solvers_are_unaffected():
    from heatsim.grid import Grid1D
    from heatsim.solvers import explicit_step, crank_nicolson_step

    x = np.linspace(0.0, LENGTH, 401)
    from heatsim.analytical import gaussian_point_source
    u0 = gaussian_point_source(x, T_START, ALPHA, X0)

    g1 = Grid1D(LENGTH, 401, ALPHA, u0)
    explicit_step(g1, 1e-6)
    g2 = Grid1D(LENGTH, 401, ALPHA, u0)
    crank_nicolson_step(g2, 1e-6)

    assert np.all(np.isfinite(g1.u))
    assert np.all(np.isfinite(g2.u))