import numpy as np
import pytest

from heatsim.grid import Grid1D
from heatsim.solvers import (
    crank_nicolson_step,
    run_crank_nicolson,
    explicit_step,
    max_stable_dt,
    thomas_solve,
)
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


def _relative_rmse(u_numerical, x):
    exact = gaussian_point_source(x, T_END, ALPHA, X0)
    rmse = np.sqrt(np.mean((u_numerical - exact) ** 2))
    return rmse / exact.max()


def test_thomas_solve_matches_dense_solve():
    rng = np.random.default_rng(0)
    n = 8
    diag = rng.uniform(4, 6, n)
    sub = np.concatenate(([0.0], rng.uniform(-1.0, -0.5, n - 1)))
    sup = np.concatenate((rng.uniform(-1.0, -0.5, n - 1), [0.0]))
    rhs = rng.uniform(-1.0, 1.0, n)

    dense = np.diag(diag) + np.diag(sub[1:], -1) + np.diag(sup[:-1], 1)
    expected = np.linalg.solve(dense, rhs)
    actual = thomas_solve(sub, diag, sup, rhs)

    assert actual == pytest.approx(expected, abs=1e-10)


def test_matches_gaussian_point_source():
    n_points = 401
    grid = _seeded_grid(n_points)
    dt_scale = max_stable_dt(grid, safety=0.9)
    n_steps = int(np.ceil((T_END - T_START) / dt_scale))

    run_crank_nicolson(grid, T_START, T_END, n_steps=n_steps)
    x = np.linspace(0.0, LENGTH, n_points)

    assert _relative_rmse(grid.u, x) < 1e-3


def test_is_second_order_accurate_in_space():
    errors = []
    for n_points in (101, 201, 401, 801):
        grid = _seeded_grid(n_points)
        x = np.linspace(0.0, LENGTH, n_points)
        run_crank_nicolson(grid, T_START, T_END, dt=1e-5)
        errors.append(_relative_rmse(grid.u, x))

    orders = [np.log2(a / b) for a, b in zip(errors, errors[1:])]
    assert all(o > 1.8 for o in orders), orders


def test_is_second_order_accurate_in_time():
    n_points = 1601
    x = np.linspace(0.0, LENGTH, n_points)
    errors = []
    for n_steps in (4, 8, 16, 32):
        grid = _seeded_grid(n_points)
        run_crank_nicolson(grid, T_START, T_END, n_steps=n_steps)
        errors.append(_relative_rmse(grid.u, x))

    orders = [np.log2(a / b) for a, b in zip(errors, errors[1:])]
    assert all(o > 1.7 for o in orders), orders


def test_insulated_boundaries_conserve_heat():
    grid = _seeded_grid(401)
    before = grid.total_heat()
    run_crank_nicolson(grid, T_START, T_END, boundary="neumann", n_steps=200)

    assert grid.total_heat() == pytest.approx(before, rel=1e-10)


@pytest.mark.parametrize("cfl_multiple", [2, 5, 10, 50])
def test_stable_well_past_explicit_cfl_limit(cfl_multiple):
    n_points = 401
    grid = _seeded_grid(n_points)
    x = np.linspace(0.0, LENGTH, n_points)

    dt_cfl = max_stable_dt(grid, safety=1.0)
    dt = dt_cfl * cfl_multiple
    n_steps = int(np.ceil((T_END - T_START) / dt))

    run_crank_nicolson(grid, T_START, T_END, n_steps=n_steps)

    assert np.all(np.isfinite(grid.u))
    assert _relative_rmse(grid.u, x) < 0.05


def test_explicit_actually_diverges_at_the_same_large_dt():
    n_points = 401
    grid = _seeded_grid(n_points)
    dt_cfl = max_stable_dt(grid, safety=1.0)
    dt = dt_cfl * 10
    peak_before = np.abs(grid.u).max()

    n_steps = int(np.ceil((T_END - T_START) / dt))
    for _ in range(n_steps):
        explicit_step(grid, dt)

    assert np.abs(grid.u).max() > 100 * peak_before