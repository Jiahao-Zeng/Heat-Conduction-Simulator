import numpy as np
import pytest

from heatsim.grid import Grid2D
from heatsim.solvers import (
    explicit_step_2d,
    run_explicit_2d,
    run_crank_nicolson_2d,
    max_stable_dt_2d,
)
from heatsim.analytical import gaussian_point_source_2d_anisotropic

LENGTH = 1.0
ALPHA_X = 0.02
ALPHA_Y = 0.005
X0 = Y0 = LENGTH / 2.0
T_START = 0.02
T_END = 0.05


def _grid_xy(n, length_x=LENGTH, length_y=LENGTH):
    xs = np.linspace(0.0, length_x, n)
    ys = np.linspace(0.0, length_y, n)
    return np.meshgrid(xs, ys, indexing="ij")


def _anisotropic_grid(n, u0, alpha_x=ALPHA_X, alpha_y=ALPHA_Y,
                      length_x=LENGTH, length_y=LENGTH):
    return Grid2D(length_x, length_y, n, n, initial_temperature=u0,
                  k=alpha_x, k_y=alpha_y, rho_c=1.0)


def _seeded_anisotropic_grid(n):
    X, Y = _grid_xy(n)
    u0 = gaussian_point_source_2d_anisotropic(
        X, Y, T_START, ALPHA_X, ALPHA_Y, X0, Y0)
    return _anisotropic_grid(n, u0)


def _point_source_error(u, n):
    X, Y = _grid_xy(n)
    exact = gaussian_point_source_2d_anisotropic(
        X, Y, T_END, ALPHA_X, ALPHA_Y, X0, Y0)
    return np.sqrt(np.mean((u - exact) ** 2)) / exact.max()


def test_crank_nicolson_matches_anisotropic_gaussian_point_source():
    n = 81
    grid = _seeded_anisotropic_grid(n)
    n_steps = int(np.ceil((T_END - T_START) / max_stable_dt_2d(grid, safety=0.9)))

    run_crank_nicolson_2d(grid, T_START, T_END, n_steps=n_steps)

    assert _point_source_error(grid.u, n) < 1e-2


def test_explicit_matches_anisotropic_gaussian_point_source():
    n = 81
    grid = _seeded_anisotropic_grid(n)

    run_explicit_2d(grid, T_START, T_END, safety=0.9)

    assert _point_source_error(grid.u, n) < 1e-3


MODE_M, MODE_N = 1, 2
LX, LY = 1.0, 0.75


def _exact_anisotropic_fourier_mode(X, Y, t, alpha_x=ALPHA_X, alpha_y=ALPHA_Y):
    kx = MODE_M * np.pi / LX
    ky = MODE_N * np.pi / LY
    decay_rate = alpha_x * kx ** 2 + alpha_y * ky ** 2
    return np.sin(kx * X) * np.sin(ky * Y) * np.exp(-decay_rate * t)


@pytest.mark.parametrize("solver", ["explicit", "crank_nicolson"])
def test_matches_anisotropic_exact_fourier_mode(solver):
    n = 81
    t0, t1 = 0.0, 0.5
    X, Y = _grid_xy(n, LX, LY)
    u0 = _exact_anisotropic_fourier_mode(X, Y, t0)
    exact = _exact_anisotropic_fourier_mode(X, Y, t1)

    grid = _anisotropic_grid(n, u0, length_x=LX, length_y=LY)
    if solver == "explicit":
        run_explicit_2d(grid, t0, t1, safety=0.9)
    else:
        run_crank_nicolson_2d(grid, t0, t1, n_steps=200)

    rel = np.sqrt(np.mean((grid.u - exact) ** 2)) / np.max(np.abs(exact))
    assert rel < 1e-3


def test_cfl_timestep_accounts_for_the_faster_axis():
    n = 61
    X, Y = _grid_xy(n)
    u0 = np.sin(np.pi * X) * np.sin(np.pi * Y)
    grid = _anisotropic_grid(n, u0, alpha_x=0.001, alpha_y=0.05)

    dt = max_stable_dt_2d(grid, safety=1.0)
    true_limit = 1.0 / (2.0 * (0.001 / grid.dx ** 2 + 0.05 / grid.dy ** 2))
    assert dt <= true_limit

    peak_before = np.abs(grid.u).max()
    for _ in range(400):
        explicit_step_2d(grid, max_stable_dt_2d(grid, safety=0.9))
    assert np.all(np.isfinite(grid.u))
    assert np.abs(grid.u).max() <= peak_before


def test_isotropic_grid_is_unchanged_by_the_k_y_parameter():
    n = 41
    X, Y = _grid_xy(n)
    u0 = np.sin(np.pi * X) * np.sin(np.pi * Y)

    implied = Grid2D(LENGTH, LENGTH, n, n, initial_temperature=u0,
                     k=0.01, rho_c=1.0)
    spelled_out = Grid2D(LENGTH, LENGTH, n, n, initial_temperature=u0,
                         k=0.01, k_y=0.01, rho_c=1.0)

    run_crank_nicolson_2d(implied, 0.0, 0.1, n_steps=20)
    run_crank_nicolson_2d(spelled_out, 0.0, 0.1, n_steps=20)

    assert np.allclose(implied.u, spelled_out.u, rtol=0, atol=1e-14)


def test_copy_preserves_anisotropy_and_independence():
    grid = _anisotropic_grid(11, 0.0)
    clone = grid.copy()

    assert np.allclose(clone.k, ALPHA_X)
    assert np.allclose(clone.k_y, ALPHA_Y)

    clone.k_y[:] = 999.0
    assert np.allclose(grid.k_y, ALPHA_Y)
    assert not np.shares_memory(clone.k, clone.k_y)


def test_isotropic_copy_keeps_the_two_axes_tied_together():
    grid = Grid2D(LENGTH, LENGTH, 11, 11, k=0.01, rho_c=1.0)
    clone = grid.copy()

    assert (grid.k_y is grid.k) == (clone.k_y is clone.k)


def test_k_y_cannot_be_combined_with_alpha():
    with pytest.raises(ValueError):
        Grid2D(LENGTH, LENGTH, 11, 11, alpha=0.01, k_y=0.02)


def test_negative_k_y_is_rejected():
    with pytest.raises(ValueError):
        Grid2D(LENGTH, LENGTH, 11, 11, k=0.01, k_y=-1.0, rho_c=1.0)