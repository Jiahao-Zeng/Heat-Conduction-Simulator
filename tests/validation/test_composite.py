import numpy as np
import pytest

from heatsim.grid import Grid1D, Grid2D
from heatsim.solvers import (
    run_explicit,
    run_crank_nicolson,
    explicit_step,
    explicit_step_2d,
    crank_nicolson_step,
    max_stable_dt,
    max_stable_dt_2d,
)

LENGTH = 1.0
N = 101
HOT, COLD = 100.0, 0.0


def _two_layer(k_left, k_right, rho_c_left, rho_c_right, n=N):
    k = np.where(np.arange(n) < n // 2, k_left, k_right).astype(float)
    rho_c = np.where(np.arange(n) < n // 2, rho_c_left, rho_c_right).astype(float)
    u = np.zeros(n)
    u[0], u[-1] = HOT, COLD
    return Grid1D(LENGTH, n, initial_temperature=u, k=k, rho_c=rho_c)


def _series_resistance_profile(grid, k_left, k_right):
    n = grid.n_points
    interface = 0.5 * (grid.x[n // 2 - 1] + grid.x[n // 2])
    flux = (HOT - COLD) / (interface / k_left + (LENGTH - interface) / k_right)

    left = HOT - flux * grid.x / k_left
    right = (HOT - flux * interface / k_left
             - flux * (grid.x - interface) / k_right)
    return np.where(grid.x < interface, left, right)


def test_identical_conductivity_gives_a_linear_steady_state():
    grid = _two_layer(k_left=10.0, k_right=10.0,
                      rho_c_left=1.0, rho_c_right=8.0)
    run_explicit(grid, 0.0, 2.0)

    linear = HOT + (COLD - HOT) * grid.x / LENGTH
    assert grid.u == pytest.approx(linear, abs=1e-6)


def test_steady_state_matches_series_resistance():
    grid = _two_layer(k_left=20.0, k_right=2.0,
                      rho_c_left=3.0, rho_c_right=3.0)
    run_explicit(grid, 0.0, 3.0)

    expected = _series_resistance_profile(grid, 20.0, 2.0)
    assert grid.u == pytest.approx(expected, abs=1e-6)


def test_crank_nicolson_reaches_the_same_steady_state():
    explicit = _two_layer(20.0, 2.0, 3.0, 3.0)
    run_explicit(explicit, 0.0, 3.0)

    implicit = _two_layer(20.0, 2.0, 3.0, 3.0)
    run_crank_nicolson(implicit, 0.0, 3.0, n_steps=3000)

    assert implicit.u == pytest.approx(explicit.u, abs=1e-6)


def test_energy_is_conserved_with_non_uniform_heat_capacity():
    x = np.linspace(0.0, LENGTH, N)
    grid = Grid1D(LENGTH, N,
                  initial_temperature=np.exp(-3.0 * x) + 0.2 * x ** 2,
                  k=np.full(N, 2.0),
                  rho_c=np.linspace(1.0, 5.0, N))
    before = grid.total_heat()

    dt = max_stable_dt(grid)
    for _ in range(5000):
        explicit_step(grid, dt, boundary="neumann")

    assert grid.total_heat() == pytest.approx(before, rel=1e-10)


def test_alpha_is_derived_from_k_and_rho_c():
    grid = _two_layer(20.0, 2.0, 4.0, 1.0)
    assert grid.alpha == pytest.approx(grid.k / grid.rho_c)


def test_alpha_shorthand_matches_explicit_unit_heat_capacity():
    x = np.linspace(0.0, LENGTH, N)
    u0 = np.exp(-3.0 * x)

    shorthand = Grid1D(LENGTH, N, 0.01, u0)
    explicit_form = Grid1D(LENGTH, N, initial_temperature=u0,
                           k=np.full(N, 0.01), rho_c=np.ones(N))

    dt = max_stable_dt(shorthand)
    for _ in range(500):
        explicit_step(shorthand, dt)
        explicit_step(explicit_form, dt)

    assert shorthand.u == pytest.approx(explicit_form.u, abs=0.0)


def test_2d_composite_reduces_to_the_1d_solver():
    n = 61
    x = np.linspace(0.0, LENGTH, n)
    k_x = np.where(x < 0.5, 20.0, 2.0)
    rho_c_x = np.where(x < 0.5, 1.0, 8.0)
    u_x = np.exp(-4.0 * x) + 0.3 * x ** 2

    one_d = Grid1D(LENGTH, n, initial_temperature=u_x, k=k_x, rho_c=rho_c_x)
    two_d = Grid2D(LENGTH, LENGTH, n, n,
                   initial_temperature=np.repeat(u_x[:, None], n, axis=1),
                   k=np.repeat(k_x[:, None], n, axis=1),
                   rho_c=np.repeat(rho_c_x[:, None], n, axis=1))

    dt = min(max_stable_dt(one_d), max_stable_dt_2d(two_d))
    for _ in range(2000):
        explicit_step(one_d, dt, boundary="neumann")
        explicit_step_2d(two_d, dt, boundary="neumann")

    assert two_d.u[:, 0] == pytest.approx(one_d.u, abs=1e-12)
    assert np.max(np.abs(two_d.u - two_d.u[:, :1])) == 0.0


LAYER_K1, LAYER_K2 = 0.5, 8.0
LAYER_INTERFACE = 0.4
LAYER_T0, LAYER_TL = 0.0, 100.0
LAYER_LENGTH = 1.0


def _layered_steady_profile(xs, interface):
    flux = (LAYER_TL - LAYER_T0) / (interface / LAYER_K1
                                    + (LAYER_LENGTH - interface) / LAYER_K2)
    return np.where(
        xs < interface,
        LAYER_T0 + flux * xs / LAYER_K1,
        LAYER_T0 + flux * interface / LAYER_K1
        + flux * (xs - interface) / LAYER_K2)


def _layered_grid(n):
    xs = np.linspace(0.0, LAYER_LENGTH, n)
    dx = LAYER_LENGTH / (n - 1)
    conductivity = np.where(xs < LAYER_INTERFACE, LAYER_K1, LAYER_K2)

    profile = _layered_steady_profile(xs, LAYER_INTERFACE - dx / 2.0)
    grid = Grid1D(LAYER_LENGTH, n, initial_temperature=profile.copy(),
                  k=conductivity, rho_c=np.full(n, 1e3))
    return grid, profile


@pytest.mark.parametrize("n", [101, 201, 401])
def test_two_layer_steady_state_is_a_discrete_fixed_point(n):
    grid, profile = _layered_grid(n)

    for _ in range(200):
        crank_nicolson_step(grid, 1.0, boundary="dirichlet")

    assert np.abs(grid.u - profile).max() < 1e-8


def test_two_layer_steady_state_is_a_fixed_point_for_the_explicit_solver():
    grid, profile = _layered_grid(201)
    dt = max_stable_dt(grid, safety=0.9)

    for _ in range(5000):
        explicit_step(grid, dt, boundary="dirichlet")

    assert np.abs(grid.u - profile).max() < 1e-9


def test_heat_flux_is_continuous_across_a_material_interface():
    grid, _ = _layered_grid(201)

    flux = grid.face_k * np.diff(grid.u) / grid.dx

    assert flux.max() - flux.min() < 1e-8 * abs(flux.mean())