import numpy as np
import pytest

from heatsim.grid import Grid1D, harmonic_face_mean
from heatsim.solvers import (
    explicit_step,
    crank_nicolson_step,
    run_explicit,
    run_crank_nicolson,
    max_stable_dt,
)


LENGTH = 1.0
ALPHA = 0.01
N = 401


def _asymmetric_profile(n=N):
    x = np.linspace(0.0, LENGTH, n)
    return np.exp(-3.0 * x) + 0.2 * x ** 2


def test_total_heat_is_conserved_with_asymmetric_profile():
    grid = Grid1D(LENGTH, N, ALPHA, _asymmetric_profile())
    before = grid.total_heat()

    dt = max_stable_dt(grid)
    for _ in range(20_000):
        explicit_step(grid, dt, boundary="neumann")

    assert grid.u[0] != pytest.approx(_asymmetric_profile()[0], rel=1e-3), \
        "endpoints should have moved, or this test proves nothing"
    assert grid.total_heat() == pytest.approx(before, rel=1e-10)


def test_crank_nicolson_also_conserves_heat_asymmetrically():
    grid = Grid1D(LENGTH, N, ALPHA, _asymmetric_profile())
    before = grid.total_heat()

    run_crank_nicolson(grid, 0.0, 0.5, boundary="neumann", n_steps=500)

    assert grid.total_heat() == pytest.approx(before, rel=1e-10)


@pytest.mark.parametrize("step", [explicit_step, crank_nicolson_step])
def test_steps_mutate_u_in_place(step):
    grid = Grid1D(LENGTH, N, ALPHA, _asymmetric_profile())
    held_reference = grid.u

    step(grid, 1e-6)

    assert held_reference is grid.u


def test_face_conductivity_is_cached_and_invalidated():
    grid = Grid1D(LENGTH, 11, ALPHA)
    first = grid.face_k
    assert grid.face_k is first, "should be cached, not recomputed"

    grid.k[:] = ALPHA * 4
    grid.invalidate_material()

    assert grid.face_k == pytest.approx(ALPHA * 4)
    assert grid.alpha == pytest.approx(ALPHA * 4)


def test_changing_material_changes_crank_nicolson_result():
    baseline = Grid1D(LENGTH, N, ALPHA * 4, _asymmetric_profile())
    crank_nicolson_step(baseline, 1e-4)

    grid = Grid1D(LENGTH, N, ALPHA, _asymmetric_profile())
    crank_nicolson_step(grid, 1e-4)
    grid.u[:] = _asymmetric_profile()
    grid.k[:] = ALPHA * 4
    grid.invalidate_material()
    crank_nicolson_step(grid, 1e-4)

    assert grid.u == pytest.approx(baseline.u, rel=1e-12)


def test_harmonic_face_mean_handles_zero_conductivity():
    alpha = np.array([0.01, 0.0, 0.01])
    faces = harmonic_face_mean(alpha)

    assert np.all(np.isfinite(faces))
    assert faces == pytest.approx(0.0)


def test_zero_length_interval_does_not_divide_by_zero():
    grid = Grid1D(LENGTH, N, ALPHA, _asymmetric_profile())
    _, _, n_steps = run_explicit(grid, 0.05, 0.05)
    assert n_steps >= 1


def test_timestep_larger_than_interval_does_not_divide_by_zero():
    grid = Grid1D(LENGTH, N, ALPHA, _asymmetric_profile())
    _, dt_used, n_steps = run_crank_nicolson(grid, 0.05, 0.10, dt=1.0)
    assert n_steps == 1
    assert dt_used == pytest.approx(0.05)


def test_backwards_interval_is_rejected():
    grid = Grid1D(LENGTH, N, ALPHA, _asymmetric_profile())
    with pytest.raises(ValueError):
        run_explicit(grid, 0.10, 0.05)


@pytest.mark.parametrize("safety", [0.1, 0.9])
def test_passing_both_dt_and_safety_is_rejected(safety):
    grid = Grid1D(LENGTH, N, ALPHA, _asymmetric_profile())
    with pytest.raises(ValueError):
        run_explicit(grid, 0.0, 0.01, safety=safety, dt=1e-4)


def test_crank_nicolson_requires_exactly_one_of_dt_or_n_steps():
    grid = Grid1D(LENGTH, N, ALPHA, _asymmetric_profile())
    with pytest.raises(ValueError):
        run_crank_nicolson(grid, 0.0, 0.01)
    with pytest.raises(ValueError):
        run_crank_nicolson(grid, 0.0, 0.01, dt=1e-4, n_steps=10)


def test_invalid_grid_arguments_are_rejected():
    with pytest.raises(ValueError):
        Grid1D(LENGTH, 2, ALPHA) # too few points
    with pytest.raises(ValueError):
        Grid1D(0.0, N, ALPHA) # non-positive length
    with pytest.raises(ValueError):
        Grid1D(LENGTH, N, -ALPHA) # negative conductivity
    with pytest.raises(ValueError):
        Grid1D(LENGTH, N) # no material given
    with pytest.raises(ValueError):
        Grid1D(LENGTH, N, ALPHA, k=ALPHA) # both forms at once
    with pytest.raises(ValueError):
        Grid1D(LENGTH, N, k=ALPHA, rho_c=0.0) # non-positive heat capacity


def test_copy_is_independent():
    grid = Grid1D(LENGTH, N, ALPHA, _asymmetric_profile())
    clone = grid.copy()
    clone.u[:] = 0.0
    clone.k[:] = 999.0

    assert np.any(grid.u != 0.0)
    assert grid.k == pytest.approx(ALPHA)

def _neumann_orders(step_runner):
    length, alpha, k_val, t_end = 1.0, 1.0e-2, 0.6, 5.0
    kappa = np.pi / length
    errors = []
    for n in [51, 101, 201, 401]:
        xs = np.linspace(0.0, length, n)
        grid = Grid1D(length, n, initial_temperature=np.cos(kappa * xs),
                      k=k_val, rho_c=k_val / alpha)
        step_runner(grid, t_end)
        exact = np.cos(kappa * xs) * np.exp(-alpha * kappa ** 2 * t_end)
        errors.append(np.sqrt(np.mean((grid.u - exact) ** 2)))
    return [np.log2(errors[i] / errors[i + 1]) for i in range(len(errors) - 1)]


def test_explicit_neumann_boundary_is_second_order_accurate():
    orders = _neumann_orders(
        lambda g, t_end: run_explicit(g, 0.0, t_end, safety=0.9,
                                      boundary="neumann"))
    assert all(o > 1.85 for o in orders), orders


def test_crank_nicolson_neumann_boundary_is_second_order_accurate():
    orders = _neumann_orders(
        lambda g, t_end: run_crank_nicolson(g, 0.0, t_end,
                                            boundary="neumann", n_steps=4000))
    assert all(o > 1.85 for o in orders), orders


def test_total_heat_uses_half_size_boundary_control_volumes():
    grid = Grid1D(1.0, 5, initial_temperature=1.0, k=1.0, rho_c=1.0)
    assert grid.total_heat() == pytest.approx(1.0)