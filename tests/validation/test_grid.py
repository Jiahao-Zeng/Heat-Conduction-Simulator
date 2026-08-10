import numpy as np
import pytest

from heatsim.grid import Grid1D, face_diffusivity
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


def test_face_diffusivity_is_cached_and_invalidated():
    grid = Grid1D(LENGTH, 11, ALPHA)
    first = grid.face_alpha
    assert grid.face_alpha is first, "should be cached, not recomputed"

    grid.alpha[:] = ALPHA * 4
    grid.invalidate_material()

    assert grid.face_alpha == pytest.approx(ALPHA * 4)


def test_changing_material_changes_crank_nicolson_result():
    baseline = Grid1D(LENGTH, N, ALPHA * 4, _asymmetric_profile())
    crank_nicolson_step(baseline, 1e-4)

    grid = Grid1D(LENGTH, N, ALPHA, _asymmetric_profile())
    crank_nicolson_step(grid, 1e-4)          # populates the cache
    grid.u[:] = _asymmetric_profile()        # reset the field
    grid.alpha[:] = ALPHA * 4
    grid.invalidate_material()
    crank_nicolson_step(grid, 1e-4)

    assert grid.u == pytest.approx(baseline.u, rel=1e-12)


def test_face_diffusivity_handles_zero_conductivity():
    alpha = np.array([0.01, 0.0, 0.01])
    faces = face_diffusivity(alpha)

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


def test_passing_both_dt_and_safety_is_rejected():
    grid = Grid1D(LENGTH, N, ALPHA, _asymmetric_profile())
    with pytest.raises(ValueError):
        run_explicit(grid, 0.0, 0.01, safety=0.1, dt=1e-4)


def test_crank_nicolson_requires_exactly_one_of_dt_or_n_steps():
    grid = Grid1D(LENGTH, N, ALPHA, _asymmetric_profile())
    with pytest.raises(ValueError):
        run_crank_nicolson(grid, 0.0, 0.01)
    with pytest.raises(ValueError):
        run_crank_nicolson(grid, 0.0, 0.01, dt=1e-4, n_steps=10)


def test_invalid_grid_arguments_are_rejected():
    with pytest.raises(ValueError):
        Grid1D(LENGTH, 2, ALPHA)
    with pytest.raises(ValueError):
        Grid1D(0.0, N, ALPHA)
    with pytest.raises(ValueError):
        Grid1D(LENGTH, N, -ALPHA)


def test_copy_is_independent():
    grid = Grid1D(LENGTH, N, ALPHA, _asymmetric_profile())
    clone = grid.copy()
    clone.u[:] = 0.0
    clone.alpha[:] = 999.0

    assert np.any(grid.u != 0.0)
    assert grid.alpha == pytest.approx(ALPHA)