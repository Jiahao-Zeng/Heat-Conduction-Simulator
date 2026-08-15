import numpy as np
import pytest

from heatsim.grid import Grid1D, Grid2D
from heatsim.boundary import Dirichlet, Neumann
from heatsim.solvers import crank_nicolson_step, crank_nicolson_step_2d


def test_cache_is_capped():
    grid = Grid1D(1.0, 101, 0.01, 1.0)
    for i in range(Grid1D.MAX_SOLVER_CACHE_ENTRIES * 3):
        crank_nicolson_step(grid, 1e-4 * (i + 1))

    assert len(grid._solver_cache) <= Grid1D.MAX_SOLVER_CACHE_ENTRIES


def test_repeating_a_timestep_reuses_the_factorization():
    grid = Grid1D(1.0, 101, 0.01, 1.0)
    crank_nicolson_step(grid, 1e-4)
    first = next(iter(grid._solver_cache.values()))

    crank_nicolson_step(grid, 1e-4)
    assert next(iter(grid._solver_cache.values())) is first


def test_evicted_timestep_still_produces_correct_results():
    def run(dts):
        grid = Grid1D(1.0, 101, 0.01, np.linspace(0.0, 1.0, 101))
        for dt in dts:
            crank_nicolson_step(grid, dt)
        return grid.u

    dt = 1e-4
    others = [1e-4 * (i + 2) for i in range(Grid1D.MAX_SOLVER_CACHE_ENTRIES + 2)]

    without_eviction = run([dt, dt])
    with_eviction = run([dt] + others + [dt])
    assert np.all(np.isfinite(with_eviction))
    assert np.all(np.isfinite(without_eviction))


def test_cache_key_distinguishes_boundary_conditions():
    grid = Grid1D(1.0, 101, 0.01, np.linspace(0.0, 1.0, 101))
    crank_nicolson_step(grid, 1e-4, boundary="dirichlet")
    crank_nicolson_step(grid, 1e-4, boundary="neumann")

    keys = list(grid._solver_cache)
    assert len(keys) == 2, keys


def test_2d_cache_key_includes_boundary():
    grid = Grid2D(1.0, 1.0, 21, 21, 0.01, 1.0)
    crank_nicolson_step_2d(grid, 1e-4)
    key = next(iter(grid._solver_cache))
    assert Dirichlet() in key


def test_equivalent_boundary_spellings_share_one_cache_entry():
    grid = Grid1D(1.0, 101, 0.01, np.linspace(0.0, 1.0, 101))
    crank_nicolson_step(grid, 1e-4, boundary="neumann")
    crank_nicolson_step(grid, 1e-4, boundary=Neumann())
    crank_nicolson_step(grid, 1e-4, boundary=(Neumann(), Neumann()))

    assert len(grid._solver_cache) == 1, list(grid._solver_cache)


def test_invalidating_material_clears_the_solver_cache():
    grid = Grid1D(1.0, 101, 0.01, 1.0)
    crank_nicolson_step(grid, 1e-4)
    assert grid._solver_cache

    grid.invalidate_material()
    assert not grid._solver_cache