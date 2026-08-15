import numpy as np
import pytest

from heatsim.grid import Grid1D, Grid2D
from heatsim.boundary import Dirichlet, Neumann, Convective
from heatsim.solvers import (explicit_step, run_explicit, max_stable_dt,
                             crank_nicolson_step, explicit_step_2d,
                             crank_nicolson_step_2d)
from heatsim.analytical import semi_infinite_convective


ALPHA = 1.0e-5
K = 0.6
H = 15.0
T_I = 20.0
T_INF = 100.0
LENGTH = 0.2
T_END = 50.0


def _grid(n):
    xs = np.linspace(0.0, LENGTH, n)
    u0 = np.full(n, T_I)
    grid = Grid1D(LENGTH, n, initial_temperature=u0, k=K, rho_c=K / ALPHA)
    return grid, xs


def _exact(xs, t):
    return semi_infinite_convective(xs, t, ALPHA, H, K, T_I, T_INF)


def test_explicit_matches_semi_infinite_convective_solution():
    n = 401
    grid, xs = _grid(n)
    boundary = (Convective(h=H, u_inf=T_INF), Dirichlet())

    run_explicit(grid, 0.0, T_END, safety=0.9, boundary=boundary)

    exact = _exact(xs, T_END)
    rel = np.sqrt(np.mean((grid.u - exact) ** 2)) / (T_INF - T_I)
    assert rel < 1e-3


def test_convective_boundary_reduces_to_neumann_at_h_zero():
    n = 81
    grid_conv, _ = _grid(n)
    grid_neu, _ = _grid(n)

    run_explicit(grid_conv, 0.0, 5.0, safety=0.9,
                boundary=(Convective(h=0.0, u_inf=T_INF), Dirichlet()))
    run_explicit(grid_neu, 0.0, 5.0, safety=0.9,
                boundary=(Neumann(), Dirichlet()))

    assert np.array_equal(grid_conv.u, grid_neu.u)


def test_convective_matches_dirichlet_in_the_large_h_limit():
    n = 161
    grid, xs = _grid(n)
    huge_h = 1.0e6

    run_explicit(grid, 0.0, 5.0, safety=0.9,
                boundary=(Convective(h=huge_h, u_inf=T_INF), Dirichlet()))

    assert np.all(np.isfinite(grid.u))
    assert grid.u.max() <= T_INF + 1e-6
    assert grid.u.min() >= T_I - 1e-6
    assert grid.u[0] == pytest.approx(T_INF, abs=1e-2)


@pytest.mark.parametrize("biot", [0.0, 0.5, 1.0, 2.0, 2.5, 3.0, 3.5, 4.0,
                                  5.0, 6.0, 8.0, 10.0, 25.0, 100.0, 1e4])
def test_explicit_stays_stable_across_the_per_cell_biot_range(biot):
    n = 161
    grid, xs = _grid(n)
    h = biot * K / grid.dx

    run_explicit(grid, 0.0, 2.0, safety=0.9,
                boundary=(Convective(h=h, u_inf=T_INF), Dirichlet()))

    assert np.all(np.isfinite(grid.u))
    assert grid.u.max() <= T_INF + 1e-6
    assert grid.u.min() >= T_I - 1e-6


@pytest.mark.parametrize("biot", [1.0, 3.0, 5.0, 10.0])
def test_explicit_stable_with_convection_on_both_sides(biot):
    n = 161
    grid, xs = _grid(n)
    h = biot * K / grid.dx
    bc = (Convective(h=h, u_inf=T_INF), Convective(h=h, u_inf=T_INF))

    run_explicit(grid, 0.0, 2.0, safety=0.9, boundary=bc)

    assert np.all(np.isfinite(grid.u))
    assert grid.u.max() <= T_INF + 1e-6
    assert grid.u.min() >= T_I - 1e-6


def test_string_and_object_dirichlet_give_identical_results():
    n = 81
    grid_str, _ = _grid(n)
    grid_obj, _ = _grid(n)

    run_explicit(grid_str, 0.0, 5.0, safety=0.9, boundary="dirichlet")
    run_explicit(grid_obj, 0.0, 5.0, safety=0.9,
                boundary=(Dirichlet(), Dirichlet()))

    assert np.array_equal(grid_str.u, grid_obj.u)


def test_string_and_object_neumann_give_identical_results():
    n = 81
    grid_str, _ = _grid(n)
    grid_obj, _ = _grid(n)

    run_explicit(grid_str, 0.0, 5.0, safety=0.9, boundary="neumann")
    run_explicit(grid_obj, 0.0, 5.0, safety=0.9,
                boundary=(Neumann(), Neumann()))

    assert np.array_equal(grid_str.u, grid_obj.u)


def test_unknown_boundary_string_is_rejected():
    grid, _ = _grid(21)
    with pytest.raises(ValueError):
        explicit_step(grid, 0.001, boundary="bogus")


@pytest.mark.parametrize("spelling", ["dirichlet", "neumann"])
def test_every_1d_solver_accepts_string_and_object_forms_identically(spelling):
    objects = {"dirichlet": Dirichlet(), "neumann": Neumann()}
    obj = objects[spelling]

    for step, kwargs in ((explicit_step, {}), (crank_nicolson_step, {})):
        g_str, _ = _grid(41)
        g_obj, _ = _grid(41)
        g_tup, _ = _grid(41)
        g_str.u[20] = 80.0
        g_obj.u[20] = 80.0
        g_tup.u[20] = 80.0

        step(g_str, 1e-3, boundary=spelling, **kwargs)
        step(g_obj, 1e-3, boundary=obj, **kwargs)
        step(g_tup, 1e-3, boundary=(obj, obj), **kwargs)

        assert np.array_equal(g_str.u, g_obj.u), step.__name__
        assert np.array_equal(g_str.u, g_tup.u), step.__name__


@pytest.mark.parametrize("spelling", ["dirichlet", "neumann"])
def test_explicit_2d_accepts_string_and_object_forms_identically(spelling):
    objects = {"dirichlet": Dirichlet(), "neumann": Neumann()}
    obj = objects[spelling]

    def seeded():
        u0 = np.zeros((21, 21))
        u0[10, 10] = 50.0
        return Grid2D(1.0, 1.0, 21, 21, initial_temperature=u0,
                      k=K, rho_c=K / ALPHA)

    g_str, g_obj = seeded(), seeded()
    explicit_step_2d(g_str, 1e-3, boundary=spelling)
    explicit_step_2d(g_obj, 1e-3, boundary=obj)

    assert np.array_equal(g_str.u, g_obj.u)


def test_crank_nicolson_2d_accepts_the_dirichlet_object():
    g_str = Grid2D(1.0, 1.0, 21, 21, initial_temperature=1.0,
                   k=K, rho_c=K / ALPHA)
    g_obj = Grid2D(1.0, 1.0, 21, 21, initial_temperature=1.0,
                   k=K, rho_c=K / ALPHA)

    crank_nicolson_step_2d(g_str, 1e-3, boundary="dirichlet")
    crank_nicolson_step_2d(g_obj, 1e-3, boundary=Dirichlet())

    assert np.array_equal(g_str.u, g_obj.u)


def test_solvers_without_convective_support_say_so_clearly():
    g1, _ = _grid(41)
    g2 = Grid2D(1.0, 1.0, 21, 21, initial_temperature=1.0, k=K, rho_c=K / ALPHA)
    conv = Convective(h=10.0, u_inf=T_INF)

    with pytest.raises(ValueError, match="does not support"):
        crank_nicolson_step(g1, 1e-3, boundary=conv)
    with pytest.raises(ValueError, match="does not support"):
        explicit_step_2d(g2, 1e-3, boundary=conv)
    with pytest.raises(ValueError, match="only supports Dirichlet"):
        crank_nicolson_step_2d(g2, 1e-3, boundary=conv)


def test_uniform_only_solvers_reject_asymmetric_boundaries():
    g1, _ = _grid(41)
    g2 = Grid2D(1.0, 1.0, 21, 21, initial_temperature=1.0, k=K, rho_c=K / ALPHA)
    mixed = (Dirichlet(), Neumann())

    with pytest.raises(ValueError, match="both sides"):
        crank_nicolson_step(g1, 1e-3, boundary=mixed)
    with pytest.raises(ValueError, match="both sides"):
        explicit_step_2d(g2, 1e-3, boundary=mixed)


def test_negative_h_is_rejected():
    with pytest.raises(ValueError):
        Convective(h=-1.0, u_inf=T_INF)


def _convergence_orders(errors):
    return [np.log2(errors[i] / errors[i + 1]) for i in range(len(errors) - 1)]


def test_convective_boundary_is_second_order_accurate():
    errors = []
    for n in [101, 201, 401, 801]:
        grid, xs = _grid(n)
        run_explicit(grid, 0.0, T_END, safety=0.9,
                    boundary=(Convective(h=H, u_inf=T_INF), Dirichlet()))
        exact = _exact(xs, T_END)
        errors.append(np.sqrt(np.mean((grid.u - exact) ** 2)))

    orders = _convergence_orders(errors)
    assert all(o > 1.85 for o in orders), orders
    assert orders[-1] == pytest.approx(2.0, abs=0.15)