from dataclasses import dataclass

import numpy as np
import pytest

from heatsim.grid import Grid1D, Grid2D
from heatsim.boundary import Dirichlet, Neumann, Convective
from heatsim.solvers import (explicit_step, run_explicit, max_stable_dt,
                             crank_nicolson_step, run_crank_nicolson,
                             explicit_step_2d, crank_nicolson_step_2d,
                             max_monotone_dt, _crank_nicolson_general_step)
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
    g2 = Grid2D(1.0, 1.0, 21, 21, initial_temperature=1.0, k=K, rho_c=K / ALPHA)
    conv = Convective(h=10.0, u_inf=T_INF)

    with pytest.raises(ValueError, match="does not support"):
        explicit_step_2d(g2, 1e-3, boundary=conv)
    with pytest.raises(ValueError, match="only supports Dirichlet"):
        crank_nicolson_step_2d(g2, 1e-3, boundary=conv)


def test_uniform_only_solvers_reject_asymmetric_boundaries():
    g2 = Grid2D(1.0, 1.0, 21, 21, initial_temperature=1.0, k=K, rho_c=K / ALPHA)
    mixed = (Dirichlet(), Neumann())

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


def test_crank_nicolson_matches_semi_infinite_convective_solution():
    n = 401
    grid, xs = _grid(n)
    boundary = (Convective(h=H, u_inf=T_INF), Dirichlet())

    run_crank_nicolson(grid, 0.0, T_END, boundary=boundary, n_steps=4000)

    exact = _exact(xs, T_END)
    rel = np.sqrt(np.mean((grid.u - exact) ** 2)) / (T_INF - T_I)
    assert rel < 1e-3


def test_crank_nicolson_convective_boundary_is_second_order_accurate():
    errors = []
    for n in [101, 201, 401, 801]:
        grid, xs = _grid(n)
        run_crank_nicolson(grid, 0.0, T_END, n_steps=4000,
                          boundary=(Convective(h=H, u_inf=T_INF), Dirichlet()))
        errors.append(np.sqrt(np.mean((grid.u - _exact(xs, T_END)) ** 2)))

    orders = _convergence_orders(errors)
    assert all(o > 1.85 for o in orders), orders


@pytest.mark.parametrize("bc", [Dirichlet(), Neumann()])
def test_general_path_reproduces_the_uniform_paths(bc):
    xs = np.linspace(0.0, 1.0, 101)
    u0 = np.exp(-3.0 * xs) + 0.2 * xs ** 2

    legacy = Grid1D(1.0, 101, initial_temperature=u0.copy(), k=0.6, rho_c=60.0)
    general = Grid1D(1.0, 101, initial_temperature=u0.copy(), k=0.6, rho_c=60.0)

    for _ in range(50):
        crank_nicolson_step(legacy, 1e-3, boundary=bc)
        _crank_nicolson_general_step(general, 1e-3, (bc, bc))

    assert np.allclose(legacy.u, general.u, rtol=0, atol=1e-12)


def test_crank_nicolson_stays_bounded_far_beyond_the_explicit_limit():
    n = 161
    grid, xs = _grid(n)
    boundary = (Convective(h=1e4, u_inf=T_INF), Dirichlet())
    explicit_limit = max_stable_dt(grid, safety=0.9, boundary=boundary)

    run_crank_nicolson(grid, 0.0, 2.0, boundary=boundary,
                      dt=200.0 * explicit_limit)

    assert np.all(np.isfinite(grid.u))


@pytest.mark.parametrize("h", [1e2, 1e3, 1e4, 1e6])
def test_max_monotone_dt_keeps_crank_nicolson_free_of_overshoot(h):
    n = 161
    grid, xs = _grid(n)
    boundary = (Convective(h=h, u_inf=T_INF), Dirichlet())
    dt = max_monotone_dt(grid, boundary=boundary, safety=0.9)

    for _ in range(40):
        crank_nicolson_step(grid, dt, boundary=boundary)
        assert grid.u.max() <= T_INF + 1e-9
        assert grid.u.min() >= T_I - 1e-9


def test_max_monotone_dt_is_not_needlessly_small():
    n = 41

    def factory():
        return Grid1D(1.0, n, initial_temperature=np.zeros(n),
                      k=0.6, rho_c=1e4)

    dt = max_monotone_dt(factory(), boundary="neumann", safety=0.9)
    assert _amplification_min_eigenvalue(factory, n, 1.5 * dt, "neumann") < 0.0


def _amplification_min_eigenvalue(grid_factory, n, dt, boundary):
    matrix = np.zeros((n, n))
    for j in range(n):
        grid = grid_factory()
        grid.u[:] = np.eye(n)[j]
        crank_nicolson_step(grid, dt, boundary=boundary)
        matrix[:, j] = grid.u
    return np.linalg.eigvals(matrix).real.min()


_MONOTONE_MATERIALS = {
    "uniform": (lambda n: np.full(n, 0.6), lambda n: np.full(n, 1e4)),
    "conductive_skin": (
        lambda n: np.concatenate([[200.0, 200.0], np.full(n - 2, 0.1)]),
        lambda n: np.full(n, 1e4)),
    "light_skin": (
        lambda n: np.full(n, 0.6),
        lambda n: np.concatenate([[1e2, 1e2], np.full(n - 2, 1e4)])),
    "graded": (lambda n: np.linspace(0.1, 50.0, n),
               lambda n: np.linspace(1e3, 1e6, n)),
}

_MONOTONE_BOUNDARIES = {
    "neumann": "neumann",
    "dirichlet": "dirichlet",
    "convective_weak": (Convective(h=1e3, u_inf=0.0), Dirichlet()),
    "convective_stiff": (Convective(h=1e6, u_inf=0.0), Dirichlet()),
    "both_convective": (Convective(h=1e4, u_inf=0.0),
                        Convective(h=1e4, u_inf=0.0)),
}


@pytest.mark.parametrize("material", list(_MONOTONE_MATERIALS))
@pytest.mark.parametrize("boundary_name", list(_MONOTONE_BOUNDARIES))
def test_max_monotone_dt_admits_no_oscillatory_modes(material, boundary_name):
    n = 41
    k_fn, rho_c_fn = _MONOTONE_MATERIALS[material]
    boundary = _MONOTONE_BOUNDARIES[boundary_name]

    def factory():
        return Grid1D(1.0, n, initial_temperature=np.zeros(n),
                      k=k_fn(n), rho_c=rho_c_fn(n))

    dt = max_monotone_dt(factory(), boundary=boundary, safety=0.9)
    assert _amplification_min_eigenvalue(factory, n, dt, boundary) >= -1e-12


@pytest.mark.parametrize("boundary", ["dirichlet", "neumann"])
def test_max_monotone_dt_is_finite_without_convection(boundary):
    grid, _ = _grid(81)
    dt = max_monotone_dt(grid, boundary=boundary)
    assert np.isfinite(dt)
    assert dt > 0.0


@pytest.mark.parametrize("boundary", ["dirichlet", "neumann"])
def test_crank_nicolson_does_not_ring_on_sharp_data_at_the_monotone_limit(boundary):
    n = 81
    grid, _ = _grid(n)
    grid.u[n // 2] = T_INF
    dt = max_monotone_dt(grid, boundary=boundary)

    for _ in range(60):
        crank_nicolson_step(grid, dt, boundary=boundary)
        assert grid.u.min() >= T_I - 1e-9
        assert grid.u.max() <= T_INF + 1e-9


def test_convection_tightens_the_monotone_limit():
    grid, _ = _grid(81)
    plain = max_monotone_dt(grid, boundary="neumann")
    convective = max_monotone_dt(
        grid, boundary=(Convective(h=1e5, u_inf=T_INF), Dirichlet()))
    assert convective < plain


def test_general_path_rejects_unhandled_boundary_types():
    @dataclass(frozen=True)
    class _Unhandled:
        pass

    grid, _ = _grid(21)
    with pytest.raises(TypeError):
        _crank_nicolson_general_step(grid, 1e-3, (_Unhandled(), Dirichlet()))


def test_zero_h_convective_matches_neumann_in_crank_nicolson():
    n = 101
    g_conv, _ = _grid(n)
    g_neu, _ = _grid(n)
    g_conv.u[:] = np.exp(-3.0 * np.linspace(0.0, 1.0, n))
    g_neu.u[:] = g_conv.u

    for _ in range(30):
        crank_nicolson_step(g_conv, 1e-3,
                           boundary=(Convective(h=0.0, u_inf=999.0), Dirichlet()))
        crank_nicolson_step(g_neu, 1e-3, boundary=(Neumann(), Dirichlet()))

    assert np.array_equal(g_conv.u, g_neu.u)


@pytest.mark.parametrize("solver", ["explicit", "crank_nicolson"])
def test_convective_boundary_is_mirror_symmetric(solver):
    n = 401
    conv = Convective(h=H, u_inf=T_INF)

    def run(bc):
        grid, _ = _grid(n)
        if solver == "explicit":
            run_explicit(grid, 0.0, T_END, safety=0.9, boundary=bc)
        else:
            run_crank_nicolson(grid, 0.0, T_END, boundary=bc, n_steps=8000)
        return grid.u.copy()

    left = run((conv, Dirichlet()))
    right = run((Dirichlet(), conv))
    assert np.allclose(left, right[::-1], rtol=0, atol=1e-11)


def test_convective_boundaries_reach_the_exact_robin_steady_state():
    n = 81
    h1, t1, h2, t2 = 100.0, 0.0, 100.0, 100.0
    grid, xs = _grid(n)
    grid.u[:] = 50.0
    boundary = (Convective(h=h1, u_inf=t1), Convective(h=h2, u_inf=t2))

    flux = -(K / LENGTH) * (t2 - t1) / (1.0 + (K / LENGTH) * (1.0 / h1 + 1.0 / h2))
    u0, u_l = t1 - flux / h1, t2 + flux / h2
    expected = u0 + (u_l - u0) * xs / LENGTH

    dt = max_monotone_dt(grid, boundary=boundary)
    run_crank_nicolson(grid, 0.0, 60000.0, boundary=boundary,
                      n_steps=int(np.ceil(60000.0 / dt)))

    assert np.allclose(grid.u, expected, rtol=0, atol=1e-6)


def test_convective_boundary_energy_balance_closes():
    n = 201
    grid, _ = _grid(n)
    boundary = (Convective(h=H, u_inf=T_INF), Neumann())
    before = grid.total_heat()

    n_steps, dt = 4000, 0.5
    influx = 0.0
    for _ in range(n_steps):
        u0_before = grid.u[0]
        crank_nicolson_step(grid, dt, boundary=boundary)
        influx += 0.5 * (H * (T_INF - u0_before)
                         + H * (T_INF - grid.u[0])) * dt

    change = grid.total_heat() - before
    assert change == pytest.approx(influx, rel=1e-8)


def _skin_grid(n=81):
    conductivity = np.full(n, 20.0)
    conductivity[0] = 0.05
    capacity = np.full(n, 1.0e6)
    capacity[0] = 1.0e3
    grid = Grid1D(LENGTH, n, initial_temperature=np.full(n, T_I),
                  k=conductivity, rho_c=capacity)
    return grid


@pytest.mark.parametrize("h", [1e2, 1e3, 1e4])
def test_convective_timestep_uses_boundary_local_material(h):
    grid = _skin_grid()
    boundary = (Convective(h=h, u_inf=T_INF), Dirichlet())

    dt = max_stable_dt(grid, safety=0.9, boundary=boundary)
    local_limit = (grid.rho_c[0] * grid.dx ** 2
                   / (2.0 * grid.face_k[0] + h * grid.dx))
    assert dt <= local_limit

    for _ in range(400):
        explicit_step(grid, dt, boundary=boundary)
    assert np.all(np.isfinite(grid.u))
    assert grid.u.max() <= T_INF + 1e-6
    assert grid.u.min() >= T_I - 1e-6


def test_convective_skin_would_diverge_on_the_interior_material_bound():
    grid = _skin_grid()
    h = 1e3
    boundary = (Convective(h=h, u_inf=T_INF), Dirichlet())

    correct = max_stable_dt(grid, safety=0.9, boundary=boundary)
    interior_based = min(
        0.9 * grid.dx ** 2 / (2.0 * float(np.max(grid.alpha))),
        0.9 * grid.rho_c[1] * grid.dx ** 2
        / (2.0 * grid.face_k[1] + h * grid.dx))

    assert correct < interior_based