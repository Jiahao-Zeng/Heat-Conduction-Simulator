import numpy as np

from heatsim.grid import harmonic_face_mean, face_diffusivity  # re-exported
from heatsim.boundary import (Dirichlet, Neumann, Convective,
                              normalize_boundary, uniform_boundary)

_UNSET = object()


def max_stable_dt(grid, safety=0.9, boundary="dirichlet"):
    dt = safety * grid.dx ** 2 / (2.0 * _max_diffusivity(grid))
    left, right = normalize_boundary(boundary)
    face_k = grid.face_k
    sides = ((left, grid.rho_c[0], face_k[0]),
             (right, grid.rho_c[-1], face_k[-1]))
    for side, rho_c_b, k_face_b in sides:
        if isinstance(side, Convective) and side.h > 0.0:
            bound = (rho_c_b * grid.dx ** 2
                     / (2.0 * k_face_b + side.h * grid.dx))
            dt = min(dt, safety * bound)
    return dt


def _max_diffusivity(grid):
    alpha_max = float(np.max(grid.alpha))
    if alpha_max <= 0.0:
        raise ValueError("cannot derive a timestep for zero diffusivity")
    return alpha_max


def _step_count(t_start, t_end, dt):
    if t_end < t_start:
        raise ValueError("t_end must not precede t_start")
    if dt <= 0.0:
        raise ValueError("dt must be positive")
    return max(1, int(np.ceil((t_end - t_start) / dt)))


def _explicit_plan(grid, t_start, t_end, dt, safety, max_dt_fn, boundary):
    if dt is not None and safety is not _UNSET:
        raise ValueError("pass either dt or safety, not both")
    if dt is None:
        dt = max_dt_fn(grid, safety=0.9 if safety is _UNSET else safety,
                        boundary=boundary)
    n_steps = _step_count(t_start, t_end, dt)
    return n_steps, (t_end - t_start) / n_steps


def _implicit_plan(t_start, t_end, dt, n_steps):
    if (dt is None) == (n_steps is None):
        raise ValueError("provide exactly one of dt or n_steps")
    if n_steps is None:
        n_steps = _step_count(t_start, t_end, dt)
    elif n_steps < 1:
        raise ValueError("n_steps must be at least 1")
    return n_steps, (t_end - t_start) / n_steps


def _march(grid, step, n_steps, dt_used, boundary):
    for _ in range(n_steps):
        step(grid, dt_used, boundary=boundary)
    return grid, dt_used, n_steps


def _explicit_boundary_term(bc, u_boundary):
    if isinstance(bc, (Dirichlet, Neumann)):
        return 0.0
    if isinstance(bc, Convective):
        return bc.h * (bc.u_inf - u_boundary)
    raise TypeError(f"unsupported boundary condition: {bc!r}")


def explicit_step(grid, dt, boundary="dirichlet"):
    u = grid.u
    dx = grid.dx
    left, right = normalize_boundary(boundary)

    flux = grid.face_k * np.diff(u) / dx

    divergence = np.empty_like(u)
    divergence[1:-1] = np.diff(flux)
    if not isinstance(left, Dirichlet):
        divergence[0] = flux[0] + _explicit_boundary_term(left, u[0])
    if not isinstance(right, Dirichlet):
        divergence[-1] = -flux[-1] + _explicit_boundary_term(right, u[-1])

    u[1:-1] += (dt / dx) * grid.inv_rho_c[1:-1] * divergence[1:-1]
    if not isinstance(left, Dirichlet):
        u[0] += (dt / (dx * grid.cell_weight[0])) * grid.inv_rho_c[0] * divergence[0]
    if not isinstance(right, Dirichlet):
        u[-1] += (dt / (dx * grid.cell_weight[-1])) * grid.inv_rho_c[-1] * divergence[-1]

    return u


def run_explicit(grid, t_start, t_end, safety=_UNSET, boundary="dirichlet", dt=None):
    n_steps, dt_used = _explicit_plan(
        grid, t_start, t_end, dt, safety, max_stable_dt, boundary)
    return _march(grid, explicit_step, n_steps, dt_used, boundary)


def tridiagonal_factor(sub, diag, sup):
    n = np.shape(diag)[-1]
    a = np.array(sub, dtype=float)
    b = np.array(diag, dtype=float)
    c = np.array(sup, dtype=float)
    a[..., 0] = 0.0
    c[..., -1] = 0.0

    levels = []
    h = 1
    while h < n:
        a_lo, b_lo, c_lo = (_shift_down(a, h, 0.0), _shift_down(b, h, 1.0),
                            _shift_down(c, h, 0.0))
        a_hi, b_hi, c_hi = (_shift_up(a, h, 0.0), _shift_up(b, h, 1.0),
                            _shift_up(c, h, 0.0))

        alpha = -a / b_lo
        beta = -c / b_hi

        a = alpha * a_lo
        b = b + alpha * c_lo + beta * a_hi
        c = beta * c_hi

        levels.append((alpha, beta, h))
        h *= 2

    return levels, b


def _shift_down(arr, h, fill):
    out = np.full_like(arr, fill)
    out[..., h:] = arr[..., :-h]
    return out


def _shift_up(arr, h, fill):
    out = np.full_like(arr, fill)
    out[..., :-h] = arr[..., h:]
    return out


def tridiagonal_solve(factors, rhs):
    levels, diag_final = factors
    d = np.array(rhs, dtype=float)

    for alpha, beta, h in levels:
        d = d + alpha * _shift_down(d, h, 0.0) + beta * _shift_up(d, h, 0.0)

    return d / diag_final


def thomas_solve(sub, diag, sup, rhs):
    n = len(diag)
    c = np.empty(n)
    d = np.empty(n)

    pivot = diag[0]
    if pivot == 0.0:
        raise ZeroDivisionError("zero pivot in tridiagonal solve")
    c[0] = sup[0] / pivot
    d[0] = rhs[0] / pivot

    for i in range(1, n):
        pivot = diag[i] - sub[i] * c[i - 1]
        if pivot == 0.0:
            raise ZeroDivisionError("zero pivot in tridiagonal solve")
        if i < n - 1:
            c[i] = sup[i] / pivot
        d[i] = (rhs[i] - sub[i] * d[i - 1]) / pivot

    x = np.empty(n)
    x[-1] = d[-1]
    for i in range(n - 2, -1, -1):
        x[i] = d[i] - c[i] * x[i + 1]
    return x


def _tridiagonal_from_rates(r_left, r_right):
    sub = np.concatenate(
        [np.zeros_like(r_left[..., :1]), -r_left[..., 1:]], axis=-1)
    diag = 1.0 + r_left + r_right
    sup = np.concatenate(
        [-r_right[..., :-1], np.zeros_like(r_right[..., :1])], axis=-1)
    return sub, diag, sup


def _crank_nicolson_rates(grid, dt, bc):
    scale = dt / (2.0 * grid.dx ** 2)
    face = grid.face_k

    if isinstance(bc, Dirichlet):
        inv_rho_c = grid.inv_rho_c[1:-1]
        return scale * face[:-1] * inv_rho_c, scale * face[1:] * inv_rho_c
    if isinstance(bc, Neumann):
        padded = np.concatenate(([0.0], face, [0.0]))
        inv_rho_c = grid.inv_rho_c / grid.cell_weight
        return scale * padded[:-1] * inv_rho_c, scale * padded[1:] * inv_rho_c
    raise ValueError(
        f"crank_nicolson_step does not support {bc!r} yet")


def crank_nicolson_step(grid, dt, boundary="dirichlet"):
    u = grid.u
    bc = uniform_boundary(boundary, "crank_nicolson_step")
    if not isinstance(bc, (Dirichlet, Neumann)):
        raise ValueError(f"crank_nicolson_step does not support {bc!r} yet")

    key = ("cn1d", float(dt), bc)
    cached = grid._solver_cache.get(key)
    if cached is None:
        r_left, r_right = _crank_nicolson_rates(grid, dt, bc)
        factors = tridiagonal_factor(*_tridiagonal_from_rates(r_left, r_right))
        cached = grid._cache_solver(key, (factors, r_left, r_right))
    factors, r_left, r_right = cached

    if isinstance(bc, Dirichlet):
        rhs = (r_left * u[0:-2]
               + (1.0 - r_left - r_right) * u[1:-1]
               + r_right * u[2:])
        rhs[0] += r_left[0] * u[0]
        rhs[-1] += r_right[-1] * u[-1]
        u[1:-1] = tridiagonal_solve(factors, rhs)
    else:
        rhs = (1.0 - r_left - r_right) * u
        rhs[1:] += r_left[1:] * u[:-1]
        rhs[:-1] += r_right[:-1] * u[1:]
        u[:] = tridiagonal_solve(factors, rhs)

    return u


def run_crank_nicolson(grid, t_start, t_end, boundary="dirichlet",
                       dt=None, n_steps=None):
    n_steps, dt_used = _implicit_plan(t_start, t_end, dt, n_steps)
    return _march(grid, crank_nicolson_step, n_steps, dt_used, boundary)


def max_stable_dt_2d(grid, safety=0.9, boundary="dirichlet"):
    normalize_boundary(boundary)
    return safety / (2.0 * _max_diffusivity(grid)
                     * (1.0 / grid.dx ** 2 + 1.0 / grid.dy ** 2))


def _axis_divergence(face_k, u, spacing, axis):
    gradient = np.diff(u, axis=axis) / spacing
    flux = face_k * gradient
    return np.diff(flux, axis=axis) / spacing


def explicit_step_2d(grid, dt, boundary="dirichlet"):
    u = grid.u
    dx, dy = grid.dx, grid.dy
    bc = uniform_boundary(boundary, "explicit_step_2d")

    if isinstance(bc, Dirichlet):
        div_x = _axis_divergence(grid.face_k_x, u, dx, axis=0)
        div_y = _axis_divergence(grid.face_k_y, u, dy, axis=1)
        u[1:-1, 1:-1] += (dt * grid.inv_rho_c[1:-1, 1:-1]
                          * (div_x[:, 1:-1] + div_y[1:-1, :]))
    elif isinstance(bc, Neumann):
        flux_x = grid.face_k_x * np.diff(u, axis=0) / dx
        flux_y = grid.face_k_y * np.diff(u, axis=1) / dy
        div = (np.diff(np.pad(flux_x, ((1, 1), (0, 0))), axis=0)
               / (dx * grid.cell_weight_x)
               + np.diff(np.pad(flux_y, ((0, 0), (1, 1))), axis=1)
               / (dy * grid.cell_weight_y))
        u += dt * grid.inv_rho_c * div
    else:
        raise ValueError(f"explicit_step_2d does not support {bc!r} yet")

    return u


def run_explicit_2d(grid, t_start, t_end, safety=_UNSET,
                    boundary="dirichlet", dt=None):
    n_steps, dt_used = _explicit_plan(
        grid, t_start, t_end, dt, safety, max_stable_dt_2d, boundary)
    return _march(grid, explicit_step_2d, n_steps, dt_used, boundary)


def _adi_rates(face_k_along, inv_rho_c_interior, dt_half, spacing):
    scale = dt_half / spacing ** 2
    return (scale * face_k_along[..., :-1] * inv_rho_c_interior,
            scale * face_k_along[..., 1:] * inv_rho_c_interior)


def crank_nicolson_step_2d(grid, dt, boundary="dirichlet"):
    bc = uniform_boundary(boundary, "crank_nicolson_step_2d")
    if not isinstance(bc, Dirichlet):
        raise ValueError(
            f"crank_nicolson_step_2d only supports Dirichlet boundaries "
            f"(got {bc!r})")

    u = grid.u
    dx, dy = grid.dx, grid.dy
    dt_half = dt / 2.0

    key = ("cn2d", float(dt), bc)
    cached = grid._solver_cache.get(key)
    if cached is None:
        inv_rho_c_int = grid.inv_rho_c[1:-1, 1:-1]
        rate_x = _adi_rates(
            np.moveaxis(grid.face_k_x, 0, -1)[1:-1, :],
            np.moveaxis(inv_rho_c_int, 0, -1), dt_half, dx)
        rate_y = _adi_rates(
            grid.face_k_y[1:-1, :], inv_rho_c_int, dt_half, dy)
        cached = grid._cache_solver(key, (
            tridiagonal_factor(*_tridiagonal_from_rates(*rate_x)), rate_x,
            tridiagonal_factor(*_tridiagonal_from_rates(*rate_y)), rate_y))
    factors_x, (rlx, rrx), factors_y, (rly, rry) = cached

    ly_un = _axis_divergence(grid.face_k_y, u, dy, axis=1)
    u_row = np.moveaxis(u, 0, -1)
    ly_row = np.moveaxis(ly_un, 0, -1)

    rhs_x = (u_row[1:-1, 1:-1]
             + dt_half * np.moveaxis(grid.inv_rho_c, 0, -1)[1:-1, 1:-1]
             * ly_row[:, 1:-1])
    rhs_x[:, 0] += rlx[:, 0] * u_row[1:-1, 0]
    rhs_x[:, -1] += rrx[:, -1] * u_row[1:-1, -1]

    u_star = u.copy()
    u_star[1:-1, 1:-1] = np.moveaxis(tridiagonal_solve(factors_x, rhs_x), -1, 0)

    lx_star = _axis_divergence(grid.face_k_x, u_star, dx, axis=0)

    rhs_y = (u_star[1:-1, 1:-1]
             + dt_half * grid.inv_rho_c[1:-1, 1:-1] * lx_star[:, 1:-1])
    rhs_y[:, 0] += rly[:, 0] * u_star[1:-1, 0]
    rhs_y[:, -1] += rry[:, -1] * u_star[1:-1, -1]

    u[1:-1, 1:-1] = tridiagonal_solve(factors_y, rhs_y)
    return u


def run_crank_nicolson_2d(grid, t_start, t_end, boundary="dirichlet",
                          dt=None, n_steps=None):
    n_steps, dt_used = _implicit_plan(t_start, t_end, dt, n_steps)
    return _march(grid, crank_nicolson_step_2d, n_steps, dt_used, boundary)