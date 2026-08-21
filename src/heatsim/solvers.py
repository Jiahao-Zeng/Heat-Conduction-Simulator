import numpy as np

from heatsim.grid import harmonic_face_mean, face_diffusivity
from heatsim.boundary import (Dirichlet, Neumann, Convective,
                              normalize_boundary, normalize_boundary_2d)

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


def max_monotone_dt(grid, boundary="dirichlet", safety=0.9):
    left, right = normalize_boundary(boundary)
    face_k = grid.face_k
    dx2 = grid.dx ** 2

    interior = np.min(grid.rho_c[1:-1] / (face_k[:-1] + face_k[1:]))
    dt = interior * dx2

    sides = ((left, grid.rho_c[0], face_k[0]),
             (right, grid.rho_c[-1], face_k[-1]))
    for side, rho_c_b, k_face_b in sides:
        if isinstance(side, Dirichlet):
            continue
        h = side.h if isinstance(side, Convective) else 0.0
        dt = min(dt, rho_c_b * dx2 / (2.0 * (k_face_b + h * grid.dx)))

    return safety * dt


def _crank_nicolson_general_rates(grid, dt, left, right):
    for side in (left, right):
        if not isinstance(side, (Dirichlet, Neumann, Convective)):
            raise TypeError(f"unsupported boundary condition: {side!r}")

    scale = dt / (2.0 * grid.dx ** 2)
    padded = np.concatenate(([0.0], grid.face_k, [0.0]))
    inv_rho_c = grid.inv_rho_c / grid.cell_weight

    r_left = scale * padded[:-1] * inv_rho_c
    r_right = scale * padded[1:] * inv_rho_c
    forcing = np.zeros_like(r_left)

    if isinstance(left, Convective):
        r_left[0] = scale * left.h * grid.dx * inv_rho_c[0]
        forcing[0] = 2.0 * r_left[0] * left.u_inf
    elif isinstance(left, Dirichlet):
        r_left[0] = 0.0
        r_right[0] = 0.0

    if isinstance(right, Convective):
        r_right[-1] = scale * right.h * grid.dx * inv_rho_c[-1]
        forcing[-1] = 2.0 * r_right[-1] * right.u_inf
    elif isinstance(right, Dirichlet):
        r_left[-1] = 0.0
        r_right[-1] = 0.0

    return r_left, r_right, forcing


def _crank_nicolson_general_step(grid, dt, sides):
    u = grid.u
    key = ("cn1d-general", float(dt), sides)
    cached = grid._solver_cache.get(key)
    if cached is None:
        r_left, r_right, forcing = _crank_nicolson_general_rates(
            grid, dt, *sides)
        factors = tridiagonal_factor(*_tridiagonal_from_rates(r_left, r_right))
        cached = grid._cache_solver(key, (factors, r_left, r_right, forcing))
    factors, r_left, r_right, forcing = cached

    rhs = (1.0 - r_left - r_right) * u + forcing
    rhs[1:] += r_left[1:] * u[:-1]
    rhs[:-1] += r_right[:-1] * u[1:]
    u[:] = tridiagonal_solve(factors, rhs)
    return u


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
    left, right = normalize_boundary(boundary)

    if left != right or isinstance(left, Convective):
        return _crank_nicolson_general_step(grid, dt, (left, right))

    bc = left
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


def _axis_stable_rate(bc_lo, bc_hi, spacing, alpha_max,
                       face_k_lo, face_k_hi, weight_lo, weight_hi,
                       rho_c_lo, rho_c_hi):
    rate = 2.0 * alpha_max / spacing ** 2
    if isinstance(bc_lo, Convective) and bc_lo.h > 0.0:
        rate = max(rate, (2.0 * face_k_lo + bc_lo.h * spacing)
                   / (rho_c_lo * weight_lo * spacing ** 2))
    if isinstance(bc_hi, Convective) and bc_hi.h > 0.0:
        rate = max(rate, (2.0 * face_k_hi + bc_hi.h * spacing)
                   / (rho_c_hi * weight_hi * spacing ** 2))
    return rate


def max_stable_dt_2d(grid, safety=0.9, boundary="dirichlet"):
    x_lo, x_hi, y_lo, y_hi = normalize_boundary_2d(boundary, "max_stable_dt_2d")
    alpha_max = _max_diffusivity(grid)

    rate_x = _axis_stable_rate(
        x_lo, x_hi, grid.dx, alpha_max,
        float(grid.face_k_x[0, :].max()), float(grid.face_k_x[-1, :].max()),
        float(grid.cell_weight_x[0, 0]), float(grid.cell_weight_x[-1, 0]),
        float(grid.rho_c[0, :].min()), float(grid.rho_c[-1, :].min()))
    rate_y = _axis_stable_rate(
        y_lo, y_hi, grid.dy, alpha_max,
        float(grid.face_k_y[:, 0].max()), float(grid.face_k_y[:, -1].max()),
        float(grid.cell_weight_y[0, 0]), float(grid.cell_weight_y[0, -1]),
        float(grid.rho_c[:, 0].min()), float(grid.rho_c[:, -1].min()))

    return safety / (rate_x + rate_y)


def _axis_divergence(face_k, u, spacing, axis):
    gradient = np.diff(u, axis=axis) / spacing
    flux = face_k * gradient
    return np.diff(flux, axis=axis) / spacing


def explicit_step_2d(grid, dt, boundary="dirichlet"):
    u = grid.u
    dx, dy = grid.dx, grid.dy
    x_lo, x_hi, y_lo, y_hi = normalize_boundary_2d(boundary, "explicit_step_2d")
    for bc in (x_lo, x_hi, y_lo, y_hi):
        if not isinstance(bc, (Dirichlet, Neumann, Convective)):
            raise ValueError(f"explicit_step_2d does not support {bc!r} yet")

    flux_x = grid.face_k_x * np.diff(u, axis=0) / dx
    div_x = np.empty_like(u)
    div_x[1:-1, :] = np.diff(flux_x, axis=0)
    div_x[0, :] = flux_x[0, :] + _explicit_boundary_term(x_lo, u[0, :])
    div_x[-1, :] = -flux_x[-1, :] + _explicit_boundary_term(x_hi, u[-1, :])

    flux_y = grid.face_k_y * np.diff(u, axis=1) / dy
    div_y = np.empty_like(u)
    div_y[:, 1:-1] = np.diff(flux_y, axis=1)
    div_y[:, 0] = flux_y[:, 0] + _explicit_boundary_term(y_lo, u[:, 0])
    div_y[:, -1] = -flux_y[:, -1] + _explicit_boundary_term(y_hi, u[:, -1])

    delta = dt * grid.inv_rho_c * (div_x / (dx * grid.cell_weight_x)
                                   + div_y / (dy * grid.cell_weight_y))

    u[1:-1, 1:-1] += delta[1:-1, 1:-1]
    if not isinstance(x_lo, Dirichlet):
        u[0, 1:-1] += delta[0, 1:-1]
    if not isinstance(x_hi, Dirichlet):
        u[-1, 1:-1] += delta[-1, 1:-1]
    if not isinstance(y_lo, Dirichlet):
        u[1:-1, 0] += delta[1:-1, 0]
    if not isinstance(y_hi, Dirichlet):
        u[1:-1, -1] += delta[1:-1, -1]
    if not isinstance(x_lo, Dirichlet) and not isinstance(y_lo, Dirichlet):
        u[0, 0] += delta[0, 0]
    if not isinstance(x_lo, Dirichlet) and not isinstance(y_hi, Dirichlet):
        u[0, -1] += delta[0, -1]
    if not isinstance(x_hi, Dirichlet) and not isinstance(y_lo, Dirichlet):
        u[-1, 0] += delta[-1, 0]
    if not isinstance(x_hi, Dirichlet) and not isinstance(y_hi, Dirichlet):
        u[-1, -1] += delta[-1, -1]

    return u


def run_explicit_2d(grid, t_start, t_end, safety=_UNSET,
                    boundary="dirichlet", dt=None):
    n_steps, dt_used = _explicit_plan(
        grid, t_start, t_end, dt, safety, max_stable_dt_2d, boundary)
    return _march(grid, explicit_step_2d, n_steps, dt_used, boundary)


def _cn2d_axis_rates(grid, dt_half, axis, bc_lo, bc_hi):
    if axis == 0:
        spacing, face, weight = grid.dx, grid.face_k_x, grid.cell_weight_x
        pad = np.zeros((1, grid.ny))
        padded = np.concatenate([pad, face, pad], axis=0)
    else:
        spacing, face, weight = grid.dy, grid.face_k_y, grid.cell_weight_y
        pad = np.zeros((grid.nx, 1))
        padded = np.concatenate([pad, face, pad], axis=1)

    scale = dt_half / spacing ** 2
    inv = grid.inv_rho_c / weight
    lo_slice = (slice(0, -1), slice(None)) if axis == 0 else (slice(None), slice(0, -1))
    hi_slice = (slice(1, None), slice(None)) if axis == 0 else (slice(None), slice(1, None))
    r_lo = scale * padded[lo_slice] * inv
    r_hi = scale * padded[hi_slice] * inv
    forcing = np.zeros_like(r_lo)

    edge_lo = (0, slice(None)) if axis == 0 else (slice(None), 0)
    edge_hi = (-1, slice(None)) if axis == 0 else (slice(None), -1)

    if isinstance(bc_lo, Convective):
        r_lo[edge_lo] = scale * bc_lo.h * spacing * inv[edge_lo]
        forcing[edge_lo] = r_lo[edge_lo] * bc_lo.u_inf
    elif isinstance(bc_lo, Dirichlet):
        r_lo[edge_lo] = 0.0
        r_hi[edge_lo] = 0.0

    if isinstance(bc_hi, Convective):
        r_hi[edge_hi] = scale * bc_hi.h * spacing * inv[edge_hi]
        forcing[edge_hi] = r_hi[edge_hi] * bc_hi.u_inf
    elif isinstance(bc_hi, Dirichlet):
        r_lo[edge_hi] = 0.0
        r_hi[edge_hi] = 0.0

    if axis == 0:
        return r_lo, r_hi, forcing
    return r_lo, r_hi, forcing


def _apply_axis_rates(u_along, r_lo, r_hi, forcing):
    out = -(r_lo + r_hi) * u_along + forcing
    out[..., 1:] += r_lo[..., 1:] * u_along[..., :-1]
    out[..., :-1] += r_hi[..., :-1] * u_along[..., 1:]
    return out


def _cn2d_fixed_mask(grid, x_lo, x_hi, y_lo, y_hi):
    mask = np.zeros((grid.nx, grid.ny), dtype=bool)
    if isinstance(x_lo, Dirichlet):
        mask[0, :] = True
    if isinstance(x_hi, Dirichlet):
        mask[-1, :] = True
    if isinstance(y_lo, Dirichlet):
        mask[:, 0] = True
    if isinstance(y_hi, Dirichlet):
        mask[:, -1] = True
    return mask


def crank_nicolson_step_2d(grid, dt, boundary="dirichlet"):
    sides = normalize_boundary_2d(boundary, "crank_nicolson_step_2d")
    x_lo, x_hi, y_lo, y_hi = sides
    for bc in sides:
        if not isinstance(bc, (Dirichlet, Neumann, Convective)):
            raise ValueError(f"crank_nicolson_step_2d does not support {bc!r} yet")

    u = grid.u
    dt_half = dt / 2.0

    key = ("cn2d", float(dt), sides)
    cached = grid._solver_cache.get(key)
    if cached is None:
        rx = _cn2d_axis_rates(grid, dt_half, 0, x_lo, x_hi)
        ry = _cn2d_axis_rates(grid, dt_half, 1, y_lo, y_hi)
        rx_moved = tuple(np.moveaxis(a, 0, -1) for a in rx)
        cached = grid._cache_solver(key, (
            tridiagonal_factor(*_tridiagonal_from_rates(
                rx_moved[0], rx_moved[1])), rx, rx_moved,
            tridiagonal_factor(*_tridiagonal_from_rates(ry[0], ry[1])), ry,
            _cn2d_fixed_mask(grid, x_lo, x_hi, y_lo, y_hi)))
    factors_x, (rlx, rrx, fx), rx_moved, factors_y, (rly, rry, fy), fixed = cached

    original = u.copy()

    y_explicit = _apply_axis_rates(u, rly, rry, fy)
    rhs_x = u + y_explicit + fx
    rhs_x[fixed] = original[fixed]
    u_star = np.moveaxis(
        tridiagonal_solve(factors_x, np.moveaxis(rhs_x, 0, -1)), -1, 0)
    u_star[fixed] = original[fixed]

    x_explicit = np.moveaxis(
        _apply_axis_rates(np.moveaxis(u_star, 0, -1), *rx_moved), -1, 0)
    rhs_y = u_star + x_explicit + fy
    rhs_y[fixed] = original[fixed]
    u[:] = tridiagonal_solve(factors_y, rhs_y)
    u[fixed] = original[fixed]

    return u


def run_crank_nicolson_2d(grid, t_start, t_end, boundary="dirichlet",
                          dt=None, n_steps=None):
    n_steps, dt_used = _implicit_plan(t_start, t_end, dt, n_steps)
    return _march(grid, crank_nicolson_step_2d, n_steps, dt_used, boundary)