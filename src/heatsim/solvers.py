import numpy as np

from heatsim.grid import face_diffusivity  # re-exported; see grid.py


def max_stable_dt(grid, safety=0.9):
    alpha_max = float(np.max(grid.alpha))
    if alpha_max <= 0.0:
        raise ValueError("cannot derive a timestep for zero diffusivity")
    return safety * grid.dx ** 2 / (2.0 * alpha_max)


def _step_count(t_start, t_end, dt):
    if t_end < t_start:
        raise ValueError("t_end must not precede t_start")
    if dt <= 0.0:
        raise ValueError("dt must be positive")
    return max(1, int(np.ceil((t_end - t_start) / dt)))


def explicit_step(grid, dt, boundary="dirichlet"):
    u = grid.u
    dx = grid.dx

    flux = grid.face_alpha * np.diff(u) / dx

    if boundary == "neumann":
        divergence = np.empty_like(u)
        divergence[0] = flux[0]
        divergence[1:-1] = np.diff(flux)
        divergence[-1] = -flux[-1]
        u += (dt / dx) * divergence
    elif boundary == "dirichlet":
        u[1:-1] += (dt / dx) * np.diff(flux)
    else:
        raise ValueError(f"unknown boundary condition: {boundary!r}")

    return u


def run_explicit(grid, t_start, t_end, safety=0.9, boundary="dirichlet", dt=None):
    if dt is not None and safety != 0.9:
        raise ValueError("pass either dt or safety, not both")
    if dt is None:
        dt = max_stable_dt(grid, safety=safety)

    n_steps = _step_count(t_start, t_end, dt)
    dt_used = (t_end - t_start) / n_steps

    for _ in range(n_steps):
        explicit_step(grid, dt_used, boundary=boundary)

    return grid, dt_used, n_steps


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
        a_lo = _shift_down(a, h, 0.0)
        b_lo = _shift_down(b, h, 1.0)
        c_lo = _shift_down(c, h, 0.0)
        a_hi = _shift_up(a, h, 0.0)
        b_hi = _shift_up(b, h, 1.0)
        c_hi = _shift_up(c, h, 0.0)

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
        d_lo = _shift_down(d, h, 0.0)
        d_hi = _shift_up(d, h, 0.0)
        d = d + alpha * d_lo + beta * d_hi

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


def _crank_nicolson_matrix(grid, dt, boundary):
    dx = grid.dx
    face = grid.face_alpha  # length n - 1

    if boundary == "dirichlet":
        r_left = dt / (2.0 * dx ** 2) * face[:-1]   # face(i-1, i), i = 1..n-2
        r_right = dt / (2.0 * dx ** 2) * face[1:]   # face(i, i+1), i = 1..n-2
    elif boundary == "neumann":
        r = dt / (2.0 * dx ** 2) * np.concatenate(([0.0], face, [0.0]))
        r_left, r_right = r[:-1], r[1:]
    else:
        raise ValueError(f"unknown boundary condition: {boundary!r}")

    sub = np.concatenate(([0.0], -r_left[1:]))
    diag = 1.0 + r_left + r_right
    sup = np.concatenate((-r_right[:-1], [0.0]))
    return sub, diag, sup, r_left, r_right


def crank_nicolson_step(grid, dt, boundary="dirichlet"):
    u = grid.u

    key = ("cn", float(dt), boundary)
    cached = grid._solver_cache.get(key)
    if cached is None:
        sub, diag, sup, r_left, r_right = _crank_nicolson_matrix(grid, dt, boundary)
        cached = (tridiagonal_factor(sub, diag, sup), r_left, r_right)
        grid._solver_cache[key] = cached
    factors, r_left, r_right = cached

    if boundary == "dirichlet":
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


def run_crank_nicolson(grid, t_start, t_end, boundary="dirichlet", dt=None, n_steps=None):

    if (dt is None) == (n_steps is None):
        raise ValueError("provide exactly one of dt or n_steps")

    if n_steps is None:
        n_steps = _step_count(t_start, t_end, dt)
    elif n_steps < 1:
        raise ValueError("n_steps must be at least 1")

    dt_used = (t_end - t_start) / n_steps

    for _ in range(n_steps):
        crank_nicolson_step(grid, dt_used, boundary=boundary)

    return grid, dt_used, n_steps