"""
The tridiagonal linear solver underneath Crank-Nicolson.

The solvers use parallel cyclic reduction (PCR) because it vectorizes
in NumPy, where the Thomas algorithm's sequential sweep does not. PCR
is the less obvious of the two, so these tests pin it against the
Thomas reference implementation and against a general dense solve.

Batching matters beyond 1D: a 2D ADI scheme solves one tridiagonal
system per grid line, and doing them as a single batched call rather
than a Python loop is what keeps 2D Crank-Nicolson fast enough to run
live in the Phase 8 GUI.
"""

import numpy as np
import pytest

from heatsim.solvers import (
    tridiagonal_factor,
    tridiagonal_solve,
    thomas_solve,
)


def _random_system(rng, shape):
    """A diagonally dominant tridiagonal system of the given shape.

    Diagonal dominance mirrors the Crank-Nicolson matrices the solvers
    actually build, and is the condition under which both algorithms
    are stable without pivoting.
    """
    diag = rng.uniform(4.0, 6.0, shape)
    sub = rng.uniform(-1.0, -0.5, shape)
    sup = rng.uniform(-1.0, -0.5, shape)
    sub[..., 0] = 0.0
    sup[..., -1] = 0.0
    rhs = rng.uniform(-1.0, 1.0, shape)
    return sub, diag, sup, rhs


def _dense_solve(sub, diag, sup, rhs):
    matrix = (np.diag(diag)
              + np.diag(sub[1:], -1)
              + np.diag(sup[:-1], 1))
    return np.linalg.solve(matrix, rhs)


# Sizes deliberately include powers of two and values just above and
# below them: PCR's reduction proceeds in strides of 1, 2, 4, ..., so
# a mishandled final partial level would show up at n = 2^k +/- 1.
@pytest.mark.parametrize("n", [3, 4, 5, 7, 8, 9, 15, 16, 17, 101, 401])
def test_matches_dense_solve(n):
    rng = np.random.default_rng(n)
    sub, diag, sup, rhs = _random_system(rng, (n,))

    result = tridiagonal_solve(tridiagonal_factor(sub, diag, sup), rhs)

    assert result == pytest.approx(_dense_solve(sub, diag, sup, rhs), abs=1e-10)


@pytest.mark.parametrize("n", [5, 17, 401, 1601])
def test_matches_thomas_reference(n):
    rng = np.random.default_rng(n + 1)
    sub, diag, sup, rhs = _random_system(rng, (n,))

    fast = tridiagonal_solve(tridiagonal_factor(sub, diag, sup), rhs)
    reference = thomas_solve(sub, diag, sup, rhs)

    assert fast == pytest.approx(reference, abs=1e-10)


def test_solves_a_batch_of_independent_systems():
    """Each leading-axis entry must be solved with its own matrix, not
    with a shared one -- in 2D every grid line has different material
    properties, so getting this wrong would be silently plausible."""
    rng = np.random.default_rng(11)
    n_systems, n = 8, 201
    sub, diag, sup, rhs = _random_system(rng, (n_systems, n))

    batched = tridiagonal_solve(tridiagonal_factor(sub, diag, sup), rhs)

    for k in range(n_systems):
        one = thomas_solve(sub[k], diag[k], sup[k], rhs[k])
        assert batched[k] == pytest.approx(one, abs=1e-10)


def test_one_matrix_broadcasts_over_many_right_hand_sides():
    """A single factorization reused across a batch of right-hand
    sides -- the uniform-material case, where every grid line shares
    the same matrix."""
    rng = np.random.default_rng(12)
    n = 201
    sub, diag, sup, _ = _random_system(rng, (n,))
    many_rhs = rng.uniform(-1.0, 1.0, (5, n))

    factors = tridiagonal_factor(sub, diag, sup)
    batched = tridiagonal_solve(factors, many_rhs)

    for k in range(5):
        one = thomas_solve(sub, diag, sup, many_rhs[k])
        assert batched[k] == pytest.approx(one, abs=1e-10)


def test_factorization_is_reusable_across_right_hand_sides():
    """Factoring is done once per run and reused every timestep, so a
    factorization must not be consumed or mutated by solving."""
    rng = np.random.default_rng(13)
    n = 101
    sub, diag, sup, rhs = _random_system(rng, (n,))
    factors = tridiagonal_factor(sub, diag, sup)

    first = tridiagonal_solve(factors, rhs)
    other_rhs = rng.uniform(-1.0, 1.0, n)
    tridiagonal_solve(factors, other_rhs)
    again = tridiagonal_solve(factors, rhs)

    assert again == pytest.approx(first, abs=1e-12)


def test_solving_does_not_modify_the_right_hand_side():
    rng = np.random.default_rng(14)
    n = 101
    sub, diag, sup, rhs = _random_system(rng, (n,))
    original = rhs.copy()

    tridiagonal_solve(tridiagonal_factor(sub, diag, sup), rhs)

    assert rhs == pytest.approx(original, abs=0.0)


def test_identity_system_returns_the_right_hand_side():
    n = 33
    sub = np.zeros(n)
    sup = np.zeros(n)
    diag = np.ones(n)
    rhs = np.arange(n, dtype=float)

    result = tridiagonal_solve(tridiagonal_factor(sub, diag, sup), rhs)

    assert result == pytest.approx(rhs, abs=1e-12)