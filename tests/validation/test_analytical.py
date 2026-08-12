import numpy as np
import pytest

from heatsim.analytical import (
    gaussian_point_source,
    gaussian_point_source_2d,
    gaussian_point_source_2d_anisotropic,
)

ALPHA_X, ALPHA_Y = 0.023, 0.0071
X0, Y0, T, TOTAL_HEAT = 0.31, -0.42, 0.037, 1.7


def test_anisotropic_source_integrates_to_the_total_heat():
    n, half_width = 3001, 3.0
    xs = np.linspace(X0 - half_width, X0 + half_width, n)
    ys = np.linspace(Y0 - half_width, Y0 + half_width, n)
    X, Y = np.meshgrid(xs, ys, indexing="ij")

    u = gaussian_point_source_2d_anisotropic(
        X, Y, T, ALPHA_X, ALPHA_Y, X0, Y0, TOTAL_HEAT)
    mass = np.trapezoid(np.trapezoid(u, ys, axis=1), xs)

    assert mass == pytest.approx(TOTAL_HEAT, rel=1e-9)


def test_anisotropic_source_reduces_to_the_isotropic_solution():
    alpha = 0.019
    rng = np.random.default_rng(0)
    X = rng.uniform(-2.0, 2.0, 500)
    Y = rng.uniform(-2.0, 2.0, 500)

    anisotropic = gaussian_point_source_2d_anisotropic(
        X, Y, T, alpha, alpha, X0, Y0, TOTAL_HEAT)
    isotropic = gaussian_point_source_2d(X, Y, T, alpha, X0, Y0, TOTAL_HEAT)

    assert np.allclose(anisotropic, isotropic,
                       rtol=1e-12, atol=1e-12 * np.abs(isotropic).max())


def test_anisotropic_source_is_the_product_of_two_1d_solutions():
    rng = np.random.default_rng(1)
    X = rng.uniform(-2.0, 2.0, 500)
    Y = rng.uniform(-2.0, 2.0, 500)

    combined = gaussian_point_source_2d_anisotropic(
        X, Y, T, ALPHA_X, ALPHA_Y, X0, Y0, TOTAL_HEAT)
    separated = (TOTAL_HEAT
                 * gaussian_point_source(X, T, ALPHA_X, X0)
                 * gaussian_point_source(Y, T, ALPHA_Y, Y0))

    assert np.allclose(combined, separated,
                       rtol=1e-12, atol=1e-12 * np.abs(separated).max())


def test_anisotropic_source_satisfies_the_anisotropic_heat_equation():
    rng = np.random.default_rng(2)
    X = rng.uniform(-1.0, 1.0, 400)
    Y = rng.uniform(-1.0, 1.0, 400)
    h, dt = 1e-4, 1e-6

    def u(x, y, t):
        return gaussian_point_source_2d_anisotropic(
            x, y, t, ALPHA_X, ALPHA_Y, X0, Y0, TOTAL_HEAT)

    u_t = (u(X, Y, T + dt) - u(X, Y, T - dt)) / (2.0 * dt)
    u_xx = (u(X + h, Y, T) - 2.0 * u(X, Y, T) + u(X - h, Y, T)) / h ** 2
    u_yy = (u(X, Y + h, T) - 2.0 * u(X, Y, T) + u(X, Y - h, T)) / h ** 2

    residual = np.abs(u_t - ALPHA_X * u_xx - ALPHA_Y * u_yy).max()
    assert residual / np.abs(u_t).max() < 1e-4


@pytest.mark.parametrize("bad_t", [0.0, -1.0])
def test_anisotropic_source_rejects_non_positive_time(bad_t):
    with pytest.raises(ValueError):
        gaussian_point_source_2d_anisotropic(
            0.0, 0.0, bad_t, ALPHA_X, ALPHA_Y, X0, Y0)