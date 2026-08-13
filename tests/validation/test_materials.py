import numpy as np
import pytest

from heatsim.grid import Grid1D, Grid2D
from heatsim.materials import Material, MATERIALS, build_grid_1d, build_grid_2d


ANISOTROPIC_NAMES = [name for name, m in MATERIALS.items() if m.is_anisotropic]
ISOTROPIC_NAMES = [name for name, m in MATERIALS.items() if not m.is_anisotropic]


def test_registry_keys_match_material_names():
    for key, material in MATERIALS.items():
        assert key == material.name


def test_registry_contains_at_least_two_anisotropic_materials():
    assert len(ANISOTROPIC_NAMES) >= 2


def test_pine_wood_is_anisotropic_with_along_grain_faster_than_across():
    pine = MATERIALS["pine_wood"]
    assert pine.is_anisotropic
    assert pine.k > pine.k_y
    assert 1.0 < pine.k / pine.k_y < 3.0


def test_graphite_is_far_more_anisotropic_than_wood():
    graphite = MATERIALS["pyrolytic_graphite"]
    pine = MATERIALS["pine_wood"]
    assert graphite.is_anisotropic
    assert graphite.k > graphite.k_y
    assert 100.0 < graphite.k / graphite.k_y < 1000.0
    assert (graphite.k / graphite.k_y) > (pine.k / pine.k_y)


@pytest.mark.parametrize("name", ISOTROPIC_NAMES)
def test_isotropic_materials_have_no_k_y(name):
    assert not MATERIALS[name].is_anisotropic
    assert MATERIALS[name].k_y is None


@pytest.mark.parametrize("name", ANISOTROPIC_NAMES)
def test_anisotropic_materials_have_distinct_axis_conductivities(name):
    material = MATERIALS[name]
    assert material.is_anisotropic
    assert material.k != material.k_y


@pytest.mark.parametrize("material", MATERIALS.values(), ids=list(MATERIALS))
def test_every_registered_material_has_positive_properties(material):
    assert material.k > 0.0
    assert material.rho_c > 0.0
    if material.k_y is not None:
        assert material.k_y > 0.0


def test_metals_conduct_better_than_insulators_and_gases():
    for metal in ("copper", "aluminum", "gold", "silver", "steel"):
        for insulator in ("glass", "concrete", "pine_wood", "air"):
            assert MATERIALS[metal].k > MATERIALS[insulator].k


def test_air_is_the_worst_conductor_in_the_registry():
    air_k = MATERIALS["air"].k
    assert all(air_k <= m.k for m in MATERIALS.values())


def test_build_grid_2d_from_isotropic_material_matches_manual_construction():
    material = MATERIALS["copper"]
    built = build_grid_2d(material, 1.0, 1.0, 11, 11, initial_temperature=5.0)
    manual = Grid2D(1.0, 1.0, 11, 11, initial_temperature=5.0,
                    k=material.k, rho_c=material.rho_c)

    assert np.array_equal(built.k, manual.k)
    assert np.array_equal(built.rho_c, manual.rho_c)
    assert not built.is_anisotropic


@pytest.mark.parametrize("name", ANISOTROPIC_NAMES)
def test_build_grid_2d_from_anisotropic_material_is_actually_anisotropic(name):
    material = MATERIALS[name]
    grid = build_grid_2d(material, 1.0, 1.0, 11, 11)

    assert grid.is_anisotropic
    assert np.allclose(grid.k, material.k)
    assert np.allclose(grid.k_y, material.k_y)
    assert np.allclose(grid.face_k_x, material.k)
    assert np.allclose(grid.face_k_y, material.k_y)


def test_build_grid_1d_uses_the_primary_axis_conductivity():
    pine = MATERIALS["pine_wood"]
    grid = build_grid_1d(pine, 1.0, 11)

    assert np.allclose(grid.k, pine.k)
    assert np.allclose(grid.rho_c, pine.rho_c)


def test_pine_wood_heat_spreads_faster_along_the_grain_than_across_it():
    from heatsim.solvers import run_explicit_2d

    pine = MATERIALS["pine_wood"]
    n = 81
    grid = build_grid_2d(pine, 1.0, 1.0, n, n)
    cx = cy = n // 2
    grid.u[cx, cy] = 1.0 / (grid.dx * grid.dy)

    run_explicit_2d(grid, 0.0, 5.0, safety=0.9)

    marginal_x = grid.u.sum(axis=1) * grid.dy
    marginal_y = grid.u.sum(axis=0) * grid.dx
    spread_x = np.sum(marginal_x * (grid.x - grid.x[cx]) ** 2) * grid.dx
    spread_y = np.sum(marginal_y * (grid.y - grid.y[cy]) ** 2) * grid.dy

    assert spread_x > spread_y


def test_material_rejects_non_positive_properties():
    with pytest.raises(ValueError):
        Material("bad", k=0.0, rho_c=1.0)
    with pytest.raises(ValueError):
        Material("bad", k=1.0, rho_c=-1.0)
    with pytest.raises(ValueError):
        Material("bad", k=1.0, rho_c=1.0, k_y=-1.0)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_material_rejects_non_finite_properties(bad):
    with pytest.raises(ValueError):
        Material("bad", k=bad, rho_c=1.0)
    with pytest.raises(ValueError):
        Material("bad", k=1.0, rho_c=bad)
    with pytest.raises(ValueError):
        Material("bad", k=1.0, rho_c=1.0, k_y=bad)


def test_material_rejects_empty_name():
    with pytest.raises(ValueError):
        Material("", k=1.0, rho_c=1.0)


def test_k_y_equal_to_k_is_treated_as_isotropic():
    material = Material("degenerate", k=5.0, rho_c=1.0, k_y=5.0)
    assert not material.is_anisotropic
    assert "k_y" not in material.kwargs_2d()

    grid = build_grid_2d(material, 1.0, 1.0, 11, 11)
    assert not grid.is_anisotropic
    assert grid.k_y is grid.k