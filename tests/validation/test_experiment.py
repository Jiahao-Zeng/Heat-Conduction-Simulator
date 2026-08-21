import numpy as np
import pytest

from heatsim.boundary import Convective, Dirichlet, Neumann
from heatsim.experiment import (ExperimentConfig, build_experiment_grid,
                                resolve_timestep, run_experiment, sweep)
from heatsim.materials import MATERIALS
from heatsim.analytical import semi_infinite_convective

BASE = dict(material="copper", length=0.5, n_points=101, t_end=200.0,
            initial_temperature=20.0, left_value=100.0)


def _config(**overrides):
    merged = dict(BASE)
    merged.update(overrides)
    return ExperimentConfig(**merged)


@pytest.mark.parametrize("overrides", [
    dict(material="unobtainium"),
    dict(solver="magic"),
    dict(n_points=2),
    dict(length=0.0),
    dict(length=float("nan")),
    dict(t_end=0.0),
    dict(t_end=float("inf")),
    dict(n_snapshots=1),
    dict(dt=1e-3, n_steps=10),
    dict(dt=-1.0),
    dict(n_steps=0),
    dict(boundary="bogus"),
])
def test_invalid_configurations_are_rejected(overrides):
    with pytest.raises(ValueError):
        _config(**overrides)


def test_every_registered_material_is_accepted():
    for name in MATERIALS:
        _config(material=name)


def test_snapshots_span_the_full_interval():
    result = run_experiment(_config(n_snapshots=6))

    assert result.times[0] == pytest.approx(0.0)
    assert result.times[-1] == pytest.approx(result.config.t_end)
    assert np.all(np.diff(result.times) > 0.0)
    assert result.snapshots.shape == (len(result.times), BASE["n_points"])


def test_snapshot_cadence_does_not_change_the_physics():
    sparse = run_experiment(_config(n_snapshots=2))
    dense = run_experiment(_config(n_snapshots=50))

    assert sparse.dt == dense.dt
    assert sparse.n_steps == dense.n_steps
    assert np.array_equal(sparse.final, dense.final)


@pytest.mark.parametrize("n_snapshots", [4, 5, 6, 7, 9, 13])
def test_snapshots_land_on_the_nearest_step_to_the_requested_times(n_snapshots):
    config = _config(n_snapshots=n_snapshots)
    result = run_experiment(config)
    requested = np.linspace(0.0, config.t_end, n_snapshots)

    assert len(result.times) == n_snapshots
    assert np.abs(result.times - requested).max() <= result.dt / 2.0 + 1e-12


def test_first_snapshot_is_the_initial_condition():
    config = _config(n_snapshots=4)
    result = run_experiment(config)
    expected = build_experiment_grid(config).u

    assert np.array_equal(result.initial, expected)


def test_more_snapshots_than_steps_is_handled():
    result = run_experiment(_config(t_end=1.0, n_snapshots=500, dt=0.25))

    assert result.n_steps == 4
    assert len(result.times) == len(np.unique(result.times))
    assert result.times[-1] == pytest.approx(1.0)


def test_recorded_scalars_match_the_snapshots():
    result = run_experiment(_config(n_snapshots=5))

    assert result.min_temperature == pytest.approx(result.snapshots.min(axis=1))
    assert result.max_temperature == pytest.approx(result.snapshots.max(axis=1))


def test_mean_temperature_is_volume_weighted():
    config = _config(n_snapshots=2)
    grid = build_experiment_grid(config)
    weight = grid.cell_weight
    expected = float(np.sum(weight * grid.u) / weight.sum())

    result = run_experiment(config)

    assert result.mean_temperature[0] == pytest.approx(expected)
    assert result.mean_temperature[0] != pytest.approx(float(grid.u.mean()))


def test_mean_temperature_is_second_order_in_space():
    values = []
    for n_points in (101, 201, 401, 801):
        values.append(
            run_experiment(_config(n_points=n_points,
                                   n_snapshots=2)).mean_temperature[-1])
    diffs = [abs(values[i + 1] - values[i]) for i in range(len(values) - 1)]
    orders = [np.log2(diffs[i] / diffs[i + 1]) for i in range(len(diffs) - 1)]

    assert all(o > 1.85 for o in orders), orders


def test_insulated_experiment_conserves_heat_across_snapshots():
    result = run_experiment(
        _config(boundary="neumann", left_value=None, t_end=50.0,
                n_snapshots=8))

    assert result.total_heat == pytest.approx(result.total_heat[0], rel=1e-12)


def test_runner_reproduces_the_analytical_convective_solution():
    material = MATERIALS["water"]
    alpha = material.k / material.rho_c
    t_i, t_inf, h = 20.0, 100.0, 15.0
    config = ExperimentConfig(
        material="water", length=0.05, n_points=401, t_end=200.0,
        initial_temperature=t_i, n_snapshots=3,
        boundary=(Convective(h=h, u_inf=t_inf), Dirichlet()))

    result = run_experiment(config)

    exact = semi_infinite_convective(result.x, result.times[-1], alpha, h,
                                     material.k, t_i, t_inf)
    assert np.abs(result.final - exact).max() < 1e-3


def test_both_solvers_agree():
    explicit = run_experiment(_config(solver="explicit", n_snapshots=2))
    implicit = run_experiment(_config(solver="crank_nicolson", n_snapshots=2))

    assert np.abs(explicit.final - implicit.final).max() < 1e-2


def test_conductive_materials_heat_the_rod_faster():
    results = {r.parameters["material"]: r
               for r in sweep(_config(n_snapshots=2),
                              material=["copper", "steel", "pine_wood"])}

    copper = results["copper"].mean_temperature[-1]
    steel = results["steel"].mean_temperature[-1]
    wood = results["pine_wood"].mean_temperature[-1]
    assert copper > steel > wood


def test_dirichlet_end_is_held_at_its_initial_value():
    result = run_experiment(_config(n_snapshots=5))

    assert np.all(result.snapshots[:, 0] == pytest.approx(BASE["left_value"]))


def test_explicit_uses_the_stability_limit_by_default():
    config = _config(n_snapshots=2)
    grid = build_experiment_grid(config)
    from heatsim.solvers import max_stable_dt

    assert resolve_timestep(config, grid) == pytest.approx(
        max_stable_dt(grid, safety=config.safety, boundary=config.boundary))


def test_crank_nicolson_uses_the_monotone_limit_by_default():
    config = _config(solver="crank_nicolson", n_snapshots=2)
    grid = build_experiment_grid(config)
    from heatsim.solvers import max_monotone_dt

    assert resolve_timestep(config, grid) == pytest.approx(
        max_monotone_dt(grid, boundary=config.boundary, safety=config.safety))


def test_explicit_variants_stay_within_the_stability_limit():
    for material in MATERIALS:
        result = run_experiment(_config(material=material, t_end=10.0,
                                        n_snapshots=2))
        assert np.all(np.isfinite(result.final))


def test_sweep_covers_the_cartesian_product():
    results = sweep(_config(n_snapshots=2),
                    material=["copper", "steel"], n_points=[51, 101])

    combinations = {(r.parameters["material"], r.parameters["n_points"])
                    for r in results}
    assert len(results) == 4
    assert combinations == {("copper", 51), ("copper", 101),
                            ("steel", 51), ("steel", 101)}


def test_sweep_tags_each_result_and_applies_the_override():
    results = sweep(_config(n_snapshots=2), n_points=[51, 101])

    for result in results:
        assert result.config.n_points == result.parameters["n_points"]
        assert result.x.size == result.parameters["n_points"]


def test_sweep_applies_every_swept_parameter_to_the_config():
    results = sweep(_config(n_snapshots=2),
                    material=["steel", "pine_wood"],
                    n_points=[51, 101],
                    t_end=[25.0, 50.0])

    assert len(results) == 8
    for result in results:
        assert result.config.material == result.parameters["material"]
        assert result.config.n_points == result.parameters["n_points"]
        assert result.config.t_end == result.parameters["t_end"]
        assert result.x.size == result.parameters["n_points"]
        assert result.times[-1] == pytest.approx(result.parameters["t_end"])

    by_material = {}
    for result in results:
        by_material.setdefault(result.parameters["material"], []).append(
            result.mean_temperature[-1])
    assert by_material["steel"] != by_material["pine_wood"]


def test_sweep_leaves_the_base_config_untouched():
    base = _config(n_snapshots=2)
    sweep(base, material=["steel", "pine_wood"])

    assert base.material == "copper"


def test_sweep_rejects_unknown_fields():
    with pytest.raises(ValueError, match="unknown config field"):
        sweep(_config(n_snapshots=2), thickness=[1.0])


def test_sweep_rejects_empty_parameter_lists():
    with pytest.raises(ValueError, match="no values"):
        sweep(_config(n_snapshots=2), n_points=[])


def test_sweep_requires_a_parameter():
    with pytest.raises(ValueError, match="at least one parameter"):
        sweep(_config(n_snapshots=2))


def test_sweep_over_boundary_conditions():
    boundaries = ["dirichlet", "neumann",
                  (Convective(h=5000.0, u_inf=0.0), Dirichlet())]
    results = sweep(_config(left_value=None, initial_temperature=100.0,
                            t_end=500.0, n_snapshots=2),
                    boundary=boundaries)

    assert len(results) == 3
    dirichlet, neumann, convective = results
    assert dirichlet.mean_temperature[-1] == pytest.approx(100.0)
    assert neumann.mean_temperature[-1] == pytest.approx(100.0)
    assert convective.mean_temperature[-1] < 90.0
    assert convective.total_heat[-1] < convective.total_heat[0]


def test_cold_dirichlet_end_cools_the_rod():
    hot = run_experiment(_config(left_value=None, initial_temperature=100.0,
                                 t_end=50.0, n_snapshots=2))
    cooled = run_experiment(_config(left_value=0.0, initial_temperature=100.0,
                                    t_end=50.0, n_snapshots=2))

    assert hot.mean_temperature[-1] == pytest.approx(100.0)
    assert cooled.mean_temperature[-1] < hot.mean_temperature[-1]


def test_config_is_frozen_and_hashable():
    config = _config()
    {config}
    with pytest.raises(Exception):
        config.material = "steel"


def test_explicit_dt_override_is_honoured():
    config = _config(dt=0.05, n_snapshots=2)
    result = run_experiment(config)

    assert result.n_steps == int(np.ceil(config.t_end / 0.05))
    assert result.dt == pytest.approx(config.t_end / result.n_steps)
    assert result.dt <= 0.05


def test_n_steps_override_is_honoured():
    config = _config(solver="crank_nicolson", n_steps=500, n_snapshots=2)
    result = run_experiment(config)

    assert result.n_steps == 500
    assert result.dt == pytest.approx(config.t_end / 500)


def test_both_endpoint_values_seed_the_initial_condition():
    config = _config(left_value=100.0, right_value=5.0,
                     initial_temperature=20.0)
    grid = build_experiment_grid(config)

    assert grid.u[0] == pytest.approx(100.0)
    assert grid.u[-1] == pytest.approx(5.0)
    assert np.all(grid.u[1:-1] == pytest.approx(20.0))


def test_a_convective_boundary_tightens_the_explicit_timestep():
    plain = _config(boundary="dirichlet")
    convective = _config(
        boundary=(Convective(h=1.0e5, u_inf=100.0), Dirichlet()))

    dt_plain = resolve_timestep(plain, build_experiment_grid(plain))
    dt_conv = resolve_timestep(convective, build_experiment_grid(convective))

    assert dt_conv < dt_plain


def test_results_carry_the_config_that_produced_them():
    results = sweep(_config(n_snapshots=2), n_points=[51, 101])

    for result in results:
        assert result.config.n_points == result.parameters["n_points"]
        assert result.x.shape == (result.config.n_points,)
        assert result.snapshots.shape[1] == result.config.n_points


def test_resolved_timestep_is_never_exceeded():
    config = _config(n_points=11, t_end=1.0, dt=0.4, n_snapshots=2)
    grid = build_experiment_grid(config)
    limit = resolve_timestep(config, grid)

    result = run_experiment(config)

    assert result.dt <= limit + 1e-12
    assert result.n_steps * result.dt == pytest.approx(config.t_end)


def test_crank_nicolson_default_differs_from_the_stability_limit():
    from heatsim.solvers import max_monotone_dt, max_stable_dt

    config = _config(solver="crank_nicolson",
                     boundary=(Convective(h=1.0e4, u_inf=100.0), Dirichlet()))
    grid = build_experiment_grid(config)

    monotone = max_monotone_dt(grid, boundary=config.boundary,
                               safety=config.safety)
    stable = max_stable_dt(grid, safety=config.safety,
                           boundary=config.boundary)

    assert monotone != pytest.approx(stable)
    assert resolve_timestep(config, grid) == pytest.approx(monotone)