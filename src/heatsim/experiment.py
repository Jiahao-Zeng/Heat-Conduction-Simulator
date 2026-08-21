import itertools
import math
from dataclasses import dataclass, field, replace
from typing import Optional, Tuple

import numpy as np

from .boundary import Convective, Dirichlet, Neumann, normalize_boundary
from .materials import MATERIALS, build_grid_1d
from .solvers import (crank_nicolson_step, explicit_step, max_monotone_dt,
                      max_stable_dt)


SOLVERS = ("explicit", "crank_nicolson")


@dataclass(frozen=True)
class ExperimentConfig:
    material: str
    length: float
    n_points: int
    t_end: float
    initial_temperature: float = 0.0
    left_value: Optional[float] = None
    right_value: Optional[float] = None
    boundary: object = "dirichlet"
    solver: str = "explicit"
    safety: float = 0.9
    dt: Optional[float] = None
    n_steps: Optional[int] = None
    n_snapshots: int = 11

    def __post_init__(self):
        if self.material not in MATERIALS:
            raise ValueError(
                f"unknown material {self.material!r}; "
                f"known materials: {sorted(MATERIALS)}")
        if self.solver not in SOLVERS:
            raise ValueError(
                f"unknown solver {self.solver!r}; expected one of {SOLVERS}")
        if self.n_points < 3:
            raise ValueError("n_points must be at least 3")
        if not math.isfinite(self.length) or self.length <= 0.0:
            raise ValueError("length must be positive and finite")
        if not math.isfinite(self.t_end) or self.t_end <= 0.0:
            raise ValueError("t_end must be positive and finite")
        if self.n_snapshots < 2:
            raise ValueError("n_snapshots must be at least 2")
        if self.dt is not None and self.n_steps is not None:
            raise ValueError("provide at most one of dt or n_steps")
        if self.dt is not None and self.dt <= 0.0:
            raise ValueError("dt must be positive")
        if self.n_steps is not None and self.n_steps < 1:
            raise ValueError("n_steps must be at least 1")
        normalize_boundary(self.boundary)


@dataclass(frozen=True)
class ExperimentResult:
    config: ExperimentConfig
    parameters: dict
    x: np.ndarray
    times: np.ndarray
    snapshots: np.ndarray
    total_heat: np.ndarray
    min_temperature: np.ndarray
    max_temperature: np.ndarray
    mean_temperature: np.ndarray
    dt: float
    n_steps: int

    @property
    def final(self):
        return self.snapshots[-1]

    @property
    def initial(self):
        return self.snapshots[0]


def build_experiment_grid(config):
    u0 = np.full(config.n_points, float(config.initial_temperature))
    if config.left_value is not None:
        u0[0] = float(config.left_value)
    if config.right_value is not None:
        u0[-1] = float(config.right_value)
    return build_grid_1d(MATERIALS[config.material], config.length,
                         config.n_points, initial_temperature=u0)


def resolve_timestep(config, grid):
    if config.dt is not None:
        return float(config.dt)
    if config.n_steps is not None:
        return config.t_end / config.n_steps
    if config.solver == "explicit":
        return max_stable_dt(grid, safety=config.safety,
                             boundary=config.boundary)
    return max_monotone_dt(grid, boundary=config.boundary,
                           safety=config.safety)


def _snapshot_indices(n_steps, n_snapshots):
    raw = np.linspace(0, n_steps, n_snapshots)
    return np.unique(np.rint(raw).astype(int))


def run_experiment(config, parameters=None):
    grid = build_experiment_grid(config)
    step = explicit_step if config.solver == "explicit" else crank_nicolson_step

    dt = resolve_timestep(config, grid)
    n_steps = max(1, int(math.ceil(config.t_end / dt)))
    dt_used = config.t_end / n_steps
    wanted = _snapshot_indices(n_steps, config.n_snapshots)

    times, fields = [], []
    total_heat, lows, highs, means = [], [], [], []
    weight = grid.cell_weight
    weight_total = float(weight.sum())

    def record(step_index):
        times.append(step_index * dt_used)
        fields.append(grid.u.copy())
        total_heat.append(grid.total_heat())
        lows.append(float(grid.u.min()))
        highs.append(float(grid.u.max()))
        means.append(float(np.sum(weight * grid.u) / weight_total))

    pending = set(int(i) for i in wanted)
    if 0 in pending:
        record(0)
    for index in range(1, n_steps + 1):
        step(grid, dt_used, boundary=config.boundary)
        if index in pending:
            record(index)

    return ExperimentResult(
        config=config,
        parameters=dict(parameters or {}),
        x=grid.x.copy(),
        times=np.asarray(times, dtype=float),
        snapshots=np.asarray(fields, dtype=float),
        total_heat=np.asarray(total_heat, dtype=float),
        min_temperature=np.asarray(lows, dtype=float),
        max_temperature=np.asarray(highs, dtype=float),
        mean_temperature=np.asarray(means, dtype=float),
        dt=dt_used,
        n_steps=n_steps,
    )


def sweep(base_config, **parameter_lists):
    if not parameter_lists:
        raise ValueError("sweep needs at least one parameter to vary")

    valid = set(ExperimentConfig.__dataclass_fields__)
    unknown = sorted(set(parameter_lists) - valid)
    if unknown:
        raise ValueError(
            f"unknown config field(s) {unknown}; valid fields: {sorted(valid)}")
    for name, values in parameter_lists.items():
        if len(values) == 0:
            raise ValueError(f"parameter {name!r} has no values to sweep")

    names = sorted(parameter_lists)
    results = []
    for combination in itertools.product(*(parameter_lists[n] for n in names)):
        parameters = dict(zip(names, combination))
        config = replace(base_config, **parameters)
        results.append(run_experiment(config, parameters=parameters))
    return results