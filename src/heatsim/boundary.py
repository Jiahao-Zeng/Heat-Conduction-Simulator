import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Dirichlet:
    pass


@dataclass(frozen=True)
class Neumann:
    pass


@dataclass(frozen=True)
class Convective:
    h: float
    u_inf: float

    def __post_init__(self):
        if not math.isfinite(self.h) or self.h < 0.0:
            raise ValueError("h must be a finite, non-negative number")
        if not math.isfinite(self.u_inf):
            raise ValueError("u_inf must be a finite number")


_BOUNDARY_TYPES = (Dirichlet, Neumann, Convective)

_STRING_ALIASES = {
    "dirichlet": Dirichlet(),
    "neumann": Neumann(),
}


def normalize_boundary(boundary):
    if isinstance(boundary, str):
        try:
            bc = _STRING_ALIASES[boundary]
        except KeyError:
            raise ValueError(f"unknown boundary condition: {boundary!r}")
        return (bc, bc)

    if isinstance(boundary, _BOUNDARY_TYPES):
        return (boundary, boundary)

    if (isinstance(boundary, tuple) and len(boundary) == 2
            and all(isinstance(side, _BOUNDARY_TYPES) for side in boundary)):
        return boundary

    raise ValueError(f"unknown boundary condition: {boundary!r}")


def uniform_boundary(boundary, context):
    left, right = normalize_boundary(boundary)
    if left != right:
        raise ValueError(
            f"{context} requires the same boundary condition on both sides "
            f"(got {left!r} and {right!r})")
    return left


def normalize_boundary_2d(boundary, context):
    if isinstance(boundary, tuple) and len(boundary) == 2:
        x_spec, y_spec = boundary
    else:
        x_spec = y_spec = boundary
    try:
        x_left, x_right = normalize_boundary(x_spec)
    except ValueError as exc:
        raise ValueError(f"{context} (x-axis): {exc}") from None
    try:
        y_bottom, y_top = normalize_boundary(y_spec)
    except ValueError as exc:
        raise ValueError(f"{context} (y-axis): {exc}") from None
    return x_left, x_right, y_bottom, y_top