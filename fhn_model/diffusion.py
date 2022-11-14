from typing import Tuple, List
from enum import Enum
import torch as t
from torch.distributions import uniform
import numpy as np


class Direction2D(Enum):
    Center = 0
    North_West = 1
    North_East = 2
    South_West = 3
    South_East = 4
    North = 5
    West = 6
    East = 7
    South = 8


def translate(grid: t.Tensor, v: Direction2D) -> t.Tensor:
    translate_north = translate_west = [*t.arange(1, grid.shape[1]), 0]
    translate_south = translate_east = [-1, *t.arange(grid.shape[1] - 1)]

    if v == Direction2D.North_West:
        return grid[:, translate_west][translate_north, :]
    elif v == Direction2D.West:
        return grid[:, translate_west]
    elif v == Direction2D.South_West:
        return grid[:, translate_west][translate_south, :]
    elif v == Direction2D.North:
        return grid[translate_north, :]
    elif v == Direction2D.Center:
        return grid
    elif v == Direction2D.South:
        return grid[translate_south, :]
    elif v == Direction2D.North_East:
        return grid[translate_north, :][:, translate_east]
    elif v == Direction2D.East:
        return grid[:, translate_east]
    elif v == Direction2D.South_East:
        return grid[translate_south, :][:, translate_east]
    else:
        raise Exception("Invalid translation direction")


def excite_particles(state: t.Tensor, N: int) -> Tuple[t.Tensor, t.Tensor]:
    """
    Returns the adjusted Lattice + auxiliary grid of "excited" particles E.
    """
    grid_dim = state.shape[-1]
    Xi = (
        uniform.Uniform(0, 1)
        .sample_n(2 * (grid_dim**2))
        .reshape(2, grid_dim, grid_dim)
    )
    E = t.heaviside(state - N * Xi, values=t.zeros(state.shape))
    E[state==0] = 0
    return state - E, E


def accomodate_particles(grid: t.Tensor, E: t.Tensor) -> t.Tensor:
    """
    Merges E into the current grid state.
    """
    return grid + E


def diffuse(state: t.Tensor, N: int, D_A, D_B) -> t.Tensor:
    state, E = excite_particles(state, N)
    # translate the excited A species
    p_A = [
        1 - D_A,
        *[D_A * (np.sqrt(2) - 1) / 4 for i in range(4)],
        *[D_A * (np.sqrt(2) - 1) * np.sqrt(2) / 4 for i in range(4)],
    ]
    direction_A = np.random.choice(list(Direction2D), p=p_A)
    E[0] = translate(E[0], direction_A)

    # translate the excited B species
    p_B = [
        1 - D_B,
        *[D_B * (np.sqrt(2) - 1) / 4 for i in range(4)],
        *[D_B * (np.sqrt(2) - 1) * np.sqrt(2) / 4 for i in range(4)],
    ]
    direction_B = np.random.choice(list(Direction2D), p=p_B)
    E[1] = translate(E[1], direction_B)

    # accomodate the translated particles at their new positions
    state = accomodate_particles(state, E)
    return state
