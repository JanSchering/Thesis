import sys

sys.path.insert(0, "../")
import torch as t
import numpy as np
from volume import volume, H_volume
from volume_diff import H_volume_diff


def test_volume():
    grid = t.zeros((1, 3, 3))
    grid[0, :, 0] = 1
    grid[0, 0, 2] = 1
    assert volume(grid, cell_id=1) == 4


def test_H_volume():
    grid = t.zeros((1, 9, 9))
    grid[0, 0, 0] = 1
    grid[0, 1, 0] = 1
    grid[0, 2, 1] = 1
    grid[0, 3, 2] = 1
    grid[0, 4, 3] = 1
    grid[0, 5, 4] = 1
    grid[0, 6, 5] = 1
    print(grid)
    target_volumes = [9]
    scaling_factor = 0.6

    assert H_volume(grid, target_volumes, scaling_factor) == 0.6 * (7.0 - 9.0) ** 2


def test_H_volume_diff_1():
    """Check if the function replicates the behavior of the original."""
    grid = t.zeros((1, 9, 9))
    grid[0, 0, 0] = 1
    grid[0, 1, 0] = 1
    grid[0, 2, 1] = 1
    grid[0, 3, 2] = 1
    grid[0, 4, 3] = 1
    grid[0, 5, 4] = 1
    grid[0, 6, 5] = 1
    print(grid)
    target_volumes = t.tensor((9,))
    scaling_factor = 0.6

    assert (
        H_volume_diff(grid, target_volumes, scaling_factor, device="cpu")
        == 0.6 * (7.0 - 9.0) ** 2
    )
