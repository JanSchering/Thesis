import sys

sys.path.insert(0, "../")
import torch as t
from adhesion import adhesion


def test_adhesion_1():
    """
    [   [0, 0, 0, 0, 0]
        [0, 0, 1, 0, 0]
        [0, 1, 1, 1, 0]
        [0, 0, 1, 0, 0]
        [0, 0, 0, 0, 0]   ]
    """
    grid = t.zeros((1, 5, 5))
    grid[0, 2, 1] = 1
    grid[0, 2, 2] = 1
    grid[0, 2, 3] = 1
    grid[0, 1, 2] = 1
    grid[0, 3, 2] = 1

    assert adhesion(grid, 1) == 24
    assert adhesion(grid, 0) == 24


def test_adhesion_2():
    """
    [   [0, 0, 0, 0, 0]
        [0, 2, 1, 1, 0]
        [2, 1, 1, 1, 0]
        [0, 2, 1, 1, 0]
        [0, 0, 0, 0, 0]   ]
    """
    grid = t.zeros((1, 5, 5))
    grid[0, 1, 1] = 2
    grid[0, 1, 2] = 1
    grid[0, 1, 3] = 1
    grid[0, 2, 0] = 2
    grid[0, 2, 1] = 1
    grid[0, 2, 2] = 1
    grid[0, 2, 3] = 1
    grid[0, 3, 1] = 2
    grid[0, 3, 2] = 1
    grid[0, 3, 3] = 1

    assert adhesion(grid, 1) == 28
    assert adhesion(grid, 2) == 20
    assert adhesion(grid, 0) == 34


def test_adhesion_3():
    """
    [   [0, 3, 0, 3, 0]
        [0, 2, 1, 1, 0]
        [2, 1, 1, 1, 0]
        [0, 2, 1, 1, 0]
        [1, 0, 1, 0, 0]   ]
    """
    grid = t.zeros((1, 5, 5))
    grid[0, 0, 1] = 3
    grid[0, 0, 3] = 3
    grid[0, 1, 1] = 2
    grid[0, 1, 2] = 1
    grid[0, 1, 3] = 1
    grid[0, 2, 0] = 2
    grid[0, 2, 1] = 1
    grid[0, 2, 2] = 1
    grid[0, 2, 3] = 1
    grid[0, 3, 1] = 2
    grid[0, 3, 2] = 1
    grid[0, 3, 3] = 1
    grid[0, 4, 0] = 1
    grid[0, 4, 2] = 1

    assert adhesion(grid, 1) == 40
    assert adhesion(grid, 3) == 16
    assert adhesion(grid, 2) == 20
    assert adhesion(grid, 0) == 44
