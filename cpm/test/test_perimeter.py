import sys

sys.path.insert(0, "../")
import torch as t
from perimeter import perimeter


def test_perimeter_1():
    """
    [   [0,0,0,0,0,0]
        [0,0,0,0,0,0]
        [0,0,0,1,0,0]
        [0,0,1,1,0,0]
        [0,0,0,0,0,0]
        [0,0,0,0,0,0]]
    """
    grid = t.zeros((1, 6, 6))
    grid[0, 2, 3] = 1
    grid[0, 3, 2] = 1
    grid[0, 3, 3] = 1

    assert perimeter(grid, cell_id=1) == 18


def test_perimeter_2():
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

    assert perimeter(grid, cell_id=1) == 28
    assert perimeter(grid, cell_id=2) == 20
