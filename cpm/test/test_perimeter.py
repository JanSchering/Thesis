import sys

sys.path.insert(0, "../")
import torch as t
from perimeter import perimeter
from perimeter_diff import H_perimeter


def init_test_1():
    """
    [   [0,0,0,0,0,0]
        [0,0,0,0,0,0]
        [0,0,0,1,0,0]
        [0,0,1,1,0,0]
        [0,0,0,0,0,0]
        [0,0,0,0,0,0]]
    """
    batch = t.zeros((1, 6, 6))
    batch[0, 2, 3] = 1
    batch[0, 3, 2] = 1
    batch[0, 3, 3] = 1

    return batch


def test_perimeter_1():
    batch = init_test_1()
    assert perimeter(batch, cell_id=1) == 18


def test_H_perimeter_diff_1():
    batch = init_test_1()

    h_perim = H_perimeter(
        batch,
        target_perimeter=t.tensor((16.0,)),
    )

    assert t.isclose(h_perim, t.tensor(4.0))


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


def test_H_perimeter_diff_2():
    grid = t.zeros((2, 5, 5))
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

    h_perims = H_perimeter(grid, t.tensor((2.0, 5.0)))

    assert t.all(t.isclose(h_perims, t.tensor((901.0, 29.0))))

    h_perims = H_perimeter(grid, t.tensor((2.0,)))

    assert t.all(t.isclose(h_perims, t.tensor((1000.0, 8.0))))
