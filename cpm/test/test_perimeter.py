from cell_typing import CellKind, CellMap
from perimeter_diff import H_perimeter
from perimeter import perimeter
import torch as t
import sys

sys.path.insert(0, "../")


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

    test_cellkind = CellKind(
        target_perimeter=t.tensor(16.0), target_volume=None)

    test_cellmap = CellMap()
    test_cellmap.add(1, test_cellkind)

    h_perim = H_perimeter(
        batch,
        cell_map=test_cellmap,
        target_perimeter=None
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

    test_cellkind1 = CellKind(
        target_perimeter=t.tensor(2.0), target_volume=None)
    test_cellkind2 = CellKind(
        target_perimeter=t.tensor(5.0), target_volume=None)

    test_cellmap = CellMap()
    test_cellmap.add(cell_id=1, cell_type=test_cellkind1)
    test_cellmap.add(cell_id=2, cell_type=test_cellkind2)

    h_perims = H_perimeter(grid, cell_map=test_cellmap, target_perimeter=None)

    assert t.all(t.isclose(h_perims, t.tensor((901.0, 29.0))))
