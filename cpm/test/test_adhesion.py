import sys

sys.path.insert(0, "../")
import torch as t
from adhesion import contact_points, adhesion
from adhesion_diff import adhesion_energy
from cell_typing import CellKind, CellMap


def test_adh_energy_1():
    """
    [
       [[0 0 0 0 0]
        [0 2 1 1 0]
        [2 1 1 1 0]
        [0 2 1 1 0]
        [0 0 0 0 0]],

       [[0 0 0 0 0]
        [0 0 0 0 0]
        [0 0 0 0 0]
        [0 0 0 0 0]
        [0 0 0 0 0]]
    ]
    """
    batch = t.zeros((2, 5, 5))
    batch[0, 1, 1] = 2
    batch[0, 2, 0] = 2
    batch[0, 3, 1] = 2

    batch[0, 1, 2] = 1
    batch[0, 1, 3] = 1
    batch[0, 2, 1] = 1
    batch[0, 2, 2] = 1
    batch[0, 2, 3] = 1
    batch[0, 3, 2] = 1
    batch[0, 3, 3] = 1

    cell1_adhesion = {0: 34.0, 1: 56.0, 2: 56.0}
    cell1 = CellKind(
        type_id=1,
        target_perimeter=None,
        target_volume=None,
        lambda_volume=None,
        adhesion_cost=cell1_adhesion,
    )

    cell2_adhesion = {0: 34.0, 1: 56.0, 2: 56.0}
    cell2 = CellKind(
        type_id=2,
        target_perimeter=None,
        target_volume=None,
        lambda_volume=None,
        adhesion_cost=cell2_adhesion,
    )
    cell_map = CellMap()
    cell_map.add(cell_id=1, cell_type=cell1)
    cell_map.add(cell_id=2, cell_type=cell2)

    adhesive_energy = adhesion_energy(batch, cell_map)

    assert t.all(t.isclose(adhesive_energy, t.tensor((1548.0, 0.0))))


def test_adh_energy_2():
    """
    [   [0, 0, 0, 0, 0]
        [0, 0, 1, 0, 0]
        [0, 1, 1, 1, 0]
        [0, 0, 1, 0, 0]
        [0, 0, 0, 0, 0]   ]
    """
    batch = t.zeros((1, 5, 5))
    batch[0, 2, 1] = 1
    batch[0, 2, 2] = 1
    batch[0, 2, 3] = 1
    batch[0, 1, 2] = 1
    batch[0, 3, 2] = 1

    assert contact_points(batch, 1, 0) == 24

    cell1_adhesion = {0: t.tensor(25.0)}
    cell1 = CellKind(
        type_id=1,
        target_perimeter=None,
        target_volume=None,
        lambda_volume=None,
        adhesion_cost=cell1_adhesion,
    )
    cell_map = CellMap()
    cell_map.add(cell_id=1, cell_type=cell1)

    assert t.isclose(
        adhesion_energy(batch, cell_map),
        t.tensor((24.0 * 25.0,), dtype=t.float),
    )


def test_adh_energy_3():
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

    assert contact_points(grid, 1, 2) == 9
    assert contact_points(grid, 1, 3) == 6
    assert contact_points(grid, 2, 3) == 1
    assert contact_points(grid, 1, 0) == 25
    assert contact_points(grid, 2, 0) == 10
    assert contact_points(grid, 3, 0) == 9

    penalties_1 = {0: 10, 2: 20, 3: 30}
    penalties_2 = {0: 40, 1: 50, 3: 60}
    penalties_3 = {0: 70, 1: 80, 2: 90}

    assert adhesion(grid, cell_id=1, penalties=penalties_1) == 25 * 10 + 9 * 20 + 6 * 30
    assert adhesion(grid, cell_id=2, penalties=penalties_2) == 10 * 40 + 9 * 50 + 1 * 60
    assert adhesion(grid, cell_id=3, penalties=penalties_3) == 9 * 70 + 6 * 80 + 1 * 90

    cell1_adhesion = {0: t.tensor(10.0), 1: t.tensor(30.0)}
    cell1 = CellKind(
        type_id=1,
        target_perimeter=None,
        target_volume=None,
        lambda_volume=None,
        adhesion_cost=cell1_adhesion,
    )
    cell_map = CellMap()
    cell_map.add(cell_id=1, cell_type=cell1)
    cell_map.add(cell_id=2, cell_type=cell1)
    cell_map.add(cell_id=3, cell_type=cell1)

    assert t.isclose(
        adhesion_energy(grid, cell_map),
        t.tensor(
            (
                25.0 * 10.0
                + 10.0 * 10.0
                + 9.0 * 10.0
                + 9.0 * 30.0
                + 6.0 * 30.0
                + 1.0 * 30.0
            )
        ),
    )
