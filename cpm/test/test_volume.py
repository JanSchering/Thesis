from cell_typing import CellKind, CellMap
from volume_diff import volume_energy
from volume import volume, H_volume
import numpy as np
import torch as t
import sys

sys.path.insert(0, "../")


def test_volume():
    grid = t.zeros((1, 3, 3))
    grid[0, :, 0] = 1
    grid[0, 0, 2] = 1
    assert volume(grid, cell_id=1) == 4


def test_volume_energy_1():
    batch = t.zeros((2, 3, 3))
    batch[0, :, 0] = 1
    batch[0, 0, 2] = 1
    batch[0, 2, 2] = 2

    batch[1, :, :] = 3

    cell1 = CellKind(
        type_id=1,
        target_perimeter=None,
        lambda_perimeter=None,
        target_volume=t.tensor(4.0),
        lambda_volume=t.tensor(1.0),
        adhesion_cost=None,
    )
    cell2 = CellKind(
        type_id=1,
        target_perimeter=None,
        lambda_perimeter=None,
        target_volume=t.tensor(2.0),
        lambda_volume=t.tensor(2.0),
        adhesion_cost=None,
    )
    cell3 = CellKind(
        type_id=1,
        target_perimeter=None,
        lambda_perimeter=None,
        target_volume=t.tensor(1.0),
        lambda_volume=t.tensor(3.0),
        adhesion_cost=None,
    )
    cell_map = CellMap()
    cell_map.add(cell_id=1, cell_type=cell1)
    cell_map.add(cell_id=2, cell_type=cell2)
    cell_map.add(cell_id=3, cell_type=cell3)

    vol_e = volume_energy(batch, cell_map=cell_map)

    assert t.all(t.isclose(vol_e, t.tensor([5.0, 216.0])))


def test_volume_energy_2():
    batch = t.zeros((1, 9, 9))
    batch[0, 0, 0] = 1
    batch[0, 1, 0] = 1
    batch[0, 2, 1] = 1
    batch[0, 3, 2] = 1
    batch[0, 4, 3] = 1
    batch[0, 5, 4] = 1
    batch[0, 6, 5] = 1

    cell1 = CellKind(
        type_id=1,
        target_perimeter=None,
        lambda_perimeter=None,
        target_volume=t.tensor(9.0),
        lambda_volume=t.tensor(0.6),
        adhesion_cost=None,
    )
    cell_map = CellMap()
    cell_map.add(cell_id=1, cell_type=cell1)

    vol_e = volume_energy(batch, cell_map=cell_map)

    assert t.isclose(vol_e, t.tensor(0.6 * (7.0 - 9.0) ** 2))


def test_volume_energy_3():
    batch = t.zeros((1, 9, 9))
    batch[0, 0, 0] = 1
    batch[0, 1, 0] = 1
    batch[0, 2, 1] = 1
    batch[0, 3, 2] = 1
    batch[0, 4, 3] = 1
    batch[0, 5, 4] = 1
    batch[0, 6, 5] = 1

    batch[0, 1, 1] = 2
    batch[0, 2, 2] = 2

    batch[0, 4, 8] = 3
    batch[0, 1, 2] = 3
    batch[0, 2, 0] = 3
    batch[0, 4, 6] = 4

    lambda_volume = t.tensor(0.3)
    cell1 = CellKind(
        type_id=1,
        target_perimeter=None,
        lambda_perimeter=None,
        target_volume=t.tensor(9.0),
        lambda_volume=lambda_volume,
        adhesion_cost=None,
    )
    cell2 = CellKind(
        type_id=2,
        target_perimeter=None,
        lambda_perimeter=None,
        target_volume=t.tensor(5.0),
        lambda_volume=lambda_volume,
        adhesion_cost=None,
    )
    cell3 = CellKind(
        type_id=3,
        target_perimeter=None,
        lambda_perimeter=None,
        target_volume=t.tensor(7.0),
        lambda_volume=lambda_volume,
        adhesion_cost=None,
    )
    cell4 = CellKind(
        type_id=4,
        target_perimeter=None,
        lambda_perimeter=None,
        target_volume=t.tensor(2.0),
        lambda_volume=lambda_volume,
        adhesion_cost=None,
    )

    cell_map = CellMap()
    cell_map.add(cell_id=1, cell_type=cell1)
    cell_map.add(cell_id=2, cell_type=cell2)
    cell_map.add(cell_id=3, cell_type=cell3)
    cell_map.add(cell_id=4, cell_type=cell4)

    vol_e_cell_1 = lambda_volume * (7.0 - 9.0) ** 2
    vol_e_cell_2 = lambda_volume * (5.0 - 2.0) ** 2
    vol_e_cell_3 = lambda_volume * (7.0 - 3.0) ** 2
    vol_e_cell_4 = lambda_volume * (2.0 - 1.0) ** 2
    expected_vol_e = t.tensor(
        vol_e_cell_1 + vol_e_cell_2 + vol_e_cell_3 + vol_e_cell_4)

    assert t.isclose(volume_energy(batch, cell_map), expected_vol_e)


def test_volume_energy_3():
    batch = t.zeros((3, 3, 3))
    batch[0, 0] = 1
    batch[0, 1] = 2
    batch[0, 2] = 3
    batch[1, 1] = 3
    batch[1, 2] = 2
    batch[2, :, 2] = 1

    lambda_volume = t.tensor(0.6)
    cell1 = CellKind(
        type_id=1,
        target_perimeter=None,
        lambda_perimeter=None,
        target_volume=t.tensor(6.0),
        lambda_volume=lambda_volume,
        adhesion_cost=None,
    )
    cell2 = CellKind(
        type_id=2,
        target_perimeter=None,
        lambda_perimeter=None,
        target_volume=t.tensor(6.0),
        lambda_volume=lambda_volume,
        adhesion_cost=None,
    )
    cell3 = CellKind(
        type_id=3,
        target_perimeter=None,
        lambda_perimeter=None,
        target_volume=t.tensor(6.0),
        lambda_volume=lambda_volume,
        adhesion_cost=None,
    )

    cell_map = CellMap()
    cell_map.add(cell_id=1, cell_type=cell1)
    cell_map.add(cell_id=2, cell_type=cell2)
    cell_map.add(cell_id=3, cell_type=cell3)

    # calculate volume energy for sample 1
    vol_e_1_1 = lambda_volume * (6.0 - 3.0) ** 2
    vol_e_1_2 = lambda_volume * (6.0 - 3.0) ** 2
    vol_e_1_3 = lambda_volume * (6.0 - 3.0) ** 2
    expected_vol_e_1 = vol_e_1_1 + vol_e_1_2 + vol_e_1_3

    # calculate the H vol for sample 2
    vol_e_2_1 = lambda_volume * (6.0 - 0.0) ** 2
    vol_e_2_2 = lambda_volume * (6.0 - 3.0) ** 2
    vol_e_2_3 = lambda_volume * (6.0 - 3.0) ** 2
    expected_vol_e_2 = vol_e_2_1 + vol_e_2_2 + vol_e_2_3

    # calculate the H vol for sample 3
    vol_e_3_1 = lambda_volume * (6.0 - 3.0) ** 2
    vol_e_3_2 = lambda_volume * (6.0 - 0.0) ** 2
    vol_e_3_3 = lambda_volume * (6.0 - 0.0) ** 2
    expected_vol_e_3 = vol_e_3_1 + vol_e_3_2 + vol_e_3_3

    expected_result = t.tensor(
        (expected_vol_e_1, expected_vol_e_2, expected_vol_e_3))

    assert t.all(
        t.isclose(
            volume_energy(batch, cell_map),
            expected_result,
        )
    )
