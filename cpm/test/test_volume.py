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


def init_test_1():
    batch = t.zeros((1, 9, 9))
    batch[0, 0, 0] = 1
    batch[0, 1, 0] = 1
    batch[0, 2, 1] = 1
    batch[0, 3, 2] = 1
    batch[0, 4, 3] = 1
    batch[0, 5, 4] = 1
    batch[0, 6, 5] = 1
    print(batch)
    target_volumes = [9]
    scaling_factor = 0.6

    return batch, target_volumes, scaling_factor


def test_H_volume_1():
    batch, target_volumes, scaling_factor = init_test_1()

    assert H_volume(batch, target_volumes, scaling_factor) == 0.6 * (7.0 - 9.0) ** 2
    assert (
        H_volume_diff(batch, target_volumes, scaling_factor, device="cpu")
        == 0.6 * (7.0 - 9.0) ** 2
    )


def init_test_2():
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

    target_volumes = t.tensor((9, 5, 7, 2))
    scaling_factor = 0.3

    return batch, target_volumes, scaling_factor


def test_H_volume_2():
    batch, target_volumes, scaling_factor = init_test_2()

    h_vol_cell_1 = scaling_factor * (7.0 - 9.0) ** 2
    h_vol_cell_2 = scaling_factor * (5.0 - 2.0) ** 2
    h_vol_cell_3 = scaling_factor * (7.0 - 3.0) ** 2
    h_vol_cell_4 = scaling_factor * (2.0 - 1.0) ** 2
    expected_h_vol = t.tensor(h_vol_cell_1 + h_vol_cell_2 + h_vol_cell_3 + h_vol_cell_4)

    assert t.isclose(H_volume(batch, target_volumes, scaling_factor)[0], expected_h_vol)
    assert t.isclose(
        H_volume_diff(batch, target_volumes, scaling_factor, device="cpu")[0],
        expected_h_vol,
    )


def init_test_3():
    batch = t.zeros((3, 3, 3))
    batch[0, 0] = 1
    batch[0, 1] = 2
    batch[0, 2] = 3
    batch[1, 1] = 3
    batch[1, 2] = 2
    batch[2, :, 2] = 1
    target_volumes = t.tensor((6, 6, 6))
    scaling_factor = 0.6

    return batch, target_volumes, scaling_factor


def H_volume_test_3():
    batch, target_volumes, scaling_factor = init_test_3()

    # calculate H vol for sample 1
    h_vol_1_1 = scaling_factor * (6.0 - 3.0) ** 2
    h_vol_1_2 = scaling_factor * (6.0 - 3.0) ** 2
    h_vol_1_3 = scaling_factor * (6.0 - 3.0) ** 2
    expected_h_vol_1 = h_vol_1_1 + h_vol_1_2 + h_vol_1_3

    # calculate the H vol for sample 2
    h_vol_2_1 = scaling_factor * (6.0 - 0.0) ** 2
    h_vol_2_2 = scaling_factor * (6.0 - 3.0) ** 2
    h_vol_2_3 = scaling_factor * (6.0 - 3.0) ** 2
    expected_h_vol_2 = h_vol_2_1 + h_vol_2_2 + h_vol_2_3

    # calculate the H vol for sample 3
    h_vol_3_1 = scaling_factor * (6.0 - 3.0) ** 2
    h_vol_3_2 = scaling_factor * (6.0 - 0.0) ** 2
    h_vol_3_3 = scaling_factor * (6.0 - 0.0) ** 2
    expected_h_vol_3 = h_vol_3_1 + h_vol_3_2 + h_vol_3_3

    expected_result = t.tensor((expected_h_vol_1, expected_h_vol_2, expected_h_vol_3))

    assert t.all(
        t.isclose(H_volume(batch, target_volumes, scaling_factor), expected_result)
    )
    assert t.all(
        t.isclose(
            H_volume_diff(batch, target_volumes, scaling_factor, device="cpu"),
            expected_result,
        )
    )
