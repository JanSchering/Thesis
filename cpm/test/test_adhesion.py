import sys

sys.path.insert(0, "../")
import torch as t
from adhesion import contact_points, adhesion
from adhesion_diff import adhesion_energy


def test_adh_diff_initial():
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

    adhesive_energy = adhesion_energy(grid, t.tensor((34.0,)), t.tensor((56,)))

    assert t.all(t.isclose(adhesive_energy, t.tensor((1548.0, 0.0))))


def init_case_1():
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
    return grid


def test_contact_points_1():
    grid = init_case_1()
    assert contact_points(grid, 1, 0) == 24


def test_adhesion_1():
    grid = init_case_1()
    penalties = {0: 25}
    assert adhesion(grid, 1, penalties) == 24 * 25


def test_adhesion_diff_1():
    grid = init_case_1()
    cell_bg_penalty = t.tensor((25.0,))

    assert t.isclose(
        adhesion_energy(grid, cell_bg_penalty, cell_cell_penalty=None),
        t.tensor((24.0 * 25.0,), dtype=t.float),
    )


def init_case_2():
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
    return grid


def test_contact_points_2():
    grid = init_case_2()
    assert contact_points(grid, 1, 2) == 7
    assert contact_points(grid, 1, 0) == 21
    assert contact_points(grid, 2, 0) == 13


def test_adhesion_2():
    grid = init_case_2()
    penalties_1 = {0: 34, 2: 56}
    penalties_2 = {0: 20, 1: 56}

    assert adhesion(grid, cell_id=1, penalties=penalties_1) == 34 * 21 + 56 * 7
    assert adhesion(grid, cell_id=2, penalties=penalties_2) == 20 * 13 + 56 * 7


def test_adhesion_diff_3():
    grid = init_case_2()

    cell_bg_penalty = t.tensor((34.0,))
    cell_cell_penalty = t.tensor((56.0,))

    assert t.isclose(
        adhesion_energy(grid, cell_bg_penalty, cell_cell_penalty),
        t.tensor((34.0 * 21.0 + 34.0 * 13.0 + 56.0 * 7.0)),
    )


def init_case_3():
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
    return grid


def test_contact_points_3():
    grid = init_case_3()

    assert contact_points(grid, 1, 2) == 9
    assert contact_points(grid, 1, 3) == 6
    assert contact_points(grid, 2, 3) == 1
    assert contact_points(grid, 1, 0) == 25
    assert contact_points(grid, 2, 0) == 10
    assert contact_points(grid, 3, 0) == 9


def test_adhesion_3():
    grid = init_case_3()
    penalties_1 = {0: 10, 2: 20, 3: 30}
    penalties_2 = {0: 40, 1: 50, 3: 60}
    penalties_3 = {0: 70, 1: 80, 2: 90}

    assert adhesion(grid, cell_id=1, penalties=penalties_1) == 25 * 10 + 9 * 20 + 6 * 30
    assert adhesion(grid, cell_id=2, penalties=penalties_2) == 10 * 40 + 9 * 50 + 1 * 60
    assert adhesion(grid, cell_id=3, penalties=penalties_3) == 9 * 70 + 6 * 80 + 1 * 90


def test_adhesion_diff_3():
    grid = init_case_3()

    cell_bg_penalty = t.tensor((10.0,))
    cell_cell_penalty = t.tensor((30.0,))

    assert t.isclose(
        adhesion_energy(grid, cell_bg_penalty, cell_cell_penalty),
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
