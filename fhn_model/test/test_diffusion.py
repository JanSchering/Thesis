import sys

sys.path.insert(0, "../")
import torch as t
from diffusion import translate, Direction2D


def test_translate():
    s1 = t.arange(9).reshape((3, 3))

    expectation_north_west = t.tensor(
        [
            [
                [s1[1, 1], s1[1, 2], s1[1, 0]],
                [s1[2, 1], s1[2, 2], s1[2, 0]],
                [s1[0, 1], s1[0, 2], s1[0, 0]],
            ],
        ]
    )

    expectation_north = t.tensor(
        [
            [
                [s1[1, 0], s1[1, 1], s1[1, 2]],
                [s1[2, 0], s1[2, 1], s1[2, 2]],
                [s1[0, 0], s1[0, 1], s1[0, 2]],
            ],
        ]
    )

    expectation_north_east = t.tensor(
        [
            [
                [s1[1, 2], s1[1, 0], s1[1, 1]],
                [s1[2, 2], s1[2, 0], s1[2, 1]],
                [s1[0, 2], s1[0, 0], s1[0, 1]],
            ],
        ]
    )

    expectation_west = t.tensor(
        [
            [
                [s1[0, 1], s1[0, 2], s1[0, 0]],
                [s1[1, 1], s1[1, 2], s1[1, 0]],
                [s1[2, 1], s1[2, 2], s1[2, 0]],
            ],
        ]
    )

    expectation_east = t.tensor(
        [
            [
                [s1[0, 2], s1[0, 0], s1[0, 1]],
                [s1[1, 2], s1[1, 0], s1[1, 1]],
                [s1[2, 2], s1[2, 0], s1[2, 1]],
            ],
        ]
    )

    expectation_south_west = t.tensor(
        [
            [
                [s1[2, 1], s1[2, 2], s1[2, 0]],
                [s1[0, 1], s1[0, 2], s1[0, 0]],
                [s1[1, 1], s1[1, 2], s1[1, 0]],
            ],
        ]
    )

    expectation_south = t.tensor(
        [
            [
                [s1[2, 0], s1[2, 1], s1[2, 2]],
                [s1[0, 0], s1[0, 1], s1[0, 2]],
                [s1[1, 0], s1[1, 1], s1[1, 2]],
            ],
        ]
    )

    expectation_south_east = t.tensor(
        [
            [
                [s1[2, 2], s1[2, 0], s1[2, 1]],
                [s1[0, 2], s1[0, 0], s1[0, 1]],
                [s1[1, 2], s1[1, 0], s1[1, 1]],
            ],
        ]
    )

    assert t.all(translate(s1, Direction2D.North_West) == expectation_north_west)
    assert t.all(translate(s1, Direction2D.North) == expectation_north)
    assert t.all(translate(s1, Direction2D.North_East) == expectation_north_east)
    assert t.all(translate(s1, Direction2D.West) == expectation_west)
    assert t.all(translate(s1, Direction2D.Center) == s1)
    assert t.all(translate(s1, Direction2D.East) == expectation_east)
    assert t.all(translate(s1, Direction2D.South_West) == expectation_south_west)
    assert t.all(translate(s1, Direction2D.South) == expectation_south)
    assert t.all(translate(s1, Direction2D.South_East) == expectation_south_east)
