import sys

sys.path.insert(0, "../")
import torch as t
from hamiltonian import nabla_hamiltonian


def test_nabla_hamiltonian():
    cell_bg_adhesion = 1
    target_volume = 3
    vol_scaling = 1
    target_perimeter = 8
    perim_scaling = 1

    cell_params = {
        "cell_id": 1,
        "adhesion_penalties": {0: cell_bg_adhesion},
        "target_volume": target_volume,
        "vol_scaling": vol_scaling,
        "target_perimeter": target_perimeter,
        "perim_scaling": perim_scaling,
    }

    grid = t.zeros((1, 3, 3))
    grid[0, 1, :] = 1
    target_coords = (0, 1)

    h_diff = nabla_hamiltonian(
        grid, target_coords[0], target_coords[1], grid[0, 1, 1].clone(), [cell_params]
    )

    assert h_diff == 47.0
