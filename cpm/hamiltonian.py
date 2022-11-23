from typing import List
import torch as t
import sys
from adhesion import adhesion
from volume import volume
from perimeter import perimeter
from local_types import Cell_Type


def nabla_hamiltonian(
    grid: t.Tensor,
    row: int,
    column: int,
    target_id: int,
    cell_params: List[Cell_Type],
    use_adhesion=True,
    use_volume=True,
    use_perimeter=True,
) -> t.Tensor:
    """Calculate the change in the hamiltonian of the <grid> when
    cell (<row>, <column>) changes from its current ID to <target_id>.

    Args:
        grid: 2D tensor that represents the lattice of the CPM.
        row: the row of the target pixel.
        column: the column of the target pixel.
        target_id: the ID that is supposed to be copied onto the target pixel.
    """
    grid_adjusted = grid.clone().detach()
    print("gets here")
    grid_adjusted[0, row, column] = target_id

    adhesion_current = 0.0
    adhesion_adjusted = 0.0
    volume_h_current = 0.0
    volume_h_adjusted = 0.0
    perimeter_h_current = 0.0
    perimeter_h_adjusted = 0.0
    for cell in cell_params:
        if use_adhesion:
            # calculate the adhesion component of the hamiltonian
            adhesion_penalties = cell["adhesion_penalties"]
            adhesion_current += adhesion(grid, cell["cell_id"], adhesion_penalties)
            adhesion_adjusted += adhesion(
                grid_adjusted, cell["cell_id"], adhesion_penalties
            )
        print("adhesion successful")
        if use_volume:
            # calculate the volume component of the hamiltonian
            target_volume = cell["target_volume"]
            scaling_factor_v = cell["vol_scaling"]
            volume_h_current += (
                scaling_factor_v * (volume(grid, cell["cell_id"]) - target_volume) ** 2
            )
            volume_h_adjusted += (
                scaling_factor_v
                * (volume(grid_adjusted, cell["cell_id"]) - target_volume) ** 2
            )
        print("volume successful")
        if use_perimeter:
            # calculate the perimeter component of the hamiltonian
            target_perimeter = cell["target_perimeter"]
            scaling_factor_p = cell["perim_scaling"]
            perimeter_h_current += (
                scaling_factor_p
                * (perimeter(grid, cell["cell_id"]) - target_perimeter) ** 2
            )
            print("first perim successful")
            perimeter_h_adjusted += (
                scaling_factor_p
                * (perimeter(grid_adjusted, cell["cell_id"]) - target_perimeter) ** 2
            )
        print("perimeter successful")

    nabla_adhesion = adhesion_adjusted - adhesion_current
    nabla_volume = volume_h_adjusted - volume_h_current
    nabla_perimeter = perimeter_h_adjusted - perimeter_h_current

    return nabla_adhesion + nabla_volume + nabla_perimeter
