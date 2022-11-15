from typing import List
import torch as t


def volume(grid: t.Tensor, cell_id: int) -> t.Tensor:
    """
    Calculate the volume of the cell with ID <cell_id> on the 2D
    lattice <grid>.
    """
    cell_mask = grid == cell_id
    return t.sum(cell_mask, dim=(-1, -2))


def H_volume(
    grid: t.Tensor, target_volumes: List[float], scaling_factor: float
) -> t.Tensor:
    """
    Calculates the Hamiltonian volume energy on the lattice <grid> given the <target_volumes>
    for each cell on the lattice.
    """
    total_energy = 0
    for i, target_vol in enumerate(target_volumes):
        cell_id = i + 1
        current_vol = volume(grid, cell_id)
        total_energy += scaling_factor * (current_vol - target_vol) ** 2
    return total_energy
