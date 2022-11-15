from typing import List
import torch as t
import sys

sys.path.insert(0, "../si_model")
from risk_conv import risk_convolution2D


def contact_points(grid: t.Tensor, cell_id: int, target_id: int) -> t.Tensor:
    """
    Find the number of contact points on the 2D lattice <grid> between <cell_id>
    and <target_id>.
    """
    cell_mask = (grid == cell_id).double()
    target_mask = (grid == target_id).double()
    active_neighbors = risk_convolution2D(target_mask)
    valid_neighbors = active_neighbors * cell_mask

    return t.sum(valid_neighbors, dim=(-1, -2))


def adhesion(grid: t.Tensor, cell_id: int, penalties: dict) -> t.Tensor:
    """
    Calculate the adhesion of cell <cell_id> on the grid given some <penalties>.

    grid:       The CPM lattice.
    cell_id:    The ID of the cell to calculate the adhesion energy for.
    penalties:  penalty J for each interface between <cell_id> and other cell IDs.
    """
    total_adhesion = 0
    for target_id in penalties:
        penalty = penalties[target_id]
        total_adhesion += contact_points(grid, cell_id, target_id) * penalty
    return total_adhesion
