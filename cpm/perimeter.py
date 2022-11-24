import torch as t
import sys

sys.path.insert(0, "../si_model")
from risk_conv import risk_convolution2D


def perimeter(grid: t.Tensor, cell_id: int) -> t.Tensor:
    """Find the perimeter of the cell <cell_id> on the 2D lattice <grid>.

    Args:
        grid (t.Tensor): The lattice.
        cell_id (int): The ID of the cell to find the perimeter for.

    Returns:
        t.Tensor: the perimeter of the cell.
    """
    cell_mask = (grid != cell_id).double()
    active_neighbors = risk_convolution2D(cell_mask)
    valid_neighbors = active_neighbors * (1 - cell_mask)

    return t.sum(valid_neighbors, dim=(-1, -2))
