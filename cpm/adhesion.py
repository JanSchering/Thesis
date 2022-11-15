# %%
import torch as t
import sys

sys.path.insert(0, "../si_model")
from risk_conv import risk_convolution2D


def adhesion(grid: t.Tensor, cell_id: t.Tensor):
    cell_mask = (grid != cell_id).double()
    active_neighbors = risk_convolution2D(cell_mask)
    valid_neighbors = active_neighbors * (1 - cell_mask)

    return t.sum(valid_neighbors, dim=(-1, -2))
