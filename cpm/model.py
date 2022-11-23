from typing import List
import torch as t
import numpy as np
import random
from hamiltonian import nabla_hamiltonian
from local_types import Cell_Type


def model(
    grid: t.Tensor, cell_params: List[Cell_Type], temperature: t.Tensor
) -> t.Tensor:
    _, num_rows, num_cols = grid.shape

    # pick the source cell
    src_row_idx = np.random.randint(0, num_rows)
    src_col_idx = np.random.randint(0, num_cols)
    src_pixel = grid[0, src_row_idx, src_col_idx]

    # pick the target cell from the Moore neighborhood of the source cell
    move_x, move_y = random.choice(
        [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    )
    target_row_idx = src_row_idx + move_x
    target_col_idx = src_col_idx + move_y

    # use periodic torus for cells at the edges
    if target_row_idx == num_rows:
        target_row_idx = 0
    elif target_row_idx == -1:
        target_row_idx = num_rows - 1

    if target_col_idx == num_cols:
        target_col_idx = 0
    elif target_col_idx == -1:
        target_col_idx = num_cols - 1

    target_pixel = grid[0, target_row_idx, target_col_idx]

    # if the cells have the same ID, nothing changes
    if src_pixel == target_pixel:
        return grid

    h_diff = nabla_hamiltonian(
        grid, target_row_idx, target_col_idx, src_pixel, cell_params
    )

    if h_diff <= 0:
        grid[0, target_row_idx, target_pixel] = src_pixel
    else:
        p_copy = t.exp(-h_diff / temperature)
        threshold = t.rand((1))
        print(
            p_copy,
            threshold,
            p_copy - threshold,
            (p_copy - threshold) >= 0,
            ((p_copy - threshold) >= 0).float(),
            ((p_copy - threshold) >= 0).float().item(),
        )
        new_val = ((p_copy - threshold) >= 0).float().item()
        grid[0, target_row_idx, target_pixel] = new_val
