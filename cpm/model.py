from typing import List, Tuple
import torch as t
import numpy as np
import random
from hamiltonian import nabla_hamiltonian
from local_types import Cell_Type


def pick_cells(grid: t.Tensor) -> t.Tensor:
    _, num_rows, num_cols = grid.shape

    # pick the source cell
    src_row_idx = np.random.randint(0, num_rows)
    src_col_idx = np.random.randint(0, num_cols)

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

    return (src_row_idx, src_col_idx), (target_row_idx, target_col_idx)


def model(
    grid: t.Tensor,
    src_coords: Tuple[int, int],
    target_coords: Tuple[int, int],
    cell_params: List[Cell_Type],
    temperature: t.Tensor,
) -> t.Tensor:
    src_row_idx, src_col_idx = src_coords
    target_row_idx, target_col_idx = target_coords

    # pick the cells
    src_pixel = grid[0, src_row_idx, src_col_idx].clone()
    target_pixel = grid[0, target_row_idx, target_col_idx].clone()

    # if the cells have the same ID, nothing changes
    if src_pixel == target_pixel:
        return grid

    h_diff = nabla_hamiltonian(
        grid, target_row_idx, target_col_idx, src_pixel, cell_params
    )

    print(f"h_diff: {h_diff}")
    print(f"src: ({src_row_idx, src_col_idx})")
    print(f"target: ({target_row_idx, target_col_idx})")

    if h_diff <= 0:
        grid[0, target_row_idx, target_col_idx] = src_pixel
    else:
        p_copy = t.exp(-h_diff / temperature)
        print(f"p_copy: {p_copy}")
        threshold = t.rand((1))
        copy_success = ((p_copy - threshold) >= 0).float().item()
        if copy_success:
            print("copy success!")
            grid[0, target_row_idx, target_col_idx] = src_pixel

    return grid
