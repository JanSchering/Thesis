from typing import List, Tuple
import torch as t
import numpy as np
import random
from hamiltonian import nabla_hamiltonian
from local_types import Cell_Type


def pick_cells(grid: t.Tensor) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    """Picks the coordinates for a 'source' cell on the grid at random. Then
    chooses the coordinates for a 'target' cell from the Moore neighborhood of the
    source pixel. Uses a periodic torus.

    Args:
        grid (t.Tensor): The grid to pick coordinates from.

    Returns:
        Tuple[Tuple[int, int], Tuple[int, int]]: The coordinates of the source and the target pixel.
    """
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


def prob_copy(h_diff: t.Tensor, temperature: t.Tensor) -> t.Tensor:
    """Calculate the probability of the copy attempt succeeding based on the change
    in the global hamiltonian energy <h_diff> and the temperature of the model <temperature>.

    Args:
        h_diff (t.Tensor): The change in the global hamiltonian energy caused by the copy attempt.
        temperature (t.Tensor): The model temperature (reaction likelihood scaling factor).

    Returns:
        t.Tensor: The probability of success for the copy attempt.
    """
    return t.exp(-h_diff / temperature)


def model(
    grid: t.Tensor,
    src_coords: Tuple[int, int],
    target_coords: Tuple[int, int],
    cell_params: List[Cell_Type],
    temperature: t.Tensor,
) -> t.Tensor:
    """CPM model. tries to copy the identity of the pixel at (<src_coords>) to
    to the pixel at (<target_coords>) based on the model temperature <temperature>
    and the change in the global hamiltonian energy that a successful copy attempt would
    cause.

    Args:
        grid (t.Tensor): The current state of the grid
        src_coords (Tuple[int, int]): The coordinates of the source pixel.
        target_coords (Tuple[int, int]): The coordinates of the target pixel.
        cell_params (List[Cell_Type]): The parameters for the cells on the grid.
        temperature (t.Tensor): The model temperature (reaction likelihood scaling factor).

    Returns:
        t.Tensor: The state of the grid after the copy attempt.
    """
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
        p_copy = prob_copy(h_diff, temperature)
        print(f"p_copy: {p_copy}")
        threshold = t.rand((1))
        copy_success = ((p_copy - threshold) >= 0).float().item()
        if copy_success:
            print("copy success!")
            grid[0, target_row_idx, target_col_idx] = src_pixel

    return grid
