from typing import Tuple
import random
import torch as t
import numpy as np
from cell_typing import CellMap
from adhesion_diff import adhesion_energy
from perimeter_diff import perimeter_energy
from volume_diff import volume_energy
from ste_func import STEFunction, STELogicalOr


def pick_cells(grid: t.Tensor) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    """Picks the coordinates for a 'source' cell on the grid at random. Then
    chooses the coordinates for a 'target' cell from the Moore neighborhood of the
    source pixel. Uses a periodic torus.

    Args:
        grid (t.Tensor): The grid to pick coordinates from.

    Returns:
        Tuple[Tuple[int, int], Tuple[int, int]]: The coordinates of the source and the target pixel.
    """
    num_rows, num_cols = grid.shape

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


def hamiltonian_energy(
    batch: t.Tensor,
    cell_map: CellMap,
    use_volume: bool,
    use_perimeter: bool,
    use_adhesion: bool,
):
    e = 0.0
    stats = {}
    if use_volume:
        vol_e = volume_energy(batch, cell_map)
        e += vol_e
        stats["volume_energy"] = vol_e
    if use_perimeter:
        perim_e = perimeter_energy(batch, cell_map)
        e += perim_e
        stats["perimeter_energy"] = perim_e
    if use_adhesion:
        adh_e = adhesion_energy(batch, cell_map)
        e += adh_e
        stats["adh_e"] = adh_e

    stats["h"] = e
    return e, stats


def prob_copy(h_diff, temperature):
    return t.exp(-h_diff / temperature)


def model(batch, cell_map, temperature):
    src_coords_batch = []
    target_coords_batch = []
    for grid in batch:
        valid_coords = False
        while not valid_coords:
            src_coords, target_coords = pick_cells(grid)
            valid_coords = grid[src_coords] != grid[target_coords]
        src_coords_batch.append(src_coords)
        target_coords_batch.append(target_coords)

    src_coords_batch = np.hstack(
        (
            np.arange(batch.shape[0]).reshape((batch.shape[0], 1)),
            np.array(src_coords_batch),
        )
    )
    target_coords_batch = np.hstack(
        (
            np.arange(batch.shape[0]).reshape((batch.shape[0], 1)),
            np.array(target_coords_batch),
        )
    )

    target_batch = batch.clone()
    target_x = target_coords_batch[:, 0]
    target_y = target_coords_batch[:, 1]
    target_z = target_coords_batch[:, 2]
    src_x = src_coords_batch[:, 0]
    src_y = src_coords_batch[:, 1]
    src_z = src_coords_batch[:, 2]
    target_batch[target_x, target_y, target_z] *= 0
    target_batch[target_x, target_y, target_z] += batch[src_x, src_y, src_z]

    h_current, stats_current = hamiltonian_energy(
        batch, cell_map, use_volume=True, use_perimeter=True, use_adhesion=True
    )
    h_adjusted, stats_adjusted = hamiltonian_energy(
        target_batch, cell_map, use_volume=True, use_perimeter=True, use_adhesion=True
    )

    # NOTE: should be a vector, 1 val per sample in batch
    h_diff = h_adjusted - h_current

    # NOTE: should be a vector of probabilities
    p_copy = prob_copy(h_diff, temperature)
    residual = t.rand(size=(batch.shape[0],))

    aux1 = 0 ** (t.sqrt(h_diff**2) - h_diff)
    aux2 = STEFunction.apply(p_copy - residual)
    copy_success = STELogicalOr.apply((t.cat((aux1, aux2))))

    stats = {}
    stats["src_pixel"] = src_coords_batch
    stats["target_pixel"] = target_coords_batch
    stats["current"] = stats_current
    stats["adjusted"] = stats_adjusted
    stats["h_diff"] = h_diff
    stats["p_copy"] = p_copy
    stats["success"] = copy_success

    return copy_success * target_batch + (1 - copy_success) * batch, stats
