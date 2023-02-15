
import torch as t
import numpy as np
from cell_typing import CellMap
from adhesion_diff import adhesion_energy
from perimeter_diff import perimeter_energy
from volume_diff import volume_energy
from cpm_model import pick_cells
from ste_func import STEFunction, STELogicalOr


def hamiltonian_energy(
    batch: t.Tensor,
    cell_map: CellMap,
    use_volume: bool,
    use_perimeter: bool,
    use_adhesion: bool
):
    e = 0.
    if use_volume:
        e += volume_energy(batch, cell_map)
    if use_perimeter:
        e += perimeter_energy(batch, cell_map)
    if use_adhesion:
        e += adhesion_energy(batch, cell_map)

    return e


def prob_copy(h_diff, temperature):
    return t.exp(-h_diff / temperature)


def model(batch, cell_map, temperature):
    src_coords_batch, target_coords_batch = []
    for grid in batch:
        valid_coords = False
        while not valid_coords:
            src_coords, target_coords = pick_cells(grid)
            valid_coords = grid[src_coords] == grid[target_coords]
        src_coords_batch.append(src_coords)
        target_coords_batch.append(target_coords_batch)

    src_coords_batch = np.concatenate(
        (np.arange(batch.shape[0]), src_coords_batch), axis=1)
    target_coords_batch = np.concatenate(
        (np.arange(batch.shape[0]), target_coords_batch), axis=1)

    target_batch = batch.clone()
    target_batch[target_coords] *= 0
    target_batch[target_coords] += src_coords

    h_current = hamiltonian_energy(batch, cell_map)
    h_adjusted = hamiltonian_energy(batch, cell_map)

    # NOTE: should be a vector, 1 val per sample in batch
    h_diff = h_adjusted - h_current

    # NOTE: should be a vector of probabilities
    p_copy = prob_copy(h_diff, temperature)
    residual = t.rand(size=(batch.shape[0],))

    aux1 = (0 ** (t.sqrt(h_diff**2) - h_diff))
    aux2 = STEFunction.apply(p_copy - residual)
    copy_success = STELogicalOr(t.cat((aux1, aux2)))

    return copy_success * target_batch + (1-copy_success) * batch
