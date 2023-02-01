from typing import List
import torch as t


def H_volume_diff(
    batch: t.Tensor,
    target_volumes: t.Tensor,
    scaling_factor: t.Tensor,
    device: t.device,
) -> t.Tensor:
    """Calculates the Hamiltonian volume energy on the lattice <grid> given the <target_volumes>
    for each cell on the lattice. Uses differentiable transformations.

    Args:
        batch (t.Tensor): Batch of CPM lattices
        target_volumes (t.Tensor): The target volume of each cell on the lattice. Shape (N_cells,)
        scaling_factor (t.Tensor): Scaling factor of the volume penalty.
        device (t.device): The device to store the Torch tensors on.

    Returns:
        t.Tensor: The Hamiltonian volume energy for each of the CPM lattices.
    """
    batch_size = batch.shape[0]

    h_volume = t.zeros((batch_size,), device=device)
    for i, target_volume in enumerate(target_volumes):
        # transform the lattices such that all cells occupied by
        # <cell_id> become equal to 1, while all others become equal
        # to zero
        masked_grid = 0 ** ((batch - (i + 1)) ** 2)
        h_volume += (
            scaling_factor * (t.sum(masked_grid, dim=(1, 2)) - target_volume) ** 2
        )

    return h_volume
