from typing import List
import torch as t
from cell_typing import CellMap, CellKind
from operator import itemgetter
import functorch as funcT


def volume(cell_id: t.Tensor, batch: t.Tensor):
    return t.sum(0 ** ((batch - cell_id) ** 2), dim=(1, 2))


id_batched_volume = funcT.vmap(volume, in_dims=(0, None))


def volume_energy(
    batch: t.Tensor,
    cell_map: CellMap,
) -> t.Tensor:
    """Calculates the volume energy of each CPM grid in the <batch>.

    Args:
        batch (t.Tensor): Batch of CPM lattices.
        cell_map (CellMap): Maps each cell on the CPM to a CellKind providing the parameters.

    Returns:
        t.Tensor: The volume energy for each of the CPM lattices.
    """
    cell_IDs = t.unique(batch)[1:]

    print(cell_IDs)

    if cell_IDs.size()[0] == 1:
        target_cellkind = cell_map.get_map()[cell_IDs[0].int().item()]
        target_vol = target_cellkind.target_volume.unsqueeze(0)
        lambda_vol = target_cellkind.lambda_volume.unsqueeze(0)
    else:
        target_cellkind = itemgetter(*cell_IDs.int().tolist())(cell_map.get_map())
        target_vol = t.tensor([c.target_volume for c in target_cellkind])
        lambda_vol = t.tensor([c.lambda_volume for c in target_cellkind])

    print(target_vol)
    print(lambda_vol)
    print(id_batched_volume(cell_IDs, batch))
    volumes = id_batched_volume(cell_IDs, batch).T

    print(volumes)

    mse_volume = (volumes - target_vol) ** 2

    print(mse_volume)

    return t.sum(mse_volume.T * lambda_vol.unsqueeze(0).T, dim=0)
