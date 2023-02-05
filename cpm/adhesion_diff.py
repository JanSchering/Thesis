import torch as t
import functorch as funcT
import sys

sys.path.insert(0, "../si_model")
from periodic_padding import periodic_padding


def row_adh_contacts(
    row: t.Tensor, target_center: t.Tensor, target_contact: t.Tensor
) -> t.Tensor:
    """HELPER_FUNCTION: Takes an unfolded 3x3 convolution <row> (shape (9,)). Then,
    if the center of the convolution (<row>[4]) is equal to <target_center>, sums up
    all occurences of <target_contact> in <row>. (Calculates the contact points between
    <target_center> and <target_contact> within a Moore neighboorhood). If the convolution center
    is not equal to <target_center>, returns 0 as no contact is made.

    Args:
        row (t.Tensor): Unfolded 3x3 convolution block.
        target_center (t.Tensor): The expected center of the convolution.
        target_contact (t.Tensor): The pixel value to sum the number of contacts over.

    Returns:
        t.Tensor: Number of contacts between <target_center> and <target_contact>
    """
    # First part of the function converts every pixel entry in <row> to 1,
    # if the entry is equal to <target_contact>, 0 otherwise then sums over the tensor
    # second part checks if the center of <row> has the desired value and zeroes out the sum
    # otherwise
    return t.sum((0 ** ((row - target_contact) ** 2))) * 0 ** (
        (row[4] - target_center) ** 2
    )


# Vectorizes the <row_adh_contacts> function such that it can be applied to a list of
# unfolded convolution blocks at once.
matrix_adh_contacts = funcT.vmap(row_adh_contacts, in_dims=(0, None, None))


def sum_adh_contacts(
    grid: t.Tensor, target_center: t.Tensor, target_contact: t.Tensor
) -> t.Tensor:
    """HELPER-FUNCTION: For a list of unfolded convolution blocks <grid>,
    calculate how many contact points exist between <target_center> and <target_contact>.

    Args:
        grid (t.Tensor): List of unfolded 3x3 convolution blocks, shape (num_convolutions, 9)
        target_center (t.Tensor): The expected center value of the convolution.
        target_contact (t.Tensor): The pixel value to sum the number of contacts over.

    Returns:
        t.Tensor: The number of contact points between <target_center> and <target_contact>
    """
    return t.sum(matrix_adh_contacts(grid, target_center, target_contact))


# Vectorize the <sum_adh_contacts> function such that it can be applied to a batch of grids
adhesion = funcT.vmap(sum_adh_contacts, in_dims=(0, None, None))
# Vectorize <adhesion> over the possible contact targets for the same cell
contact_batched_adhesion = funcT.vmap(adhesion, in_dims=(None, None, 0))


def calc_adh_energy(
    batch: t.Tensor,
    target_center: t.Tensor,
    target_contacts: t.Tensor,
    cell_bg_penalty: t.Tensor,
    cell_cell_penalty: t.Tensor,
) -> t.Tensor:
    """HELPER-FUNCTION: Calculate the adhesive energy between <target_center> and its <target_contacts>,
    using the number of contact points and the penalties per contact <cell_bg_penalty>, <cell_cell_penalty>,
    for a <batch> of grids. Expects the batch to be reshaped into a list of unfolded 3x3 convolution blocks.

    Args:
        batch (t.Tensor): batch where each sample is a list of unfolded 3x3 convolution blocks, shape (batch_size, num_convolutions, 9).
        target_center (t.Tensor): The cell to calculate the adhesive energy for.
        target_contacts (t.Tensor): list of possible contact points (cell IDs and background).
        cell_bg_penalty (t.Tensor): The penalty per contact point with the background.
        cell_cell_penalty (t.Tensor): The penalty per contact point with a foreign cell.

    Returns:
        _type_: _description_
    """
    # Get the number of contact points to each possible contact target
    adhesive_contacts = contact_batched_adhesion(batch, target_center, target_contacts)

    # prep penalty vector,
    # NOTE: we halve the cell_cell penalty, as the mirrored calculation is bound to appear.
    # I.e. when calculating the adhesion energy, we will calculate it for both (1, 3) and (3, 1).
    # This holds true for all cell_cell contacts. Needs to be revised if we have to apply a different
    # penalty depending on (source, target) combination.
    if cell_cell_penalty:
        cell_cell_penalty /= 2
    if target_contacts.size()[0] == 1:
        penalties = cell_bg_penalty
    elif cell_cell_penalty.size()[0] >= 1 and target_contacts.size()[0] > 1:
        penalties = t.cat(
            (cell_bg_penalty, cell_cell_penalty.repeat(target_contacts.size()[0] - 1))
        )
    else:
        penalties = t.cat((cell_bg_penalty, cell_cell_penalty))

    return t.sum(adhesive_contacts * penalties.unsqueeze(0).T, dim=0)


# Vectorize the <calc_adh_energy> function, such that it can be applied for a batch of <target_center>, <target_contacts>
# combinations
centroid_batched_adh_energy = funcT.vmap(
    calc_adh_energy, in_dims=(None, 0, 0, None, None)
)


def adhesion_energy(
    batch: t.Tensor,
    cell_bg_penalty: t.Tensor,
    cell_cell_penalty: t.Tensor,
) -> t.Tensor:
    """Calculate the adhesive energy for a batch of CPM grids.

    Args:
        batch (t.Tensor): Batch of CPM grids, shape (batch_size, grid_size, grid_size)
        cell_bg_penalty (t.Tensor): The energy penalty to apply per contact point of a cell with the background.
        cell_cell_penalty (t.Tensor): The energy penalty to apply per contact point of a cell with another, different cell.

    Returns:
        t.Tensor: The adhesive energy per sample in the batch
    """
    # ensure correct data type of the tensors
    batch = batch.float()
    if cell_cell_penalty:
        cell_cell_penalty = cell_cell_penalty.float()
    # define an unfolding operator
    unfold_transform = t.nn.Unfold(kernel_size=3)
    # provide a periodic torus padding to the grid
    padded_batch = periodic_padding(batch)
    # we need to add a channel dimension because unfold expects vectors of shape (N,C,H,W)
    padded_batch = padded_batch.unsqueeze(1)
    # apply the unfolding operator on the padded grid, this provides all convolution blocks
    unfolded_batch = unfold_transform(padded_batch)
    # turn each convolution block into a row
    batch_reshaped = unfolded_batch.permute(0, 2, 1)
    # get a list of all cells on the grid
    cell_IDs = t.unique(batch)
    # make a list of target contacts for each cell_ID
    target_contacts = t.tensor(
        [
            (t.tensor(0.0), *cell_IDs[1 : i + 1], *cell_IDs[i + 2 :])
            for i in range(cell_IDs[1:].size()[0])
        ]
    )

    return t.sum(
        centroid_batched_adh_energy(
            batch_reshaped,
            cell_IDs[1:],
            target_contacts,
            cell_bg_penalty,
            cell_cell_penalty,
        ),
        dim=0,
    )
