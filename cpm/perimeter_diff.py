import torch as t
import functorch as funcT
import sys

sys.path.insert(0, "../si_model")
from periodic_padding import periodic_padding


def row_contacts(row: t.Tensor, target: t.Tensor) -> t.Tensor:
    """HELPER FUNCTION: Takes in an unfolded 3x3 convolution block (tensor of shape (9,)).
    Then, if the center of the convolution is equal to <target> returns the number of entries
    with differing values (i.e. the number of contact points the center has to 'foreign' pixels).
    Else, returns 0.

    Args:
        row (t.Tensor): Unfolded 3x3 convolution block, shape (9,)
        target (t.Tensor): The expected value of the center of the convolution

    Returns:
        t.Tensor: The number of contacts if center equal to <target>, else 0
    """
    # first part of the function sets all pixels to 1 if they are different from the target,
    # 0 otherwise. The resulting array is summed up. Second half is checking if the center
    # corresponds to the target and zeroes out the result if not.
    return t.sum(1 - (0 ** ((row - target) ** 2))) * 0 ** ((row[4] - target) ** 2)


# vactorize the row_contacts function in order to process all convolution blocks of a grid at once
matrix_contacts = funcT.vmap(row_contacts, in_dims=(0, None))


def sum_contacts(grid: t.Tensor, target: t.Tensor) -> t.Tensor:
    """HELPER-FUNCTION: Get the sum of contact points between pixels with value <target> and foreign
    pixels (i.e. other pixel values). Expects an unfolded convolution Matrix, where each row corresponds
    to an unfolded 3x3 convolution block on the original grid.

    Args:
        grid (t.Tensor): Matrix of unfolded 3x3 convolution blocks. Each row corresponds to an unfolded 3x3 convolution
        target (t.Tensor): The pixel value to find the number of foreign contact points for

    Returns:
        t.Tensor: Number of foreign contact points with <target>.
    """
    return t.sum(matrix_contacts(grid, target))


# in order to process a whole batch of grids, we vectorize the sum_contacts function another time
perimeter = funcT.vmap(sum_contacts, in_dims=(0, None))
# Vectorizes the process for multiple target values
id_batched_perimeter = funcT.vmap(perimeter, in_dims=(None, 0))


def H_perimeter(batch: t.Tensor, target_perimeter: t.Tensor):
    """Calculate the Hamiltonian perimeter energy for each CPM in <batch>.

    Args:
        batch (t.Tensor): Batch of CPM grids
        target_perimeter (t.Tensor): The desired perimeter of the cells. Can either be a single value of
        shape (1,) or specify a target perimeter for each cell with shape (N_cells,)

    Returns:
        t.Tensor: the Hamiltonian perimeter energy for each CPM, shape (batch_size,)
    """
    # ensure that the batch is using float
    batch = batch.float()
    # get the IDs of the cells on the grid
    cell_IDs = t.unique(batch)
    # check the shape of the input and adjust if necessary
    if target_perimeter.size()[0] == 1:
        target_perimeter = target_perimeter.repeat(cell_IDs.shape[0] - 1)
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
    # apply the ID batched perimeter function to find the perimeter for each cell ID
    # NOTE: this includes the perimeter for the Background (ID = 0) which should be ignored in the
    # proceeding calculations as it shouldn't influence the Hamiltonian
    cell_perimeters = id_batched_perimeter(batch_reshaped, cell_IDs).T[:, 1:]

    # take the square difference between current perimeter and target for every cell and sum up the result
    return t.sum((cell_perimeters - target_perimeter) ** 2, dim=1)
