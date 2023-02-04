# %%
import torch as t
import functorch as funcT
import sys

sys.path.insert(0, "../si_model")
from periodic_padding import periodic_padding


def row_adh_contacts(
    row: t.Tensor, target_center: t.Tensor, target_contact: t.Tensor
) -> t.Tensor:
    return t.sum((0 ** ((row - target_contact) ** 2))) * 0 ** (
        (row[4] - target_center) ** 2
    )


matrix_adh_contacts = funcT.vmap(row_adh_contacts, in_dims=(0, None, None))


def sum_adh_contacts(grid: t.Tensor, target_center: t.Tensor, target_contact: t.Tensor):
    return t.sum(matrix_adh_contacts(grid, target_center, target_contact))


adhesion = funcT.vmap(sum_adh_contacts, in_dims=(0, None, None))
# vectorize over the possible contact targets for the same cell
contact_batched_adhesion = funcT.vmap(adhesion, in_dims=(None, None, 0))


def calc_adh_energy(
    batch, target_center, target_contacts, cell_bg_penalty, cell_cell_penalty
):
    batch = batch.float()
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

    adhesive_contacts = contact_batched_adhesion(
        batch_reshaped, target_center, target_contacts
    )

    # prep penalty vector
    cell_cell_penalty /= 2
    if target_contacts.size()[0] == 1:
        penalties = cell_bg_penalty
    elif cell_cell_penalty.size()[0] >= 1 and target_contacts.size()[0] > 1:
        penalties = t.cat(
            (cell_bg_penalty, cell_cell_penalty.repeat(target_contacts.size()[0] - 1))
        )
    else:
        penalties = t.cat((cell_bg_penalty, cell_cell_penalty))

    print("penalties", penalties)
    print("target center", target_center)
    print("target contacts", target_contacts)
    print("adhesive contacts", adhesive_contacts.T)

    return t.sum(adhesive_contacts * penalties.unsqueeze(0).T, dim=0)


centroid_batched_adh_energy = funcT.vmap(
    calc_adh_energy, in_dims=(None, 0, 0, None, None)
)


# %%
grid = t.zeros((2, 5, 5))
grid[0, 1, 1] = 2
grid[0, 1, 2] = 1
grid[0, 1, 3] = 1
grid[0, 2, 0] = 2
grid[0, 2, 1] = 1
grid[0, 2, 2] = 1
grid[0, 2, 3] = 1
grid[0, 3, 1] = 2
grid[0, 3, 2] = 1
grid[0, 3, 3] = 1

t.sum(
    centroid_batched_adh_energy(
        grid,
        t.tensor((1.0, 2.0)),
        t.tensor([(0.0, 2.0), (0.0, 1.0)]),
        t.tensor((34.0,)),
        t.tensor((56,)),
    ),
    dim=0,
)
# %%
