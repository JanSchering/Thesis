# %%
from typing import Tuple
import torch as t
from torch.distributions import uniform
from torch.nn.functional import conv2d, gumbel_softmax
from periodic_padding import periodic_padding
from ste_func import STEFunction


def translate_gumbel(grids: t.Tensor, D: t.Tensor) -> Tuple[t.Tensor, t.Tensor]:
    """Translate grid according to a 9-direction scheme with probabilities based on the
    diffusion coefficient <D>

    Args:
        grids (t.Tensor): Tensor of shape (N_grids,height,width)
        D (t.Tensor): Diffusion coefficient

    Returns:
        (Tuple[t.Tensor, t.Tensor]): The translated grids + the kernels used for each translation
    """
    # Get the number of grids in the batch
    num_grids, height, width = grids.shape
    # Calculate the log-probability of moving horizontally or vertically
    p_sides = t.log(D * (t.sqrt(t.tensor(2.0)) - 1) * t.sqrt(t.tensor(2.0)) / 4)
    # Calculate the log-probability of moving along the diagonal
    p_diag = t.log(D * (t.sqrt(t.tensor(2.0)) - 1) / 4)
    # Calculate the log-probability of not moving
    p_center = t.log(1 - D)

    # build a log-probability matrix that will be used to sample a translation kernel
    # from the Gumbel-softmax
    kernel_logits = t.zeros((1, 1, 3, 3))
    kernel_logits[:, :, 0, 0] += p_diag
    kernel_logits[:, :, 0, 2] += p_diag
    kernel_logits[:, :, 2, 0] += p_diag
    kernel_logits[:, :, 2, 2] += p_diag
    kernel_logits[:, :, 0, 1] += p_sides
    kernel_logits[:, :, 2, 1] += p_sides
    kernel_logits[:, :, 1, 0] += p_sides
    kernel_logits[:, :, 1, 2] += p_sides
    kernel_logits[:, :, 1, 1] += p_center

    # sample a translation kernel for each grid
    kernels = t.cat(
        [
            gumbel_softmax(
                logits=kernel_logits.flatten(), tau=1.0, hard=True
            ).unflatten(dim=0, sizes=(1, 1, 3, 3))
            for i in range(num_grids)
        ],
        dim=0,
    )
    if grids.is_cuda:
        kernels = kernels.cuda()

    assert kernels.shape == (num_grids, 1, 3, 3)

    padded_grids = periodic_padding(grids).float()
    expanded_grids = t.unsqueeze(padded_grids, -1)
    transposed_grids = t.permute(expanded_grids, (0, 3, 1, 2))

    # move batch dim into channels
    transposed_grids = transposed_grids.view(1, -1, height + 2, width + 2)

    return (
        conv2d(
            transposed_grids, kernels, stride=1, padding="valid", groups=num_grids
        ).squeeze(),
        kernels,
    )


# %%


def excite_particles_STE(batch: t.Tensor, N: int) -> Tuple[t.Tensor, t.Tensor]:
    """
    Returns the adjusted Lattice + auxiliary grid of "excited" particles E.
    """
    batch_size, num_grids, height, width = batch.shape

    Xi = uniform.Uniform(0, 1).sample_n((batch_size, num_grids, height, width))

    if batch.is_cuda:
        Xi = Xi.cuda()

    E = STEFunction.apply(batch - N * Xi)
    # if a cell is filled, E has to be 1 at that cell
    E[batch == N] += 1 - E[batch == N]
    # if a cell is empty, E has to be 0 at that cell
    E[batch == 0] *= 0

    return batch - E, E


def accommodate_particles(batch: t.Tensor, E_A: t.Tensor, E_B: t.Tensor) -> t.Tensor:
    """
    Merges E into the current grid state.
    """
    batch[:,0] += E_A
    batch[:,1] += E_B
    return batch


def diffuse_STE(grid: t.Tensor, N: int, D_A, D_B) -> t.Tensor:
    grid, E = excite_particles_STE(grid, N)

    # translate the excited particles of the A species
    E_A = translate_gumbel(E[0], D_A)
    # translate the excited particles of the B species
    E_B = translate_gumbel(E[1], D_B)

    # accommodate the translated particles at their new positions
    return accommodate_particles(grid, E_A, E_B)
