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
    # set torch device
    device = t.device("cuda" if t.cuda.is_available() else "cpu")
    # define a probability factor for convenience
    p_factor = t.sqrt(t.tensor(2.0, device=device))
    # Get the number of grids in the batch
    num_grids, height, width = grids.shape
    # Calculate the log-probability of moving horizontally or vertically
    p_sides = t.log(D * (p_factor - 1) * p_factor / 4)
    # Calculate the log-probability of moving along the diagonal
    p_diag = t.log(D * (p_factor - 1) / 4)
    # Calculate the log-probability of not moving
    p_center = t.log(1 - D)

    # build a log-probability matrix that will be used to sample a translation kernel
    # from the Gumbel-softmax
    kernel_logits = t.zeros((1, 1, 3, 3), device=device)
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


def excite_particles_STE(batch: t.Tensor, N: int) -> Tuple[t.Tensor, t.Tensor]:
    """Returns the adjusted Lattice + auxiliary grid of "excited" particles E.

    Args:
        batch The lattices, shape (batch_size, 2, lattice_size, lattice_size)
        N (int): The maximum occupation number per lattice cell

    Returns:
        Tuple[t.Tensor, t.Tensor]: [Grid after excitement step, Excited Lattice E]
    """
    # set torch device
    device = t.device("cuda" if t.cuda.is_available() else "cpu")
    # get the dimensionality of the batch
    batch_size, num_grids, height, width = batch.shape
    # sample a random number for each lattice cell in the batch,
    # used to determine whether a particle of a given cell becomes excited
    Xi = uniform.Uniform(0, 1).sample((batch_size, num_grids, height, width)).to(device)
    # Apply heaviside to get the grid of excited particles E
    E = STEFunction.apply(batch - N * Xi)
    # if a cell is filled, E has to be 1 at that cell
    E[batch == N] += 1 - E[batch == N]
    # if a cell is empty, E has to be 0 at that cell
    E[batch == 0] *= 0

    return batch - E, E


def accommodate_particles(batch: t.Tensor, E_A: t.Tensor, E_B: t.Tensor) -> t.Tensor:
    """Merges the batch of excited lattices E back into the current grid state.

    Args:
        batch (t.Tensor): The lattices, shape (batch_size, 2, lattice_size, lattice_size)
        E_A (t.Tensor): The excited lattices for the A species
        E_B (t.Tensor): The excited lattices for the B species

    Returns:
        t.Tensor: _description_
    """
    batch[:, 0] += E_A
    batch[:, 1] += E_B
    return batch


def diffuse_STE(batch: t.Tensor, N: int, D_A: t.Tensor, D_B: t.Tensor) -> t.Tensor:
    """Perform a step of diffusion for a batch of FHN lattices

    Args:
        batch (t.Tensor): The lattices, shape (batch_size, 2, lattice_size, lattice_size)
        N (int): The maximum occupation number per lattice cell
        D_A (t.Tensor): The diffusion coefficient of the A species
        D_B (t.Tensor): The diffusion coefficient of the B species

    Returns:
        t.Tensor: The batch after the diffusion step
    """
    # perform a step of particle excitation
    batch, E = excite_particles_STE(batch, N)

    # translate the excited particles of the A species
    E_A, _ = translate_gumbel(E[:, 0], D_A)
    # translate the excited particles of the B species
    E_B, _ = translate_gumbel(E[:, 1], D_B)

    # accommodate the translated particles at their new positions
    return accommodate_particles(batch, E_A, E_B)
