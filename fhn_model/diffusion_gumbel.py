from typing import Tuple
import torch as t
from torch.distributions import uniform
from torch.nn.functional import conv2d, gumbel_softmax
from periodic_padding import periodic_padding
from ste_func import STEFunction


def translate_gumbel(grid: t.Tensor, D: t.Tensor):
    """Translate grid according to a 9-direction scheme with probabilities based on the
    diffusion coefficient <D>

    Args:
        D (t.Tensor): Diffusion coefficient
    """
    grid = grid.unsqueeze(0)
    p_sides = t.log(D * (t.sqrt(t.tensor(2.0)) - 1) * t.sqrt(t.tensor(2.0)) / 4)
    p_diag = t.log(D * (t.sqrt(t.tensor(2.0)) - 1) / 4)
    p_center = t.log(1 - D)

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

    kernel = gumbel_softmax(
        logits=kernel_logits.flatten(), tau=1.0, hard=True
    ).unflatten(dim=0, sizes=(1, 1, 3, 3))
    if grid.is_cuda:
        kernel = kernel.cuda()

    padded_grid = periodic_padding(grid).float()
    expanded_grid = t.unsqueeze(padded_grid, -1)
    transposed_grid = t.permute(expanded_grid, (0, 3, 1, 2))

    return conv2d(transposed_grid, kernel, stride=1, padding="valid").squeeze()


def excite_particles_STE(state: t.Tensor, N: int) -> Tuple[t.Tensor, t.Tensor]:
    """
    Returns the adjusted Lattice + auxiliary grid of "excited" particles E.
    """
    grid_dim = state.shape[-1]
    Xi = (
        uniform.Uniform(0, 1)
        .sample_n(2 * (grid_dim**2))
        .reshape(2, grid_dim, grid_dim)
    )
    if state.is_cuda:
        Xi = Xi.cuda()

    E = STEFunction.apply(state - N * Xi)
    # if a cell is filled, E has to be 1 at that cell
    E[state == N] += 1 - E[state == N]
    # if a cell is empty, E has to be 0 at that cell
    E[state == 0] *= 0

    return state - E, E


def accommodate_particles(grid: t.Tensor, E_A: t.Tensor, E_B: t.Tensor) -> t.Tensor:
    """
    Merges E into the current grid state.
    """
    grid[0] += E_A
    grid[1] += E_B
    return grid


def diffuse_STE(grid: t.Tensor, N: int, D_A, D_B) -> t.Tensor:
    grid, E = excite_particles_STE(grid, N)

    # translate the excited particles of the A species
    E_A = translate_gumbel(E[0], D_A)
    # translate the excited particles of the B species
    E_B = translate_gumbel(E[1], D_B)

    # accommodate the translated particles at their new positions
    return accommodate_particles(grid, E_A, E_B)
