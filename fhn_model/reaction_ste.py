from typing import Callable, List
import torch as t
from torch.distributions import uniform
from ste_func import STEFunction


def sigma_STE(
    cells: t.Tensor,
    channel: int,
    N: int,
    gamma: float,
    rate_coefficient: float,
    probability_func: Callable,
):
    num_cells = cells.shape[-1]
    # use the probability function to get the reaction prob. for each of the cells
    reaction_probs = probability_func(cells, N, gamma, rate_coefficient)
    # randomly sample a threshold value for each cell to compare the prob. against
    thresholds = uniform.Uniform(0, 1).sample_n(num_cells)
    if cells.is_cuda:
        thresholds = thresholds.cuda()
    # for channel 0 and 2, the number of particles for species A should increase according to the prob.
    if channel == 0 or channel == 2:
        cells[0] += STEFunction.apply(reaction_probs - thresholds)
    # for channel 1 and 3, the number of particles for species A should decrease according to the prob.
    elif channel == 1 or channel == 3:
        cells[0] -= STEFunction.apply(reaction_probs - thresholds)
    # for channel 4, the number of particles for species B should increase according to the prob.
    elif channel == 4:
        cells[1] += STEFunction.apply(reaction_probs - thresholds)
    # for channel 5, the number of particles for species B should decrease according to the prob.
    elif channel == 5:
        cells[1] -= STEFunction.apply(reaction_probs - thresholds)
    return cells


def rho_STE(
    batch: t.Tensor,
    N: int,
    gamma: float,
    rate_coefficients: List[float],
    probability_funcs: List[Callable],
    num_reaction_channels: int,
):
    batch_size, grids_per_el, height, width = batch.shape
    # for each cell on the lattice, choose a reaction channel
    channels = t.randint(high=num_reaction_channels, size=(batch_size, height, width))
    if batch.is_cuda:
        channels = channels.cuda()
    # iterate over each reaction channel
    for channel_idx in range(num_reaction_channels):
        # get the reaction probability function of the current channel
        p_func = probability_funcs[channel_idx]
        # get the rate coefficient of the current channel
        rate_coefficient = rate_coefficients[channel_idx]
        # mask out all cells that use a different reaction channel
        channel_mask = channels == channel_idx
        # move the batch dimension in to match the masking
        batch = batch.permute(1, 0, 2, 3)
        cells = batch[:, channel_mask].detach().clone().float()
        # run the local reaction operator on each cell and update their state
        batch[:, channel_mask] = sigma_STE(
            cells, channel_idx, N, gamma, rate_coefficient, p_func
        )
        # move the batch back to its original shape
        batch = batch.permute(1, 0, 2, 3)
    return batch
