from typing import Callable, List
import torch as t
from torch.distributions import uniform


def p1(cells: t.Tensor, N: int, gamma: float, k1: float) -> t.Tensor:
    n = cells[0]
    k1_bar = k1 / ((N - 1) * (N - 2))
    return gamma * k1_bar * n * (n - 1) * (N - n)


def p2(cells: t.Tensor, N: int, gamma: float, k1_star: float) -> t.Tensor:
    n = cells[0]
    k1_star_bar = k1_star / ((N - 1) * (N - 2))
    return gamma * k1_star_bar * n * (N - n) * (N - 1 - n)


def p3(cells: t.Tensor, N: int, gamma: float, k2: float) -> t.Tensor:
    n = cells[0]
    m = cells[1]
    k2_bar = k2 / N
    return gamma * k2_bar * (N - n) * m


def p4(cells: t.Tensor, N: int, gamma: float, k2_star: float) -> t.Tensor:
    n = cells[0]
    m = cells[1]
    k2_star_bar = k2_star / N
    return gamma * k2_star_bar * n * (N - m)


def p5(cells: t.Tensor, N: int, gamma: float, k3: float) -> t.Tensor:
    n = cells[0]
    m = cells[1]
    k3_bar = k3 / N
    return gamma * k3_bar * (N - n) * (N - m)


def p6(cells: t.Tensor, N: int, gamma: float, k3_star: float) -> t.Tensor:
    n = cells[0]
    m = cells[1]
    k3_star_bar = k3_star / N
    return gamma * k3_star_bar * n * m


def sigma(
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
    # edge values to use for the heaviside at x=0
    edge_vals = t.zeros(num_cells)
    if cells.is_cuda:
        thresholds = thresholds.cuda()
        edge_vals = edge_vals.cuda()
    # for channel 0 and 2, the number of particles for species A should increase according to the prob.
    if channel == 0 or channel == 2:
        cells[0] += t.heaviside(reaction_probs - thresholds, values=edge_vals)
    # for channel 1 and 3, the number of particles for species A should decrease according to the prob.
    elif channel == 1 or channel == 3:
        cells[0] -= t.heaviside(reaction_probs - thresholds, values=edge_vals)
    # for channel 4, the number of particles for species B should increase according to the prob.
    elif channel == 4:
        cells[1] += t.heaviside(reaction_probs - thresholds, values=edge_vals)
    # for channel 5, the number of particles for species B should decrease according to the prob.
    elif channel == 5:
        cells[1] -= t.heaviside(reaction_probs - thresholds, values=edge_vals)
    return cells


def rho(
    grid: t.Tensor,
    N: int,
    gamma: float,
    rate_coefficients: List[float],
    probability_funcs: List[Callable],
    num_reaction_channels: int,
):
    grid_size = grid.shape[-1]
    # for each cell on the lattice, choose a reaction channel
    channels = t.randint_like(t.zeros(grid_size, grid_size), high=num_reaction_channels)
    if grid.is_cuda:
        channels = channels.cuda()
    # iterate over each reaction channel 
    for channel_idx in range(num_reaction_channels):
        # get the reaction probability function of the current channel
        p_func = probability_funcs[channel_idx]
        # get the rate coefficient of the current channel 
        rate_coefficient = rate_coefficients[channel_idx]
        # mask out all cells that use a different reaction channel
        channel_mask = channels == channel_idx
        cells = grid[:, channel_mask].detach().clone()
        # run the local reaction operator on each cell and update their state 
        grid[:, channel_mask] = sigma(
            cells, channel_idx, N, gamma, rate_coefficient, p_func
        )
    return grid
