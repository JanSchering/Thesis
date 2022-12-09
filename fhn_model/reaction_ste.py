from typing import Callable
import torch as t
from torch.distributions import uniform
from ste_func import STEFunction


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
