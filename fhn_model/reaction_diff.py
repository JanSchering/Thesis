import torch as t
from torch.distributions import uniform
from typing import List, Callable
from ste_func import STEFunction
import numpy as np
from tqdm import tqdm


def p1_(batch: t.Tensor, N: int, gamma: float, k1: float) -> t.Tensor:
    n = batch[:, 0]
    # k1_bar = k1 / ((N - 1) * (N - 2))
    return gamma * (k1 / ((N - 1) * (N - 2))) * n * (n - 1) * (N - n)


def p2_(batch: t.Tensor, N: int, gamma: float, k1_star: float) -> t.Tensor:
    n = batch[:, 0]
    # k1_star_bar = k1_star / ((N - 1) * (N - 2))
    return gamma * (k1_star / ((N - 1) * (N - 2))) * n * (N - n) * (N - 1 - n)


def p3_(cells: t.Tensor, N: int, gamma: float, k2: float) -> t.Tensor:
    n = cells[:, 0]
    m = cells[:, 1]
    # k2_bar = k2 / N
    return gamma * (k2 / N) * (N - n) * m


def p4_(cells: t.Tensor, N: int, gamma: float, k2_star: float) -> t.Tensor:
    n = cells[:, 0]
    m = cells[:, 1]
    # k2_star_bar = k2_star / N
    return gamma * (k2_star / N) * n * (N - m)


def p5_(cells: t.Tensor, N: int, gamma: float, k3: float) -> t.Tensor:
    n = cells[:, 0]
    m = cells[:, 1]
    # k3_bar = k3 / N
    return gamma * (k3 / N) * (N - n) * (N - m)


def p6_(cells: t.Tensor, N: int, gamma: float, k3_star: float) -> t.Tensor:
    n = cells[:, 0]
    m = cells[:, 1]
    # k3_star_bar = k3_star / N
    return gamma * (k3_star / N) * n * m


def rho_STE(
    batch: t.Tensor,
    N: int,
    gamma: t.Tensor,
    k1: t.Tensor,
    k1_bar: t.Tensor,
    k2: t.Tensor,
    k2_bar: t.Tensor,
    k3: t.Tensor,
    k3_bar: t.Tensor,
):
    # set torch device
    device = t.device("cuda" if t.cuda.is_available() else "cpu")
    batch_size, grids_per_el, height, width = batch.shape
    # for each cell on the lattice, choose a reaction channel
    channels = t.randint(high=6, size=(batch_size, height, width), device=device)
    if batch.is_cuda:
        channels = channels.cuda()

    # --------------------------------------------------------------
    # handle reaction 1
    # --------------------------------------------------------------
    p_r1 = p1_(batch, N, gamma, k1)
    p_r1_expand = t.stack(
        (p_r1, t.zeros(batch_size, height, width, device=device)), dim=1
    )
    chnl_msk = t.stack(
        (channels == 0, t.zeros(batch_size, height, width, device=device)), dim=1
    )
    thresholds = uniform.Uniform(0, 1).sample((batch_size, 2, height, width)).to(device)
    # batch += STEFunction.apply(p_r1_expand - thresholds) * chnl_msk
    p_r1_res = STEFunction.apply(p_r1_expand - thresholds) * chnl_msk

    # --------------------------------------------------------------
    # handle reaction 2
    # --------------------------------------------------------------
    p_r2 = p2_(batch, N, gamma, k1_bar)
    p_r2_expand = t.stack(
        (p_r2, t.zeros(batch_size, height, width, device=device)), dim=1
    )
    chnl_msk = t.stack(
        (channels == 1, t.zeros(batch_size, height, width, device=device)), dim=1
    )
    thresholds = uniform.Uniform(0, 1).sample((batch_size, 2, height, width)).to(device)
    # batch -= STEFunction.apply(p_r2_expand - thresholds) * chnl_msk
    p_r2_res = STEFunction.apply(p_r2_expand - thresholds) * chnl_msk

    # --------------------------------------------------------------
    # handle reaction 3
    # --------------------------------------------------------------
    p_r3 = p3_(batch, N, gamma, k2)
    p_r3_expand = t.stack(
        (p_r3, t.zeros(batch_size, height, width, device=device)), dim=1
    )
    chnl_msk = t.stack(
        (channels == 2, t.zeros(batch_size, height, width, device=device)), dim=1
    )
    thresholds = uniform.Uniform(0, 1).sample((batch_size, 2, height, width)).to(device)
    # batch += STEFunction.apply(p_r3_expand - thresholds) * chnl_msk
    p_r3_res = STEFunction.apply(p_r3_expand - thresholds) * chnl_msk

    # ---------------------------------------------------------------
    # handle reaction 4
    # ---------------------------------------------------------------
    p_r4 = p4_(batch, N, gamma, k2_bar)
    p_r4_expand = t.stack(
        (p_r4, t.zeros(batch_size, height, width, device=device)), dim=1
    )
    chnl_msk = t.stack(
        (channels == 3, t.zeros(batch_size, height, width, device=device)), dim=1
    )
    thresholds = uniform.Uniform(0, 1).sample((batch_size, 2, height, width)).to(device)
    # batch -= STEFunction.apply(p_r4_expand - thresholds) * chnl_msk
    p_r4_res = STEFunction.apply(p_r4_expand - thresholds) * chnl_msk

    # ----------------------------------------------------------------
    # handle reaction 5
    # ----------------------------------------------------------------
    p_r5 = p5_(batch, N, gamma, k3)
    p_r5_expand = t.stack(
        (t.zeros(batch_size, height, width, device=device), p_r5), dim=1
    )
    chnl_msk = t.stack(
        (t.zeros(batch_size, height, width, device=device), channels == 4), dim=1
    )
    thresholds = uniform.Uniform(0, 1).sample((batch_size, 2, height, width)).to(device)
    # batch += STEFunction.apply(p_r5_expand - thresholds) * chnl_msk
    p_r5_res = STEFunction.apply(p_r5_expand - thresholds) * chnl_msk

    # -----------------------------------------------------------------
    # handle reaction 6
    # -----------------------------------------------------------------
    p_r6 = p6_(batch, N, gamma, k3_bar)
    p_r6_expand = t.stack(
        (t.zeros(batch_size, height, width, device=device), p_r6), dim=1
    )
    chnl_msk = t.stack(
        (t.zeros(batch_size, height, width, device=device), channels == 5), dim=1
    )
    # batch -= STEFunction.apply(p_r6_expand - thresholds) * chnl_msk
    p_r6_res = STEFunction.apply(p_r6_expand - thresholds) * chnl_msk

    return batch + p_r1_res - p_r2_res + p_r3_res - p_r4_res + p_r5_res - p_r6_res
