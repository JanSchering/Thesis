import torch as t
from torch.distributions import uniform
from diffusion import diffuse
from reaction import rho, p1, p2, p3, p4, p5, p6
from diffusion_gumbel import diffuse_STE
import numpy as np
from tqdm import tqdm
import os
from os import path, getcwd
import time
import matplotlib.pyplot as plt

# ------------------------------
# Helper to create a dataset from a sequence
# -----------------------------


def chop_and_shuffle_data(sequence, shuffle=True):
    """
    Chop the training data into a set of state transitions and shuffle the resulting set.

    sequences (np.ndarray): matrix of shape (n_sequences, steps_per_seq, grid_height, grid_width)
    """
    device = t.device("cuda" if t.cuda.is_available() else "cpu")
    steps_per_seq, _, grid_height, grid_width = sequence.shape
    # each transition consists of 2 states
    indexer = np.arange(2)[None, :] + np.arange(steps_per_seq - 1)[:, None]
    chopped_set = np.zeros([(steps_per_seq - 1), 2, 2, grid_height, grid_width])
    chopped_set = sequence.detach().numpy()[indexer]
    if shuffle:
        np.random.shuffle(chopped_set)
    return t.tensor(chopped_set, device=device)


# ------------------------------
# Helper to generate sequences
# -----------------------------


def generate_sequence(
    grid,
    num_steps,
    N,
    use_diffusion=True,
    DA=None,
    DB=None,
    use_reaction=False,
    gamma=None,
    k1=None,
    k1_bar=None,
    k2=None,
    k2_bar=None,
    k3=None,
    k3_bar=None,
    create_vis=False,
    save_steps=False,
    create_seq=True,
    im_path=None,
):
    grid = grid.float()
    timestamp = str(time.time())
    if create_vis:
        if not im_path:
            im_path = path.join(getcwd(), "vis", timestamp)
        os.mkdir(im_path)
    if save_steps:
        data_path = path.join(getcwd(), "data", timestamp)
        os.mkdir(data_path)
        os.mkdir(path.join(data_path, "batch_0"))
        t.save(grid, path.join(data_path, "batch_0", "0.pt"))
        batch_counter = 0
    if create_seq:
        sequence = t.zeros((num_steps, *grid.shape))
        sequence[0] = grid.detach().clone()

    for i in tqdm(range(num_steps)):
        if use_diffusion:
            grid = diffuse(grid, N, DA, DB)
        if use_reaction:
            rate_coefficients = t.tensor([k1, k1_bar, k2, k2_bar, k3, k3_bar])
            probability_funcs = [p1, p2, p3, p4, p5, p6]
            grid = rho(
                grid,
                N,
                gamma,
                rate_coefficients,
                probability_funcs,
                num_reaction_channels=6,
            )
        if create_seq:
            sequence[i] = grid.detach().clone()
        if create_vis and i % 100 == 0:
            fig, axs = plt.subplots(1, 2)
            axs[0].imshow(
                grid[0].cpu(),
                cmap="Greys",
                interpolation="nearest",
                vmin=0,
                vmax=N,
            )
            axs[0].set_title("A species")
            axs[0].axis("off")

            axs[1].imshow(
                grid[1].cpu(),
                cmap="Greys",
                interpolation="nearest",
                vmin=0,
                vmax=N,
            )
            axs[1].set_title("B species")
            axs[1].axis("off")

            plt.savefig(
                path.join(im_path, f"{i}.png"), bbox_inches="tight", pad_inches=0
            )
            plt.close(fig)
        if save_steps:
            if (i + 1) % 100 == 0:
                batch_counter += 1
                os.mkdir(path.join(data_path, f"batch_{batch_counter}"))
            t.save(
                grid,
                path.join(data_path, f"batch_{batch_counter}", f"{(i+1) % 100}.pt"),
            )

    if create_seq:
        return sequence


# ------------------------------
# Simple distance functions
# -----------------------------


def dim_specific_MSE(X, Y, dim: int):
    return t.mean(t.sum((X[:, dim] - Y[:, dim]) ** 2, dim=((1, 2))))


def dim_specific_dist(X, D1, D2, dim: int):
    return (dim_specific_MSE(X, D1, dim) - dim_specific_MSE(X, D2, dim)) ** 2


def dataset_dist(X, D1, D2):
    species_A_dist = dim_specific_dist(X, D1, D2, dim=0)
    species_B_dist = dim_specific_dist(X, D1, D2, dim=1)
    return species_A_dist + species_B_dist
