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
    Chop the training data into a set of state transitions and shuffle the resulting set (sliding window approach).

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
    grid: t.Tensor,
    num_steps: int,
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
    """Helper function that simulates the FHN model on a provided starting state for a given
    number of steps.

    Args:
        grid: The FHN lattice, shape (2, lattice_size, lattice_size)
        num_steps: The number of simulation steps to perform
        N: The maximum occupation number of particles per lattice cell
        use_diffusion (bool, optional): Whether to perform diffusion in the simulation. Defaults to True.
        DA (t.Tensor, optional): The diffusion coefficient of the A species. Defaults to None.
        DB (t.Tensor, optional): The diffusion coefficient of the B species. Defaults to None.
        use_reaction (bool, optional): Whether to perform chemical reaction in the simulation. Defaults to False.
        gamma (t.Tensor, optional): Reaction time scale factor. Defaults to None.
        k1 t.Tensor, optional): Rate coefficient of the first reaction channel. Defaults to None.
        k1_bar (t.Tensor, optional): Rate coefficient of the second reaction channel. Defaults to None.
        k2 (t.Tensor, optional): Rate coefficient of the third reaction channel. Defaults to None.
        k2_bar (t.Tensor, optional): Rate coefficient of the 4th reaction channel. Defaults to None.
        k3 (t.Tensor, optional): Rate coefficient of the fifth reaction channel. Defaults to None.
        k3_bar (t.Tensor, optional): Rate coefficient of the sixth reaction channel. Defaults to None.
        create_vis (bool, optional): Whether to create an image of the lattice, performed every 100 steps. Defaults to False.
        save_steps (bool, optional): Whether to save the simulation steps, stored as ".pt" files. Defaults to False.
        create_seq (bool, optional): Whether to store the simulation process as a torch Tensor. Defaults to True.
        im_path (str, optional): The path to store the visualizations under. Defaults to None.

    Returns:
        t.Tensor | None: The sequence of transitions, if <create_seq>
    """
    # turn the dtype of the grid to float to avoid dtype issues (e.g. with grids auto-initialized to double)
    grid = grid.float()
    # timestamp of the simulation start, used for storage of simulation artifacts
    timestamp = str(time.time())
    # initialize the visualization folder if necessary
    if create_vis:
        if not im_path:
            im_path = path.join(getcwd(), "vis", timestamp)
        os.mkdir(im_path)
    # initialize the data folder if necessary
    if save_steps:
        data_path = path.join(getcwd(), "data", timestamp)
        os.mkdir(data_path)
        os.mkdir(path.join(data_path, "batch_0"))
        t.save(grid.cpu(), path.join(data_path, "batch_0", "0.pt"))
        batch_counter = 0
    # initialize the sequence tensor if necessary
    if create_seq:
        sequence = t.zeros((num_steps, *grid.shape))
        sequence[0] = grid.detach().clone()

    # perform <num_steps> of FHN simulation
    for i in tqdm(range(num_steps)):
        # perform a step of diffusion if necessary
        if use_diffusion:
            grid = diffuse(grid, N, DA, DB)
        # perform a step of chemical reaction if necessary
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
        # store the new state
        if create_seq:
            sequence[i] = grid.detach().clone()
        # create a visualization of the updated state
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
        # store the updated state as a torch data file
        if save_steps:
            if (i + 1) % 100 == 0:
                batch_counter += 1
                os.mkdir(path.join(data_path, f"batch_{batch_counter}"))
            t.save(
                grid.cpu(),
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
