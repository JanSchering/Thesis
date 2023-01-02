import torch as t
from torch.distributions import uniform
from diffusion import diffuse
from diffusion_gumbel import diffuse_STE
import numpy as np
from tqdm import tqdm

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
    react=False,
    gamma=None,
    k1=None,
    k1_bar=None,
    k2=None,
    k2_bar=None,
    k3=None,
    k3_bar=None,
    num_reaction_channels=None,
):
    grid = grid.float()
    sequence = t.zeros((num_steps, *grid.shape))

    for i in tqdm(range(num_steps)):
        sequence[i] = grid.detach().clone()
        if use_diffusion:
            grid = diffuse(grid, N, DA, DB)

    return sequence


# ------------------------------
# Simple distance functions
# -----------------------------


def MSE(X, Y):
    return t.mean(t.sum((X - Y) ** 2, dim=((1, 2, 3))))


def dist(X, D1, D2):
    mse_D1 = MSE(X, D1)
    # print(mse_D1)
    mse_D2 = MSE(X, D2)
    # print(mse_D2)
    return (mse_D1 - mse_D2) ** 2


def dim_specific_MSE(X, Y, dim: int):
    return t.mean(t.sum((X[:, dim] - Y[:, dim]) ** 2, dim=((1, 2))))


def dim_specific_dist(X, D1, D2, dim: int):
    return (dim_specific_MSE(X, D1, dim) - dim_specific_MSE(X, D2, dim)) ** 2
