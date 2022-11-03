import torch as t
import numpy as np


def chop_and_shuffle_data(sequences, shuffle=True):
    """
    Chop the training data into a set of state transitions and shuffle the resulting set.

    sequences (np.ndarray): matrix of shape (n_sequences, steps_per_seq, grid_height, grid_width)
    """
    n_sequences, steps_per_seq, grid_height, grid_width = sequences.shape
    # each transition consists of 2 states
    indexer = np.arange(2)[None, :] + np.arange(steps_per_seq - 1)[:, None]
    chopped_set = np.zeros(
        [(steps_per_seq - 1) * n_sequences, 2, grid_height, grid_width]
    )
    for idx, sequence in enumerate(sequences):
        start_idx = idx * (steps_per_seq - 1)
        chopped_set[
            start_idx : start_idx + steps_per_seq - 1, :, :, :
        ] = sequence.detach().numpy()[indexer]
    if shuffle:
        np.random.shuffle(chopped_set)
    return t.tensor(chopped_set)


def heaviside(x, k):
    return 1 / (1 + t.exp(-2 * k * x))
