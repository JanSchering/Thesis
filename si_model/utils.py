from typing import Callable
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
            start_idx: start_idx + steps_per_seq - 1, :, :, :
        ] = sequence.detach().numpy()[indexer]
    if shuffle:
        np.random.shuffle(chopped_set)
    return t.tensor(chopped_set)


def pre_process(data: t.Tensor):
    dset = data.numpy()
    mask = np.all(dset[:, 0] == 1, axis=(-1, -2))

    num_faulty_entries = mask[mask == True].shape[0]
    print(f"remove {num_faulty_entries} entries from the dataset")

    dset = np.delete(dset, mask, axis=0)
    dataset = t.from_numpy(dset)

    print(dataset.size())
    return dataset


def gaussian_pdf(mu: t.Tensor, sigma_sq: t.Tensor) -> Callable:
    """
    Produces a PDF for a Gaussian normal distribution N(<mu>, sqrt(<sigma_sq>)).
    mu: Mean of the distribution

    sigma_sq: variance of the distribution
    """
    # assume scalar mean
    assert mu.numel() == 1
    # assume scalar variance
    assert sigma_sq.numel() == 1

    def pdf(x: t.Tensor) -> t.Tensor:
        """
        Calculate the probability density at points <x> for a Gaussian distribution
        N(<mu>,<sigma>)

        x (torch.Tensor): The points to evaluate the density at.
        """
        return (1 / (t.sqrt(2 * t.tensor(np.pi)) * t.sqrt(sigma_sq))) * t.exp(
            -((x - mu) ** 2) / (2 * sigma_sq)
        )

    return pdf


def heaviside(x, k):
    return 1 / (1 + t.exp(-2 * k * x))
