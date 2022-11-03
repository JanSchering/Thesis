import torch as t
import numpy as np
from utils import chop_and_shuffle_data


def test_chop_and_shuffle():
    test_data = t.arange(24).reshape(2, 3, 2, 2)
    n_sequences, steps_per_seq, grid_height, grid_width = test_data.shape
    chopped = chop_and_shuffle_data(test_data, shuffle=False)
    assert chopped.shape == (
        n_sequences * (steps_per_seq - 1),
        2,
        grid_height,
        grid_width,
    )
    assert np.all(chopped[0, 0].detach().numpy() == np.arange(4).reshape(2, 2))
    assert np.all(chopped[0, 1].detach().numpy() == (np.arange(4) + 4).reshape(2, 2))
    assert np.all(chopped[1, 0].detach().numpy() == (np.arange(4) + 4).reshape(2, 2))
    assert np.all(chopped[1, 1].detach().numpy() == (np.arange(4) + 8).reshape(2, 2))
