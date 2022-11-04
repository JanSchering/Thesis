import torch as t
from periodic_padding import periodic_padding


def test_periodic_padding1():
    size = 9
    batch_size = 2

    batch = t.zeros((batch_size, size, size))

    batch[0, 0, 0] = 1
    batch[0, 0, 1] = 1
    batch[0, 0, 2] = 1
    batch[0, 1, 0] = 1
    batch[0, 1, 2] = 1
    batch[0, 2, 0] = 1
    batch[0, 2, 1] = 1
    batch[0, 2, 2] = 1

    padded_batch = periodic_padding(batch)

    # Torus wrapping around the right edge providing active padding
    assert padded_batch[0, 1, -1] == 1.0
    assert padded_batch[0, 2, -1] == 1.0
    assert padded_batch[0, 3, -1] == 1.0
    # Below and above should not be active
    assert padded_batch[0, 0, -1] == 0.0
    assert padded_batch[0, 4, -1] == 0.0
    # Torus wrapping around the bottom providing active cells
    assert padded_batch[0, -1, 1] == 1.0
    assert padded_batch[0, -1, 2] == 1.0
    assert padded_batch[0, -1, 1] == 1.0
    # Left and right should not be active
    assert padded_batch[0, -1, 0] == 0.0
    assert padded_batch[0, -1, 4] == 0.0
    # Torus wrapping around the left edge should be inactive
    assert padded_batch[0, 0, 0] == 0.0
    assert padded_batch[0, 1, 0] == 0.0
    assert padded_batch[0, 2, 0] == 0.0
    assert padded_batch[0, 3, 0] == 0.0
    assert padded_batch[0, 4, 0] == 0.0
    # Bottom right corner should be active
    assert padded_batch[0, -1, -1] == 1.0
