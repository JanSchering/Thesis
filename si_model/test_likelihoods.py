import torch as t
from likelihoods import spread_likelihood, transition_likelihood


def init_test_grids(size, batch_size):
    batch = t.zeros((batch_size, size, size))
    batch[:, size // 2, size // 2] = 1
    return batch


def test_spread_likelihood_1():
    size = 9
    batch_size = 2
    beta = 0.35

    batch = init_test_grids(size, batch_size)
    batch[0, 0, 0] = 1
    batch[0, 0, 1] = 1
    batch[0, 0, 2] = 1
    batch[0, 1, 0] = 1
    batch[0, 1, 2] = 1
    batch[0, 2, 0] = 1
    batch[0, 2, 1] = 1
    batch[0, 2, 2] = 1

    likelihoods = spread_likelihood(batch, beta)

    def likelihood(beta, n):
        return 1 - (1 - beta) ** n

    assert "{:0.4f}".format(likelihoods[0, 0, 0]) == "{:0.4f}".format(
        likelihood(beta, 2)
    )
    assert "{:0.4f}".format(likelihoods[0, 0, 1]) == "{:0.4f}".format(
        likelihood(beta, 4)
    )
    assert "{:0.4f}".format(likelihoods[0, 0, 2]) == "{:0.4f}".format(
        likelihood(beta, 2)
    )
    assert "{:0.4f}".format(likelihoods[0, 1, 0]) == "{:0.4f}".format(
        likelihood(beta, 4)
    )
    assert "{:0.4f}".format(likelihoods[0, 1, 1]) == "{:0.4f}".format(
        likelihood(beta, 8)
    )
    assert "{:0.4f}".format(likelihoods[0, -1, 1]) == "{:0.4f}".format(
        likelihood(beta, 3)
    )
