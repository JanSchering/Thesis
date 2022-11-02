import torch as t
from risk_conv import risk_convolution2D


def calculate_spread_likelihood(batch: t.Tensor, beta: t.Tensor):
    """
    calculates the likelihood of each cell in the batch transitioning to an
    activated state at the next time step:
    1-(1-ß)^N_k(n)
    """
    risk_conv = risk_convolution2D(batch).squeeze(1)
    return 1.0 - (1.0 - beta) ** risk_conv


if __name__ == "__main__":
    print("testing the spread likelihood function...")
    size = 9
    batch_size = 2
    beta = 0.35

    batch = t.zeros((batch_size, size, size))
    batch[:, size // 2, size // 2] = 1

    batch[0, 0, 0] = 1
    batch[0, 0, 1] = 1
    batch[0, 0, 2] = 1
    batch[0, 1, 0] = 1
    batch[0, 1, 2] = 1
    batch[0, 2, 0] = 1
    batch[0, 2, 1] = 1
    batch[0, 2, 2] = 1

    likelihoods = calculate_spread_likelihood(batch, beta)

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

    print("all tests passed successfully.")
