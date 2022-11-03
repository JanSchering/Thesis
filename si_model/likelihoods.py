import torch as t
from risk_conv import risk_convolution2D


def spread_likelihood(batch: t.Tensor, beta: t.Tensor):
    """
    calculates the likelihood of each cell in the batch transitioning to an
    activated state at the next time step:
    1-(1-ß)^N_k(n)
    """
    risk_conv = risk_convolution2D(batch).squeeze(1)
    return 1.0 - (1.0 - beta) ** risk_conv


def transition_likelihood(
    spread_likelihood: t.Tensor, x_t: t.Tensor, x_tt: t.Tensor
) -> t.Tensor:
    """
    Returns the cell-wise likelihood of transitioning from state <x_t> to state <x_tt> given a
    spread-likelihood matrix <spread_likelihood>.

    spread_likelihood:  The cell-wise likelihood for becoming/being infected at the next time step.
    x_t:                The input state (time t).
    x_tt:               The output stateof (time t+1).
    """
    return (1 - x_t) * (1 - x_tt) * (1 - spread_likelihood) + (
        x_tt * spread_likelihood * (1 - x_t) + x_t
    )


def total_likelihood(transition_likelihood: t.Tensor) -> t.Tensor:
    """
    Calculate the element-wise product along the spatial dimensions (-1, -2) of the matrix.

    x_t (np.ndarray): The previous state of the grid (time step t).
    x_tt (np.ndarray): The current state of the grid (time step t+1).
    beta (float): The diffusion coefficient of the PCA.
    """
    return t.sum(t.prod(t.prod(transition_likelihood, dim=-1), dim=-1))
