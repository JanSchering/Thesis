from typing import Callable, Tuple, List
import torch as t
import numpy as np
from tqdm import tqdm
import random
from torch.distributions import normal
from helpers import gaussian_pdf
import matplotlib.pyplot as plt

Sampler = Callable[[], t.Tensor]
Input = t.Tensor
Observations = t.Tensor
Parameters = t.Tensor
Output = t.Tensor


def abc_mcmc(
    X: t.Tensor,
    Y_obs: t.Tensor,
    theta: t.Tensor,
    model: Callable[[Input, Parameters], Output],
    distance_func: Callable[[Input, t.Tensor, t.Tensor], t.Tensor],
    calc_acceptance_rate: Callable,
    epsilon: float,
    q: Callable[[t.Tensor], t.Tensor],
    N: int,
) -> Tuple[List[np.ndarray], List[np.ndarray], List[float]]:
    """
    Perform approximate Bayesian computation via Markov Chain Monte Carlo sampling.

    X:                      The Model inputs.
    Y_obs:                  The observed outputs produced from the inputs.
    theta:                  The Model parameters.
    model:                  The Model.
    distance_func:          Calculates the distance between two given sets of observations (given the inputs).
    calc_acceptance_rate:   Calculates the (e.g. Metropolis-Hastings) acceptance rate for drawn samples to correct for bias.
    epsilon:                The upper threshold on the distance.
    q:                      Proposal distribution from which proposal thetas can be sampled.
    N:                      The number of MCMC iterations to run
    """
    # List of all accepted theta values
    thetas = [theta.detach().clone().cpu().numpy()]
    # History of all thetas
    theta_hist = [theta.detach().clone().cpu().numpy()]
    # Track how many steps are necessary to generate an accepted sample
    steps = 1
    # History of necessary steps per sample
    step_hist = []
    # run the MCMC algorithm for N steps
    for _ in tqdm(range(N)):
        # generate a proposal theta* ~ q(theta*|theta) from proposal density q
        theta_star = q(theta)
        if X.is_cuda:
            theta_star = theta_star.cuda()
        # use theta* to generate a dataset
        Y_sim = model(X, theta_star)
        # calculate the distance between the observed dataset and the dataset generated by the model using theta*
        dist = distance_func(X, Y_sim, Y_obs)
        # calculate the (Metropolis-Hastings) acceptance rate for theta*
        alpha = calc_acceptance_rate(theta_star, theta, epsilon, dist)
        # With probability <alpha>, accept theta*
        if random.random() <= alpha:
            # theta_i+1 = theta*
            theta = theta_star.detach().clone()
            # collect the sample
            thetas.append(theta_star.detach().clone().cpu().numpy())
            # track the number of steps
            step_hist.append(steps)
            steps = 1
        else:
            steps += 1
        theta_hist.append(theta.detach().clone().cpu().numpy())
    return thetas, theta_hist, step_hist


if __name__ == "__main__":
    """
    Test ABC-MCMC on the toy example defined in https://www.pnas.org/doi/pdf/10.1073/pnas.0607208104
    """
    sigma = t.tensor(0.15)
    epsilon = 0.025
    theta_init = t.tensor(0.0)
    N = 10_000

    def q(theta: t.Tensor):
        return normal.Normal(theta, sigma).sample()

    def gauss_likelihood(x: t.Tensor, y: t.Tensor) -> t.Tensor:
        pdf = gaussian_pdf(y, sigma**2)
        return pdf(x)

    def calc_distance(X: t.Tensor, Y_sim: t.Tensor, Y_obs: t.Tensor) -> t.Tensor:
        if random.random() > 0.5:
            return t.abs(t.mean(Y_sim))
        else:
            return t.abs(Y_sim[0])

    def uniform_likelihood(low: float, high: float, x: t.Tensor) -> float:
        if x > high or x < low:
            return 0
        else:
            return 1 / (high - low)

    def calc_alpha(
        theta_star: t.Tensor, theta: t.Tensor, epsilon: float, dist: t.Tensor
    ) -> t.Tensor:
        if dist <= epsilon and uniform_likelihood(-10, 10, theta_star) > 0:
            ratio = gauss_likelihood(theta, theta_star) / gauss_likelihood(
                theta_star, theta
            )
            return t.min(t.tensor([t.tensor(1.0), ratio]))
        else:
            return t.tensor(0.0)

    # according to the paper, the dataset should be 100 samples drawn from N(theta,1)
    def model(X, theta) -> t.Tensor:
        n = normal.Normal(theta, 1)
        return n.sample_n(100)

    thetas, theta_hist, step_hist = abc_mcmc(
        None, None, theta_init, model, calc_distance, calc_alpha, epsilon, q, N
    )

    print(f"accepted {len(thetas)} samples")

    # Visualize the results and compare to the expected posterior
    pdf_1 = gaussian_pdf(t.tensor(0), t.tensor(1 / 100))
    pdf_2 = gaussian_pdf(t.tensor(0), t.tensor(1))

    def posterior(theta: float) -> float:
        return (1 / 2) * pdf_1(theta) + (1 / 2) * pdf_2(theta)

    test_vals = t.linspace(-3, 3, 100)

    fig, axs = plt.subplots(2)
    axs[0].hist(thetas, density=True, bins=30)
    axs[0].plot(test_vals, [posterior(test_val) for test_val in test_vals], color="red")
    axs[1].plot(np.arange(len(theta_hist)), theta_hist)
    axs.flat[0].set_xlim(-4, 4)
    plt.show()
