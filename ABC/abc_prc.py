from typing import List, Callable, Tuple
import torch as t
from torch.distributions import normal, uniform
from tqdm import tqdm
import matplotlib.pyplot as plt
import random
from helpers import gaussian_pdf

Sampler = Callable[[], t.Tensor]
Input = t.Tensor
Observations = t.Tensor
Parameters = t.Tensor
Output = t.Tensor


def abc_prc(
    X: t.Tensor,
    Y_obs: t.Tensor,
    model: Callable[[Input, Parameters], Output],
    mu_1: Sampler,
    kernel: Callable[[t.Tensor], t.Tensor],
    distance_func: Callable[[Input, t.Tensor, t.Tensor], t.Tensor],
    epsilons: List[float],
    N: int,
) -> Tuple[t.Tensor, List[t.Tensor]]:
    """
    Perform approximate Bayesian computation through Sequential Monte Carlo with Particle
    Rejection Control as defined in https://www.pnas.org/doi/pdf/10.1073/pnas.0607208104.

    X:              The Model inputs.
    Y_obs:          The observed outputs (based on the inputs).
    model:          The Model.
    mu_1:           Initial sampling distribution over theta.
    kernel:         Markov transition kernel to move theta.
    distance_func:  Calculates the distance between two sets of observations (given a set of inputs).
    epsilons:       A list of increasingly stricter distance thresholds.
    """
    thetas = mu_1(N)
    theta_hist = [thetas.detach().clone().numpy()]
    for _, epsilon in enumerate(tqdm(epsilons)):
        for i in range(N):
            close_enough = False
            theta_star = theta_ss = Y_sim = None
            while not close_enough:
                # Sample from previous population
                theta_star = thetas[t.randint(len(thetas), (1,))]
                # Move the particle with a gaussian markov kernel
                theta_ss = kernel(theta_star)
                # sample Y_sim using beta**
                Y_sim = model(X, theta_ss)
                # check the distance
                close_enough = distance_func(X, Y_sim, Y_obs) <= epsilon
            thetas[i] = theta_ss.detach().clone()
        theta_hist.append(thetas.detach().clone().numpy())
    return thetas, theta_hist


if __name__ == "__main__":
    """
    Test ABC-PRC on the toy example defined in https://www.pnas.org/doi/pdf/10.1073/pnas.0607208104
    """
    # Define the initial sampler
    mu_1 = uniform.Uniform(t.tensor(-10.0), t.tensor(10.0)).sample_n

    def distance_func(X: t.Tensor, Y_sim: t.Tensor, Y_obs: t.Tensor) -> t.Tensor:
        if random.random() > 0.5:
            return t.abs(t.mean(Y_sim))
        else:
            return t.abs(Y_sim[0])

    # according to the paper, the dataset should be 100 samples drawn from N(theta,1)
    def model(X: t.Tensor, theta: t.Tensor) -> t.Tensor:
        n = normal.Normal(theta, 1.0)
        return n.sample_n(100)

    def kernel(theta: t.Tensor) -> t.Tensor:
        return normal.Normal(theta, t.tensor(1.0)).sample()

    N = 1000
    epsilons = [2, 0.5, 0.025]

    thetas, theta_hist = abc_prc(
        None, None, model, mu_1, kernel, distance_func, epsilons, N
    )

    # define the expected posterior for reference
    pdf_1 = gaussian_pdf(t.tensor(0), t.tensor(1 / 100))
    pdf_2 = gaussian_pdf(t.tensor(0), t.tensor(1))

    def posterior(theta: float) -> float:
        return (1 / 2) * pdf_1(theta) + (1 / 2) * pdf_2(theta)

    test_vals = t.linspace(-3, 3, 100)

    fig, axs = plt.subplots(2, 2, sharex=True, sharey=True)

    axs[0, 0].hist(theta_hist[0], density=True, bins=30)
    axs[0, 0].set(ylabel="density", title="Population 0")
    axs[0, 0].plot(
        test_vals, [posterior(test_val) for test_val in test_vals], color="red"
    )
    axs[0, 1].hist(theta_hist[1], density=True, bins=30)
    axs[0, 1].set(title="Population 1")
    axs[0, 1].plot(
        test_vals, [posterior(test_val) for test_val in test_vals], color="red"
    )
    axs[1, 0].hist(theta_hist[2], density=True, bins=30)
    axs[1, 0].set(xlabel="theta", ylabel="density", title="Population 2")
    axs[1, 0].plot(
        test_vals, [posterior(test_val) for test_val in test_vals], color="red"
    )
    axs[1, 1].hist(theta_hist[3], density=True, bins=30)
    axs[1, 1].set(xlabel="theta", title="Population 3")
    axs[1, 1].plot(
        test_vals, [posterior(test_val) for test_val in test_vals], color="red"
    )

    plt.xlim(-3, 3)
    plt.show()
