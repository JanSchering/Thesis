from typing import List, Callable, Tuple
import torch as t
from torch.distributions import normal, uniform
from tqdm import tqdm
import matplotlib.pyplot as plt
import random

Sampler = Callable[[], t.Tensor]
Input = t.Tensor
Observations = t.Tensor
Parameters = t.Tensor
Output = t.Tensor


def ABC_PRC(
    X: t.Tensor,
    Y_obs: t.Tensor,
    model: Callable[[Input, Parameters], Output],
    mu_1: Sampler,
    kernel: Callable,
    distance_func: Callable,
    epsilons: List[float],
    N: int,
) -> Tuple[List[float], List[List[float]]]:
    thetas = mu_1(1000)
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
    # Define the initial sampler
    mu_1 = uniform.Uniform(t.tensor(-10.0), t.tensor(10.0)).sample_n

    def distance_func(X: t.tensor, Y_sim: t.tensor, Y_obs: t.tensor) -> t.Tensor:
        if random.random() > 0.5:
            return t.abs(t.mean(Y_sim))
        else:
            return t.abs(Y_sim[0])

    # according to the paper, the dataset should be 100 samples drawn from N(theta,1)
    def model(X: t.tensor, theta: t.tensor) -> t.Tensor:
        n = normal.Normal(theta, 1.0)
        return n.sample_n(100)

    def kernel(theta: t.tensor) -> t.Tensor:
        return normal.Normal(theta, t.tensor(1.0)).sample()

    N = 1000
    epsilons = [2, 0.5, 0.025]

    thetas, theta_hist = ABC_PRC(
        None, None, model, mu_1, kernel, distance_func, epsilons, N
    )

    fig, axs = plt.subplots(2, 2, sharex=True, sharey=True)

    axs[0, 0].hist(theta_hist[0], density=True, bins=30)
    axs[0, 0].set(ylabel="density", title="Population 0")
    axs[0, 1].hist(theta_hist[1], density=True, bins=30)
    axs[0, 1].set(title="Population 1")
    axs[1, 0].hist(theta_hist[2], density=True, bins=30)
    axs[1, 0].set(xlabel="theta", ylabel="density", title="Population 2")
    axs[1, 1].hist(theta_hist[3], density=True, bins=30)
    axs[1, 1].set(xlabel="theta", title="Population 3")

    plt.xlim(-3, 3)
    plt.show()
