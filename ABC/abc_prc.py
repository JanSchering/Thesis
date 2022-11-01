from typing import List, Callable, Tuple
import torch as t
from torch.distributions import normal
from tqdm import tqdm

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
    theta_hist = [thetas.detach().numpy()]
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
        theta_hist.append(thetas.detach().numpy())
    return thetas, theta_hist
