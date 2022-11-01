from typing import Callable, Tuple
import torch as t
from tqdm import tqdm
import random
from torch.distributions import uniform, normal
import matplotlib.pyplot as plt
from helpers import gaussian_pdf

Sampler = Callable[[], t.Tensor]
Input = t.Tensor
Observations = t.Tensor
Parameters = t.Tensor
Output = t.Tensor


def ABC_REJ(
    X: t.Tensor,
    Y_obs: t.Tensor,
    epsilon: float,
    prior: Sampler,
    model: Callable[[Input, Parameters], Output],
    distance_func: Callable[[Input, t.Tensor, t.Tensor], t.Tensor],
) -> Tuple(t.Tensor, float):
    """
    Perform approximate Bayesian computation through rejection sampling.

    X:              The Model inputs.
    Y_obs:          The dataset of observations.
    epsilon:        The rejection threshold.
    prior:          A prior distribution over the model parameters theta.
    model:          A model parameterized by a set of parameters theta that produces observations (given some input).
    distance_func:  Calculates the distance between two given dataset. Additionally takes the underlying input as a parameter.

    Returns (t.Tensor, float):
        theta: A sample from the posterior p(theta|Y_obs).
        num_stps: The amount of sampling steps necessary to produce the accepted sample.
    """
    num_steps = 1
    theta = None
    while True:
        # Sample theta from the prior distribution over theta 
        theta = prior()
        # Use the sampled theta to generate a dataset
        Y_sim = model(X, theta)
        # Calculate the distance between the observed dataset and the one generated with theta
        distance = distance_func(X, Y_sim, Y_obs)
        # If distance below threshold, accept the sample
        if distance <= epsilon:
            break
        # Else, continue the algorithm
        else:
            num_steps += 1
    return theta, num_steps


if __name__ == "__main__":
    """
    Test ABC-REJ on the toy example defined in https://www.pnas.org/doi/pdf/10.1073/pnas.0607208104
    """
    print("run ABC-REJ for simple toy example")

    # theta ~ U(-10,10)
    prior = uniform.Uniform(t.tensor([-10.0]), t.tensor([10.0]))

    # according to the paper, the dataset should be 100 samples drawn from N(theta,1)
    def model(X, theta):
        n = normal.Normal(theta, 1)
        return n.sample_n(100)

    # distance function as defined in the paper
    def calc_distance(X: t.Tensor, Y_sim: t.Tensor, Y_obs: t.Tensor) -> float:
        if random.random() > 0.5:
            return t.abs(t.mean(Y_sim))
        else:
            return t.abs(Y_sim[0])

    samples = []
    required_steps = []
    epsilon = epsilon = 0.025

    # In the paper, 1,000 accepted samples are drawn
    for i in tqdm(range(1000)):
        sample, num_steps = ABC_REJ(
            None, None, epsilon, prior.sample, model, calc_distance
        )
        samples.append(sample)
        required_steps.append(num_steps)

    print(
        f"average # steps for acceptance: {t.mean(t.tensor(required_steps, dtype=t.float))}"
    )

    # Visualize the results and compare to the expected posterior
    pdf_1 = gaussian_pdf(t.tensor(0), t.tensor(1 / 100))
    pdf_2 = gaussian_pdf(t.tensor(0), t.tensor(1))

    def posterior(theta: float) -> float:
        return (1 / 2) * pdf_1(theta) + (1 / 2) * pdf_2(theta)

    thetas = t.linspace(-3, 3, 100)

    n, bins, _ = plt.hist(t.tensor(samples), density=True, bins=30)
    plt.plot(thetas, [posterior(theta) for theta in thetas], color="red")
    plt.xlim(-4, 4)
    plt.show()
