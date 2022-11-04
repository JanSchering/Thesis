# %%
import sys

sys.path.insert(0, "../abc")
import torch as t
import numpy as np
import matplotlib.pyplot as plt
from model import init_grids, model
from vis_utils import visualize_batch, visualize_sequence
from utils import chop_and_shuffle_data, heaviside
from likelihoods import spread_likelihood

# %%
def model(grids: t.Tensor, beta: t.Tensor) -> t.Tensor:
    healthy_mask = grids == 0

    likelihoods = spread_likelihood(grids, beta)
    infection_update = t.rand(*grids.shape) <= likelihoods * healthy_mask
    return grids + infection_update.detach().clone().numpy().astype(int)


def modelv2(grids: t.Tensor, beta: t.Tensor) -> t.Tensor:
    healthy_mask = 1 - grids
    likelihoods = spread_likelihood(grids, beta)
    residuals = likelihoods - t.rand(*grids.shape)
    update_mask = heaviside(residuals, k=20) * healthy_mask
    return grids + update_mask


def S1(X: t.Tensor, Y: t.Tensor) -> t.Tensor:
    return t.sum((1 - X) * Y, axis=(-1, -2))


def calc_distance(X, Y_sim: t.Tensor, Y_obs: t.Tensor) -> float:
    # get statistics of the simulated set
    s_sim = S1(X, Y_sim)
    # get statistics of the observed set
    s_obs = S1(X, Y_obs)
    # calculate the mean ratio
    return (1 - t.mean(s_sim / (s_obs + 1))) ** 2


#%%
x_t = t.zeros((1, 3, 3))
x_t[0, 1, 1] = 1

test = modelv2(x_t, t.tensor(0.2))
plt.imshow(test[0], cmap="Greys", interpolation="nearest", vmin=0, vmax=1)
# %%
test2 = model(x_t, t.tensor(0.6))
plt.imshow(test2[0], cmap="Greys", interpolation="nearest", vmin=0, vmax=1)

# %%
calc_distance(x_t, test, test2)
# %%
beta = t.tensor(0.2)
beta.requires_grad_()
test = modelv2(x_t, beta)
distance = calc_distance(x_t, test, test2)
gradient = t.autograd.grad(distance, beta)
gradient
# %%
betas = t.linspace(0.0, 1.0, 1000)
grads = []
for beta in betas:
    beta.requires_grad_()
    x_tt = modelv2(x_t, beta)
    dist = calc_distance(x_t, x_tt, test2)
    gradient = t.autograd.grad(dist, beta)[0]
    grads.append(gradient.detach().clone())
# %%
plt.scatter(betas, grads)
# %%
from math import factorial


def savgol_filter(y, window_size, order, deriv=0, rate=1):
    try:
        window_size = np.abs(int(window_size))
        order = np.abs(int(order))
    except ValueError:
        raise ValueError("window_size and order have to be of type int")
    if window_size % 2 != 1 or window_size < 1:
        raise TypeError("window_size size must be a positive odd number")
    if window_size < order + 2:
        raise TypeError("window_size is too small for the polynomials order")
    order_range = range(order + 1)
    half_window = (window_size - 1) // 2
    # precompute coefficients
    b = np.mat(
        [[k**i for i in order_range] for k in range(-half_window, half_window + 1)]
    )
    m = np.linalg.pinv(b).A[deriv] * rate**deriv * factorial(deriv)
    # pad the signal at the extremes with
    # values taken from the signal itself
    firstvals = y[0] - np.abs(y[1 : half_window + 1][::-1] - y[0])
    lastvals = y[-1] + np.abs(y[-half_window - 1 : -1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    return np.convolve(m[::-1], y, mode="valid")


# %%

grad_list = np.array([tnsor.numpy() for tnsor in grads])

# %%
plt.plot(betas, grads, label="gradient w.r.t beta")
smoothed = savgol_filter(grad_list, 101, 3)
plt.plot(betas, smoothed, label="smoothed gradient")
plt.xlabel("beta")
plt.ylabel("gradient")
plt.legend()
plt.show()
# %%
def update(
    beta: t.Tensor, x_t: t.Tensor, x_tt: t.Tensor, learning_rate: float
) -> t.Tensor:
    beta.requires_grad_()
    x_tt = modelv2(x_t, beta)
    dist = calc_distance(x_t, x_tt, test2)
    gradient = t.autograd.grad(dist, beta)[0]
    return beta - learning_rate * gradient


beta = t.tensor(0.2)
betas = []
for i in range(100):
    beta = update(beta, x_t, test2, 0.01)
    betas.append(beta.detach())

plt.plot(np.arange(len(betas)), t.tensor(betas))

# %%
