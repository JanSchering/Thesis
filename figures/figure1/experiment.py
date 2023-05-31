from distance_funcs import mean_sq_distance
from likelihoods import spread_likelihood, transition_likelihood, neg_log_likelihood
from utils import chop_and_shuffle_data, pre_process
from model import init_grids, model
from tqdm import tqdm
import imageio
from matplotlib import ticker
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.distributions import uniform, normal
import torch as t
from typing import Tuple
import sys

sys.path.insert(0, "../../si_model")


###
#   Set a random seed for torch and numpy
###

t.manual_seed(0)
np.random.seed(0)

###
#   Define the differentiable model
##


def model_gumbel(grids: t.Tensor, beta: t.Tensor) -> t.Tensor:
    healthy_mask = 1 - grids
    # calculate the likelihood of spread
    likelihoods = spread_likelihood(grids, beta)
    # print(likelihoods.flatten().unsqueeze(1).shape)
    flattened = likelihoods.flatten().unsqueeze(0)
    #print(t.concat([t.log(flattened), t.log(1-flattened)], dim=0).T.shape)
    update = t.nn.functional.gumbel_softmax(
        t.concat([t.log(flattened), t.log(1-flattened)]).T, hard=True)
    # print(update.shape)
    unflattened = update[:, 0].unflatten(
        dim=0, sizes=grids.shape) * healthy_mask
    # apply the update to the current state
    return grids + unflattened


###
# Define the Dataset class
###

class CustomDataset(Dataset):
    def __init__(self, beta_mu=None, beta_sigma=None, num_sequences=None, steps_per_sequence=None, grid_size=None, load_path=None):
        if load_path:
            self.data = t.load(load_path)
        else:
            sequences = t.zeros(
                (num_sequences, steps_per_sequence, grid_size, grid_size))
            self.betas = []

            for seq_idx in tqdm(range(num_sequences)):
                sequence = init_grids(grid_size, 1)
                for step_idx in range(steps_per_sequence):
                    beta = t.clip(normal.Normal(
                        beta_mu, beta_sigma).sample(), 0, 1)
                    sequences[seq_idx, step_idx, :, :] = sequence
                    sequence = model_gumbel(sequence, beta)
                    self.betas.append(beta.detach().cpu())

            dataset = chop_and_shuffle_data(sequences, shuffle=False)
            self.data = pre_process(dataset)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return self.data[idx], 0


###
#   Create Experiment Data
###

beta_mu = 0.5
beta_sigma = 0.1
num_sequences = 100
steps_per_sequence = 30
grid_size = 64

training_data = CustomDataset(beta_mu=beta_mu, beta_sigma=beta_sigma,
                              num_sequences=num_sequences, steps_per_sequence=steps_per_sequence, grid_size=grid_size)
training_loader = DataLoader(training_data, batch_size=32, shuffle=True)
t.save(training_data.data, "./data/training_set.pt")

###
#   Visualize sample sequence for the given mu, sigma
###

grids = init_grids(grid_size, 1)
for i in range(steps_per_sequence):
    fig, axs = plt.subplots(figsize=(3, 3), dpi=300)
    axs.imshow(grids[0], cmap="Greys", interpolation="nearest", vmin=0, vmax=1)
    axs.get_xaxis().set_visible(False)
    axs.get_yaxis().set_visible(False)
    fig.tight_layout()
    fig.savefig(f"./images/sequence/{i}.png")
    plt.close(fig)
    beta = t.clip(normal.Normal(beta_mu, beta_sigma).sample(), 0, 1)
    grids = model_gumbel(grids, beta)


images = []
for i in range(steps_per_sequence):
    images.append(imageio.imread(f"./images/sequence/{i}.png"))
imageio.mimsave(f"./gif/{beta_mu}_{beta_sigma}.gif", images)

###
#   Define Minibatch training loop
###


def train_loop(dataloader, optimizer):
    size = len(dataloader.dataset)
    betas = []
    distances = []
    for batch, (transitions, _) in enumerate(dataloader):
        X = transitions[:, 0]
        Y_obs = transitions[:, 1]
        # Compute prediction and loss
        Y_sim = model_gumbel(X, beta)
        dist = mean_sq_distance(X, Y_sim, Y_obs)
        distances.append(dist.detach().clone().cpu().numpy())
        # Backpropagation
        optimizer.zero_grad()
        dist.backward()
        optimizer.step()
        betas.append(beta.detach().clone().cpu().numpy())
    return betas, distances

###
#   Run Sampling-based training
###


beta = uniform.Uniform(t.tensor(0.), t.tensor(1.)).sample()
beta.requires_grad_()
lr = 1e-7
epochs = 50
optimizer = t.optim.SGD([beta], lr=lr)

beta_hist = []
dist_hist = []
for epoch in tqdm(range(epochs)):
    betas, distances = train_loop(training_loader, optimizer)
    beta_hist = beta_hist + betas
    dist_hist = dist_hist + distances

np.save(
    f"./data/sample_based/betas_multibeta_mean{beta_mu}_sig{beta_sigma}.npy", np.array(beta_hist))
np.save(
    f"./data/sample_based/dist_multibeta_mean{beta_mu}_sig{beta_sigma}.npy", np.array(dist_hist))

###
#   Define update function using gradient of the likelihood
###


def update(beta: t.Tensor, X: t.Tensor, Y_obs: t.Tensor, optimizer) -> t.Tensor:
    # calculate the spread likelihood for the given beta
    L = spread_likelihood(X, beta)
    # calculate the transition likelihood matrix for the given beta
    P = transition_likelihood(L, X, Y_obs)
    # calculate the total likelihood of the transition
    neg_log_l = neg_log_likelihood(P)

    optimizer.zero_grad()
    neg_log_l.backward()
    optimizer.step()

    return beta, neg_log_l


def train_epoch(train_loader, beta: t.Tensor, optimizer):
    """
    Execute the training loop.
    """
    betas = []
    likelihoods = []
    for batch, (transitions, _) in enumerate(train_loader):
        X = transitions[:, 0]
        Y_obs = transitions[:, 1]
        beta, neg_log_likelihood = update(beta, X, Y_obs, optimizer)
        likelihoods.append(neg_log_likelihood.detach().clone().numpy())
        betas.append(beta.detach().clone().numpy())

    return betas, likelihoods

###
#   Run Likelihood-based training
###


beta = t.tensor(
    np.load(f"./data/sample_based/betas_multibeta_mean0.5_sig0.1.npy")[0])
beta.requires_grad_()
lr = 1e-7
epochs = 50
optimizer = t.optim.SGD([beta], lr=lr)

beta_hist = []
likelihood_hist = []
for epoch in tqdm(range(epochs)):
    betas, likelihoods = train_epoch(training_loader, beta, optimizer)
    beta_hist = beta_hist + betas
    likelihood_hist = likelihood_hist + likelihoods

np.save(
    f"./data/likelihood_based/betas_multibeta_mean{beta_mu}_sig{beta_sigma}.npy", np.array(beta_hist))
np.save(
    f"./data/likelihood_based/neg_log_l_multibeta_mean{beta_mu}_sig{beta_sigma}.npy", np.array(likelihood_hist))
