# ensure path to si modules
import sys

sys.path.insert(0, "../../si_model")
# import modules

from os import getcwd, path, listdir, mkdir

import torch as t
from torch.distributions import uniform, normal
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt

import imageio
from tqdm import tqdm

from model import init_grids, model
from utils import chop_and_shuffle_data, pre_process
from likelihoods import spread_likelihood, transition_likelihood, neg_log_likelihood

mpl.rcParams["axes.linewidth"] = 1
mpl.rcParams["text.usetex"] = True
mpl.rcParams["font.family"] = "Helvetica"
mpl.rcParams["font.size"] = 12
mpl.rcParams["text.latex.preamble"] = "\\usepackage{amssymb}"

###
#   Set a random seed for torch and numpy
###

t.manual_seed(1)
np.random.seed(1)

###
#   Define storage path
###
data_path = path.join(getcwd(), "data")
if not "data" in listdir(getcwd()):
    mkdir(data_path)

###
#   Define the differentiable model
##


class STEFunction(t.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return (input > 0).float()

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


def model_STE(grids: t.Tensor, beta: t.Tensor) -> t.Tensor:
    # produces a mask that is 1 for all healthy cells, 0 for the infected cells
    healthy_mask = 1 - grids
    # calculate the likelihood of spread
    likelihoods = spread_likelihood(grids, beta)
    # print(t.unique(likelihoods))
    # compare each likelihood to a random value ~ U(0,1) to get the residual values
    if grids.is_cuda:
        residuals = likelihoods - t.rand(*grids.shape).cuda()
    else:
        residuals = likelihoods - t.rand(*grids.shape)
    # apply the heaviside to the residuals,
    #   if the residual is positive, the cell should be infected ( < spread_likelihood)
    #   if the residual is negative, the cell should stay healthy ( > spread_likelihood)
    #   if the cell was already infected, no update should be applied
    update_mask = STEFunction.apply(residuals) * healthy_mask
    # apply the update to the current state
    return grids + update_mask


###
# Define the Dataset class
###


class CustomDataset(Dataset):
    def __init__(
        self,
        beta_mu=None,
        beta_sigma=None,
        num_sequences=None,
        steps_per_sequence=None,
        grid_size=None,
        load_path=None,
    ):
        if load_path:
            self.data = t.load(load_path)
        else:
            sequences = t.zeros(
                (num_sequences, steps_per_sequence, grid_size, grid_size)
            )
            self.betas = []

            for seq_idx in tqdm(range(num_sequences)):
                sequence = init_grids(grid_size, 1)
                for step_idx in range(steps_per_sequence):
                    beta = t.clip(normal.Normal(beta_mu, beta_sigma).sample(), 0, 1)
                    sequences[seq_idx, step_idx, :, :] = sequence
                    sequence = model(sequence, beta)
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

# training_data = CustomDataset(
#    beta_mu=beta_mu,
#    beta_sigma=beta_sigma,
#    num_sequences=num_sequences,
#    steps_per_sequence=steps_per_sequence,
#    grid_size=grid_size,
# )
data = t.load("./data/training_set.pt")
print(data.shape)
training_loader = DataLoader(data, batch_size=32, shuffle=True)
# t.save(training_data.data, path.join(data_path, "training_set.pt"))

###
#   Visualize sample sequence for the given mu, sigma
###
create_vis = False

if create_vis:
    vis_path = path.join(getcwd(), "vis")
    sequence_path = path.join(vis_path, "sequence")
    if not "vis" in listdir(getcwd()):
        mkdir(vis_path)
    if not "sequence" in listdir(vis_path):
        mkdir(sequence_path)

    grids = init_grids(grid_size, 1)
    for i in tqdm(range(steps_per_sequence)):
        fig, axs = plt.subplots(figsize=(3, 3), dpi=300)
        axs.imshow(grids[0], cmap="Greys", interpolation="nearest", vmin=0, vmax=1)
        axs.get_xaxis().set_visible(False)
        axs.get_yaxis().set_visible(False)
        fig.tight_layout()
        fig.savefig(path.join(sequence_path, f"{i}.png"))
        plt.close(fig)
        beta = t.clip(normal.Normal(beta_mu, beta_sigma).sample(), 0, 1)
        grids = model_STE(grids, beta)

    images = []
    for i in range(steps_per_sequence):
        images.append(imageio.imread(path.join(sequence_path, f"{i}.png")))
    imageio.mimsave(path.join(vis_path, f"{beta_mu}_{beta_sigma}.gif"), images)

###
#   Define Minibatch training loop
###


def S1(X: t.Tensor, Y: t.Tensor) -> t.Tensor:
    return t.sum((1 - X) * Y, axis=(-1, -2))


def mean_sq_distance(X: t.Tensor, Y_sim: t.Tensor, Y_obs: t.Tensor) -> t.Tensor:
    # get statistics of the simulated set
    s_sim = S1(X, Y_sim)
    # get statistics of the observed set
    s_obs = S1(X, Y_obs)
    # print(t.unique(Y_sim))
    # print(s_sim)
    # calculate the mean square difference btw. the statistics
    return t.mean((s_sim - s_obs) ** 2)


def train_loop(beta, dataloader, optimizer):
    size = len(dataloader.dataset)
    betas = []
    distances = []
    for batch, (transitions) in enumerate(dataloader):
        # print(f"------------- {batch} ----------------")
        X = transitions[:, 0]
        Y_obs = transitions[:, 1]
        # Compute prediction and loss
        Y_sim = model_STE(X, beta)
        dist = mean_sq_distance(X, Y_sim, Y_obs)
        distances.append(dist.detach().clone().cpu().numpy())
        # print(" Before ------------")
        # print(beta)
        # print(dist)
        # print("------------")
        # Backpropagation
        optimizer.zero_grad()
        dist.backward()
        optimizer.step()
        # print(" After ------------")
        # print(beta)
        # print(dist)
        # print(beta.grad)
        # print("------------")
        betas.append(beta.detach().clone().cpu().numpy())
    return betas, distances


###
#   Run Sampling-based training
###
sample_train = True

if sample_train:
    sample_based_path = path.join(data_path, "sample_based")
    if not "sample_based" in listdir(data_path):
        mkdir(sample_based_path)

    # beta = uniform.Uniform(t.tensor(0.0), t.tensor(1.0)).sample()
    beta = t.tensor(0.79744744)
    beta.requires_grad_()
    lr = 1e-7
    epochs = 50
    optimizer = t.optim.SGD([beta], lr=lr)

    beta_hist = []
    dist_hist = []
    for epoch in tqdm(range(epochs)):
        # print(epoch)
        betas, distances = train_loop(beta, training_loader, optimizer)
        beta_hist = beta_hist + betas
        dist_hist = dist_hist + distances

    np.save(path.join(sample_based_path, "param_trace2.npy"), np.array(beta_hist))
    np.save(path.join(sample_based_path, "losses2.npy"), np.array(dist_hist))

###
#   Define update function using gradient of the likelihood
###

likelihood_based_path = path.join(data_path, "likelihood_based")
if not "likelihood_based" in listdir(data_path):
    mkdir(likelihood_based_path)


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
    for batch, (transitions) in enumerate(train_loader):
        X = transitions[:, 0]
        Y_obs = transitions[:, 1]
        beta, neg_log_likelihood = update(beta, X, Y_obs, optimizer)
        likelihoods.append(neg_log_likelihood.detach().clone().numpy())
        betas.append(beta.detach().clone().numpy())

    return betas, likelihoods


###
#   Run Likelihood-based training
###
likelihood_train = True

if likelihood_train:
    beta = t.tensor(np.load(path.join(sample_based_path, "param_trace.npy"))[0])
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

    np.save(path.join(likelihood_based_path, "param_trace2.npy"), np.array(beta_hist))
    np.save(path.join(likelihood_based_path, "losses2.npy"), np.array(likelihood_hist))

###
#   Evaluate the likelihood of the dataset for beta=[0,1]
###
# likelihood_contour_path = path.join(data_path, "likelihood_contour")
# if not "likelihood_contour" in listdir(data_path):
#    mkdir(likelihood_contour_path)


# def get_neg_log_l(beta, X, Y_obs):
#    # calculate the spread likelihood for the given beta
#    L = spread_likelihood(X, beta).type(t.double)
#    # calculate the transition likelihood matrix for the given beta
#    P = transition_likelihood(L, X, Y_obs)
# calculate the total likelihood of the transition
#    neg_log_l = neg_log_likelihood(P)
#    return neg_log_l


# beta_vals = t.linspace(0, 1, 1000)

# mean_likelihoods = []
# for beta_val in tqdm(beta_vals):
#    likelihoods = []
#    for i in range(5):
#        for batch, (transitions, _) in enumerate(training_loader):
#            X = transitions[:, 0]
#            Y_obs = transitions[:, 1]
#            likelihood = get_neg_log_l(beta_val, X, Y_obs)
#            # print(likelihood)
#            likelihoods.append(likelihood.detach().numpy())
#    mean_likelihoods.append(np.mean(np.array(likelihoods[likelihoods != float("inf")])))

# np.save(path.join(likelihood_contour_path, "beta_vals.npy"), beta_vals.detach().numpy())
# np.save(
#    path.join(likelihood_contour_path, "likelihoods.npy"), np.array(mean_likelihoods)
# )
