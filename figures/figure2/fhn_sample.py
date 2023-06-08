import sys
sys.path.insert(0, '../../fhn_model/')
from os import path, getcwd, listdir, mkdir
from tqdm import tqdm

import torch as t
import numpy as np

from diffusion import diffuse
from reaction import rho, p1, p2, p3, p4, p5, p6

###
#   Set a random seed for torch and numpy
###

t.manual_seed(0)
np.random.seed(0)

###
#   Set torch device
###

device = t.device("cuda") if t.cuda.is_available() else t.device("cpu")

###
#   Ensure storage folder exists
###
data_path = path.join(getcwd(), "data")
fhn_sample_path = path.join(data_path, "fhn_sample")
if not "data" in listdir(getcwd()):
    mkdir(data_path)
if not "fhn_sample" in listdir(data_path):
    mkdir(fhn_sample_path)
    
###
#   Initialize the FHN lattice
###

grid = t.zeros((2, 64, 64), device=device)
grid[:] = 25
grid[0, 29:35] = 40
grid = grid.float()

###
#   Set simulation params
###

gamma = 0.005
k1 = k1_bar = 0.98
k2 = k2_bar = 0.1
k3 = k3_bar = 0.2
rate_coefficients = t.tensor([k1, k1_bar, k2, k2_bar, k3, k3_bar], device=device)
probability_funcs = [p1, p2, p3, p4, p5, p6]

N = 50
num_steps = 10_000
DA = 0.1
DB = 0.4

###
#   Run simulation
###

sequence = np.zeros((num_steps+1, *grid.shape))
sequence[0] = grid.detach().clone().cpu().numpy()

for i in tqdm(range(num_steps)):
    # perform a step of diffusion
    grid = diffuse(grid, N, DA, DB)
    grid = rho(
        grid,
        N,
        gamma,
        rate_coefficients,
        probability_funcs,
        num_reaction_channels=6,
    )
    sequence[i+1] = grid.detach().clone().cpu().numpy()
    
np.save(path.join(fhn_sample_path, "sequence.npy"), sequence)