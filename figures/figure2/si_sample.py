# ensure path to si modules
import sys
sys.path.insert(0, "../../si_model")
from os import getcwd, path, listdir, mkdir
# import modules
import torch as t
import numpy as np
from tqdm import tqdm
from model import init_grids, model

data_path = path.join(getcwd(), "data")
si_path = path.join(data_path, "si_sample")
if not "data" in listdir(getcwd()):
    mkdir(data_path)
if not "si_sample" in listdir(data_path):
    mkdir(si_path)

###
#   Set a random seed for torch and numpy
###

t.manual_seed(0)
np.random.seed(0)

beta = t.tensor(0.5)
grid_size = 64
num_steps = 50
grid = init_grids(grid_size, 1)
sequence = np.zeros((num_steps+1,grid_size,grid_size))
sequence[0] = grid.detach().clone().numpy()
for step_idx in tqdm(range(num_steps)):
    grid = model(grid, beta)
    sequence[step_idx+1] = grid.detach().clone().numpy()

np.save(path.join(si_path, "sequence.npy"), sequence)