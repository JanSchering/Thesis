import sys
sys.path.insert(0, '../../cpm_r_model/')
from os import path, getcwd, listdir, mkdir
from tqdm import tqdm

import torch as t
import numpy as np
import random

from model import model

###
#   Set random seeds
##

t.manual_seed(0)
np.random.seed(0)
random.seed(0)

###
#   Set torch device
###

device = t.device("cuda") if t.cuda.is_available() else t.device("cpu")

###
#   Ensure storage path is available
###

data_path = path.join(getcwd(), "data")
cpm_r_sample_path = path.join(data_path, "cpm_r_sample")
if "data" not in listdir(getcwd()):
    mkdir(data_path)
if "cpm_r_sample" not in listdir(data_path):
    mkdir(cpm_r_sample_path)

###
#   Set simulation parameters
###

temperature = t.tensor(1., device=device)
target_vol = 1.
grid_size = 32
batch = t.zeros(1,grid_size,grid_size, device=device)
batch[:,grid_size//2,grid_size//2] += 1

num_steps = 1000

###
#   Run simulation
###

states = np.zeros((num_steps+1,grid_size,grid_size))
states[0] = batch[0]
for i in tqdm(range(num_steps)):
    batch = model(batch, temperature)
    if t.any(t.sum(batch, dim=(-1,-2)) == 0) or t.any(t.sum(batch, dim=(-1,-2)) > 2):
        print("ISSUE DETECTED, STOP SIM")
        break
    else:
        states[i+1] = batch[0].detach().clone().cpu().numpy()
        
np.save(path.join(cpm_r_sample_path, "sequence.npy"), states)