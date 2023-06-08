import sys
sys.path.insert(0, '../../cpm_r_model/')
from os import path, getcwd, listdir, mkdir
from tqdm import tqdm

import torch as t
import numpy as np
import random

from model import model
from PIL import Image

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

temperature = t.tensor(4., device=device)
target_vol = 1.
grid_size = 128
batch = t.zeros(5,grid_size,grid_size, device=device)
batch[0,grid_size//2,grid_size//2] += 1
batch[1,grid_size//4,grid_size//4] += 1
batch[2,grid_size//4,(grid_size//4)*3] += 1 
batch[3,(grid_size//4)*3,grid_size//4] += 1
batch[4,(grid_size//4)*3,(grid_size//4)*3] += 1

num_steps = 100_000

col_idxs = np.tile(np.arange(grid_size), (grid_size,1)) 
row_idxs = np.tile(np.arange(grid_size), (grid_size,1)).T

def centroid(state):
    x = np.sum(state * col_idxs, axis=(-1,-2)) / np.sum(state, axis=(-1,-2))
    y = np.sum(state * row_idxs, axis=(-1,-2)) / np.sum(state, axis=(-1,-2))
    return np.hstack([x[:, np.newaxis],y[:, np.newaxis]])

###
#   Run simulation
###
centroids = np.zeros((num_steps+1,5,2))
centroids[0] = centroid(batch.detach().clone().cpu().numpy())
for i in tqdm(range(num_steps)):
    batch = model(batch, temperature)
    if t.any(t.sum(batch, dim=(-1,-2)) == 0) or t.any(t.sum(batch, dim=(-1,-2)) > 2):
        print("ISSUE DETECTED, STOP SIM")
        break
    else:
        centroids[i+1] = centroid(batch.detach().clone().cpu().numpy())

#imgs = [Image.fromarray((1-states[i])*255) for i in range(states.shape[0])]
#imgs[0].save(path.join(cpm_r_sample_path, "vis.gif"), save_all=True, append_images=imgs[1:], duration=10, loop=100)
np.save(path.join(cpm_r_sample_path, f"centroids.npy"), centroids)
    