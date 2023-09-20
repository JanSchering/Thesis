import sys
import torch as t
from scipy import ndimage
from tqdm import tqdm

def get_active_pixel_idx(batch:t.Tensor):
    """"Return the coordinates of each active pixel in batch"""
    return batch.nonzero().T


def take_step(batch:t.Tensor, p_center:t.Tensor, dim=-1, device=t.device("cpu")) -> t.Tensor:
    """Perform a random step on the lattice in the given dimension"""  

    act_pixel_idx = get_active_pixel_idx(batch)
    batch_size,lattice_dim,_ = batch.shape

    # get the coordinates of the pixels left and right from  the active pixel in the given dim
    left_px = act_pixel_idx.clone()
    right_px = act_pixel_idx.clone()
    left_px[dim] -= 1
    right_px[dim] += 1
    left_px[dim][left_px[dim] < 0] = lattice_dim -1
    right_px[dim][right_px[dim] == lattice_dim] = 0

    p_left = t.log((1-p_center)/2)
    p_right = t.log((1-p_center)/2)

    # log probabilities repeated for each grid
    logits = t.hstack([p_center,p_left,p_right])
    logit_tile = logits.repeat((batch_size,1))

    # sample a one-hot direction vector (center,left,right) for each grid using gumbel softmaxing
    outcome = t.nn.functional.gumbel_softmax(logit_tile, tau=1, hard=True)

    # use the one-hot vector to take a step
    batch[act_pixel_idx[0], act_pixel_idx[1], act_pixel_idx[2]] -= 1 - outcome[:,0]
    batch[left_px[0], left_px[1], left_px[2]] += outcome[:,1]
    batch[right_px[0], right_px[1], right_px[2]] += outcome[:,2]
    return batch


def create_dist_matrix(batch:t.Tensor, device=t.device("cpu")) -> t.Tensor:
    """Create distance matrix with same width/height as reference batch"""    
    dist_matrix = ndimage.distance_transform_edt(1-batch[0].cpu(), return_indices=False)
    dist_matrix = dist_matrix**2
    dist_matrix_t = t.from_numpy(dist_matrix).to(device)
    return dist_matrix_t


def calc_stats(batch:t.Tensor, dist_matrix:t.Tensor):
    """Calculate summary statistics of the batch, I.e. Square displacements, Mean Square displacement and Standard deviation of the displacements."""  
    distances = (dist_matrix[batch.nonzero(as_tuple=True)[1:]] * batch[batch.nonzero(as_tuple=True)])
    return distances, t.mean(distances), t.std(distances)

if __name__ == "__main__":
    print(sys.argv)
    _, p, n_mcs, batch_size, lattice_dim, show_displacements = sys.argv
    p = float(p)
    n_mcs = int(n_mcs)
    batch_size = int(batch_size)
    lattice_dim = int(lattice_dim)
    show_displacements = bool(int(show_displacements))
    print(f"running simulation with p={p}, n_mcs={n_mcs}, batch_size={batch_size}, lattice_size={lattice_dim}")

    device = t.device("cuda") if t.cuda.is_available() else t.device("cpu")
    p = t.tensor(float(p), device=device)
    batch = t.zeros((batch_size,lattice_dim,lattice_dim), device=device)
    batch[:,lattice_dim//2,lattice_dim//2] = 1
    dist_mtx = create_dist_matrix(batch,device=device)

    for step in tqdm(range(n_mcs)):
        batch = take_step(batch, p, dim=-1)
        batch = take_step(batch, p, dim=-2)

    distances, msd, std_dist = calc_stats(batch, dist_mtx)
    
    print(show_displacements)
    if show_displacements:
        print("showing Square distance per grid in the batch:")
        print(distances)
        print(f"MSD: {msd}")
    else:
        print("showing coordinates of the active pixels:")
        print(get_active_pixel_idx(batch).T)