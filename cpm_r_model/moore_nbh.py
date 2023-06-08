import torch as t

###
#   Set device
###

device = t.device("cuda") if t.cuda.is_available() else t.device("cpu")

###
#   Define Moore NBH function
###

MOORE_OFFSETS = t.tensor([(1, 1), (1, -1), (1, 0), (-1, 0), (0, 0), (-1, 1), (-1, -1), (0, 1), (0, -1)], device=device)
def get_moore_nbh(batch:t.Tensor):
    _, batch_height, _ = batch.shape
    cell_pixel_coords = (batch == 1).nonzero()
    nbh_coords = (cell_pixel_coords[:, 1:].unsqueeze(1)+MOORE_OFFSETS.type(t.float).unsqueeze(0)).flatten(end_dim=1)
    idx_pad = cell_pixel_coords[:, 0].repeat_interleave(9).flatten().unsqueeze(-1)
    nbh_coords[nbh_coords == -1] = 1
    nbh_coords[nbh_coords == batch_height] = batch_height -2
    idc, counts = t.unique(idx_pad, return_counts=True)
    return t.split(t.concatenate((idx_pad, nbh_coords), dim=-1), split_size_or_sections=counts.tolist())