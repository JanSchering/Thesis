import torch as t
from torch.nn.functional import conv2d
from periodic_padding import periodic_padding


def risk_convolution2D(batch: t.Tensor):
    """
    Use a filter of shape
    | 1  1  1
    | 1  0  1
    | 1  1  1
    and apply it to the periodically padded state of the grid.
    The result will give the number of infected neighbors at each
    state.
    """
    kernel = t.ones((1, 1, 3, 3)).float()
    kernel[:, :, 1, 1] = 0
    padded_grid = periodic_padding(batch).float()
    expanded = t.unsqueeze(padded_grid, -1)
    transposed = t.permute(expanded, (0, 3, 1, 2))

    return conv2d(transposed, kernel, stride=1, padding="valid")
