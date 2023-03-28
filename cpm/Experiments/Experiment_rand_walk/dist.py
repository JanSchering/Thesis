import torch as t


def centroid(batch):
    _, grid_height, grid_width = batch.shape
    col_idxs = t.arange(grid_width).repeat(grid_width, 1) + 1
    row_idxs = t.arange(grid_height).repeat(grid_height, 1).T + 1
    x_mean = t.sum(batch * col_idxs) / t.sum(batch)
    y_mean = t.sum(batch * row_idxs) / t.sum(batch)
    return x_mean, y_mean


def euclidean_dist(centroid1, centroid2):
    c1_x, c1_y = centroid1
    c2_x, c2_y = centroid2

    return t.sqrt((c1_x - c2_x) ** 2 + (c1_y - c2_y) ** 2)
