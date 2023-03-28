import torch as t


def split_grids(batch: t.Tensor, kernel_width: int, kernel_height: int) -> t.Tensor:
    unfold_transform = t.nn.Unfold(kernel_size=(kernel_height, kernel_width))
    reflective_padding = t.nn.ReflectionPad2d(1)

    padded_batch = reflective_padding(batch).float()
    padded_batch = padded_batch.unsqueeze(1)

    return t.transpose(unfold_transform(padded_batch), dim0=1, dim1=2)


def tile_idx2conv_idx(row, col, tile_size, batch_size):
    _, batch_height, batch_width = batch_size
    tile_height, tile_width = tile_size
    tile_width -= 2
    tile_height -= 2
    tiles_per_row = batch_width - tile_width + 1
    return int(tile_height * row * tiles_per_row + tile_width * col)


def tile_coords2grid_coords(tile_idx, on_tile_coords, grid_dim, tile_dim):
    # coordinates of the pixels on the tiles
    tile_y, tile_x = on_tile_coords
    # coordinates of the tiles in the checkerboard scheme
    tile_rows = tile_idx[:, 0]
    tile_cols = tile_idx[:, 1]
    # dimensionalities of a single tile
    tile_height, tile_width = tile_dim
    _, grid_height, grid_width = grid_dim

    grid_x = tile_cols * tile_width + (tile_x - 1)
    grid_y = tile_rows * tile_height + (tile_y - 1)

    grid_x[grid_x == -1] = 1
    grid_x[grid_x == grid_width] = grid_width - 2

    grid_y[grid_y == -1] = 1
    grid_y[grid_y == grid_height] = grid_height - 2

    return t.vstack((grid_y, grid_x)).T


def indep_chkboard_sets(batch, tile_height, tile_width):
    _, batch_height, batch_width = batch.shape
    tiles_per_row = batch_width // (tile_width - 2)
    tiles_per_col = batch_height // (tile_height - 2)

    assert (
        tiles_per_row % 2 == 0
    ), "please provide a tile_width that allows for even tiling of the row"
    assert (
        tiles_per_col % 2 == 0
    ), "please provide a tile_height that allows for even tiling of the columns"

    meshgrid_coords = []
    for row_start_idx in range(2):
        for col_start_idx in range(2):
            row_idxs = t.tensor(range(row_start_idx, tiles_per_row, 2))
            col_idxs = t.tensor(range(col_start_idx, tiles_per_col, 2))
            meshgrid_coords.append(t.meshgrid((row_idxs, col_idxs), indexing="ij"))
    return meshgrid_coords


if __name__ == "__main__":
    test1 = t.arange(64).reshape(1, 8, 8).float()
    split1 = split_grids(test1, kernel_width=4, kernel_height=4)
    assert split1.shape == (1, 49, 16)
    assert t.all(
        split1[0, 0] == t.tensor([9, 8, 9, 10, 1, 0, 1, 2, 9, 8, 9, 10, 17, 16, 17, 18])
    )

    test2 = t.arange(32**2).reshape(1, 32, 32).float()
    split2 = split_grids(test2, kernel_width=6, kernel_height=6)
    assert split2.shape == (1, 841, 36)
    assert t.all(
        split2[0, 0].reshape(6, 6)
        == t.tensor(
            [
                [33, 32, 33, 34, 35, 36],
                [1, 0, 1, 2, 3, 4],
                [33, 32, 33, 34, 35, 36],
                [65, 64, 65, 66, 67, 68],
                [97, 96, 97, 98, 99, 100],
                [
                    129,
                    128,
                    129,
                    130,
                    131,
                    132,
                ],
            ]
        )
    )

    test1 = t.arange(64).reshape(1, 8, 8).float()
    split1 = split_grids(test1, kernel_width=4, kernel_height=4)
    idx = tile_idx2conv_idx(1, 0, (4, 4), test1.shape)
    assert idx == 14
    assert t.all(
        split1[0, idx]
        == t.tensor([9, 8, 9, 10, 17, 16, 17, 18, 25, 24, 25, 26, 33, 32, 33, 34])
    )

    test1 = t.arange(64).reshape(1, 8, 8).float()
    on_tile_coords = (t.tensor((1, 1, 1, 1)), t.tensor((1, 1, 1, 1)))
    on_grid_tile_idxs = t.tensor([[0, 0], [0, 2], [2, 0], [2, 2]])
    result_coords = tile_coords2grid_coords(
        on_grid_tile_idxs, on_tile_coords, grid_dim=test1.shape, tile_dim=(2, 2)
    )

    # first set of coordinates should be the top-left pixel in the top-left tile
    assert t.all(result_coords[0] == t.tensor([0, 0]))
    # second set of coordinates should be the top-left pixel of the 3. tile in the first row
    assert t.all(result_coords[1] == t.tensor([0, 4]))
    # third set of coordinates should be the top-left pixel of the 1. tile in the 3. row of the checkerboard
    assert t.all(result_coords[2] == t.tensor([4, 0]))
    # fourth set of coordinates should be top-left pixel of the 3. tile in the 3. row of the checkerboard
    assert t.all(result_coords[3] == t.tensor([4, 4]))

    on_tile_coords = (t.tensor((1, 1, 1, 1)), t.tensor((2, 2, 2, 2)))
    on_grid_tile_idxs = t.tensor([[0, 1], [0, 3], [2, 1], [2, 3]])
    result_coords = tile_coords2grid_coords(
        on_grid_tile_idxs, on_tile_coords, grid_dim=test1.shape, tile_dim=(2, 2)
    )

    # first set of coordinates should be the top-right pixel of the 2. tile in the 1. row
    assert t.all(result_coords[0] == t.tensor([0, 3]))
    # second set of coordinates should be the top-right pixel of the 4. tile in the first row
    assert t.all(result_coords[1] == t.tensor([0, 7]))
    # third set of coordinates should be the top-right pixel of the 2. tile in the 3. row
    assert t.all(result_coords[2] == t.tensor([4, 3]))
    # fourth set of coordinates should be the top-right pixel of the 4. tile in the 3. row
    assert t.all(result_coords[3] == t.tensor([4, 7]))

    on_tile_coords = (t.tensor((0, 0, 1, 3)), t.tensor((1, 0, 0, 2)))
    on_grid_tile_idxs = t.tensor([[0, 0], [0, 0], [0, 0], [3, 3]])
    result_coords = tile_coords2grid_coords(
        on_grid_tile_idxs, on_tile_coords, grid_dim=test1.shape, tile_dim=(2, 2)
    )

    # checking if the reflective boundary works as expected. Because of the padding of the tiles, setting
    # the on-tile row index to zero would bring us out of bounds. Instead, we want to be reflected back to row 2
    assert t.all(result_coords[0] == t.tensor([1, 0]))
    # checking other boundary conditions
    assert t.all(result_coords[1] == t.tensor([1, 1]))
    assert t.all(result_coords[2] == t.tensor([0, 1]))
    assert t.all(result_coords[3] == t.tensor([6, 7]))
