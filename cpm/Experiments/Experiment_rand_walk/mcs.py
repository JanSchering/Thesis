import torch as t
import random
from utils import split_grids, tile_idx2conv_idx, tile_coords2grid_coords
from update import p_update


def MCS(batch, checkerboard_sets, target_vol, temperature):
    _, batch_height, batch_width = batch.shape
    vol_kernel = t.tensor([[[[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]]]])

    steps = []
    for c, chkboard_set in enumerate(checkerboard_sets):
        print(f"-------------- Set {c} ---------------------")
        res_batch = t.zeros(*batch.shape)
        tile_idxs = t.tensor(
            [
                (chkboard_set[0][i, j], chkboard_set[1][i, j])
                for i in range(chkboard_set[0].shape[0])
                for j in range(chkboard_set[1].shape[0])
            ]
        )
        # print(f"indices of the tiles on the grid: \n {tile_idxs}")
        padded_batch = t.nn.ReflectionPad2d(1)(batch).float()
        cur_vol = t.sum(t.nn.functional.conv2d(padded_batch, vol_kernel))
        conv_stack = split_grids(batch, kernel_width=4, kernel_height=4)
        n_rows, n_cols = chkboard_set[0].shape
        conv_stack_idxs = [
            tile_idx2conv_idx(
                chkboard_set[0][i, j],
                chkboard_set[1][i, j],
                tile_size=(4, 4),
                batch_size=batch.shape,
            )
            for i in range(n_rows)
            for j in range(n_cols)
        ]
        # print(f"indices of the tiles in the conv stack: \n {conv_stack_idxs}")
        tiles = t.vstack([conv_stack[0, conv_stack_idxs]]).reshape(
            len(conv_stack_idxs), 4, 4
        )
        src_x = t.randint_like(
            t.zeros(
                tiles.shape[0],
            ),
            low=1,
            high=3,
        )
        src_x = src_x.type(t.long)
        src_y = t.randint_like(src_x, low=1, high=3)
        # For each random sample in src, we sample a random value from [-1, 0, 1]
        # and add it on to the src_idx
        step_sizes = t.tensor(
            random.choices(
                [(1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1)],
                k=tiles.shape[0],
            )
        )
        tgt_x = src_x + step_sizes[:, 1]
        tgt_y = src_y + step_sizes[:, 0]

        grid_coords_src = tile_coords2grid_coords(
            tile_idx=tile_idxs,
            on_tile_coords=(src_y, src_x),
            grid_dim=batch.shape,
            tile_dim=(2, 2),
        )
        print(f"coordinates of the source pixels on the grid: \n {grid_coords_src}")
        print(f"source pixels: {batch[0, grid_coords_src[:,0], grid_coords_src[:,1]]}")

        grid_coords_tgt = tile_coords2grid_coords(
            tile_idx=tile_idxs,
            on_tile_coords=(tgt_y, tgt_x),
            grid_dim=batch.shape,
            tile_dim=(2, 2),
        )

        print(f"coordinates of the target pixels on the grid: \n {grid_coords_tgt}")
        print(f"target pixels: {batch[0, grid_coords_tgt[:,0], grid_coords_tgt[:,1]]}")

        vol_changes = (-1 * tiles[range(tiles.shape[0]), tgt_y, tgt_x]) + tiles[
            range(tiles.shape[0]), src_y, src_x
        ]
        print("vol changes", vol_changes)
        total_vol_change = t.sum(vol_changes)
        adjusted_vol = cur_vol + total_vol_change

        if t.all(
            batch[0, grid_coords_src[:, 0], grid_coords_src[:, 1]]
            == batch[0, grid_coords_tgt[:, 0], grid_coords_tgt[:, 1]]
        ):
            print("all source IDs equivalent to target IDs")
        elif adjusted_vol > 2 or adjusted_vol <= 0:
            print("Changes would violate the hard constraints")
        elif cur_vol == 2 and total_vol_change == -1:
            print("Negative Hamiltonian, accepted")
            res_batch[0, grid_coords_tgt[:, 0], grid_coords_tgt[:, 1]] += vol_changes
        else:
            # print("on_tile source pixel coords per grid: \n",t.cat((src_x.unsqueeze(0), src_y.unsqueeze(0)), dim=0).T)
            # print("on_tile target pixel coords per grid: \n", t.cat((tgt_x.unsqueeze(0), tgt_y.unsqueeze(0)), dim=0).T)

            update_probability = p_update(adjusted_vol, target_vol, temperature)
            print(f"update probability: {update_probability}")

            # print(f"coordinates of the target pixels on the grid: \n {grid_coords_tgt}")
            # print(f"target pixels: {batch[0, grid_coords_tgt[:,0], grid_coords_tgt[:,1]]}")

            # print(update_probability)
            one_hot = t.nn.functional.gumbel_softmax(
                t.log(
                    t.cat(
                        (
                            update_probability.unsqueeze(0),
                            1 - update_probability.unsqueeze(0),
                        )
                    )
                ),
                hard=True,
            )
            print(one_hot)

            upd_val = one_hot[0] * vol_changes

            print(upd_val)

            res_batch[0, grid_coords_tgt[:, 0], grid_coords_tgt[:, 1]] += upd_val

        batch += res_batch
        steps.append(batch.detach().squeeze().clone().numpy())

    return batch, steps
