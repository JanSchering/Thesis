import torch as t


def p_update(tiles, cur_vol, target_vol, temperature, src_coords, tgt_coords):
    batch_size, _, _ = tiles.shape
    src_x, src_y = src_coords
    tgt_x, tgt_y = tgt_coords
    vol_changes = (-1 * tiles[range(batch_size), tgt_y, tgt_x]) + tiles[
        range(batch_size), src_y, src_x
    ]
    total_vol_change = t.sum(vol_changes)
    # print(f"vol change: {total_vol_change}")
    adjusted_vol = cur_vol + total_vol_change
    prob_mask = (adjusted_vol <= 2 and adjusted_vol > 0).float()
    return prob_mask * t.exp(-((target_vol - adjusted_vol) ** 2) / temperature) + (
        1 - prob_mask
    ) * t.tensor(0)
