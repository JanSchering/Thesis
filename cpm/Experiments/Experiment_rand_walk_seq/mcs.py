import torch as t
import random
from STE import STEFunction


def MCS(batch, target_vol, temperature):
    _, batch_height, batch_width = batch.shape
    cur_vol = t.sum(batch)
    src_x = t.randint(low=0, high=batch_width, size=(1,))
    src_x = src_x.type(t.long)
    src_y = t.randint_like(src_x, low=0, high=batch_height)
    print(f"coordinates of the source pixel on the grid: {src_x} {src_y}")
    step_size = t.tensor(
        random.choice(
            [(1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1)],
        )
    )
    print(f"step size: \n {step_size}")
    tgt_x = src_x + step_size[0]
    tgt_x[tgt_x == -1] = 1
    tgt_x[tgt_x == batch_width] = batch_width - 2
    tgt_y = src_y + step_size[1]
    tgt_y[tgt_y == -1] = 1
    tgt_y[tgt_y == batch_height] = batch_height - 2
    print(f"coordinates of the target pixel on the grid: {tgt_x} {tgt_y}")

    vol_change = (-1 * batch[0, tgt_y, tgt_x]) + batch[0, src_y, src_x]
    print("vol change", vol_change)
    adjusted_vol = cur_vol + vol_change

    if batch[0, tgt_y, tgt_x] == batch[0, src_y, src_x]:
        print("source is equal to target, no update")
    elif adjusted_vol > 2 or adjusted_vol <= 0:
        print("Changes would violate the hard constraints, no update")
    elif cur_vol == 2 and vol_change == -1:
        print("Negative Hamiltonian, accepted")
        batch[0, tgt_y, tgt_x] += vol_change
    else:
        update_probability = t.exp(-((target_vol - adjusted_vol) ** 2) / temperature)
        print(f"update probability: {update_probability}")

        # print(update_probability)
        # one_hot = t.nn.functional.gumbel_softmax(
        #    t.log(
        #        t.cat(
        #            (
        #                update_probability.unsqueeze(0),
        #                1 - update_probability.unsqueeze(0),
        #            )
        #        )
        #    ),
        #    hard=True,
        # )
        # print(one_hot)
        # upd_val = one_hot[0] * vol_change
        # print(upd_val)

        upd_val = STEFunction.apply(update_probability) * vol_change

        batch[0, tgt_y, tgt_x] += upd_val

    return batch
