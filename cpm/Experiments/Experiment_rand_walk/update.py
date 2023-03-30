import torch as t


def p_update(adjusted_vol, target_vol, temperature):
    print(f"adjusted vol: {adjusted_vol}")
    return t.exp(-((target_vol - adjusted_vol) ** 2) / temperature)
