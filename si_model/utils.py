import torch as t


def heaviside(x, k):
    return 1 / (1 + t.exp(-2 * k * x))
