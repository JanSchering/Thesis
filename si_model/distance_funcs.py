import torch as t

def S1(X: t.Tensor, Y: t.Tensor) -> t.Tensor:
    return t.sum((1 - X) * Y, axis=(-1, -2))


def mean_sq_distance(X: t.Tensor, Y_sim: t.Tensor, Y_obs: t.Tensor) -> t.Tensor:
    # get statistics of the simulated set
    s_sim = S1(X, Y_sim)
    # get statistics of the observed set
    s_obs = S1(X, Y_obs)
    # calculate the mean square difference btw. the statistics
    return t.mean((s_sim - s_obs)**2)
