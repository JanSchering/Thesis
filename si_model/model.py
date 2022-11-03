import torch as t
from likelihoods import spread_likelihood
from vis_utils import visualize_sequence, visualize_batch


def init_grids(size: int, batch_size: int) -> t.Tensor:
    grids = t.zeros((batch_size, size, size))
    grids[:, size // 2, size // 2] = 1
    return grids


def model(grids: t.Tensor, beta: t.Tensor) -> t.Tensor:
    healthy_mask = grids == 0

    likelihoods = spread_likelihood(grids, beta)
    infection_update = t.rand(*grids.shape) <= likelihoods * healthy_mask
    return grids + infection_update.detach().clone().numpy().astype(int)


if __name__ == "__main__":
    size = 30
    batch_size = 2
    beta = t.tensor(0.08)
    num_steps = 100

    sequences = t.zeros((batch_size, num_steps, size, size)).float()
    batch = init_grids(size, batch_size)

    for i in range(num_steps):
        batch = model(batch, beta)
        sequences[:, i, :, :] = batch.detach().clone()

    visualize_sequence(sequences[0], framerate=0.2)
