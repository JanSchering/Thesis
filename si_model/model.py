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


# ---------------------------------------------------------------


class STEFunction(t.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return (input > 0).float()

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


def model_STE(grids: t.Tensor, beta: t.Tensor) -> t.Tensor:
    # produces a mask that is 1 for all healthy cells, 0 for the infected cells
    healthy_mask = 1 - grids
    # calculate the likelihood of spread
    likelihoods = spread_likelihood(grids, beta)
    # compare each likelihood to a random value ~ U(0,1) to get the residual values
    if grids.is_cuda:
        residuals = likelihoods - t.rand(*grids.shape).cuda()
    else:
        residuals = likelihoods - t.rand(*grids.shape)
    # apply the heaviside to the residuals,
    #   if the residual is positive, the cell should be infected ( < spread_likelihood)
    #   if the residual is negative, the cell should stay healthy ( > spread_likelihood)
    #   if the cell was already infected, no update should be applied
    update_mask = STEFunction.apply(residuals) * healthy_mask
    # apply the update to the current state
    return grids + update_mask


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
