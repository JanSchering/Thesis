import torch as t
from torch.nn.functional import conv2d
from periodic_padding import periodic_padding


def risk_convolution2D(batch: t.Tensor):
    """
    Use a filter of shape
    | 1  1  1
    | 1  0  1
    | 1  1  1
    and apply it to the periodically padded state of the grid.
    The result will give the number of infected neighbors at each
    state.
    """
    kernel = t.ones((1, 1, 3, 3)).float()
    kernel[:, :, 1, 1] = 0
    padded_grid = periodic_padding(batch).float()
    expanded = t.unsqueeze(padded_grid, -1)
    transposed = t.permute(expanded, (0, 3, 1, 2))

    return conv2d(transposed, kernel, stride=1, padding="valid")


if __name__ == "__main__":
    print("testing the risk convolution function...")
    size = 9
    batch_size = 2
    beta = 0.35

    batch = t.zeros((batch_size, size, size))
    batch[:, size // 2, size // 2] = 1

    batch[0, 0, 0] = 1
    batch[0, 0, 1] = 1
    batch[0, 0, 2] = 1
    batch[0, 1, 0] = 1
    batch[0, 1, 2] = 1
    batch[0, 2, 0] = 1
    batch[0, 2, 1] = 1
    batch[0, 2, 2] = 1

    conv_result = risk_convolution2D(batch)

    assert conv_result[0, 0, 0, 0] == 2
    assert conv_result[0, 0, 0, 1] == 4
    assert conv_result[0, 0, 0, 2] == 2
    assert conv_result[0, 0, 1, 0] == 4
    assert conv_result[0, 0, 1, 1] == 8
    assert conv_result[0, 0, 1, 2] == 4
    assert conv_result[0, 0, 2, 0] == 2
    assert conv_result[0, 0, -1, 1] == 3
    assert conv_result[0, 0, -1, 0] == 2
    assert conv_result[0, 0, -1, 2] == 2

    print("all tests passed successfully")
