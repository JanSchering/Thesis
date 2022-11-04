import sys
sys.path.insert(0, '../')
import torch as t
from risk_conv import risk_convolution2D


def test_risk_conv1():
    size = 9
    batch_size = 2

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
