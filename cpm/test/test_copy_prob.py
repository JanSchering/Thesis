import sys

sys.path.insert(0, "../")
import torch as t
from cpm_model import prob_copy


def test_prob_copy():
    temperature = t.tensor(27.0)
    h_diff = t.tensor(47.0)

    p_copy = prob_copy(h_diff, temperature)

    assert f"{p_copy:.4}" == "0.1754"
