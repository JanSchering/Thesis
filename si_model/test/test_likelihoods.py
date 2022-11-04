import sys
sys.path.insert(0, '../')
import torch as t
from likelihoods import spread_likelihood, transition_likelihood, total_likelihood


def init_test_grids(size, batch_size):
    batch = t.zeros((batch_size, size, size))
    batch[:, size // 2, size // 2] = 1
    return batch


def init_case_1():
    """
    Transition:

    x_t =   [0 0 0] x_tt =  [0 1 0] beta = 0.1
            [0 1 0]         [0 1 0]
            [0 0 0]         [0 1 1]
    """
    size = 3
    num_grids = 2
    beta = t.tensor(0.1)

    grids = init_test_grids(size, num_grids)
    x_t = grids[0].unsqueeze(0)
    x_tt = grids[1].unsqueeze(0)
    x_tt[0, 1, 1] = 1.0
    x_tt[0, 2, 1] = 1.0
    x_tt[0, 2, 2] = 1.0
    x_tt[0, 0, 1] = 1.0
    return x_t, x_tt, beta


def init_case_2():
    x_t = t.zeros((2, 3, 3))
    x_t[:, 1, 1] = 1

    x_tt = t.zeros((2, 3, 3))
    x_tt[:, 1, 1] = 1

    x_tt[0, 2, 1] = 1
    x_tt[0, 2, 2] = 1
    x_tt[0, 0, 1] = 1

    x_tt[1, 0, 0] = 1
    x_tt[1, 2, 0] = 1
    x_tt[1, 1, 2] = 1

    beta = t.tensor(0.1)

    return x_t, x_tt, beta


def test_spread_likelihood():
    size = 9
    batch_size = 2
    beta = 0.35

    batch = init_test_grids(size, batch_size)
    batch[0, 0, 0] = 1
    batch[0, 0, 1] = 1
    batch[0, 0, 2] = 1
    batch[0, 1, 0] = 1
    batch[0, 1, 2] = 1
    batch[0, 2, 0] = 1
    batch[0, 2, 1] = 1
    batch[0, 2, 2] = 1

    likelihoods = spread_likelihood(batch, beta)

    def likelihood(beta, n):
        return 1 - (1 - beta) ** n

    assert "{:0.4f}".format(likelihoods[0, 0, 0]) == "{:0.4f}".format(
        likelihood(beta, 2)
    )
    assert "{:0.4f}".format(likelihoods[0, 0, 1]) == "{:0.4f}".format(
        likelihood(beta, 4)
    )
    assert "{:0.4f}".format(likelihoods[0, 0, 2]) == "{:0.4f}".format(
        likelihood(beta, 2)
    )
    assert "{:0.4f}".format(likelihoods[0, 1, 0]) == "{:0.4f}".format(
        likelihood(beta, 4)
    )
    assert "{:0.4f}".format(likelihoods[0, 1, 1]) == "{:0.4f}".format(
        likelihood(beta, 8)
    )
    assert "{:0.4f}".format(likelihoods[0, -1, 1]) == "{:0.4f}".format(
        likelihood(beta, 3)
    )


def test_transition_likelihood_1():
    x_t, x_tt, beta = init_case_1()

    l = spread_likelihood(x_t, beta)
    p = transition_likelihood(l, x_t, x_tt)
    print(p)

    assert "{:0.4f}".format(p[0, 0, 0]) == "0.9000"
    assert "{:0.4f}".format(p[0, 0, 1]) == "0.1000"
    assert "{:0.4f}".format(p[0, 0, 2]) == "0.9000"

    assert "{:0.4f}".format(p[0, 1, 0]) == "0.9000"
    assert "{:0.4f}".format(p[0, 1, 1]) == "1.0000"
    assert "{:0.4f}".format(p[0, 1, 2]) == "0.9000"

    assert "{:0.4f}".format(p[0, 2, 0]) == "0.9000"
    assert "{:0.4f}".format(p[0, 2, 1]) == "0.1000"
    assert "{:0.4f}".format(p[0, 2, 2]) == "0.1000"


def test_transition_likelihood_2():
    x_t, x_tt, beta = init_case_2()

    l = spread_likelihood(x_t, beta)
    p = transition_likelihood(l, x_t, x_tt)

    assert "{:0.2f}".format(p[0, 0, 0]) == "0.90"
    assert "{:0.2f}".format(p[0, 0, 1]) == "0.10"
    assert "{:0.2f}".format(p[0, 0, 2]) == "0.90"
    assert "{:0.2f}".format(p[0, 1, 0]) == "0.90"
    assert "{:0.2f}".format(p[0, 1, 1]) == "1.00"
    assert "{:0.2f}".format(p[0, 1, 2]) == "0.90"
    assert "{:0.2f}".format(p[0, 2, 0]) == "0.90"
    assert "{:0.2f}".format(p[0, 2, 1]) == "0.10"
    assert "{:0.2f}".format(p[0, 2, 2]) == "0.10"

    assert "{:0.2f}".format(p[1, 0, 0]) == "0.10"
    assert "{:0.2f}".format(p[1, 0, 1]) == "0.90"
    assert "{:0.2f}".format(p[1, 0, 2]) == "0.90"
    assert "{:0.2f}".format(p[1, 1, 0]) == "0.90"
    assert "{:0.2f}".format(p[1, 1, 1]) == "1.00"
    assert "{:0.2f}".format(p[1, 1, 2]) == "0.10"
    assert "{:0.2f}".format(p[1, 2, 0]) == "0.10"
    assert "{:0.2f}".format(p[1, 2, 1]) == "0.90"
    assert "{:0.2f}".format(p[1, 2, 2]) == "0.90"


def test_total_likelihood_1():
    x_t, x_tt, beta = init_case_1()

    l = spread_likelihood(x_t, beta)
    p = transition_likelihood(l, x_t, x_tt)
    total_lik = total_likelihood(p)

    assert "{:0.8f}".format(total_lik) == "{:0.8f}".format(0.9**5 * 0.10000002**3)


def test_total_likelihood_2():
    x_t, x_tt, beta = init_case_2()

    beta = t.tensor(0.1)

    l = spread_likelihood(x_t, beta)
    p = transition_likelihood(l, x_t, x_tt)

    assert "{:0.8f}".format(total_likelihood(p)) == "{:0.8f}".format(
        0.9**5 * 0.10000002**3 * 2
    )
