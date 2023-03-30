import torch as t


class STEFunction(t.autograd.Function):
    @staticmethod
    def forward(ctx, p):
        return (p > t.rand(1)).float()

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output
