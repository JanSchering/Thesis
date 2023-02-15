import torch as t


class STEFunction(t.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return (input > 0).float()

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


class STELogicalOr(t.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return t.any(input == 1).float()

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output
