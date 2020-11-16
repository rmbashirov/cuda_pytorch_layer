import torch.nn as nn
from torch.autograd import Function

from cuda_pytorch_layer.cuda.mult import mult_forward as mult_forward_cuda
from cuda_pytorch_layer.cuda.mult import mult_backward as mult_backward_cuda


class MultFunction(Function):
    @staticmethod
    def forward(ctx, a, b):
        ctx.save_for_backward(a, b)
        product = mult_forward_cuda(a, b)
        return product

    @staticmethod
    def backward(ctx, dresult):
        a, b = ctx.saved_tensors
        da, db = mult_backward_cuda(dresult, a, b)
        return da, db


class Mult(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, a, b):
        return MultFunction.apply(a, b)

