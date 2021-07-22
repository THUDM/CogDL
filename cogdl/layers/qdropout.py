import torch
from torch import nn
import torch.nn.functional as F
from actnn.ops import quantize_activation, dequantize_activation


class dropout_func(torch.autograd.Function):
    """
    We can implement our own custom autograd Functions by subclassing
    torch.autograd.Function and implementing the forward and backward passes
    which operate on Tensors.
    """

    @staticmethod
    def forward(ctx, input, p=0.5):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """
        out = F.dropout(input, p)
        mask = (out == 0).float()
        quantized = quantize_activation(mask, None)
        ctx.backward_tensors = quantized
        ctx.other_args = mask.shape
        return out

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """
        quantized = ctx.backward_tensors
        q_input_shape = ctx.other_args
        mask = dequantize_activation(quantized, q_input_shape).bool()
        del quantized, ctx.backward_tensors

        grad_input = grad_output.clone()
        grad_input[mask] = 0
        return grad_input, None


class QDropout(nn.Dropout):
    def __init__(self, p, inplace=False):
        super(QDropout, self).__init__(p, inplace)

    def forward(self, input):
        if self.training:
            return dropout_func.apply(input, self.p)
        else:
            return super(QDropout, self).forward(input)
