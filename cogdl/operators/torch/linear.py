import math
import torch
import torch.nn.functional as F


try:
    from actnn.ops import quantize_activation, dequantize_activation
    from actnn.conf import config
    from actnn.utils import empty_cache
except Exception:
    pass


class linear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, bias=None, scheme=None, rp_ratio=2):
        if rp_ratio > 1:
            D = input.shape[1]
            rmat = (torch.bernoulli(torch.ones((D, D // rp_ratio)).to(input.device) * 0.5) * 2.0 - 1) * math.sqrt(
                1.0 / (D // rp_ratio)
            )
            input_rp = torch.mm(input, rmat)
            quantized = quantize_activation(input_rp, scheme)
        else:
            quantized = quantize_activation(input, scheme)

        empty_cache(config.empty_cache_threshold)

        ctx.scheme = scheme
        if rp_ratio > 1:
            ctx.saved = quantized, weight, bias, rmat
            ctx.other_args = input_rp.shape
        else:
            ctx.saved = quantized, weight, bias
            ctx.other_args = input.shape
        return F.linear(input, weight, bias)

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.scheme:
            ctx.scheme.set_scale(grad_output)

        q_input_shape = ctx.other_args
        if len(ctx.saved) == 4:
            quantized, weight, bias, rmat = ctx.saved
            input_rp = dequantize_activation(quantized, q_input_shape)
            input = torch.mm(input_rp, rmat.t())
            del quantized, ctx.saved, input_rp
        else:
            quantized, weight, bias = ctx.saved
            input = dequantize_activation(quantized, q_input_shape)
            del quantized, ctx.saved

        empty_cache(config.empty_cache_threshold)

        C_in = input.shape[-1]
        C_out = grad_output.shape[-1]

        grad_output_flatten = grad_output.view(-1, C_out)
        input_flatten = input.view(-1, C_in)
        grad_input = grad_output_flatten.mm(weight)
        grad_weight = grad_output_flatten.t().mm(input_flatten)

        if bias is not None:
            grad_bias = grad_output_flatten.sum(0)
        else:
            grad_bias = None

        if ctx.scheme:
            ctx.scheme.if_allocate_perlayer()
        return grad_input, grad_weight, grad_bias, None, None
