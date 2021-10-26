from torch.utils.cpp_extension import load
import torch
import os

import fused_edgeconv


def fused_edgeconv_op(k,src_ind,h_src,h_dst):
    return FusedEdgeConvFunction.apply(k,src_ind,h_src,h_dst)

class FusedEdgeConvFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx,k,src_ind,h_src,h_dst):
        out_feat,max_idx=fused_edgeconv.edgeconv_forward(k,src_ind,h_src,h_dst)
        ctx.save_for_backward(max_idx)
        return out_feat
    
    @staticmethod
    def backward(ctx,grad_out):
        max_idx=ctx.saved_tensors[0].int()       
        grad_src=fused_edgeconv.edgeconv_backward(grad_out,max_idx)
        return None,None,grad_src,grad_out
