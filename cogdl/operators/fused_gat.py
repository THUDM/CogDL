from torch.utils.cpp_extension import load
import torch
# import dgNN
import fused_gat

def fused_gat_op(attn_row,attn_col,row_ptr,col_ind,col_ptr,row_ind,negative_slope,in_feat):
    return FusedGATFunction.apply(attn_row,attn_col,row_ptr,col_ind,col_ptr,row_ind,negative_slope,in_feat)


class FusedGATFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx,attn_row,attn_col,row_ptr,col_ind,col_ptr,row_ind,negative_slope,in_feat):
        out_feat,edge_max,edge_sum=fused_gat.gat_forward(attn_row,attn_col,row_ptr,col_ind,negative_slope,in_feat)
        ctx.save_for_backward(row_ptr,col_ind,col_ptr,row_ind,edge_max,edge_sum,in_feat,attn_row,attn_col)
        ctx.negative_slope=negative_slope      
        return out_feat
    
    @staticmethod
    def backward(ctx,grad_out):
        row_ptr,col_ind,col_ptr,row_ind,edge_max,edge_sum,in_feat,attn_row,attn_col=ctx.saved_tensors
        grad_out=grad_out.contiguous()
        # print('start backward')
        grad_feat,grad_attn_row,grad_attn_col=fused_gat.gat_backward(
            ctx.negative_slope,row_ptr,col_ind,col_ptr,row_ind,edge_max,edge_sum,in_feat,attn_row,attn_col,grad_out)
        # print('end backward')
        # print(torch.isnan(grad_feat).sum())
        # print(torch.isnan(grad_attn_row).sum())
        # print(torch.isnan(grad_attn_col).sum())
        return grad_attn_row,grad_attn_col,None,None,None,None,None,grad_feat,None
