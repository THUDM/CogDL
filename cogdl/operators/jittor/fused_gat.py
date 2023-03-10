import jittor

try:
    import fused_gatconv

    def fused_gat_func(attn_row, attn_col, row_ptr, col_ind, col_ptr, row_ind, negative_slope, in_feat):
        return FusedGATFunction.apply(attn_row, attn_col, row_ptr, col_ind, col_ptr, row_ind, negative_slope, in_feat)


except Exception:
    fused_gat_func = None


class FusedGATFunction(jittor.Function):
    @staticmethod
    def execute(self, attn_row, attn_col, row_ptr, col_ind, col_ptr, row_ind, negative_slope, in_feat):
        out_feat, edge_max, edge_sum = fused_gatconv.gat_forward(
            attn_row, attn_col, row_ptr, col_ind, negative_slope, in_feat
        )
        self.backward_fused = (row_ptr, col_ind, col_ptr, row_ind, edge_max, edge_sum, in_feat, attn_row, attn_col)
        self.negative_slope = negative_slope
        return out_feat

    @staticmethod
    def grad(self, grad_out):
        row_ptr, col_ind, col_ptr, row_ind, edge_max, edge_sum, in_feat, attn_row, attn_col = self.backward_fused
        grad_feat, grad_attn_row, grad_attn_col = fused_gatconv.gat_backward(
            self.negative_slope,
            row_ptr,
            col_ind,
            col_ptr,
            row_ind,
            edge_max,
            edge_sum,
            in_feat,
            attn_row,
            attn_col,
            grad_out,
        )
        return grad_attn_row, grad_attn_col, None, None, None, None, None, grad_feat, None
