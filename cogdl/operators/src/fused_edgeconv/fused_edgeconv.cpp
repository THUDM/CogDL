#include <torch/extension.h>
#include <iostream>
#include <vector>
#include <pybind11/pybind11.h>

std::vector<torch::Tensor> edgeconv_forward_cuda(
    const int k,
    const torch::Tensor dst_ind,
    const torch::Tensor h_src,
    const torch::Tensor h_dst);

std::vector<torch::Tensor> edgeconv_forward(
    const int k,
    const torch::Tensor src_ind,
    const torch::Tensor h_src,
    const torch::Tensor h_dst)
{
    assert(src_ind.device().type() == torch::kCUDA);
    assert(h_src.device().type() == torch::kCUDA);
    assert(h_dst.device().type() == torch::kCUDA);

    assert(src_ind.is_contiguous());
    assert(h_src.is_contiguous());
    assert(h_dst.is_contiguous());

    assert(src_ind.dtype() == torch::kInt32);
    assert(h_src.dtype() == torch::kFloat32);
    assert(h_dst.dtype() == torch::kFloat32);
    return edgeconv_forward_cuda(k, src_ind, h_src, h_dst);
}

torch::Tensor edgeconv_backward_cuda(
    const torch::Tensor grad_out,
    const torch::Tensor max_idx);

torch::Tensor edgeconv_backward(
    const torch::Tensor grad_out,
    const torch::Tensor max_idx)
{
    // printf("111111\n");
    assert(grad_out.device().type() == torch::kCUDA);
    assert(max_idx.device().type() == torch::kCUDA);

    assert(grad_out.is_contiguous());
    assert(max_idx.is_contiguous());

    assert(grad_out.dtype() == torch::kFloat32);
    assert(max_idx.dtype() == torch::kInt32);

    return edgeconv_backward_cuda(grad_out, max_idx);
}

PYBIND11_MODULE(fused_edgeconv, m)
{
    m.doc() = "fuse edgeconv into one kernel";
    m.def("edgeconv_forward", &edgeconv_forward, "fused edgeconv forward op");
    m.def("edgeconv_backward", &edgeconv_backward, "fused edgeconv backward op");
}