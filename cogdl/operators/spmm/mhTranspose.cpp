#include <torch/extension.h>
#include <iostream>
#include <vector>
#include <pybind11/pybind11.h>
#include <c10/cuda/CUDAGuard.h>

torch::Tensor mhtranspose_cuda(
    torch::Tensor permute,
    torch::Tensor attention // E * H
);

torch::Tensor mhtranspose(
    torch::Tensor permute,
    torch::Tensor attention)
{
    assert(permute.device().type() == torch::kCUDA);
    assert(attention.device().type() == torch::kCUDA);
    assert(permute.is_contiguous());
    assert(attention.is_contiguous());
    assert(permute.dtype() == torch::kInt32);
    assert(attention.dtype() == torch::kFloat32);
    const at::cuda::OptionalCUDAGuard device_guard1(device_of(permute));
    const at::cuda::OptionalCUDAGuard device_guard2(device_of(attention));
    return mhtranspose_cuda(permute, attention);
}

std::vector<torch::Tensor> csr2csc_cuda(
    torch::Tensor csrRowPtr,
    torch::Tensor csrColInd,
    torch::Tensor csrVal);

std::vector<torch::Tensor> csr2csc(
    torch::Tensor rowptr,
    torch::Tensor colind,
    torch::Tensor csr_data)
{
    assert(rowptr.device().type() == torch::kCUDA);
    assert(colind.device().type() == torch::kCUDA);
    assert(csr_data.device().type() == torch::kCUDA);
    assert(rowptr.is_contiguous());
    assert(colind.is_contiguous());
    assert(csr_data.is_contiguous());
    assert(rowptr.dtype() == torch::kInt32);
    assert(colind.dtype() == torch::kInt32);
    assert(csr_data.dtype() == torch::kInt32);
    const at::cuda::OptionalCUDAGuard device_guard1(device_of(rowptr));
    const at::cuda::OptionalCUDAGuard device_guard2(device_of(colind));
    const at::cuda::OptionalCUDAGuard device_guard3(device_of(csr_data));
    return csr2csc_cuda(rowptr, colind, csr_data);
}

PYBIND11_MODULE(mhtranspose, m)
{
    m.doc() = "mhtranspose in CSR format. ";
    m.def("mhtranspose", &mhtranspose, "CSR mhsddmm");
    m.def("csr2csc", &csr2csc, "csr2csc");
}
