#include <torch/extension.h>
#include <iostream>
#include <vector>
#include <pybind11/pybind11.h>
#include <c10/cuda/CUDAGuard.h>

torch::Tensor mhspmm_cuda(
    torch::Tensor rowptr,
    torch::Tensor colind,
    torch::Tensor attention,
    torch::Tensor infeat);

torch::Tensor mhspmm(
    torch::Tensor rowptr,
    torch::Tensor colind,
    torch::Tensor attention,
    torch::Tensor infeat)
{
    assert(rowptr.device().type() == torch::kCUDA);
    assert(colind.device().type() == torch::kCUDA);
    assert(attention.device().type() == torch::kCUDA);
    assert(infeat.device().type() == torch::kCUDA);
    assert(rowptr.is_contiguous());
    assert(colind.is_contiguous());
    assert(attention.is_contiguous());
    assert(infeat.is_contiguous());
    assert(rowptr.dtype() == torch::kInt32);
    assert(colind.dtype() == torch::kInt32);
    assert(attention.dtype() == torch::kFloat32);
    assert(infeat.dtype() == torch::kFloat32);
    const at::cuda::OptionalCUDAGuard device_guard1(device_of(rowptr));
    const at::cuda::OptionalCUDAGuard device_guard2(device_of(colind));
    const at::cuda::OptionalCUDAGuard device_guard3(device_of(attention));
    const at::cuda::OptionalCUDAGuard device_guard4(device_of(infeat));
    return mhspmm_cuda(rowptr, colind, attention, infeat);
}

PYBIND11_MODULE(mhspmm, m)
{
    m.doc() = "mhtranspose in CSR format. ";
    m.def("mhspmm", &mhspmm, "CSR mhsddmm");
}
