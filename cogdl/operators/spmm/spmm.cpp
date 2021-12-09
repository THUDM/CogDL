#include <torch/extension.h>
#include <iostream>
#include <vector>
#include <pybind11/pybind11.h>
#include <c10/cuda/CUDAGuard.h>

torch::Tensor spmm_cuda(
    torch::Tensor rowptr,
    torch::Tensor colind,
    torch::Tensor values,
    torch::Tensor dense);

torch::Tensor spmm_cuda_no_edge_value(
    torch::Tensor rowptr,
    torch::Tensor colind,
    torch::Tensor dense);

// #define CHECK_CUDA(x) TORCH_CHECK(x.type().is_cuda(), #x " must be a cuda tensor")
// #define CHECK_CONIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
// #define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONIGUOUS(x)

torch::Tensor csr_spmm(
    torch::Tensor A_rowptr,
    torch::Tensor A_colind,
    torch::Tensor A_csrVal,
    torch::Tensor B)
{
    assert(A_rowptr.device().type() == torch::kCUDA);
    assert(A_colind.device().type() == torch::kCUDA);
    assert(A_csrVal.device().type() == torch::kCUDA);
    assert(B.device().type() == torch::kCUDA);
    assert(A_rowptr.is_contiguous());
    assert(A_colind.is_contiguous());
    assert(A_csrVal.is_contiguous());
    assert(B.is_contiguous());
    assert(A_rowptr.dtype() == torch::kInt32);
    assert(A_colind.dtype() == torch::kInt32);
    assert(A_csrVal.dtype() == torch::kFloat32);
    assert(B.dtype() == torch::kFloat32);
    const at::cuda::OptionalCUDAGuard device_guard1(device_of(A_rowptr));
    const at::cuda::OptionalCUDAGuard device_guard2(device_of(A_colind));
    const at::cuda::OptionalCUDAGuard device_guard3(device_of(A_csrVal));
    const at::cuda::OptionalCUDAGuard device_guard4(device_of(B));
    return spmm_cuda(A_rowptr, A_colind, A_csrVal, B);
}

torch::Tensor csr_spmm_no_edge_value(
    torch::Tensor A_rowptr,
    torch::Tensor A_colind,
    torch::Tensor B)
{
    assert(A_rowptr.device().type() == torch::kCUDA);
    assert(A_colind.device().type() == torch::kCUDA);
    assert(B.device().type() == torch::kCUDA);
    assert(A_rowptr.is_contiguous());
    assert(A_colind.is_contiguous());
    assert(B.is_contiguous());
    assert(A_rowptr.dtype() == torch::kInt32);
    assert(A_colind.dtype() == torch::kInt32);
    assert(B.dtype() == torch::kFloat32);
    const at::cuda::OptionalCUDAGuard device_guard1(device_of(A_rowptr));
    const at::cuda::OptionalCUDAGuard device_guard2(device_of(A_colind));
    const at::cuda::OptionalCUDAGuard device_guard3(device_of(B));
    return spmm_cuda_no_edge_value(A_rowptr, A_colind, B);
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
    assert(csr_data.dtype() == torch::kFloat32);
    const at::cuda::OptionalCUDAGuard device_guard1(device_of(rowptr));
    const at::cuda::OptionalCUDAGuard device_guard2(device_of(colind));
    const at::cuda::OptionalCUDAGuard device_guard3(device_of(csr_data));
    return csr2csc_cuda(rowptr, colind, csr_data);
}

PYBIND11_MODULE(spmm, m)
{
    m.doc() = "spmm in CSR format. csr_spmm is the kernel with edge value. csr2csc provides the format transformation";
    m.def("csr_spmm", &csr_spmm, "CSR SPMM");
    m.def("csr_spmm_no_edge_value", &csr_spmm_no_edge_value, "CSR SPMM NO EDGE VALUE");
    m.def("csr2csc", &csr2csc, "csr2csc");
}
