#include <torch/extension.h>
#include <iostream>
#include <vector>
#include <pybind11/pybind11.h>
#include <c10/cuda/CUDAGuard.h>

torch::Tensor sddmm_cuda_coo(
    torch::Tensor rowind,
    torch::Tensor colind,
    torch::Tensor D1,
    torch::Tensor D2
);

torch::Tensor sddmm_cuda_csr(
    torch::Tensor rowptr,
    torch::Tensor colind,
    torch::Tensor D1,
    torch::Tensor D2
);


torch::Tensor coo_sddmm(
    torch::Tensor rowind,
    torch::Tensor colind,
    torch::Tensor D1,
    torch::Tensor D2
) {
    assert(rowind.device().type()==torch::kCUDA);
    assert(colind.device().type()==torch::kCUDA);
    assert(D1.device().type()==torch::kCUDA);
    assert(D2.device().type()==torch::kCUDA);
    assert(rowind.is_contiguous());
    assert(colind.is_contiguous());
    assert(D1.is_contiguous());
    assert(D2.is_contiguous());
    assert(rowind.dtype()==torch::kInt32);
    assert(colind.dtype()==torch::kInt32);
    assert(D1.dtype()==torch::kFloat32);
    assert(D2.dtype()==torch::kFloat32);
    const at::cuda::OptionalCUDAGuard device_guard1(device_of(rowind));
    const at::cuda::OptionalCUDAGuard device_guard2(device_of(colind));
    const at::cuda::OptionalCUDAGuard device_guard3(device_of(D1));
    const at::cuda::OptionalCUDAGuard device_guard4(device_of(D2));
    return sddmm_cuda_coo(rowind, colind, D1, D2);
}

torch::Tensor csr_sddmm(
    torch::Tensor rowptr,
    torch::Tensor colind,
    torch::Tensor D1,
    torch::Tensor D2
) {
    assert(rowptr.device().type()==torch::kCUDA);
    assert(colind.device().type()==torch::kCUDA);
    assert(D1.device().type()==torch::kCUDA);
    assert(D2.device().type()==torch::kCUDA);
    assert(rowptr.is_contiguous());
    assert(colind.is_contiguous());
    assert(D1.is_contiguous());
    assert(D2.is_contiguous());
    assert(rowptr.dtype()==torch::kInt32);
    assert(colind.dtype()==torch::kInt32);
    assert(D1.dtype()==torch::kFloat32);
    assert(D2.dtype()==torch::kFloat32);
    const at::cuda::OptionalCUDAGuard device_guard1(device_of(rowptr));
    const at::cuda::OptionalCUDAGuard device_guard2(device_of(colind));
    const at::cuda::OptionalCUDAGuard device_guard3(device_of(D1));
    const at::cuda::OptionalCUDAGuard device_guard4(device_of(D2));
    return sddmm_cuda_csr(rowptr, colind, D1, D2);
}

PYBIND11_MODULE(sddmm, m)
{
    m.doc() = "SDDMM kernel. Format of COO and CSR are provided.";
    m.def("coo_sddmm", &coo_sddmm, "COO SDDMM");
    m.def("csr_sddmm", &csr_sddmm, "CSR SDDMM");
}
