#include <torch/extension.h>
#include <iostream>
#include <vector>
#include <pybind11/pybind11.h>

torch::Tensor spmm_cpu(
    torch::Tensor rowptr,
    torch::Tensor colind,
    torch::Tensor values,
    torch::Tensor dense)
{
    const auto m = rowptr.size(0)-1;
    const auto k = dense.size(1);
    auto devid = dense.device().index();
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU, devid);
    auto out = torch::zeros({m,k}, options);
    
    int *rowptr_ptr = rowptr.data_ptr<int>();
    int *colind_ptr = colind.data_ptr<int>();
    float *values_ptr = values.data_ptr<float>();
    float *dense_ptr = dense.data_ptr<float>();
    float *out_ptr = out.data_ptr<float>();

    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < m; ++i) {
        int row_start = rowptr_ptr[i], row_end = rowptr_ptr[i + 1];
        int ik = i * k;
        for (int key = row_start; key < row_end; ++key) {
            int j = colind_ptr[key] * k;
            float val = values_ptr[key];
            for (int t = 0; t < k; ++t) {
                out_ptr[ik + t] += val * dense_ptr[j + t];
            }
        }
    }
    return out;
}

torch::Tensor csr_spmm_cpu(
    torch::Tensor A_rowptr,
    torch::Tensor A_colind,
    torch::Tensor A_csrVal,
    torch::Tensor B)
{
    assert(A_rowptr.device().type() == torch::kCPU);
    assert(A_colind.device().type() == torch::kCPU);
    assert(A_csrVal.device().type() == torch::kCPU);
    assert(B.device().type() == torch::kCPU);
    assert(A_rowptr.is_contiguous());
    assert(A_colind.is_contiguous());
    assert(A_csrVal.is_contiguous());
    assert(B.is_contiguous());
    assert(A_rowptr.dtype() == torch::kInt32);
    assert(A_colind.dtype() == torch::kInt32);
    assert(A_csrVal.dtype() == torch::kFloat32);
    assert(B.dtype() == torch::kFloat32);
    return spmm_cpu(A_rowptr, A_colind, A_csrVal, B);
}

PYBIND11_MODULE(spmm_cpu, m)
{
    m.doc() = "spmm_cpu in CSR format.";
    m.def("csr_spmm_cpu", &csr_spmm_cpu, "CSR SPMM (CPU)");
}