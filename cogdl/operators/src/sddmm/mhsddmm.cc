#include <iostream>
#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include <vector>

torch::Tensor mhsddmm_cuda(torch::Tensor rowptr, torch::Tensor colind,
                           torch::Tensor grad,   // V * H * F
                           torch::Tensor feature // V * H * F
);

torch::Tensor mhsddmm(torch::Tensor rowptr, torch::Tensor colind,
                      torch::Tensor grad,   // V * H * F
                      torch::Tensor feature // V * H * F
) {
  assert(rowptr.device().type() == torch::kCUDA);
  assert(colind.device().type() == torch::kCUDA);
  assert(grad.device().type() == torch::kCUDA);
  assert(feature.device().type() == torch::kCUDA);
  assert(rowptr.is_contiguous());
  assert(colind.is_contiguous());
  assert(grad.is_contiguous());
  assert(feature.is_contiguous());
  assert(rowptr.dtype() == torch::kInt32);
  assert(colind.dtype() == torch::kInt32);
  assert(grad.dtype() == torch::kFloat32);
  assert(feature.dtype() == torch::kFloat32);
  return mhsddmm_cuda(rowptr, colind, grad, feature);
}

PYBIND11_MODULE(mhsddmm, m) {
  m.doc() = "mhsddmm in CSR format. ";
  m.def("mhsddmm", &mhsddmm, "CSR mhsddmm");
}