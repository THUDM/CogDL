#include <iostream>
#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include <vector>

torch::Tensor mhtranspose_cuda(torch::Tensor permute,
                               torch::Tensor attention // E * H
);

torch::Tensor mhtranspose(torch::Tensor permute, torch::Tensor attention) {
  assert(permute.device().type() == torch::kCUDA);
  assert(attention.device().type() == torch::kCUDA);
  assert(permute.is_contiguous());
  assert(attention.is_contiguous());
  assert(permute.dtype() == torch::kInt32);
  assert(attention.dtype() == torch::kFloat32);
  return mhtranspose_cuda(permute, attention);
}

PYBIND11_MODULE(mhtranspose, m) {
  m.doc() = "mhtranspose in CSR format. ";
  m.def("mhtranspose", &mhtranspose, "CSR mhsddmm");
}