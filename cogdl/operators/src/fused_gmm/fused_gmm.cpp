#include <iostream>
#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include <vector>

void assertTensor(torch::Tensor &T, torch::ScalarType type)
{
  assert(T.is_contiguous());
  assert(T.device().type() == torch::kCUDA);
  assert(T.dtype() == type);
}

torch::Tensor gaussian_cuda(torch::Tensor pseudo, torch::Tensor mu,
                            torch::Tensor inv_sigma);

torch::Tensor gmmconv_cuda(torch::Tensor csrptr, torch::Tensor colind,
                           torch::Tensor node_feat, torch::Tensor pseudo,
                           torch::Tensor mu, torch::Tensor inv_sigma);

std::vector<torch::Tensor>
gmmconv_stash_cuda(torch::Tensor csrptr, torch::Tensor colind,
                   torch::Tensor node_feat, torch::Tensor pseudo,
                   torch::Tensor mu, torch::Tensor inv_sigma);

std::vector<torch::Tensor> gaussian_bp_cuda(torch::Tensor pseudo,
                                            torch::Tensor mu,
                                            torch::Tensor inv_sigma,
                                            torch::Tensor grad_out);

torch::Tensor Gaussian(torch::Tensor pseudo, torch::Tensor mu,
                       torch::Tensor inv_sigma)
{
  assertTensor(pseudo, torch::kFloat32);
  assertTensor(mu, torch::kFloat32);
  assertTensor(inv_sigma, torch::kFloat32);
  return gaussian_cuda(pseudo, mu, inv_sigma);
}

torch::Tensor GmmConv(torch::Tensor csrptr, torch::Tensor colind,
                      torch::Tensor node_feat, torch::Tensor pseudo,
                      torch::Tensor mu, torch::Tensor inv_sigma)
{
  assertTensor(csrptr, torch::kInt32);
  assertTensor(colind, torch::kInt32);
  assertTensor(node_feat, torch::kFloat32);
  assertTensor(pseudo, torch::kFloat32);
  assertTensor(mu, torch::kFloat32);
  assertTensor(inv_sigma, torch::kFloat32);
  return gmmconv_cuda(csrptr, colind, node_feat, pseudo, mu, inv_sigma);
}

std::vector<torch::Tensor> GmmStash(torch::Tensor csrptr, torch::Tensor colind,
                                    torch::Tensor node_feat,
                                    torch::Tensor pseudo, torch::Tensor mu,
                                    torch::Tensor inv_sigma)
{
  assertTensor(csrptr, torch::kInt32);
  assertTensor(colind, torch::kInt32);
  assertTensor(node_feat, torch::kFloat32);
  assertTensor(pseudo, torch::kFloat32);
  assertTensor(mu, torch::kFloat32);
  assertTensor(inv_sigma, torch::kFloat32);
  return gmmconv_stash_cuda(csrptr, colind, node_feat, pseudo, mu, inv_sigma);
}

std::vector<torch::Tensor> GaussianBp(torch::Tensor pseudo, torch::Tensor mu,
                                      torch::Tensor inv_sigma,
                                      torch::Tensor grad_out)
{
  assertTensor(pseudo, torch::kFloat32);
  assertTensor(mu, torch::kFloat32);
  assertTensor(inv_sigma, torch::kFloat32);
  assertTensor(grad_out, torch::kFloat32);
  return gaussian_bp_cuda(pseudo, mu, inv_sigma, grad_out);
}

PYBIND11_MODULE(fused_gmm, m)
{
  m.doc() = "gmmconv. ";
  m.def("gmmconv", &GmmConv, "gmmconv");
  m.def("gmmconvstash", &GmmStash, "gmmstash");
  m.def("Gaussian", &Gaussian, "Gaussian");
  m.def("GaussianBp", &GaussianBp, "Gaussian");
}