#include "../util/computeUtil.h"
#include <cuda.h>
#include <torch/types.h>

// pseudo [E, dim]
// mu [K, dim]
// sigma [K, dim]

#define CEIL(x, y) (((x) + (y)-1) / (y))

__global__ void gaussian(int kernels, int dim, float *pseudo, float *mu,
                         float *inv_sigma, float *gauss) {
  int eid = (blockIdx.x << 5) + threadIdx.x;
  int kid = threadIdx.y;
  float acc = 0;
  for (int d = 0; d < dim; ++d) {
    float tmp = pseudo[eid * dim + d] - mu[kid * dim + d];
    float sig = inv_sigma[kid * dim + d];
    acc += tmp * tmp * sig * sig;
  }
  gauss[eid * kernels + kid] = exp(-0.5 * acc);
}

__global__ void fuseGmm(int kernels, int dim, int embed, int *csrptr,
                        int *colind, float *node_feat, float *pseudo, float *mu,
                        float *inv_sigma, float *out) {
  extern __shared__ float shmem[];
  int ssize = blockDim.x;
  int rid = blockIdx.x;
  int kid = threadIdx.z;
  int fid = (threadIdx.y << 5) + threadIdx.x;
  int lb = csrptr[rid];
  int hb = csrptr[rid + 1];
  int ptr = lb;
  float acc = 0;
  // out[rid * embed + fid] = 0;
  for (; ptr < hb; ptr += ssize) {
    float gauss = 0;
    for (int d = 0; d < dim; ++d) {
      float tmp = pseudo[(ptr + threadIdx.x) * dim + d] - mu[kid * dim + d];
      float sig = inv_sigma[kid * dim + d];
      gauss += tmp * tmp * sig * sig;
    }
    gauss = exp(-0.5 * gauss);
    shmem[threadIdx.x] = gauss;
    __syncwarp();

    for (int cnt = 0; cnt < ssize && cnt + ptr < hb; cnt++) {
      int col = colind[cnt + ptr];
      acc += node_feat[col * embed * kernels + kid * embed + fid] * shmem[cnt];
    }
  }
  if (fid < embed)
    out[rid * embed * kernels + kid * embed + fid] = acc;
}

__global__ void gmm_stash(int kernels, int dim, int embed, int *csrptr,
                          int *colind, float *node_feat, float *pseudo,
                          float *mu, float *inv_sigma, float *out,
                          float *gaussian) {
  extern __shared__ float shmem[];
  int ssize = blockDim.x;
  int rid = blockIdx.x;
  int kid = threadIdx.z;
  int fid = (threadIdx.y << 5) + threadIdx.x;
  int lb = csrptr[rid];
  int hb = csrptr[rid + 1];
  int ptr = lb;
  float acc = 0;
  // out[rid * embed + fid] = 0;
  for (; ptr < hb; ptr += ssize) {
    float gauss = 0;
    for (int d = 0; d < dim; ++d) {
      float tmp = pseudo[(ptr + threadIdx.x) * dim + d] - mu[kid * dim + d];
      float sig = inv_sigma[kid * dim + d];
      gauss += tmp * tmp * sig * sig;
    }
    gauss = exp(-0.5 * gauss);
    gaussian[(ptr + threadIdx.x) * kernels + kid] = gauss;
    shmem[threadIdx.x] = gauss;
    __syncwarp();
    for (int cnt = 0; cnt < ssize && cnt + ptr < hb; cnt++) {
      int col = colind[cnt + ptr];
      acc += node_feat[col * embed * kernels + kid * embed + fid] * shmem[cnt];
    }
  }
  if (fid < embed)
    out[rid * embed * kernels + kid * embed + fid] = acc;
}

__global__ void gaussian_bp(int edges, int kernels, int dim, float *pseudo,
                            float *mu, float *inv_sigma, float *grad_gauss,
                            float *pseudo_out, float *sigma_out,
                            float *mu_out) {
  int eid = (blockIdx.x << 5) + threadIdx.x;
  int kid = threadIdx.y;
  float tmp_mout = 0, tmp_pout = 0, tmp_sout = 0, tmp_gauss = 0;
  for (int d = 0; d < dim; ++d) {
    float sig = inv_sigma[kid * dim + d];
    float pse = pseudo[eid * dim + d];
    float m = mu[kid * dim + d];
    float tmp = (pse - m) * sig;
    tmp_mout += sig * tmp;
    tmp_pout += -tmp_mout;
    tmp_sout += tmp * (m - pse);
    tmp_gauss += tmp * tmp;
  }
  tmp_gauss = exp(-0.5 * tmp_gauss) * grad_gauss[eid * kernels + kid];
  tmp_mout *= tmp_gauss;
  tmp_sout *= tmp_gauss;
  AllReduce<float>(tmp_mout, 16, 32);
  AllReduce<float>(tmp_sout, 16, 32);
  for (int d = 0; d < dim; ++d)
    atomicAdd(&pseudo_out[eid * dim + d], tmp_pout * tmp_gauss);
  if (threadIdx.x == 0) {
    for (int d = 0; d < dim; ++d) {
      atomicAdd(&sigma_out[kid * dim + d], tmp_sout);
      atomicAdd(&mu_out[kid * dim + d], tmp_mout);
    }
  }
}

torch::Tensor gaussian_cuda(torch::Tensor pseudo, torch::Tensor mu,
                            torch::Tensor inv_sigma) {
  const auto edges = pseudo.size(0);
  const auto K = mu.size(0);
  const auto dim = mu.size(1);
  auto devid = pseudo.device().index();
  auto options =
      torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA, devid);
  auto gauss = torch::empty({edges, K}, options);
  gaussian<<<dim3(CEIL(edges, 32), 1, 1), dim3(32, K, 1)>>>(
      K, dim, pseudo.data_ptr<float>(), mu.data_ptr<float>(),
      inv_sigma.data_ptr<float>(), gauss.data_ptr<float>());
  return gauss;
}

torch::Tensor gmmconv_cuda(torch::Tensor csrptr, torch::Tensor colind,
                           torch::Tensor node_feat, torch::Tensor pseudo,
                           torch::Tensor mu, torch::Tensor inv_sigma) {
  const auto edges = pseudo.size(0);
  const auto K = mu.size(0);
  const auto dim = mu.size(1);
  const auto nodes = csrptr.size(0) - 1;
  const auto embed = node_feat.size(2);
  auto devid = pseudo.device().index();
  auto options =
      torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA, devid);
  auto out = torch::empty({nodes, K, embed}, options);
  fuseGmm<<<dim3(nodes, 1, 1), dim3(32, CEIL(embed, 32), K),
            32 * sizeof(float)>>>(
      K, dim, embed, csrptr.data_ptr<int>(), colind.data_ptr<int>(),
      node_feat.data_ptr<float>(), pseudo.data_ptr<float>(),
      mu.data_ptr<float>(), inv_sigma.data_ptr<float>(), out.data_ptr<float>());
  return out;
}

std::vector<torch::Tensor>
gmmconv_stash_cuda(torch::Tensor csrptr, torch::Tensor colind,
                   torch::Tensor node_feat, torch::Tensor pseudo,
                   torch::Tensor mu, torch::Tensor inv_sigma) {
  const auto edges = pseudo.size(0);
  const auto K = mu.size(0);
  const auto dim = mu.size(1);
  const auto nodes = csrptr.size(0) - 1;
  const auto embed = node_feat.size(2);
  auto devid = pseudo.device().index();
  auto options =
      torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA, devid);
  auto out = torch::empty({nodes, K, embed}, options);
  auto gaussian = torch::empty({edges, K}, options);
  gmm_stash<<<dim3(nodes, 1, 1), dim3(32, CEIL(embed, 32), K),
              32 * sizeof(float)>>>(
      K, dim, embed, csrptr.data_ptr<int>(), colind.data_ptr<int>(),
      node_feat.data_ptr<float>(), pseudo.data_ptr<float>(),
      mu.data_ptr<float>(), inv_sigma.data_ptr<float>(), out.data_ptr<float>(),
      gaussian.data_ptr<float>());
  return {out, gaussian};
}

std::vector<torch::Tensor> gaussian_bp_cuda(torch::Tensor pseudo,
                                            torch::Tensor mu,
                                            torch::Tensor inv_sigma,
                                            torch::Tensor grad_out) {
  const auto edges = pseudo.size(0);
  const auto K = mu.size(0);
  const auto dim = mu.size(1);
  auto devid = pseudo.device().index();
  auto options =
      torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA, devid);
  auto pseudo_out = torch::empty({edges, dim}, options);
  auto sigma_out = torch::empty({K, dim}, options);
  auto mu_out = torch::empty({K, dim}, options);

  gaussian_bp<<<dim3(CEIL(edges, 32), 1, 1), dim3(32, K, 1)>>>(
      edges, K, dim, pseudo.data_ptr<float>(), mu.data_ptr<float>(),
      inv_sigma.data_ptr<float>(), grad_out.data_ptr<float>(),
      pseudo_out.data_ptr<float>(), sigma_out.data_ptr<float>(),
      mu_out.data_ptr<float>());
  return {pseudo_out, mu_out, sigma_out};
}
