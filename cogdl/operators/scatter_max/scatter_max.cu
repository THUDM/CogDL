#include <cuda.h>
#include <torch/types.h>
#include <vector>

__global__ void scatter_max_forward(const int *A_indptr, const int *A_indices,
                                    const float *B, float *C, int *max_mask) {
  int rid = blockDim.y * blockIdx.x + threadIdx.y;
  int m = gridDim.x;
  int k = blockDim.x;
  if (rid < m) {
    int lb = A_indptr[rid];
    int hb = A_indptr[(rid + 1)];
    int stride = hb - lb;
    int offset;
    int max_id;
    float acc = (stride > 0) ? FLT_MIN : 0;
    for (int ptr = lb; ptr < hb; ptr++) {
      int cid = A_indices[ptr];
      offset = cid * k + threadIdx.x;
      if (acc < B[offset]) {
        acc = B[offset];
        max_id = cid;
      }
    }
    C[(rid * k + threadIdx.x)] = acc;
    max_mask[(rid * k + threadIdx.x)] = max_id;
  }
}

__global__ void scatter_max_backward(const float *grad, float *out,
                                     int *max_mask) {
  int rid = blockDim.y * blockIdx.x + threadIdx.y;
  int m = gridDim.x;
  int k = blockDim.x;
  if (rid < m) {
    int offset = rid * k + threadIdx.x;
    int max_id;
    max_id = max_mask[offset]; // max mapping
    float grad_tmp = grad[offset];
    atomicAdd(&out[max_id * k + threadIdx.x], grad_tmp);
  }
}

std::vector<torch::Tensor> scatter_max_fp_cuda(torch::Tensor rowptr,
                                               torch::Tensor colind,
                                               torch::Tensor node_feature) {
  const long m = rowptr.size(0) - 1;
  const long k = node_feature.size(1);
  auto devid = node_feature.device().index();
  auto optionsI =
      torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA, devid);
  auto optionsF =
      torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA, devid);
  auto max_mask = torch::empty({m, k}, optionsI);
  auto out = torch::empty({m, k}, optionsF);
  scatter_max_forward<<<m, k>>>(rowptr.data_ptr<int>(), colind.data_ptr<int>(),
                                node_feature.data_ptr<float>(),
                                out.data_ptr<float>(),
                                max_mask.data_ptr<int>());
  return {out, max_mask};
}

torch::Tensor scatter_max_bp_cuda(torch::Tensor node_feature,
                                  torch::Tensor max_mask) {
  const long m = node_feature.size(0);
  const long k = node_feature.size(1);
  auto devid = node_feature.device().index();
  auto options =
      torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA, devid);
  auto out = torch::empty({m, k}, options);
  scatter_max_backward<<<m, k>>>(node_feature.data_ptr<float>(),
                                 out.data_ptr<float>(),
                                 max_mask.data_ptr<int>());
  return out;
}