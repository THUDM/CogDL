#include "../util/computeUtil.h"
#include <cuda.h>
#include <cusparse.h>
#include <torch/types.h>

__global__ void mhtranspose(const int nnz, const int h, const int *permute,
                            float *attention, float *out) {
  int hid = blockIdx.y;
  int nid = blockIdx.x * 32 + threadIdx.x;
  if (nid < nnz) {
    int idx = permute[nid];
    out[nid * h + hid] = attention[idx * h + hid];
  }
}

__global__ void mhtranspose4(const int nnz, const int h, int *permute,
                             float *attention, float *out) {
  int hid = threadIdx.y << 2;
  int nid = blockIdx.x * 32 + threadIdx.x;
  if (nid < nnz) {
    int idx = permute[nid];
    float att[4];
    Load<float4, float>(att, attention, idx * h + hid);
    Store<float4, float>(out, att, nid * h + hid);
  }
}

torch::Tensor mhtranspose_cuda(torch::Tensor permute,
                               torch::Tensor attention // E * H
) {
  const auto nnz = permute.size(0);
  const auto h = attention.size(1);
  auto devid = permute.device().index();
  auto options =
      torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA, devid);
  auto out = torch::empty({nnz, h}, options);
  if ((h & 3) == 0) {
    mhtranspose4<<<dim3(CEIL(nnz, 32), 1, 1), dim3(32, CEIL(h, 4), 1)>>>(
        nnz, h, permute.data_ptr<int>(), attention.data_ptr<float>(),
        out.data_ptr<float>());
  } else {
    mhtranspose<<<dim3(CEIL(nnz, 32), h, 1), dim3(32, 1, 1)>>>(
        nnz, h, permute.data_ptr<int>(), attention.data_ptr<float>(),
        out.data_ptr<float>());
  }
  return out;
}
