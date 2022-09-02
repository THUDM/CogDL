#include <cuda.h>
#include <torch/types.h>
#include "computeUtil.h"


__global__ void mhspmmSimple(
    int v, int nnz, int h, int f,
    int *rowptr, int *colind, float *attention /* E*H */,
    float *infeat /* V*H*F */,
    float *outfeat /* V*H*F */
)
{
  int rid = blockIdx.x;
  int hid = blockIdx.y;
  int lb = rowptr[rid];
  int hb = rowptr[(rid + 1)];
  float acc = 0;
  int offset1, offset2;
  for (int ptr = lb; ptr < hb; ptr++)
  {
    offset1 = colind[ptr] * f * h + hid * f + threadIdx.x;
    float att = attention[ptr * h + hid];
    acc = sum_reduce(acc, infeat[offset1] * att);
  }
  offset2 = rid * f * h + hid * f + threadIdx.x;
  outfeat[offset2] = acc;
}


__global__ void mhspmm_1(
  int v, int nnz, int h, int f,
  /*our spmm takes in-edge csr (row:dst, column:src)*/
  int *rowptr, int *colind, float *attention /* E*H */, //H*E
  float *infeat /* V*H*F */,
  float *outfeat /* V*H*F */
)
{
  int rid = blockIdx.x;
  int hid = threadIdx.x;
  int fid = threadIdx.y;
  int lb = rowptr[rid];
  int hb = rowptr[(rid + 1)];
  float acc = 0;
  for (int ptr = lb; ptr < hb; ptr++)
  {
    int offset1 = colind[ptr] * f * h + hid * f + fid;
    float att = attention[ptr * h + hid];
    acc += att * infeat[offset1];
  }
  outfeat[rid * h * f + hid * f + fid] = acc;
}

torch::Tensor mhspmm_cuda(
    torch::Tensor rowptr,
    torch::Tensor colind,
    torch::Tensor attention,
    torch::Tensor infeat)
{
  const auto v = rowptr.size(0) - 1;
  const auto nnz = colind.size(0);
  const auto h = attention.size(1);
  const auto f = infeat.size(2);
  auto devid = infeat.device().index();
  auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA, devid);
  auto outfeat = torch::empty({v, h, f}, options);
  if((h * f < 1024) && (f < 32))
  {
    mhspmm_1<<<dim3(v, 1, 1), dim3(h, f, 1)>>>(v, nnz, h, f, rowptr.data_ptr<int>(), colind.data_ptr<int>(),
    attention.data_ptr<float>(), infeat.data_ptr<float>(), outfeat.data_ptr<float>());
  }
  else
  {
    mhspmmSimple<<<dim3(v, h, 1), dim3(f, 1, 1)>>>(v, nnz, h, f, rowptr.data_ptr<int>(), colind.data_ptr<int>(),
    attention.data_ptr<float>(), infeat.data_ptr<float>(), outfeat.data_ptr<float>());
  }
  return outfeat;
}
