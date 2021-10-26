#include "../util/computeUtil.h"
#include <cuda.h>
#include <torch/types.h>

// #define MAX(a, b) ((a < b) ? (b) : (a))
#define LeakyRelu(x, negative_slope) ((x > 0) ? (x) : ((x)*negative_slope))
using namespace std;

__global__ void spmm_test0(int A_nrows, int B_ncols, const int *A_csrRowPtr,
                           const int *A_csrColInd, const float *A_csrVal,
                           const float *B_dnVal, float *C_dnVal)
{
  int rid = blockDim.y * blockIdx.x + threadIdx.y;
  if (rid < A_nrows)
  {
    int cid = (blockIdx.y << 5) + threadIdx.x;
    int lb = A_csrRowPtr[rid];
    int hb = A_csrRowPtr[(rid + 1)];
    int offset = 0;
    float acc = 0;
    if (blockIdx.y != gridDim.y - 1)
    {
      for (int ptr = lb; ptr < hb; ptr++)
      {
        offset = A_csrColInd[ptr] * B_ncols + cid;
        acc += A_csrVal[ptr] * B_dnVal[offset];
      }
      C_dnVal[(rid * B_ncols + cid)] = acc;
    }
    else
    {
      for (int ptr = lb; ptr < hb; ptr++)
      {
        if (cid < B_ncols)
        {
          offset = A_csrColInd[ptr] * B_ncols + cid;
        }
        acc += A_csrVal[ptr] * B_dnVal[offset];
      }
      if (cid < B_ncols)
      {
        C_dnVal[(rid * B_ncols + cid)] = acc;
      }
    }
  }
}

__global__ void spmm_test1(int A_nrows, int B_ncols, const int *A_csrRowPtr,
                           const int *A_csrColInd, const float *A_csrVal,
                           const float *B_dnVal, float *C_dnVal)
{
  extern __shared__ int sh[];
  int *colInd_sh = sh;
  float *val_sh = (float *)&sh[(blockDim.y << 5)];
  int shmem_offset = (threadIdx.y << 5);
  int thread_idx = shmem_offset + threadIdx.x;

  int rid = blockDim.y * blockIdx.x + threadIdx.y;

  if (rid < A_nrows)
  {
    int cid = (blockIdx.y << 5) + threadIdx.x;
    int lb = A_csrRowPtr[rid];
    int hb = A_csrRowPtr[(rid + 1)];
    int ptr = lb + threadIdx.x;
    int offset;
    float acc = 0;

    if (blockIdx.y != gridDim.y - 1)
    {
      for (int jj = lb; jj < hb; jj += 32)
      {
        if (ptr < hb)
        {
          val_sh[thread_idx] = A_csrVal[ptr];
          colInd_sh[thread_idx] = B_ncols * A_csrColInd[ptr];
        }
        __syncwarp();
        ptr += 32;

        for (int kk = 0; kk < 32 && jj + kk < hb; kk++)
        {
          offset = colInd_sh[(shmem_offset + kk)] + cid;
          acc += val_sh[(shmem_offset + kk)] * B_dnVal[offset];
        }
        __syncwarp();
      }
      C_dnVal[(rid * B_ncols + cid)] = acc;
    }
    else
    {
      for (int jj = lb; jj < hb; jj += 32)
      {
        if (ptr < hb)
        {
          val_sh[thread_idx] = A_csrVal[ptr];
          colInd_sh[thread_idx] = B_ncols * A_csrColInd[ptr];
        }
        __syncwarp();
        ptr += 32;

        for (int kk = 0; kk < 32 && jj + kk < hb; kk++)
        {
          offset = colInd_sh[(shmem_offset + kk)] + cid;
          if (cid < B_ncols)
          {
            acc += val_sh[(shmem_offset + kk)] * B_dnVal[offset];
          }
        }
        __syncwarp();
      }
      if (cid < B_ncols)
      {
        C_dnVal[(rid * B_ncols + cid)] = acc;
      }
    }
  }
}

__global__ void spmm_test2(int A_nrows, int B_ncols, const int *A_csrRowPtr,
                           const int *A_csrColInd, const float *A_csrVal,
                           const float *B_dnVal, float *C_dnVal)
{
  extern __shared__ int sh[];
  int *colInd_sh = sh;
  float *val_sh = (float *)&sh[(blockDim.y << 5)];
  int shmem_offset = (threadIdx.y << 5);
  int thread_idx = shmem_offset + threadIdx.x;

  int rid = blockDim.y * blockIdx.x + threadIdx.y;

  if (rid < A_nrows)
  {
    int cid = (blockIdx.y << 6) + threadIdx.x;
    int lb = A_csrRowPtr[rid];
    int hb = A_csrRowPtr[(rid + 1)];
    int ptr = lb + threadIdx.x;
    int offset;
    float acc1 = 0, acc2 = 0, val;

    if (blockIdx.y != gridDim.y - 1)
    {
      for (int jj = lb; jj < hb; jj += 32)
      {
        if (ptr < hb)
        {
          val_sh[thread_idx] = A_csrVal[ptr];
          colInd_sh[thread_idx] = B_ncols * A_csrColInd[ptr];
        }
        __syncwarp();
        ptr += 32;

        for (int kk = 0; kk < 32 && jj + kk < hb; kk++)
        {
          offset = colInd_sh[(shmem_offset + kk)] + cid;
          val = val_sh[(shmem_offset + kk)];
          acc1 += val * B_dnVal[offset];
          acc2 += val * B_dnVal[offset + 32];
        }
        __syncwarp();
      }
      offset = rid * B_ncols + cid;
      C_dnVal[offset] = acc1;
      C_dnVal[offset + 32] = acc2;
    }
    else
    {
      int nout = (B_ncols - cid + 31) / 32;
      for (int jj = lb; jj < hb; jj += 32)
      {
        if (ptr < hb)
        {
          val_sh[thread_idx] = A_csrVal[ptr];
          colInd_sh[thread_idx] = B_ncols * A_csrColInd[ptr];
        }
        __syncwarp();
        ptr += 32;

        for (int kk = 0; kk < 32 && jj + kk < hb; kk++)
        {
          val = val_sh[(shmem_offset + kk)];
          offset = colInd_sh[(shmem_offset + kk)] + cid;
          if (nout > 0)
          {
            acc1 += val * B_dnVal[offset];
          }
          if (nout > 1)
          {
            acc2 += val * B_dnVal[offset + 32];
          }
        }
        __syncwarp();
      }
      offset = rid * B_ncols + cid;
      if (nout > 0)
      {
        C_dnVal[offset] = acc1;
      }
      if (nout > 1)
      {
        C_dnVal[(offset + 32)] = acc2;
      }
    }
  }
}

void spmm_kernel(int m, int k, const int *rowptr, const int *colind,
                 const float *values, const float *dense, float *out)
{
  if (k < 32)
  {
    const int row_per_block = 128 / k;
    const int n_block = (m + row_per_block - 1) / row_per_block;
    spmm_test0<<<dim3(n_block, 1, 1), dim3(k, row_per_block, 1)>>>(
        m, k, rowptr, colind, values, dense, out);
  }
  if (k < 64)
  {
    const int tile_k = (k + 31) / 32;
    const int n_block = (m + 4 - 1) / 4;
    spmm_test1<<<dim3(n_block, tile_k, 1), dim3(32, 4, 1),
                 32 * 4 * (sizeof(int) + sizeof(float))>>>(m, k, rowptr, colind,
                                                           values, dense, out);
  }
  else
  {
    const int tile_k = (k + 63) / 64;
    const int n_block = (m + 8 - 1) / 8;
    spmm_test2<<<dim3(n_block, tile_k, 1), dim3(32, 8, 1),
                 32 * 8 * (sizeof(int) + sizeof(float))>>>(m, k, rowptr, colind,
                                                           values, dense, out);
  }
}

__global__ void fused_forward_kernel(int m, int nnz, int h, int f,
                                     const float *attn_row,
                                     const float *attn_col, const int *row_ptr,
                                     const int *col_ind, const float *in_feat,
                                     const float negative_slope,
                                     float *edge_max, float *edge_sum,
                                     float *out_feat)
{
  int rid = blockIdx.x;
  int hid = blockIdx.y;
  int lb = row_ptr[rid];
  int hb = row_ptr[rid + 1];
  int ptr = lb + threadIdx.x;
  int loop = (hb - lb + 31) / 32;
  extern __shared__ float val_sh[];
  float *attn_val_sh = val_sh;
  int *cid_sh = (int *)&val_sh[32];

  float attn_row_val = attn_row[rid * h + hid];
  // float attn_row_val=0;

  float weightMax = -1e38;
  // // computing weightMax
  for (int j = 0; j < loop; j++)
  {
    int pid = ptr + (j << 5);
    float weight = -1e38;
    if (pid < hb)
    {
      int cid = col_ind[pid];
      float attn_col_val = attn_col[cid * h + hid];
      weight = attn_row_val + attn_col_val;
      weight = LeakyRelu(weight, negative_slope);
    }
    __syncwarp();
    for (int stride = 16; stride > 0; stride >>= 1)
    {
      float tmp = __shfl_xor_sync(0xffffffff, weight, stride, 32);
      weight = MAX(tmp, weight);
    }
    weightMax = MAX(weight, weightMax);
  }
  if (threadIdx.x == 0)
    edge_max[rid * h + hid] = weightMax;

  float expAll = 0;
  for (int j = 0; j < loop; j++)
  {
    int pid = ptr + (j << 5);
    float exptmp = 0;
    if (pid < hb)
    {
      int cid = col_ind[pid];
      float attn_col_val = attn_col[cid * h + hid];
      float weight = attn_row_val + attn_col_val;
      weight = LeakyRelu(weight, negative_slope);
      exptmp = exp(weight - weightMax);
    }
    __syncwarp();
    for (int stride = 16; stride > 0; stride >>= 1)
    {
      float tmp = __shfl_xor_sync(0xffffffff, exptmp, stride, 32);
      exptmp += tmp;
    }
    expAll += exptmp;
  }
  if (threadIdx.x == 0)
    edge_sum[rid * h + hid] = expAll;

  int fid = threadIdx.y * 32 + threadIdx.x;
  // for (int fid = threadIdx.x; fid < (f + 31) / 32 * 32; fid += 32)
  {
    float acc = 0;
    for (int j = 0; j < loop; j++)
    {
      int pid = ptr + (j << 5);
      if (pid < hb)
      {
        int cid = col_ind[pid];
        float attn_col_val = attn_col[cid * h + hid];
        float weight = attn_row_val + attn_col_val;
        weight = LeakyRelu(weight, negative_slope);
        weight = exp(weight - weightMax) / expAll;
        attn_val_sh[threadIdx.x] = weight;
        cid_sh[threadIdx.x] = cid;
      }
      // else
      // {
      //     attn_val_sh[32 * hid + threadIdx.x] = 0;
      // }
      __syncwarp();
      int jj = lb + (j << 5);
      for (int kk = 0; kk < 32 && jj + kk < hb; kk++)
      {
        int cid = cid_sh[kk];
        float val = attn_val_sh[kk];
        acc += val * in_feat[cid * h * f + hid * f + fid];
      }
      __syncwarp();
    }
    if (fid < f)
      out_feat[rid * h * f + hid * f + fid] = acc;
  }
}

__global__ void fused_forward_kernel_small_f(
    int m, int nnz, int h, int f, const float *attn_row, const float *attn_col,
    const int *row_ptr, const int *col_ind, const float *in_feat,
    const float negative_slope, float *edge_max, float *edge_sum,
    float *out_feat)
{
  int rid = blockIdx.x;
  int hid = threadIdx.y;
  int lb = row_ptr[rid];
  int hb = row_ptr[rid + 1];
  int ptr = lb + threadIdx.x;
  int loop = (hb - lb + 31) / 32;
  extern __shared__ float attn_val_sh[];

  float attn_row_val = attn_row[rid * h + hid];
  // float attn_row_val=0;

  float weightMax = -1e38;
  // // computing weightMax
  for (int j = 0; j < loop; j++)
  {
    int pid = ptr + (j << 5);
    float weight = -1e38;
    if (pid < hb)
    {
      int cid = col_ind[pid];
      float attn_col_val = attn_col[cid * h + hid];
      weight = attn_row_val + attn_col_val;
      weight = LeakyRelu(weight, negative_slope);
    }
    __syncwarp();
    for (int stride = 16; stride > 0; stride >>= 1)
    {
      float tmp = __shfl_xor_sync(0xffffffff, weight, stride, 32);
      weight = MAX(tmp, weight);
    }
    weightMax = MAX(weight, weightMax);
  }
  if (threadIdx.x == 0)
    edge_max[rid * h + hid] = weightMax;

  float expAll = 0;
  for (int j = 0; j < loop; j++)
  {
    int pid = ptr + (j << 5);
    float exptmp = 0;
    if (pid < hb)
    {
      int cid = col_ind[pid];
      float attn_col_val = attn_col[cid * h + hid];
      float weight = attn_row_val + attn_col_val;
      weight = LeakyRelu(weight, negative_slope);
      exptmp = exp(weight - weightMax);
    }
    __syncwarp();
    for (int stride = 16; stride > 0; stride >>= 1)
    {
      float tmp = __shfl_xor_sync(0xffffffff, exptmp, stride, 32);
      exptmp += tmp;
    }
    expAll += exptmp;
  }
  if (threadIdx.x == 0)
    edge_sum[rid * h + hid] = expAll;

  // int fid = threadIdx.y * 32 + threadIdx.x;
  for (int fid = threadIdx.x; fid < (f + 31) / 32 * 32; fid += 32)
  {
    float acc = 0;
    for (int j = 0; j < loop; j++)
    {
      int pid = ptr + (j << 5);
      if (pid < hb)
      {
        int cid = col_ind[pid];
        float attn_col_val = attn_col[cid * h + hid];
        float weight = attn_row_val + attn_col_val;
        weight = LeakyRelu(weight, negative_slope);
        weight = exp(weight - weightMax) / expAll;
        attn_val_sh[32 * hid + threadIdx.x] = weight;
        // cid_sh[threadIdx.x] = cid;
      }
      // else
      // {
      //     attn_val_sh[32 * hid + threadIdx.x] = 0;
      // }
      __syncwarp();
      int jj = lb + (j << 5);
      for (int kk = 0; kk < 32 && jj + kk < hb; kk++)
      {
        int cid = col_ind[jj + kk];
        float val = attn_val_sh[32 * hid + kk];
        acc += val * in_feat[cid * h * f + hid * f + fid];
      }
      __syncwarp();
    }
    if (fid < f)
      out_feat[rid * h * f + hid * f + fid] = acc;
  }
}

__global__ void fused_forward_kernel_small_f_sm(
    int m, int nnz, int h, int f, const float *attn_row, const float *attn_col,
    const int *row_ptr, const int *col_ind, const float *in_feat,
    const float negative_slope, float *edge_max, float *edge_sum,
    float *out_feat)
{
  int rid = blockIdx.x;
  int hid = threadIdx.y;
  int lb = row_ptr[rid];
  int hb = row_ptr[rid + 1];
  int ptr = lb + threadIdx.x;
  int loop = (hb - lb + 31) / 32;
  extern __shared__ float edge_val_sh[];
  float *attn_val_sh = &edge_val_sh[512 * h];
  float *cid_sh = &attn_val_sh[32 * h];

  float attn_row_val = attn_row[rid * h + hid];
  // float attn_row_val=0;

  float weightMax = -1e38;
  // // computing weightMax
  for (int j = 0; j < loop; j++)
  {
    int pid = ptr + (j << 5);
    int edge_id = threadIdx.x + (j << 5);
    float weight = -1e38;
    if (pid < hb)
    {
      int cid = col_ind[pid];
      float attn_col_val = attn_col[cid * h + hid];
      weight = attn_row_val + attn_col_val;
      weight = LeakyRelu(weight, negative_slope);
      if (edge_id < 512)
      {
        edge_val_sh[edge_id * h + hid] = weight;
      }
    }
    __syncwarp();
    for (int stride = 16; stride > 0; stride >>= 1)
    {
      float tmp = __shfl_xor_sync(0xffffffff, weight, stride, 32);
      weight = MAX(tmp, weight);
    }
    weightMax = MAX(weight, weightMax);
  }
  if (threadIdx.x == 0)
    edge_max[rid * h + hid] = weightMax;

  float expAll = 0;
  for (int j = 0; j < loop; j++)
  {
    int pid = ptr + (j << 5);
    int edge_id = threadIdx.x + (j << 5);
    float exptmp = 0;
    if (pid < hb)
    {
      float weight;
      if (edge_id < 512)
      {
        weight = edge_val_sh[edge_id * h + hid];
      }
      else
      {
        int cid = col_ind[pid];
        float attn_col_val = attn_col[cid * h + hid];
        weight = attn_row_val + attn_col_val;
        weight = LeakyRelu(weight, negative_slope);
      }
      exptmp = exp(weight - weightMax);
    }
    __syncwarp();
    for (int stride = 16; stride > 0; stride >>= 1)
    {
      float tmp = __shfl_xor_sync(0xffffffff, exptmp, stride, 32);
      exptmp += tmp;
    }
    expAll += exptmp;
  }
  if (threadIdx.x == 0)
    edge_sum[rid * h + hid] = expAll;

  // int fid = threadIdx.y * 32 + threadIdx.x;
  for (int fid = threadIdx.x; fid < (f + 31) / 32 * 32; fid += 32)
  {
    float acc = 0;
    for (int j = 0; j < loop; j++)
    {
      int pid = ptr + (j << 5);
      int edge_id = threadIdx.x + (j << 5);

      if (pid < hb)
      {
        int cid = col_ind[pid];
        float weight;
        if (edge_id < 512)
        {
          weight = edge_val_sh[edge_id * h + hid];
        }
        else
        {

          float attn_col_val = attn_col[cid * h + hid];
          weight = attn_row_val + attn_col_val;
          weight = LeakyRelu(weight, negative_slope);
        }
        weight = exp(weight - weightMax) / expAll;
        attn_val_sh[32 * hid + threadIdx.x] = weight;
        cid_sh[threadIdx.x] = cid;
      }
      // else
      // {
      //     attn_val_sh[32 * hid + threadIdx.x] = 0;
      // }
      __syncwarp();
      int jj = lb + (j << 5);
      for (int kk = 0; kk < 32 && jj + kk < hb; kk++)
      {
        // int cid = col_ind[jj + kk];
        int cid = cid_sh[kk];
        float val = attn_val_sh[32 * hid + kk];
        acc += val * in_feat[cid * h * f + hid * f + fid];
      }
      __syncwarp();
    }
    if (fid < f)
      out_feat[rid * h * f + hid * f + fid] = acc;
  }
}

__global__ void mhspmm_kernel(int m, int nnz, int h, int f, const int *row_ptr,
                              const int *col_ind,
                              const float *edge_softmax_csr /* E*H */,
                              const float *in_feat /* V*H*F */,
                              float *out_feat /* V*H*F */
)
{
  int rid = blockIdx.x;
  int hid = threadIdx.y;

  int lb = row_ptr[rid];
  int hb = row_ptr[(rid + 1)];

  for (int fid = threadIdx.x; fid < f; fid += blockDim.x)
  {
    float acc = 0;
    for (int ptr = lb; ptr < hb; ptr++)
    {
      int offset1 = col_ind[ptr] * f * h + hid * f + fid;
      float att = edge_softmax_csr[ptr * h + hid];
      acc += att * in_feat[offset1];
    }
    out_feat[rid * h * f + hid * f + fid] = acc;
  }
}

void gat_forward(int m, int nnz, int h, int f, const float *attn_row,
                 const float *attn_col, const int *row_ptr, const int *col_ind,
                 float negative_slope, float *edge_max, float *edge_sum,
                 const float *in_feat, float *out_feat)
{
  if (f > 64)
    fused_forward_kernel<<<dim3(m, h, 1), dim3(32, (f + 31) / 32, 1),
                           32 * (sizeof(float) + sizeof(int))>>>(
        m, nnz, h, f, attn_row, attn_col, row_ptr, col_ind, in_feat,
        negative_slope, edge_max, edge_sum, out_feat);
  else
  {
    // fused_forward_kernel_small_f<<<dim3(m, 1, 1), dim3(32, h, 1),
    //                                32 * h * sizeof(float)>>>(
    //     m, nnz, h, f, attn_row, attn_col, row_ptr, col_ind, in_feat,
    //     negative_slope, edge_max, edge_sum, out_feat);
    fused_forward_kernel_small_f_sm<<<dim3(m, 1, 1), dim3(32, h, 1),
                                      (32 + 512) * h * sizeof(float) + 32 * sizeof(float)>>>(
        m, nnz, h, f, attn_row, attn_col, row_ptr, col_ind, in_feat,
        negative_slope, edge_max, edge_sum, out_feat);
  }
}

std::vector<torch::Tensor>
gat_forward_cuda(torch::Tensor attn_row, torch::Tensor attn_col,
                 torch::Tensor row_ptr, torch::Tensor col_ind,
                 float negative_slope, torch::Tensor in_feat)
{
  const auto m = row_ptr.size(0) - 1;
  const auto nnz = col_ind.size(0);
  const auto h = attn_row.size(1);
  const auto f = in_feat.size(2);
  auto devid = attn_row.device().index();
  auto options =
      torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA, devid);
  auto out_feat = torch::empty({m, h, f}, options);
  // auto edge_relu_csr = torch::empty({nnz, h}, options);
  // auto edge_softmax_csr = torch::empty({nnz, h}, options);
  auto edge_max = torch::empty({m, h}, options);
  auto edge_sum = torch::empty({m, h}, options);
  gat_forward(m, nnz, h, f, attn_row.data_ptr<float>(),
              attn_col.data_ptr<float>(), row_ptr.data_ptr<int>(),
              col_ind.data_ptr<int>(), negative_slope,
              edge_max.data_ptr<float>(), edge_sum.data_ptr<float>(),
              in_feat.data_ptr<float>(), out_feat.data_ptr<float>());
  return {out_feat, edge_max, edge_sum};
}

__global__ void mhspmm_backward_kernel(
    int m, int nnz, int h, int f, float negative_slope, const int *row_ptr,
    const int *col_ind, const int *col_ptr, const int *row_ind,
    // const int *permute,
    const float *edge_max, const float *edge_sum, const float *attn_row,
    const float *attn_col, const float *grad_in, float *grad_out)
{
  int cid = blockIdx.x;
  int hid = blockIdx.y;
  int lb = col_ptr[cid];
  int hb = col_ptr[(cid + 1)];
  int loop = (hb - lb + 31) / 32;
  int ptr = lb + threadIdx.x;

  float attn_col_val = attn_col[cid * h + hid];
  extern __shared__ float attn_val_sh[];
  int *rid_sh = (int *)&attn_val_sh[32];

  // for (int fid = threadIdx.x; fid < (f + 31) / 32 * 32; fid += 32)
  int fid = threadIdx.y * 32 + threadIdx.x;
  {
    float acc = 0;
    for (int j = 0; j < loop; j++)
    {
      int pid = ptr + (j << 5);
      attn_val_sh[threadIdx.x] = 0;
      if (pid < hb)
      {
        int rid = row_ind[pid];
        float attn_row_val = attn_row[rid * h + hid];
        float weight = attn_row_val + attn_col_val;
        weight = LeakyRelu(weight, negative_slope);
        float expAll = edge_sum[rid * h + hid];
        float weightMax = edge_max[rid * h + hid];
        weight = exp(weight - weightMax) / expAll;
        attn_val_sh[threadIdx.x] = weight;
        rid_sh[threadIdx.x] = rid;
      }
      __syncwarp();
      int jj = lb + (j << 5);
      for (int kk = 0; kk < 32 && jj + kk < hb; kk++)
      {
        int rid = rid_sh[kk];
        float val = attn_val_sh[kk];
        acc += val * grad_in[rid * h * f + hid * f + fid];
      }
      __syncwarp();
    }
    if (fid < f)
      grad_out[cid * h * f + hid * f + fid] = acc;
  }
}

__global__ void mhspmm_backward_kernel_small_f(
    int m, int nnz, int h, int f, float negative_slope, const int *row_ptr,
    const int *col_ind, const int *col_ptr, const int *row_ind,
    // const int *permute,
    const float *edge_max, const float *edge_sum, const float *attn_row,
    const float *attn_col, const float *grad_in, float *grad_out)
{
  int cid = blockIdx.x;
  int hid = threadIdx.y;
  int lb = col_ptr[cid];
  int hb = col_ptr[(cid + 1)];
  int loop = (hb - lb + 31) / 32;
  int ptr = lb + threadIdx.x;

  float attn_col_val = attn_col[cid * h + hid];
  extern __shared__ float attn_val_sh[];

  for (int fid = threadIdx.x; fid < (f + 31) / 32 * 32; fid += 32)
  // int fid = threadIdx.y * 32 + threadIdx.x;
  {
    float acc = 0;
    for (int j = 0; j < loop; j++)
    {
      int pid = ptr + (j << 5);
      attn_val_sh[32 * hid + threadIdx.x] = 0;
      if (pid < hb)
      {
        int rid = row_ind[pid];
        float attn_row_val = attn_row[rid * h + hid];
        float weight = attn_row_val + attn_col_val;
        weight = LeakyRelu(weight, negative_slope);
        float expAll = edge_sum[rid * h + hid];
        float weightMax = edge_max[rid * h + hid];
        weight = exp(weight - weightMax) / expAll;
        attn_val_sh[32 * hid + threadIdx.x] = weight;
      }
      __syncwarp();
      int jj = lb + (j << 5);
      for (int kk = 0; kk < 32 && jj + kk < hb; kk++)
      {
        int rid = row_ind[jj + kk];
        float val = attn_val_sh[32 * hid + kk];
        acc += val * grad_in[rid * h * f + hid * f + fid];
      }
      __syncwarp();
    }
    if (fid < f)
      grad_out[cid * h * f + hid * f + fid] = acc;
  }
}

__global__ void mhsddmm(const int v, const int f, const int h, const int nnz,
                        int *rowptr, int *colind, float *grad, float *feature,
                        float *out) // V * H * F
{
  int eid = (blockIdx.x << 4) + (threadIdx.y << 2);
  int cid = threadIdx.x;
  int hid = blockIdx.y;

  if (blockIdx.x < nnz / 16)
  {
    float multi[4] = {0, 0, 0, 0};
    int offset1[4], offset2[4];
    float D1tmp[4], D2tmp[4];

    Load<int4, int>(offset2, colind, eid);

    offset1[0] = findRow(rowptr, eid, 0, v);
    offset1[3] = findRow(rowptr, eid + 3, offset1[0], v);
    offset1[1] = findRow(rowptr, eid + 1, offset1[0], offset1[3]);
    offset1[2] = findRow(rowptr, eid + 2, offset1[1], offset1[3]);

    selfMulConst4<int>(offset1, f * h);
    selfAddConst4<int>(offset1, hid * f);
    selfMulConst4<int>(offset2, f * h);
    selfAddConst4<int>(offset2, hid * f);
    for (int i = 0; i < (f >> 5); i++)
    {
      Load4<float, float>(D1tmp, grad, offset1, cid);
      Load4<float, float>(D2tmp, feature, offset2, cid);
      Dot4<float>(multi, D1tmp, D2tmp);
      cid += 32;
    }
    int res = f & 31;
    if (res)
    {
      float D1[4] = {0, 0, 0, 0}, D2[4] = {0, 0, 0, 0};
      if (threadIdx.x < res)
      {
        Load4<float, float>(D1, grad, offset1, cid);
        Load4<float, float>(D2, feature, offset2, cid);
        Dot4<float>(multi, D1, D2);
      }
    }
    AllReduce4<float>(multi, 16, 32);
    if (threadIdx.x == 0)
    {
      out[eid * h + hid] = multi[0];
      out[(eid + 1) * h + hid] = multi[1];
      out[(eid + 2) * h + hid] = multi[2];
      out[(eid + 3) * h + hid] = multi[3];
    }
  }
  else // Dynamic parrallel?
  {
    eid = nnz - (nnz & 15) + (blockIdx.x - (nnz / 16));
    int offset1 = findRow(rowptr, eid, 0, v) * f * h + hid * f;
    int offset2 = colind[eid] * f * h + hid * f;
    float multi = 0;
    int off1 = cid = threadIdx.x;
    float D1tmp0, D2tmp0;
    for (int cc = 0; cc < (f >> 5); cc++)
    {
      D1tmp0 = grad[offset1 + cid];
      D2tmp0 = feature[offset2 + cid];
      multi += D1tmp0 * D2tmp0;
      cid += 32;
    }
    int res = f & 31;
    D1tmp0 = D2tmp0 = 0;
    if (res)
    {
      if (off1 < res)
      {
        D1tmp0 = grad[offset1 + cid];
        D2tmp0 = feature[offset2 + cid];
      }
      multi += D1tmp0 * D2tmp0;
    }
    for (int stride = 16; stride > 0; stride >>= 1)
    {
      multi += __shfl_xor_sync(0xffffffff, multi, stride, 32);
    }
    if (threadIdx.x == 0 && threadIdx.y == 0)
    {
      out[eid * h + hid] = multi;
    }
  }
}

__global__ void fused_backward_kernel(
    int m, int nnz, int h, int f, const int *row_ptr, const int *col_ind,
    const float negative_slope, const float *edge_max, const float *edge_sum,
    const float *attn_row, const float *attn_col, const float *grad_edge_csr,
    // float *grad_edge_for_gather_csr,
    float *grad_attn_row, float *grad_attn_col)
{
  int rid = blockIdx.x;
  int hid = threadIdx.y;
  int lb = row_ptr[rid];
  int hb = row_ptr[(rid + 1)];
  int loop = (hb - lb + 31) / 32;
  int ptr = lb + threadIdx.x;
  float attn_row_val = attn_row[rid * h + hid];
  float weightSum = 0;
  for (int j = 0; j < loop; j++)
  {
    int pid = ptr + (j << 5);
    float weight = 0;
    if (pid < hb)
    {
      int cid = col_ind[pid];
      float attn_col_val = attn_col[cid * h + hid];
      float val_leaky = LeakyRelu(attn_row_val + attn_col_val, negative_slope);
      // if (val_leaky != edge_relu_csr[pid * h + hid])
      //     printf("leakyrelu value error\n");
      float val_softmax =
          exp(val_leaky - edge_max[rid * h + hid]) / edge_sum[rid * h + hid];
      // if (val_softmax != edge_softmax_csr[pid * h + hid])
      //     printf("softmax value error\n");
      weight = val_softmax * grad_edge_csr[pid * h + hid];
    }
    __syncwarp();
    for (int stride = 16; stride > 0; stride >>= 1)
    {
      weight += __shfl_xor_sync(0xffffffff, weight, stride, 32);
    }
    weightSum = weight + weightSum;
  }

  float grad_row_sum = 0;
  for (int j = 0; j < loop; j++)
  {
    int pid = ptr + (j << 5);
    float grad_out = 0;
    int eid = pid * h + hid;
    if (pid < hb)
    {
      int cid = col_ind[pid];
      float attn_col_val = attn_col[cid * h + hid];
      float val_leaky = LeakyRelu(attn_row_val + attn_col_val, negative_slope);
      // if (val_leaky != edge_relu_csr[pid * h + hid])
      //     printf("leakyrelu value error\n");
      float val_softmax =
          exp(val_leaky - edge_max[rid * h + hid]) / edge_sum[rid * h + hid];
      // if (val_softmax != edge_softmax_csr[pid * h + hid])
      //     printf("softmax value error\n");
      grad_out = val_softmax * (grad_edge_csr[eid] - weightSum);
      if (val_leaky < 0)
        grad_out = grad_out * negative_slope;

      // grad_edge_for_gather_csr[eid] = grad_out;
      atomicAdd(&grad_attn_col[cid * h + hid], grad_out);
    }

    __syncwarp();
    for (int stride = 16; stride > 0; stride >>= 1)
    {
      grad_out += __shfl_xor_sync(0xffffffff, grad_out, stride, 32);
    }
    grad_row_sum += grad_out;
  }
  if (threadIdx.x == 0)
    grad_attn_row[rid * h + hid] = grad_row_sum;
}

__global__ void gather_col(int m, int nnz, int h, int f, const int *col_ptr,
                           const int *permute,
                           const float *grad_edge_for_gather_csr,
                           float *grad_attn_col)
{
  int cid = blockIdx.x;
  int hid = threadIdx.y;
  int lb = col_ptr[cid];
  int hb = col_ptr[cid + 1];
  int ptr = lb + threadIdx.x;
  int loop = (hb - lb + 31) / 32;
  float grad_col_sum = 0;
  for (int j = 0; j < loop; j++)
  {
    int pid = ptr + (j << 5);
    float grad = 0;
    if (pid < hb)
      grad = grad_edge_for_gather_csr[permute[pid] * h + hid];
    __syncwarp();

    for (int stride = 16; stride > 0; stride >>= 1)
    {
      grad += __shfl_xor_sync(0xffffffff, grad, stride, 32);
    }
    grad_col_sum += grad;
  }
  if (threadIdx.x == 0)
    if (grad_attn_col[cid * h + hid] - grad_col_sum > 1e-5)
      printf("error,%f,%f\n", grad_attn_col[cid * h + hid], grad_col_sum);
}

void gat_backward(int m, int nnz, int h, int f, float negative_slope,
                  int *row_ptr, int *col_ind, int *col_ptr, int *row_ind,
                  // int *permute,
                  float *edge_max, float *edge_sum, float *in_feat,
                  float *attn_row, float *attn_col,
                  float *grad,          // input grad
                  float *grad_edge_csr, // temp grad
                  // float *grad_edge_for_gather_csr, //temp grad
                  float *grad_feat,     // output grad
                  float *grad_attn_row, // output grad
                  float *grad_attn_col) // output grad
{
  if (f > 64)
    mhspmm_backward_kernel<<<dim3(m, h, 1), dim3(32, (f + 31) / 32, 1),
                             32 * (sizeof(float) + sizeof(int))>>>(
        m, nnz, h, f, negative_slope, row_ptr, col_ind, col_ptr, row_ind,
        edge_max, edge_sum, attn_row, attn_col, grad, grad_feat);
  else
    mhspmm_backward_kernel_small_f<<<dim3(m, 1, 1), dim3(32, h, 1),
                                     32 * h * sizeof(float)>>>(
        m, nnz, h, f, negative_slope, row_ptr, col_ind, col_ptr, row_ind,
        edge_max, edge_sum, attn_row, attn_col, grad, grad_feat);

  mhsddmm<<<dim3(nnz / 16 + (nnz & 15), h, 1), dim3(32, 4, 1)>>>(
      m, f, h, nnz, row_ptr, col_ind, grad, in_feat, grad_edge_csr);

  fused_backward_kernel<<<dim3(m, 1, 1), dim3(32, h, 1)>>>(
      m, nnz, h, f, row_ptr, col_ind, negative_slope, edge_max, edge_sum,
      attn_row, attn_col, grad_edge_csr, grad_attn_row, grad_attn_col);

  // gather_col<<<dim3(m, 1, 1), dim3(32, h, 1)>>>(m, nnz, h, f, col_ptr,
  // permute, grad_edge_for_gather_csr, grad_attn_col);
}

std::vector<torch::Tensor> gat_backward_cuda(
    float negative_slope, torch::Tensor row_ptr, torch::Tensor col_ind,
    torch::Tensor col_ptr, torch::Tensor row_ind,
    // torch::Tensor permute,
    torch::Tensor edge_max, torch::Tensor edge_sum, torch::Tensor in_feat,
    torch::Tensor attn_row, torch::Tensor attn_col, torch::Tensor grad)
{
  const auto m = row_ptr.size(0) - 1;
  const auto nnz = col_ind.size(0);
  const auto h = in_feat.size(1);
  const auto f = in_feat.size(2);
  auto devid = row_ptr.device().index();
  auto options =
      torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA, devid);
  auto grad_edge_csr = torch::empty({nnz, h}, options);
  // auto grad_edge_for_gather_csr = torch::empty({nnz, h}, options);
  auto grad_feat = torch::empty({m, h, f}, options);
  auto grad_attn_row = torch::empty({m, h}, options);
  auto grad_attn_col = torch::zeros({m, h}, options);
  // printf("gat backward cuda\n");
  gat_backward(m, nnz, h, f, negative_slope, row_ptr.data_ptr<int>(),
               col_ind.data_ptr<int>(), col_ptr.data_ptr<int>(),
               row_ind.data_ptr<int>(), edge_max.data_ptr<float>(),
               edge_sum.data_ptr<float>(), in_feat.data_ptr<float>(),
               attn_row.data_ptr<float>(), attn_col.data_ptr<float>(),
               grad.data_ptr<float>(), grad_edge_csr.data_ptr<float>(),
               grad_feat.data_ptr<float>(), grad_attn_row.data_ptr<float>(),
               grad_attn_col.data_ptr<float>());
  return {grad_feat, grad_attn_row, grad_attn_col};
}
