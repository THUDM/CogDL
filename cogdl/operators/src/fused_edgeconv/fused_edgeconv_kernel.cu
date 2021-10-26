#include <cuda.h>
#include <torch/types.h>
#include <vector>

#define MAX(a, b) ((a < b) ? (b) : (a))

__global__ void edgeconv_forward_kernel(
    const int m, const int f,
    const int k, const int *src_ind,
    const float *h_src, const float *h_dst,
    float *out_feat,
    float *max_idx)
{
    int rid = blockIdx.x;
    int lb = rid * k;
    int hb = (rid + 1) * k;
    int fid = threadIdx.y;
    int ptr = lb + threadIdx.x;
    int loop = (k+31) / 32;

    for (; fid < f; fid += 32)
    {
        float max_val = -1e39;
        float max_id = -1;
        float dst_val = h_dst[rid * f + fid];
        for (int j = 0; j < loop; j++)
        {
            int pid = ptr + (j << 5);
            float feat = -1e38;
            int src_id = -1;
            if (pid < hb)
            {
                src_id = src_ind[pid];
                feat = dst_val + h_src[src_id * f + fid];
            }
            __syncwarp();
            for (int stride = 16; stride > 0; stride >>= 1)
            {
                float tmp = __shfl_xor_sync(0xffffffff, feat, stride, 32);
                float tmp_id = __shfl_xor_sync(0xffffffff, src_id, stride, 32);
                if (tmp >= feat)
                {
                    feat = tmp;
                    src_id = tmp_id;
                }
            }
            if (feat >= max_val)
            {
                max_val = feat;
                max_id = src_id;
            }
        }
        if (threadIdx.x == 0)
        {
            out_feat[rid * f + fid] = max_val;
            max_idx[rid * f + fid] = max_id;
        }
    }
}

std::vector<torch::Tensor> edgeconv_forward_cuda(
    const int k,
    const torch::Tensor src_ind,
    const torch::Tensor h_src,
    const torch::Tensor h_dst)
{
    const auto m = h_src.size(0);
    const auto f = h_src.size(1);
    auto devid = src_ind.device().index();
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA, devid);
    auto out_feat = torch::empty({m, f}, options);
    auto max_idx = torch::empty({m, f}, options);
    dim3 grids(m, 1, 1);
    dim3 blocks;
    if (f < 32)
    {
        blocks = dim3(32, f, 1);
    }
    else
    {
        blocks = dim3(32, 32, 1);
    }
    edgeconv_forward_kernel<<<grids, blocks>>>(m, f, k, src_ind.data_ptr<int>(), h_src.data_ptr<float>(), h_dst.data_ptr<float>(), out_feat.data_ptr<float>(), max_idx.data_ptr<float>());
    return {out_feat, max_idx};
}

__global__ void edgeconv_backward_kernel(
    const int m, const int f,
    const int *max_idx,
    const float *grad_out,
    float *grad_src)
{
    int src_id = blockIdx.x;
    int fid = threadIdx.x;
    int ptr = src_id * f + fid;
    grad_src[ptr] = 0;
    atomicAdd(&grad_src[max_idx[ptr] * f + fid], grad_out[ptr]);
    // grad_src[max_idx[ptr] * f + fid] += grad_out[ptr];
}

torch::Tensor edgeconv_backward_cuda(
    const torch::Tensor grad_out,
    const torch::Tensor max_idx)
{
    auto m = grad_out.size(0);
    auto f = grad_out.size(1);
    auto devid = grad_out.device().index();
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA, devid);
    auto grad_src = torch::empty({m, f}, options);
    edgeconv_backward_kernel<<<dim3(m, 1, 1), dim3(f, 1, 1)>>>(m, f, max_idx.data_ptr<int>(), grad_out.data_ptr<float>(), grad_src.data_ptr<float>());
    return grad_src;
}

__global__ void edgeconv_forward_kernel_110(
    const int m, const int f,
    const int k, const int *src_ind,
    const float *h_src, const float *h_dst,
    float* edge_feat,
    float *out_feat,
    float *max_idx)
{
    int rid = blockIdx.x;
    int lb = rid * k;
    int hb = (rid + 1) * k;
    int fid = threadIdx.y;
    int ptr = lb + threadIdx.x;
    int loop = k / 32;

    for (; fid < f; fid += 32)
    {
        float max_val = -1e39;
        float max_id = -1;
        float dst_val = h_dst[rid * f + fid];
        for (int j = 0; j < loop; j++)
        {
            int pid = ptr + (j << 5);
            float feat = -1e38;
            int src_id = -1;
            if (pid < hb)
            {
                src_id = src_ind[pid];
                feat = dst_val + h_src[src_id * f + fid];
                edge_feat[pid*f+fid]=feat;
            }
            __syncwarp();
            for (int stride = 16; stride > 0; stride >>= 1)
            {
                float tmp = __shfl_xor_sync(0xffffffff, feat, stride, 32);
                float tmp_id = __shfl_xor_sync(0xffffffff, src_id, stride, 32);
                if (tmp >= feat)
                {
                    feat = tmp;
                    src_id = tmp_id;
                }
            }
            if (feat >= max_val)
            {
                max_val = feat;
                max_id = src_id;
            }
        }
        if (threadIdx.x == 0)
        {
            out_feat[rid * f + fid] = max_val;
            max_idx[rid * f + fid] = max_id;
        }
    }
}

std::vector<torch::Tensor> edgeconv_forward_cuda_110(
    const int k,
    const torch::Tensor src_ind,
    const torch::Tensor h_src,
    const torch::Tensor h_dst)
{
    const auto m = h_src.size(0);
    const auto f = h_src.size(1);
    auto devid = src_ind.device().index();
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA, devid);
    auto out_feat = torch::empty({m, f}, options);
    auto max_idx = torch::empty({m, f}, options);
    dim3 grids(m, 1, 1);
    dim3 blocks;
    if (f < 32)
    {
        blocks = dim3(32, f, 1);
    }
    else
    {
        blocks = dim3(32, 32, 1);
    }
    edgeconv_forward_kernel<<<grids, blocks>>>(m, f, k, src_ind.data_ptr<int>(), h_src.data_ptr<float>(), h_dst.data_ptr<float>(), out_feat.data_ptr<float>(), max_idx.data_ptr<float>());
    return {out_feat, max_idx};
}

__global__ void edgeconv_backward_kernel_110(
    const int m, const int f,
    const int *max_idx,
    const float *grad_out,
    float *grad_src)
{
    int src_id = blockIdx.x;
    int fid = threadIdx.x;
    int ptr = src_id * f + fid;
    grad_src[ptr] = 0;
    atomicAdd(&grad_src[max_idx[ptr] * f + fid], grad_out[ptr]);
    // grad_src[max_idx[ptr] * f + fid] += grad_out[ptr];
}

torch::Tensor edgeconv_backward_cuda_110(
    const torch::Tensor grad_out,
    const torch::Tensor max_idx)
{
    auto m = grad_out.size(0);
    auto f = grad_out.size(1);
    auto devid = grad_out.device().index();
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA, devid);
    auto grad_src = torch::empty({m, f}, options);
    edgeconv_backward_kernel<<<dim3(m, 1, 1), dim3(f, 1, 1)>>>(m, f, max_idx.data_ptr<int>(), grad_out.data_ptr<float>(), grad_src.data_ptr<float>());
    return grad_src;
}
