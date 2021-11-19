#include <cuda.h>
#include <torch/types.h>
#include <cusparse.h>

#define MAX(a, b) ((a < b) ? (b) : (a))

__global__ void edge_softmax(
    const int *rowptr, const float *values, float *outs, int head)
{
    int rid = blockIdx.x;
    int hid = threadIdx.y;
    int lb = rowptr[rid];
    int hb = rowptr[(rid + 1)];
    int loop = 1 + (hb - lb) / 32;
    float weightMax = -1e8;
    float expAll = 0;
    for (int j = 0; j < loop; j++)
    {
        int pid = threadIdx.x + (j << 5) + lb;
        float weight = -1e8;
        if(pid < hb)
        {
            weight = values[pid * head + hid];
        }
        __syncwarp();
        for (int stride = 16; stride > 0; stride >>= 1)
        {
            float tmp = __shfl_xor_sync(0xffffffff, weight, stride, 32);
            weight = MAX(tmp, weight);
        }
        weightMax = MAX(weight, weightMax);
    }

    for (int j = 0; j < loop; j++)
    {
        int pid = threadIdx.x + (j << 5) + lb;
        float exptmp = 0;
        if(pid < hb)
        {
            exptmp = exp(values[pid * head + hid] - weightMax);
        }
        __syncwarp();
        for (int stride = 16; stride > 0; stride >>= 1)
        {
            float tmp = __shfl_xor_sync(0xffffffff, exptmp, stride, 32);
            exptmp += tmp;
        }
        expAll += exptmp;
    }
    for (int j = 0; j < loop; j++)
    {
        int pid = threadIdx.x + (j << 5) + lb;
        float exptmp = 0;
        if(pid < hb)
        {
            exptmp = exp(values[pid * head + hid] - weightMax) / expAll;
            outs[pid * head + hid] = exptmp;
        }
    }
}


__global__ void edge_softmax_backward(
    const int *rowptr, const float *softmax, const float *grad, float *gradout, int head)
{
    int rid = blockIdx.x;
    int hid = threadIdx.y;
    int lb = rowptr[rid];
    int hb = rowptr[(rid + 1)];
    int loop = 1 + (hb - lb) / 32;
    float weightSum = 0;
    for (int j = 0; j < loop; j++)
    {
        int pid = threadIdx.x + (j << 5) + lb;
        float weight = 0;
        if(pid < hb)
        {
            weight = softmax[pid * head + hid] * grad[pid * head + hid];
        }
        __syncwarp();
        for (int stride = 16; stride > 0; stride >>= 1)
        {
            weight += __shfl_xor_sync(0xffffffff, weight, stride, 32);
        }
        weightSum = weight + weightSum;
    }

    for (int j = 0; j < loop; j++)
    {
        int pid = threadIdx.x + (j << 5) + lb;
        float exptmp = 0;
        if(pid < hb)
        {
            int nid = pid * head + hid;
            gradout[nid] = softmax[nid] * (grad[nid] - weightSum);
        }
    }
}


torch::Tensor edge_softmax_cuda(
    torch::Tensor rowptr,
    torch::Tensor weight)
{
    const auto m = rowptr.size(0) - 1;
    const auto nnz = weight.size(0);
    const auto h = weight.size(1);
    auto devid = weight.device().index();
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA, devid);
    auto outs = torch::empty({nnz, h}, options);
    edge_softmax<<<dim3(m, 1, 1), dim3(32, h, 1)>>>(rowptr.data_ptr<int>(), weight.data_ptr<float>(), outs.data_ptr<float>(), h); 
    return outs;
}

torch::Tensor edge_softmax_backward_cuda(
    torch::Tensor rowptr,
    torch::Tensor softmax,
    torch::Tensor grad)
{
    const auto m = rowptr.size(0) - 1;
    const auto nnz = softmax.size(0);
    const auto h = softmax.size(1);
    auto devid = softmax.device().index();
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA, devid);
    auto gradout = torch::empty({nnz, h}, options);
    edge_softmax_backward<<<dim3(m, 1, 1), dim3(32, h, 1)>>>(rowptr.data_ptr<int>(), softmax.data_ptr<float>(), grad.data_ptr<float>(), gradout.data_ptr<float>(), h); 
    return gradout;
}