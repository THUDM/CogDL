#include <cuda.h>
#include <torch/types.h>
#include "computeUtil.h"


__global__ void mhsddmm(const int v, const int f, const int h, const int nnz,
    int *rowptr, int *colind, float *grad,
    float *feature, float *out) // V * H * F
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
        if(res)
        {
            float D1[4] =  {0, 0, 0, 0}, D2[4] =  {0, 0, 0, 0};
            if(threadIdx.x < res)
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
        if(res)
        {
            if(off1 < res)
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


torch::Tensor mhsddmm_cuda(
    torch::Tensor rowptr,
    torch::Tensor colind, 
    torch::Tensor grad, // V * H * F
    torch::Tensor feature // V * H * F
)
{
    const auto v = rowptr.size(0) - 1; // V
    const auto nnz = colind.size(0); // E
    const auto h = feature.size(1); // H
    const auto f = feature.size(2); // F
    auto devid = feature.device().index();
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA, devid);
    auto out = torch::empty({nnz, h}, options);
    mhsddmm<<<dim3(nnz / 16 + (nnz & 15), h, 1), dim3(32, 4, 1)>>>(v, f, h, nnz,
        rowptr.data_ptr<int>(), colind.data_ptr<int>(), grad.data_ptr<float>(), feature.data_ptr<float>(), out.data_ptr<float>());
    return out;
}
