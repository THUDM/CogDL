#include <cuda.h>
#include <torch/types.h>
#include <cusparse.h>
#include "computeUtil.h"

__global__ void mhtranspose(const int nnz, const int h, const int * permute, float * attention, float * out)
{
    int hid = blockIdx.y;
    int nid = blockIdx.x * 32 + threadIdx.x;
    if(nid < nnz)
    {
        int idx = permute[nid];
        out[nid * h + hid] = attention[idx * h + hid];
    }
}

__global__ void mhtranspose4(const int nnz, const int h, int * permute, float * attention, float * out)
{
    int hid = threadIdx.y << 2;
    int nid = blockIdx.x * 32 + threadIdx.x;
    if(nid < nnz)
    {
        int idx = permute[nid];
        float att[4];
        Load<float4, float>(att, attention, idx * h + hid);
        Store<float4, float>(out, att, nid * h + hid);
    }
}

torch::Tensor mhtranspose_cuda(
    torch::Tensor permute,
    torch::Tensor attention // E * H
)
{
    const auto nnz = permute.size(0);
    const auto h = attention.size(1);
    auto devid = permute.device().index();
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA, devid);
    auto out = torch::empty({nnz, h}, options);
    if((h & 3) == 0)
    {
        mhtranspose4<<<dim3(CEIL(nnz, 32), 1, 1), dim3(32, CEIL(h, 4), 1)>>>(nnz, h, permute.data_ptr<int>(), attention.data_ptr<float>(), out.data_ptr<float>());
    }
    else
    {
        mhtranspose<<<dim3(CEIL(nnz, 32), h, 1), dim3(32, 1, 1)>>>(nnz, h, permute.data_ptr<int>(), attention.data_ptr<float>(), out.data_ptr<float>());
    }
    return out;
}

void csr2cscKernel(int m, int n, int nnz,
    int *csrRowPtr, int *csrColInd, int *csrVal,
    int *cscColPtr, int *cscRowInd, int *cscVal
)
{
    cusparseHandle_t handle;
    checkCuSparseError(cusparseCreate(&handle));
    size_t bufferSize = 0;
    void* buffer = NULL;
    checkCuSparseError(cusparseCsr2cscEx2_bufferSize(handle,
        m,
        n,
        nnz,
        csrVal,
        csrRowPtr,
        csrColInd,
        cscVal,
        cscColPtr,
        cscRowInd,
        CUDA_R_32I,
        CUSPARSE_ACTION_SYMBOLIC,
        CUSPARSE_INDEX_BASE_ZERO,
        CUSPARSE_CSR2CSC_ALG1,
        &bufferSize
    ));
    checkCudaError(cudaMalloc((void**)&buffer, bufferSize * sizeof(float)));
    checkCuSparseError(cusparseCsr2cscEx2(handle,
        m,
        n,
        nnz,
        csrVal,
        csrRowPtr,
        csrColInd,
        cscVal,
        cscColPtr,
        cscRowInd,
        CUDA_R_32I,
        CUSPARSE_ACTION_NUMERIC,
        CUSPARSE_INDEX_BASE_ZERO,
        CUSPARSE_CSR2CSC_ALG1,
        buffer
    ));
    checkCudaError(cudaFree(buffer));
}

std::vector<torch::Tensor> csr2csc_cuda(
    torch::Tensor csrRowPtr,
    torch::Tensor csrColInd,
    torch::Tensor csrVal)
{
    const auto n = csrRowPtr.size(0) - 1;
    const auto nnz = csrColInd.size(0);
    auto devid = csrRowPtr.device().index();
    auto optionsI = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA, devid);
    auto cscColPtr = torch::empty({n + 1}, optionsI);
    auto cscRowInd = torch::empty({nnz}, optionsI);
    auto cscVal = torch::empty({nnz}, optionsI);
    csr2cscKernel(n, n, nnz, csrRowPtr.data_ptr<int>(), csrColInd.data_ptr<int>(), csrVal.data_ptr<int>(),
                    cscColPtr.data_ptr<int>(), cscRowInd.data_ptr<int>(), cscVal.data_ptr<int>());
    return {cscColPtr, cscRowInd, cscVal};
}