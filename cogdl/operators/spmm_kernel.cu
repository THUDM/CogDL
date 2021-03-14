#include <cuda.h>
#include <torch/types.h>
#include <cusparse.h>

#define checkCudaError( a ) do { \
    if (cudaSuccess != (a)) { \
    fprintf(stderr, "Cuda runTime error in line %d of file %s \
    : %s \n", __LINE__, __FILE__, cudaGetErrorString(cudaGetLastError()) ); \
    exit(EXIT_FAILURE); \
    } \
} while(0)

#define checkCuSparseError( a ) do { \
    if (CUSPARSE_STATUS_SUCCESS != (a)) { \
    fprintf(stderr, "CuSparse runTime error in line %d of file %s \
    : %s \n", __LINE__, __FILE__, cudaGetErrorString(cudaGetLastError()) ); \
    exit(EXIT_FAILURE); \
    } \
} while (0)



__device__ __forceinline__ float sum_reduce(float acc, float x) {
  return acc + x;
}

__device__ __forceinline__ float sum_init() {
  return 0;
}

__global__ void topoCacheCoarsenSPMMKernel(
  int m, int k, const int* A_indptr, const int* A_indices, const float* B, float* C
) {
  extern __shared__ int sh[];
  int sm_offset = (threadIdx.y<<5);
  int thread_idx = sm_offset+threadIdx.x;

  int rid = blockDim.y*blockIdx.x+threadIdx.y;
  if (rid<m) {

    int cid = (blockIdx.y<<6)+threadIdx.x;
    int lb = A_indptr[rid];
    int hb = A_indptr[rid+1];
    int ptr = lb+threadIdx.x;
    int offset;
    float acc1 = sum_init();
    float acc2 = sum_init();
    if (blockIdx.y != gridDim.y-1) {
      for (int jj=lb; jj<hb; jj+=32) {
        if (ptr<hb) {
          sh[thread_idx] = A_indices[ptr]*k;
          // sh[thread_idx] = __ldg(A_indices+ptr)*k;
        }
        __syncwarp();
        ptr += 32;
        for (int kk=0; kk<32&&jj+kk<hb; kk++) {
          offset = sh[(sm_offset+kk)] + cid;
          acc1 = sum_reduce(acc1, B[offset]);
          acc2 = sum_reduce(acc2, B[(offset+32)]);
          // acc1 = sum_reduce(acc1, __ldg(B+offset));
          // acc2 = sum_reduce(acc2, __ldg(B+offset+32));
        }
        __syncwarp();
      }
      offset = rid*k+cid;
      C[offset] = acc1;
      C[offset+32] = acc2;
    }
    else { // threadIdx.y==blockDim.y-1
      int nout = (k-cid+31)/32;
      for (int jj=lb; jj<hb; jj+=32) {
        if (ptr<hb) {
          sh[thread_idx] = A_indices[ptr]*k;
          // sh[thread_idx] = __ldg(A_indices+ptr)*k;
        }
        __syncwarp();
        ptr += 32;
        for (int kk=0; kk<32&&jj+kk<hb; kk++) {
          offset = sh[(sm_offset+kk)] + cid;
          if (nout>0) {
          acc1 = sum_reduce(acc1, B[offset]);}
          // acc1 = sum_reduce(acc1, __ldg(B+offset)); }
          if (nout>1) {
          acc2 = sum_reduce(acc2, B[(offset+32)]);}
          // acc2 = sum_reduce(acc2, __ldg(B+offset+32));}
        }
        __syncwarp();
      }
      offset = rid*k+cid;
      if (nout>0) {
      C[offset] = acc1;}
      if (nout>1) {
      C[offset+32] = acc2;}
    }
  }
} 

__global__ void topoCacheSPMMKernel(
  int m, int k, const int* A_indptr, const int* A_indices, const float* B, float* C 
) {
  extern __shared__ int sh[];
  int sm_offset = (threadIdx.y<<5);
  int thread_idx = sm_offset + threadIdx.x;
  
  int cid = (blockIdx.y<<5)+threadIdx.x;
  int rid = blockDim.y*blockIdx.x+threadIdx.y;
    
  if (rid<m) {
    int lb = A_indptr[rid];
    int hb = A_indptr[(rid+1)];
    int offset;
    int ptr = lb+threadIdx.x;
    float acc1 = sum_init();
    if (blockIdx.y != gridDim.y-1) {
      for (int jj=lb; jj<hb; jj+=32) {
        if (ptr<hb) {
          sh[thread_idx] = A_indices[ptr]*k;
          // sh[thread_idx] = __ldg(A_indices+ptr)*k;
        }
        __syncwarp();
        ptr += 32;
        for (int kk=0; kk<32&&jj+kk<hb; kk++) {
          offset = sh[sm_offset+kk]+cid;
          acc1 = sum_reduce(acc1, B[offset]);
          // acc1 = sum_reduce(acc1, __ldg(B+offset));
        }
        __syncwarp();
      }
      offset = rid*k+cid;
      C[offset] = acc1;
    }
    else { // threadIdx.y==blockDim.y-1
      int nout = (k-cid+31)/32;
      for (int jj=lb; jj<hb; jj+=32) {
        if (ptr<hb) {
          sh[thread_idx] = A_indices[ptr]*k;
          // sh[thread_idx] = __ldg(A_indices+ptr)*k;
        }
        __syncwarp();
        ptr += 32;
        for (int kk=0; kk<32&&jj+kk<hb; kk++) {
          offset = sh[(sm_offset+kk)] + cid;
          if (nout>0) {
          acc1 = sum_reduce(acc1, B[offset]);}
          // acc1 = sum_reduce(acc1, __ldg(B+offset)); }
        }
        __syncwarp();
      }
      offset = rid*k+cid;
      if (nout>0) {
      C[offset] = acc1;}
    }
  }
}

__global__ void topoSimpleSPMMKernel(
  int m, int k, const int* A_indptr, const int* A_indices, const float* B, float* C 
) {
  int rid = blockDim.y*blockIdx.x+threadIdx.y;
  if (rid<m) {
    int lb = A_indptr[rid];
    int hb = A_indptr[(rid+1)];
    float acc1 = sum_init();
    int offset;
    for (int ptr=lb; ptr<hb; ptr++) {
      // offset = __ldg(A_indices+ptr)*k+threadIdx.x;
      // acc1 = sum_reduce(acc1, __ldg(B+offset));
      offset = A_indices[ptr]*k+threadIdx.x;
      acc1 = sum_reduce(acc1, B[offset]);
    }
    C[(rid*k+threadIdx.x)] = acc1;
  }
}

torch::Tensor spmm_cuda_no_edge_value(
    torch::Tensor rowptr,
    torch::Tensor colind,
    torch::Tensor dense
) {
    const auto m = rowptr.size(0)-1;
    const auto k = dense.size(1);
    auto devid = dense.device().index();
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA, devid);
    auto out = torch::empty({m,k}, options);
    
    if (k<32) {
        const int row_per_block = 128/k;
        const int n_block = (m+row_per_block-1)/row_per_block;
        topoSimpleSPMMKernel<<< dim3(n_block,1,1),dim3(k, row_per_block, 1)>>>(
            m, k, rowptr.data_ptr<int>(), colind.data_ptr<int>(), dense.data_ptr<float>(), out.data_ptr<float>());
        return out;
    }
    if (k<64) {
        const int tile_k = (k+31)/32;
        const int n_block = (m+3)/4;
        topoCacheSPMMKernel<<< dim3(n_block,tile_k,1), dim3(32,4,1), 128*sizeof(int)>>>(
            m, k, rowptr.data_ptr<int>(), colind.data_ptr<int>(), dense.data_ptr<float>(), out.data_ptr<float>());
        return out;
    }
    else {
        const int tile_k = (k+63)/64;
        const int n_block = (m+8-1)/8;
        topoCacheCoarsenSPMMKernel<<< dim3(n_block,tile_k,1), dim3(32,8,1), 8*32*sizeof(int)>>>(
            m, k, rowptr.data_ptr<int>(), colind.data_ptr<int>(), dense.data_ptr<float>(), out.data_ptr<float>());
        return out;
    }
}


__global__ void spmm_test0(
    int A_nrows, int B_ncols,
    int* A_csrRowPtr, int* A_csrColInd, float* A_csrVal,
    float* B_dnVal, float* C_dnVal
)
{
    int rid = blockDim.y*blockIdx.x+threadIdx.y;
    if (rid<A_nrows) {
    int cid = (blockIdx.y<<5)+threadIdx.x;
    int lb = A_csrRowPtr[rid];
    int hb = A_csrRowPtr[(rid+1)];
    int offset = 0;
    float acc=0;
    if (blockIdx.y!=gridDim.y-1){
        for (int ptr = lb; ptr<hb; ptr++) {
            offset = A_csrColInd[ptr]*B_ncols+cid;
            acc += A_csrVal[ptr]*B_dnVal[offset];
        }
        C_dnVal[(rid*B_ncols+cid)] = acc;
    }
    else {
        for (int ptr = lb; ptr<hb; ptr++) {
            if (cid<B_ncols) {
            offset = A_csrColInd[ptr]*B_ncols+cid;}
            acc += A_csrVal[ptr]*B_dnVal[offset];
        }
        if (cid<B_ncols) {
        C_dnVal[(rid*B_ncols+cid)] = acc;}
    }
    }
}

__global__ void spmm_test1(
    int A_nrows, int B_ncols,
    int* A_csrRowPtr, int* A_csrColInd, float* A_csrVal,
    float* B_dnVal, float* C_dnVal
)
{
    extern __shared__ int sh[];
    int *colInd_sh = sh;
    float *val_sh = (float *)&sh[(blockDim.y<<5)];
    int shmem_offset = (threadIdx.y<<5);
    int thread_idx = shmem_offset+threadIdx.x;

    int rid = blockDim.y*blockIdx.x+threadIdx.y;
    
    if (rid<A_nrows) {
        int cid = (blockIdx.y<<5)+threadIdx.x;
        int lb = A_csrRowPtr[rid];
        int hb = A_csrRowPtr[(rid+1)];
        int ptr = lb+threadIdx.x;
        int offset;
        float acc=0;

        if (blockIdx.y != gridDim.y-1) {
            for (int jj=lb; jj<hb; jj+=32) {
                if (ptr<hb) {
                    val_sh[thread_idx] = A_csrVal[ptr];
                    colInd_sh[thread_idx] = B_ncols*A_csrColInd[ptr];
                }
                __syncwarp();
                ptr += 32;

                for (int kk=0; kk<32&&jj+kk<hb; kk++) {
                    offset = colInd_sh[(shmem_offset+kk)] + cid;
                    acc += val_sh[(shmem_offset+kk)]*B_dnVal[offset];
                }
                __syncwarp();
            }
            C_dnVal[(rid*B_ncols+cid)] = acc;
        }
        else {
            for (int jj=lb; jj<hb; jj+=32) {
                if (ptr<hb) {
                    val_sh[thread_idx] = A_csrVal[ptr];
                    colInd_sh[thread_idx] = B_ncols*A_csrColInd[ptr];
                }
                __syncwarp();
                ptr += 32;

                for (int kk=0; kk<32&&jj+kk<hb; kk++) {
                    offset = colInd_sh[(shmem_offset+kk)] + cid;
                    if (cid<B_ncols) {
                    acc += val_sh[(shmem_offset+kk)]*B_dnVal[offset];
                    }
                }
                __syncwarp();
            }
            if (cid<B_ncols) {
            C_dnVal[(rid*B_ncols+cid)] = acc;
            }
        }
    }
}

__global__ void spmm_test2(
    int A_nrows, int B_ncols,
    int* A_csrRowPtr, int* A_csrColInd, float* A_csrVal,
    float* B_dnVal, float* C_dnVal
)
{
    extern __shared__ int sh[];
    int *colInd_sh = sh;
    float *val_sh = (float *)&sh[(blockDim.y<<5)];
    int shmem_offset = (threadIdx.y<<5);
    int thread_idx = shmem_offset+threadIdx.x;

    int rid = blockDim.y*blockIdx.x+threadIdx.y;
    
    if (rid<A_nrows) {
        int cid = (blockIdx.y<<6)+threadIdx.x;
        int lb = A_csrRowPtr[rid];
        int hb = A_csrRowPtr[(rid+1)];
        int ptr = lb+threadIdx.x;
        int offset;
        float acc1=0, acc2=0, val;

        if (blockIdx.y != gridDim.y-1) {
            for (int jj=lb; jj<hb; jj+=32) {
                if (ptr<hb) {
                    val_sh[thread_idx] = A_csrVal[ptr];
                    colInd_sh[thread_idx] = B_ncols*A_csrColInd[ptr];
                }
                __syncwarp();
                ptr += 32;

                for (int kk=0; kk<32&&jj+kk<hb; kk++) {
                    offset = colInd_sh[(shmem_offset+kk)] + cid;
                    val = val_sh[(shmem_offset+kk)];
                    acc1 += val*B_dnVal[offset];
                    acc2 += val*B_dnVal[offset+32];
                }
                __syncwarp();
            }
            offset = rid*B_ncols+cid;
            C_dnVal[offset] = acc1;
            C_dnVal[offset+32] = acc2;
        }
        else {
            int nout = (B_ncols-cid+31)/32;
            for (int jj=lb; jj<hb; jj+=32) {
                if (ptr<hb) {
                    val_sh[thread_idx] = A_csrVal[ptr];
                    colInd_sh[thread_idx] = B_ncols*A_csrColInd[ptr];
                }
                __syncwarp();
                ptr += 32;

                for (int kk=0; kk<32&&jj+kk<hb; kk++) {
                    val = val_sh[(shmem_offset+kk)];
                    offset = colInd_sh[(shmem_offset+kk)] + cid;
                    if (nout>0) {
                    acc1 += val*B_dnVal[offset];
                    }
                    if (nout>1) {
                    acc2 += val*B_dnVal[offset+32];  
                    }
                }
                __syncwarp();
            }
            offset = rid*B_ncols+cid;
            if (nout>0) {
            C_dnVal[offset] = acc1;
            }
            if (nout>1) {
            C_dnVal[(offset+32)] = acc2;
            }
        }
    }
}

void csr2cscKernel(int m, int n, int nnz,
    int *csrRowPtr, int *csrColInd, float *csrVal,
    int *cscColPtr, int *cscRowInd, float *cscVal
)
{
    cusparseHandle_t handle;
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
        CUDA_R_32F,
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
        CUDA_R_32F,
        CUSPARSE_ACTION_NUMERIC,
        CUSPARSE_INDEX_BASE_ZERO,
        CUSPARSE_CSR2CSC_ALG1,
        buffer
    ));
    checkCudaError(cudaFree(buffer));
}

torch::Tensor spmm_cuda(
    torch::Tensor rowptr,
    torch::Tensor colind,
    torch::Tensor values,
    torch::Tensor dense
) {
    const auto m = rowptr.size(0)-1;
    const auto k = dense.size(1);
    auto devid = dense.device().index();
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA, devid);
    auto out = torch::empty({m,k}, options);
    
    if (k<32) {
        const int row_per_block = 128/k;
        const int n_block = (m+row_per_block-1)/row_per_block;
        spmm_test0<<<dim3(n_block,1,1),dim3(k, row_per_block, 1)>>>(
            m, k, rowptr.data_ptr<int>(), colind.data_ptr<int>(), values.data_ptr<float>(), dense.data_ptr<float>(), out.data_ptr<float>());
        return out;
    }
    if (k<64) {
        const int tile_k = (k+31)/32;
        const int n_block = (m+4-1)/4;
        spmm_test1<<<dim3(n_block, tile_k, 1), dim3(32, 4, 1), 32*4*(sizeof(int)+sizeof(float))>>> (
            m, k, rowptr.data_ptr<int>(), colind.data_ptr<int>(), values.data_ptr<float>(), dense.data_ptr<float>(), out.data_ptr<float>());
        return out;
    }
    else {
        const int tile_k = (k+63)/64;
        const int n_block = (m+8-1)/8;
        spmm_test2<<<dim3(n_block, tile_k, 1), dim3(32, 8, 1), 32*8*(sizeof(int)+sizeof(float))>>> (
            m, k, rowptr.data_ptr<int>(), colind.data_ptr<int>(), values.data_ptr<float>(), dense.data_ptr<float>(), out.data_ptr<float>());        
        return out;
    }
}

torch::Tensor csr2csc_cuda(
  torch::Tensor csrRowPtr,
  torch::Tensor csrColInd,
  torch::Tensor csrVal,
  torch::Tensor cscColPtr,
  torch::Tensor cscRowInd
)
{
  const auto m = csrRowPtr.size(0) - 1;
  const auto n = cscColPtr.size(0) - 1;
  const auto nnz = csrColInd.size(0);
  auto devid = csrRowPtr.device().index();
  auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA, devid);
  auto cscVal = torch::empty({nnz}, options);
  csr2cscKernel(m, n, nnz, csrRowPtr.data_ptr<int>(), csrColInd.data_ptr<int>(), csrVal.data_ptr<float>(), 
  cscColPtr.data_ptr<int>(), cscRowInd.data_ptr<int>(), cscVal.data_ptr<float>());
  return cscVal;
}