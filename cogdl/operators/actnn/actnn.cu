#include <torch/extension.h>
#include <ATen/CUDAGeneratorImpl.h>
#include <THC/THCAtomics.cuh>

#include <cuda.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>

using torch::Tensor;


/****************************************/
/********* Act Quantized Dropout ********/
/****************************************/
#define ACT_QUANTIZED_DROPOUT_NUM_THREADS 512
// Compute Dropout forward and 1-bit activations (mask) and pack the mask into int32 streams
template <typename scalar_t>
__global__ void act_quantized_dropout_forward_kernel(const scalar_t* __restrict__ data,
                                                     int32_t* __restrict__ mask,
                                                     scalar_t* __restrict__ output,
                                                     std::pair<uint64_t, uint64_t> seeds,
                                                     int64_t N,
                                                     int64_t mask_len,
                                                     float dropout_p) {
  const int64_t id = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  const int64_t global_offset = (int64_t)blockIdx.x * blockDim.x / (sizeof(int32_t) * 8);
  const int shared_len = ACT_QUANTIZED_DROPOUT_NUM_THREADS / (sizeof(int32_t) * 8);
  __shared__ int mask_shared[ACT_QUANTIZED_DROPOUT_NUM_THREADS / (sizeof(int32_t) * 8)];

  if (threadIdx.x * 2 < shared_len) {
    reinterpret_cast<int2*>(mask_shared)[threadIdx.x] = make_int2(0, 0);
  }

  curandStatePhilox4_32_10_t state;
  curand_init(seeds.first, id, seeds.second, &state);
  const float noise = curand_uniform(&state);

  if (id < N) {
    bool bit = noise > dropout_p;
    if (bit) {
      output[id] = data[id] / (1.0 - dropout_p);
    } else {
      output[id] = 0.0;
    }

    __syncthreads();
    atomicOr(mask_shared + threadIdx.x % shared_len, bit << (threadIdx.x / shared_len));
    __syncthreads();
  }

  if (threadIdx.x * 2 < shared_len) {
    reinterpret_cast<int2*>(mask)[global_offset / 2 + threadIdx.x] = reinterpret_cast<int2*>(mask_shared)[threadIdx.x];
  }
}

std::pair<Tensor, Tensor> act_quantized_dropout_forward_cuda(Tensor data, float dropout_p) {
  int64_t n_elements = 1;
  for (size_t i = 0; i < data.dim(); ++i) {
    n_elements *= data.size(i);
  }

  auto options = torch::TensorOptions().dtype(torch::kInt32).device(data.device());
  int64_t mask_len = (n_elements + sizeof(int32_t) * 8 - 1) / (sizeof(int32_t) * 8);
  Tensor mask = torch::empty({mask_len}, options);
  Tensor output = torch::empty_like(data);

  int threads = ACT_QUANTIZED_DROPOUT_NUM_THREADS;
  int blocks = (n_elements + threads - 1) / threads;

  // Random number generator
  auto gen = at::check_generator<at::CUDAGeneratorImpl>(at::cuda::detail::getDefaultCUDAGenerator());
  std::pair<uint64_t, uint64_t> rng_engine_inputs;
  {
    // See Note [Acquire lock when using random generators]
    std::lock_guard<std::mutex> lock(gen->mutex_);
    rng_engine_inputs = gen->philox_engine_inputs(threads);
  }

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(data.scalar_type(), "act_quantized_dropout_forward", ([&] {
    act_quantized_dropout_forward_kernel<scalar_t><<<blocks, threads>>>(
      data.data_ptr<scalar_t>(), mask.data_ptr<int32_t>(), output.data_ptr<scalar_t>(),
      rng_engine_inputs, n_elements, mask_len, dropout_p);
  }));

  return std::make_pair(output, mask);
}

// Unpack 1-bit activations (mask) from the saved int32 stream and compute Dropout backward
template <typename scalar_t>
__global__ void act_quantized_dropout_backward_kernel(const scalar_t* __restrict__ grad_output,
                                                   int32_t* __restrict__ mask,
                                                   scalar_t* __restrict__ grad_input,
                                                   int N,
                                                   float dropout_p) {
  int64_t id = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  const int64_t global_offset = (int64_t)blockIdx.x * blockDim.x / (sizeof(int32_t) * 8);
  const int shared_len = ACT_QUANTIZED_DROPOUT_NUM_THREADS / (sizeof(int32_t) * 8);

  if (id < N) {
    bool bit =  (mask[global_offset + threadIdx.x % shared_len] >> (threadIdx.x / shared_len)) & 1;
    if (bit) {
      grad_input[id] = grad_output[id] / (1.0 - dropout_p);
    } else {
      grad_input[id] = 0.0;
    }
  }
}


Tensor act_quantized_dropout_backward_cuda(Tensor grad_output, Tensor mask, float dropout_p) {
  int64_t n_elements = 1;
  for (size_t i = 0; i < grad_output.dim(); ++i) {
    n_elements *= grad_output.size(i);
  }

  int threads = ACT_QUANTIZED_DROPOUT_NUM_THREADS;
  int blocks = (n_elements + threads - 1) / threads;

  Tensor grad_input = torch::empty_like(grad_output);

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(grad_output.scalar_type(), "act_quantized_dropout_backward", ([&] {
      act_quantized_dropout_backward_kernel<scalar_t><<<blocks, threads>>>(
        grad_output.data_ptr<scalar_t>(), mask.data_ptr<int32_t>(), grad_input.data_ptr<scalar_t>(),
        n_elements, dropout_p);
  }));

  return grad_input;
}
