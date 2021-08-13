#include <torch/extension.h>
#include <torch/torch.h>

using torch::autograd::Function;
using torch::autograd::AutogradContext;
using torch::autograd::tensor_list;
using torch::Tensor;

// Helper for type check
#define CHECK_CUDA_TENSOR_FLOAT(name)                                             \
  TORCH_CHECK(name.device().is_cuda(), #name " must be a CUDA tensor!");          \
  TORCH_CHECK(name.is_contiguous(), #name " must be contiguous!");                \
  TORCH_CHECK(name.dtype() == torch::kFloat32 || name.dtype() == torch::kFloat16, \
              "The type of " #name " is not correct!");                           \

// ActQuantizedDropout
std::pair<Tensor, Tensor> act_quantized_dropout_forward_cuda(Tensor data, float dropout_p);
Tensor act_quantized_dropout_backward_cuda(Tensor grad_output, Tensor mask, float dropout_p);

// Activation quantized dropout: use compressed bit stream to store activation
class ActQuantizedDropout : public Function<ActQuantizedDropout> {
 public:
  static Tensor forward(AutogradContext *ctx, Tensor input, float dropout_p) {
    Tensor output, mask;
    std::tie(output, mask) = act_quantized_dropout_forward_cuda(input, dropout_p);
    ctx->save_for_backward({mask});
    ctx->saved_data["dropout_p"] = dropout_p;
    return output;
  }

  static tensor_list backward(AutogradContext *ctx, tensor_list grad_outputs) {
    auto saved = ctx->get_saved_variables();
    float dropout_p = float(ctx->saved_data["dropout_p"].toDouble());
    return {act_quantized_dropout_backward_cuda(grad_outputs[0], saved[0], dropout_p), Tensor()};
  }
};

Tensor act_quantized_dropout(Tensor input, float dropout_p) {
  CHECK_CUDA_TENSOR_FLOAT(input);
  return ActQuantizedDropout::apply(input, dropout_p);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("act_quantized_dropout", &act_quantized_dropout);
}
