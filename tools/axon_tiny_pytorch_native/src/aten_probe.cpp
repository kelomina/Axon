#include "probe_common.h"

#include <ATen/Functions.h>


int wmain(int argc, wchar_t** argv) {
  try {
    const axon_tiny::Arguments arguments = axon_tiny::parse_arguments(argc, argv);
    if (arguments.help) {
      axon_tiny::print_usage("axon_tiny_aten_probe");
      return 0;
    }
    return axon_tiny::execute(
        "aten", arguments,
        [](const at::Tensor& input) {
          const std::array<float, 32> base_rows = {
              0.5F,   -0.25F, 0.125F,  0.0F,   0.75F,  -0.5F,  0.25F,  -0.125F,
              -0.25F, 0.5F,   0.0F,    0.125F, -0.5F,  0.75F, -0.125F, 0.25F,
              0.125F, 0.0F,   0.5F,    -0.25F, 0.25F,  -0.125F, 0.75F, -0.5F,
              0.0F,   0.125F, -0.25F,  0.5F,   -0.125F, 0.25F, -0.5F,  0.75F};
          std::array<float, 64> weight_values{};
          for (std::size_t row = 0; row < 4; ++row) {
            for (std::size_t duplicate = 0; duplicate < 2; ++duplicate) {
              for (std::size_t column = 0; column < 8; ++column) {
                weight_values[(row * 2 + duplicate) * 8 + column] =
                    base_rows[row * 8 + column];
              }
            }
          }
          const std::array<float, 8> bias_values = {
              0.125F, 0.125F, -0.25F, -0.25F, 0.375F, 0.375F, -0.5F, -0.5F};
          const auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCPU);
          const at::Tensor weight = at::from_blob(
              weight_values.data(), {8, 8}, options).clone();
          const at::Tensor bias = at::from_blob(
              const_cast<float*>(bias_values.data()), {8}, options).clone();
          const at::Tensor linear = at::linear(input, weight, bias);
          const at::Tensor gelu = at::gelu(linear, "none");
          auto [topk_values, topk_indices] = at::topk(gelu, 2, -1, true, true);
          return std::vector<at::Tensor>{linear, gelu, topk_values, topk_indices};
        },
        false);
  } catch (const std::exception& error) {
    std::cerr << "[Error] " << error.what() << "\n";
    return 1;
  }
}
