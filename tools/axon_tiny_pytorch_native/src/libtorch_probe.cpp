#include "probe_common.h"

#include <torch/script.h>


int wmain(int argc, wchar_t** argv) {
  try {
    const axon_tiny::Arguments arguments = axon_tiny::parse_arguments(argc, argv);
    if (arguments.help) {
      axon_tiny::print_usage("axon_tiny_libtorch_probe");
      return 0;
    }
    torch::jit::Module module = torch::jit::load(arguments.model.string(), at::kCPU);
    module.eval();
    return axon_tiny::execute(
        "libtorch", arguments, [&module](const at::Tensor& input) {
          const c10::IValue value = module.forward({input});
          if (!value.isTuple()) {
            throw std::runtime_error("TorchScript control must return a tuple");
          }
          std::vector<at::Tensor> outputs;
          for (const c10::IValue& element : value.toTupleRef().elements()) {
            if (!element.isTensor()) {
              throw std::runtime_error("TorchScript tuple contains a non-tensor");
            }
            outputs.push_back(element.toTensor());
          }
          return outputs;
        });
  } catch (const std::exception& error) {
    std::cerr << "[Error] " << error.what() << "\n";
    return 1;
  }
}
