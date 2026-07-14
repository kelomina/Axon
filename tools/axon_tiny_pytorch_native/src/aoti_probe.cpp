#include "probe_common.h"

#include <torch/csrc/inductor/aoti_package/model_package_loader.h>


int wmain(int argc, wchar_t** argv) {
  try {
    const axon_tiny::Arguments arguments = axon_tiny::parse_arguments(argc, argv);
    if (arguments.help) {
      axon_tiny::print_usage("axon_tiny_aoti_probe");
      return 0;
    }
    torch::inductor::AOTIModelPackageLoader loader(
        arguments.model.string(), "model", true, 1, -1);
    const auto metadata = loader.get_metadata();
    const auto device = metadata.find("AOTI_DEVICE_KEY");
    if (device == metadata.end() || device->second != "cpu") {
      throw std::runtime_error("AOTI package metadata is not CPU-only");
    }
    std::cout << "AOTI_DEVICE_KEY=cpu\n";
    return axon_tiny::execute(
        "aoti", arguments, [&loader](const at::Tensor& input) {
          return loader.run({input});
        });
  } catch (const std::exception& error) {
    std::cerr << "[Error] " << error.what() << "\n";
    return 1;
  }
}
