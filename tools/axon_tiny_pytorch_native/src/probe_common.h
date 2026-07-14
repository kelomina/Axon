#pragma once

#include <ATen/ATen.h>
#include <ATen/Parallel.h>
#include <c10/core/InferenceMode.h>
#include <windows.h>

#include <array>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace axon_tiny {

struct Arguments {
  fs::path model;
  fs::path input;
  fs::path output_dir;
  fs::path manifest;
  std::size_t repeat = 0;
  bool help = false;
};

inline std::string path_text(const fs::path& path) {
  const std::wstring wide = path.wstring();
  if (wide.empty()) {
    return {};
  }
  const int size = WideCharToMultiByte(
      CP_UTF8, WC_ERR_INVALID_CHARS, wide.data(), static_cast<int>(wide.size()), nullptr, 0,
      nullptr, nullptr);
  if (size <= 0) {
    throw std::runtime_error("path is not valid UTF-16");
  }
  std::string result(static_cast<std::size_t>(size), '\0');
  WideCharToMultiByte(
      CP_UTF8, WC_ERR_INVALID_CHARS, wide.data(), static_cast<int>(wide.size()), result.data(),
      size, nullptr, nullptr);
  return result;
}

inline std::string json_escape(const std::string& value) {
  std::ostringstream output;
  for (const unsigned char character : value) {
    switch (character) {
      case '"':
        output << "\\\"";
        break;
      case '\\':
        output << "\\\\";
        break;
      case '\b':
        output << "\\b";
        break;
      case '\f':
        output << "\\f";
        break;
      case '\n':
        output << "\\n";
        break;
      case '\r':
        output << "\\r";
        break;
      case '\t':
        output << "\\t";
        break;
      default:
        if (character < 0x20) {
          output << "\\u" << std::hex << std::setw(4) << std::setfill('0')
                 << static_cast<unsigned int>(character) << std::dec;
        } else {
          output << static_cast<char>(character);
        }
    }
  }
  return output.str();
}

inline void print_usage(const char* executable) {
  std::cout << "Usage: " << executable
            << " --model ARTIFACT --input INPUT_F32 --output-dir DIR --manifest FILE --repeat 3\n";
}

inline std::size_t parse_repeat(const std::wstring& value) {
  if (value.empty()) {
    throw std::runtime_error("--repeat must be a positive integer");
  }
  std::size_t result = 0;
  for (const wchar_t character : value) {
    if (character < L'0' || character > L'9') {
      throw std::runtime_error("--repeat must be a positive integer");
    }
    const std::size_t digit = static_cast<std::size_t>(character - L'0');
    if (result > ((std::numeric_limits<std::size_t>::max)() - digit) / 10) {
      throw std::runtime_error("--repeat overflows size_t");
    }
    result = result * 10 + digit;
  }
  if (result != 3) {
    throw std::runtime_error("the frozen feasibility contract requires --repeat 3");
  }
  return result;
}

inline Arguments parse_arguments(int argc, wchar_t** argv) {
  Arguments arguments;
  auto take_path = [&](int& index, fs::path& destination, const char* name) {
    if (!destination.empty()) {
      throw std::runtime_error(std::string("duplicate option: ") + name);
    }
    if (index + 1 >= argc) {
      throw std::runtime_error(std::string("missing option value: ") + name);
    }
    destination = fs::path(argv[++index]);
  };
  for (int index = 1; index < argc; ++index) {
    const std::wstring option(argv[index]);
    if (option == L"--help" || option == L"-h") {
      arguments.help = true;
    } else if (option == L"--model") {
      take_path(index, arguments.model, "--model");
    } else if (option == L"--input") {
      take_path(index, arguments.input, "--input");
    } else if (option == L"--output-dir") {
      take_path(index, arguments.output_dir, "--output-dir");
    } else if (option == L"--manifest") {
      take_path(index, arguments.manifest, "--manifest");
    } else if (option == L"--repeat") {
      if (index + 1 >= argc || arguments.repeat != 0) {
        throw std::runtime_error("invalid --repeat option");
      }
      arguments.repeat = parse_repeat(argv[++index]);
    } else {
      throw std::runtime_error("unknown option: " + path_text(fs::path(option)));
    }
  }
  if (!arguments.help && (arguments.model.empty() || arguments.input.empty() ||
                          arguments.output_dir.empty() || arguments.manifest.empty() ||
                          arguments.repeat == 0)) {
    throw std::runtime_error("all model, input, output, manifest, and repeat options are required");
  }
  return arguments;
}

inline std::array<float, 16> read_input(const fs::path& path) {
  std::error_code error;
  const auto size = fs::file_size(path, error);
  if (error || size != sizeof(float) * 16) {
    throw std::runtime_error("input must contain exactly 16 little-endian float32 values");
  }
  std::array<float, 16> values{};
  std::ifstream stream(path, std::ios::binary);
  stream.read(reinterpret_cast<char*>(values.data()), static_cast<std::streamsize>(size));
  if (!stream || stream.gcount() != static_cast<std::streamsize>(size)) {
    throw std::runtime_error("unable to read the complete input tensor");
  }
  return values;
}

inline void write_exclusive(const fs::path& path, const void* data, std::size_t size) {
  const HANDLE handle = CreateFileW(
      path.c_str(), GENERIC_WRITE, 0, nullptr, CREATE_NEW, FILE_ATTRIBUTE_NORMAL, nullptr);
  if (handle == INVALID_HANDLE_VALUE) {
    throw std::runtime_error("exclusive output create failed: " + path_text(path));
  }
  const auto close = [&]() { CloseHandle(handle); };
  const auto* bytes = static_cast<const std::uint8_t*>(data);
  std::size_t offset = 0;
  while (offset < size) {
    const DWORD chunk = static_cast<DWORD>((std::min)(
        size - offset, static_cast<std::size_t>((std::numeric_limits<DWORD>::max)())));
    DWORD written = 0;
    if (!WriteFile(handle, bytes + offset, chunk, &written, nullptr) || written != chunk) {
      close();
      DeleteFileW(path.c_str());
      throw std::runtime_error("output write failed: " + path_text(path));
    }
    offset += written;
  }
  if (!FlushFileBuffers(handle)) {
    close();
    DeleteFileW(path.c_str());
    throw std::runtime_error("output flush failed: " + path_text(path));
  }
  close();
}

inline std::vector<at::Tensor> validate_outputs(std::vector<at::Tensor> outputs) {
  if (outputs.size() != 4) {
    throw std::runtime_error("model must return four tensors");
  }
  const std::array<std::vector<std::int64_t>, 4> shapes = {
      std::vector<std::int64_t>{2, 8}, std::vector<std::int64_t>{2, 8},
      std::vector<std::int64_t>{2, 2}, std::vector<std::int64_t>{2, 2}};
  const std::array<at::ScalarType, 4> dtypes = {
      at::kFloat, at::kFloat, at::kFloat, at::kLong};
  for (std::size_t index = 0; index < outputs.size(); ++index) {
    outputs[index] = outputs[index].detach().cpu().contiguous();
    if (outputs[index].sizes().vec() != shapes[index] || outputs[index].scalar_type() != dtypes[index]) {
      throw std::runtime_error("model output contract mismatch");
    }
  }
  return outputs;
}

inline int execute(
    const char* backend,
    const Arguments& arguments,
    const std::function<std::vector<at::Tensor>(const at::Tensor&)>& runner,
    bool require_model = true) {
  if ((require_model && !fs::is_regular_file(arguments.model)) ||
      !fs::is_regular_file(arguments.input)) {
    throw std::runtime_error("required model/input path is not a regular file");
  }
  std::error_code error;
  const fs::path output_root = fs::absolute(arguments.output_dir, error);
  if (error || !fs::create_directories(output_root, error) || error) {
    throw std::runtime_error("output directory must not already exist");
  }
  const fs::path manifest = fs::absolute(arguments.manifest, error);
  if (error || manifest.parent_path() != output_root || fs::exists(manifest)) {
    throw std::runtime_error("manifest must be an absent direct child of output-dir");
  }

  std::array<float, 16> values = read_input(arguments.input);
  at::set_num_threads(1);
  at::set_num_interop_threads(1);
  c10::InferenceMode inference_guard(true);
  const at::Tensor input =
      at::from_blob(values.data(), {2, 8}, at::TensorOptions().dtype(at::kFloat)).clone();

  const std::array<const char*, 4> names = {
      "linear", "gelu", "topk_values", "topk_indices"};
  const std::array<const char*, 4> dtypes = {"float32", "float32", "float32", "int64"};
  const std::array<std::array<std::int64_t, 2>, 4> shapes = {
      std::array<std::int64_t, 2>{2, 8}, std::array<std::int64_t, 2>{2, 8},
      std::array<std::int64_t, 2>{2, 2}, std::array<std::int64_t, 2>{2, 2}};

  std::ostringstream json;
  json << "{\n  \"schema\": \"axon_tiny_pytorch_native_probe_v1\",\n"
       << "  \"backend\": \"" << json_escape(backend) << "\",\n"
       << "  \"repeat_count\": " << arguments.repeat << ",\n  \"runs\": [\n";
  for (std::size_t run_index = 0; run_index < arguments.repeat; ++run_index) {
    std::vector<at::Tensor> outputs = validate_outputs(runner(input));
    json << "    {\"index\": " << run_index << ", \"outputs\": [";
    for (std::size_t output_index = 0; output_index < outputs.size(); ++output_index) {
      const std::string filename = std::string(backend) + ".run" +
          std::to_string(run_index) + "." + names[output_index] + ".bin";
      const fs::path output_path = output_root / filename;
      const std::size_t nbytes = outputs[output_index].nbytes();
      write_exclusive(output_path, outputs[output_index].const_data_ptr(), nbytes);
      if (output_index != 0) {
        json << ",";
      }
      json << "{\"name\": \"" << names[output_index] << "\", \"dtype\": \""
           << dtypes[output_index] << "\", \"shape\": [" << shapes[output_index][0]
           << ", " << shapes[output_index][1] << "], \"file\": \""
           << json_escape(filename) << "\", \"nbytes\": " << nbytes << "}";
    }
    json << "]}" << (run_index + 1 == arguments.repeat ? "\n" : ",\n");
  }
  json << "  ]\n}\n";
  const std::string payload = json.str();
  write_exclusive(manifest, payload.data(), payload.size());
  return 0;
}

}  // namespace axon_tiny
