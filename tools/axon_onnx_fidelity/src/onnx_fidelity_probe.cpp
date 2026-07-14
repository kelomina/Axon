#include <onnxruntime_cxx_api.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <locale>
#include <optional>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

#if defined(_WIN32)
#include <windows.h>
#else
#include <cerrno>
#include <fcntl.h>
#include <unistd.h>
#endif

namespace {

namespace fs = std::filesystem;

constexpr std::size_t kByteElementCount = 8192;
constexpr std::size_t kPeElementCount = 256;
constexpr std::size_t kStatElementCount = 49;
constexpr std::size_t kMaximumRepeatCount = 1000;

static_assert(sizeof(std::int64_t) == 8, "int64 input requires 8-byte integers");
static_assert(sizeof(std::int32_t) == 4, "int32 output requires 4-byte integers");
static_assert(sizeof(float) == 4, "float32 input requires 4-byte floats");
static_assert(sizeof(double) == 8, "float64 output requires 8-byte doubles");
static_assert(sizeof(bool) == 1, "ONNX bool output requires 1-byte bool storage");

#if defined(_WIN32)
using NativeChar = wchar_t;
#else
using NativeChar = char;
#endif
using NativeString = std::basic_string<NativeChar>;

struct Arguments {
  fs::path onnx;
  fs::path byte;
  fs::path pe;
  fs::path stat;
  fs::path output_dir;
  fs::path manifest;
  std::size_t repeat = 0;
  bool help = false;
};

struct DtypeSpec {
  const char* name;
  std::size_t element_size;
};

struct OutputRecord {
  std::string name;
  std::string dtype;
  std::vector<std::int64_t> shape;
  std::string file;
  std::size_t nbytes = 0;
};

struct RunRecord {
  std::size_t index = 0;
  std::vector<OutputRecord> outputs;
};

NativeString native_ascii(const char* text) {
  NativeString result;
  for (const unsigned char character : std::string(text)) {
    result.push_back(static_cast<NativeChar>(character));
  }
  return result;
}

std::string path_text(const fs::path& path) {
#if defined(_WIN32)
  return path.u8string();
#else
  return path.string();
#endif
}

std::string json_escape(const std::string& text) {
  std::ostringstream output;
  output << std::hex << std::setfill('0');
  for (const unsigned char character : text) {
    switch (character) {
      case '"': output << "\\\""; break;
      case '\\': output << "\\\\"; break;
      case '\b': output << "\\b"; break;
      case '\f': output << "\\f"; break;
      case '\n': output << "\\n"; break;
      case '\r': output << "\\r"; break;
      case '\t': output << "\\t"; break;
      default:
        if (character < 0x20) {
          output << "\\u" << std::setw(4) << static_cast<unsigned int>(character);
        } else {
          output << static_cast<char>(character);
        }
    }
  }
  return output.str();
}

void print_usage() {
  std::cout
      << "Usage: axon_onnx_fidelity_probe --onnx MODEL --byte BYTE_I64 "
         "--pe PE_F32 --stat STAT_F32 --output-dir DIR --manifest FILE "
         "--repeat COUNT\n";
}

std::size_t parse_repeat(const NativeString& text) {
  if (text.empty()) {
    throw std::runtime_error("--repeat must be a positive decimal integer");
  }
  std::size_t result = 0;
  for (const NativeChar character : text) {
    if (character < static_cast<NativeChar>('0') ||
        character > static_cast<NativeChar>('9')) {
      throw std::runtime_error("--repeat must be a positive decimal integer");
    }
    const std::size_t digit =
        static_cast<std::size_t>(character - static_cast<NativeChar>('0'));
    if (result > ((std::numeric_limits<std::size_t>::max)() - digit) / 10) {
      throw std::runtime_error("--repeat overflows size_t");
    }
    result = result * 10 + digit;
  }
  if (result == 0 || result > kMaximumRepeatCount) {
    throw std::runtime_error("--repeat must be in the range 1..1000");
  }
  return result;
}

Arguments parse_arguments(int argc, NativeChar** argv) {
  Arguments arguments;
  std::optional<fs::path> onnx;
  std::optional<fs::path> byte;
  std::optional<fs::path> pe;
  std::optional<fs::path> stat;
  std::optional<fs::path> output_dir;
  std::optional<fs::path> manifest;
  std::optional<std::size_t> repeat;

  auto take_path = [&](int& index, std::optional<fs::path>& destination, const char* flag) {
    if (destination) {
      throw std::runtime_error(std::string("duplicate CLI option: ") + flag);
    }
    if (index + 1 >= argc) {
      throw std::runtime_error(std::string("missing value for ") + flag);
    }
    destination = fs::path(argv[++index]);
    if (destination->empty()) {
      throw std::runtime_error(std::string("empty path for ") + flag);
    }
  };

  for (int index = 1; index < argc; ++index) {
    const NativeString option(argv[index]);
    if (option == native_ascii("--help") || option == native_ascii("-h")) {
      arguments.help = true;
      continue;
    }
    if (option == native_ascii("--onnx")) {
      take_path(index, onnx, "--onnx");
    } else if (option == native_ascii("--byte")) {
      take_path(index, byte, "--byte");
    } else if (option == native_ascii("--pe")) {
      take_path(index, pe, "--pe");
    } else if (option == native_ascii("--stat")) {
      take_path(index, stat, "--stat");
    } else if (option == native_ascii("--output-dir")) {
      take_path(index, output_dir, "--output-dir");
    } else if (option == native_ascii("--manifest")) {
      take_path(index, manifest, "--manifest");
    } else if (option == native_ascii("--repeat")) {
      if (repeat) {
        throw std::runtime_error("duplicate CLI option: --repeat");
      }
      if (index + 1 >= argc) {
        throw std::runtime_error("missing value for --repeat");
      }
      repeat = parse_repeat(NativeString(argv[++index]));
    } else {
      throw std::runtime_error("unknown CLI option: " + path_text(fs::path(option)));
    }
  }

  if (arguments.help) {
    return arguments;
  }
  if (!onnx || !byte || !pe || !stat || !output_dir || !manifest || !repeat) {
    throw std::runtime_error(
        "all of --onnx, --byte, --pe, --stat, --output-dir, --manifest, and "
        "--repeat are required");
  }
  arguments.onnx = std::move(*onnx);
  arguments.byte = std::move(*byte);
  arguments.pe = std::move(*pe);
  arguments.stat = std::move(*stat);
  arguments.output_dir = std::move(*output_dir);
  arguments.manifest = std::move(*manifest);
  arguments.repeat = *repeat;
  return arguments;
}

fs::path prepare_output_directory(const fs::path& requested) {
  std::error_code error;
  fs::path absolute = fs::absolute(requested, error);
  if (error) {
    throw std::runtime_error("cannot resolve --output-dir: " + error.message());
  }
  fs::create_directories(absolute, error);
  if (error) {
    throw std::runtime_error("cannot create --output-dir: " + error.message());
  }
  fs::path canonical = fs::weakly_canonical(absolute, error);
  if (error || !fs::is_directory(canonical)) {
    throw std::runtime_error("--output-dir is not a usable directory");
  }
  return canonical;
}

fs::path confine_manifest_path(
    const fs::path& output_directory,
    const fs::path& requested) {
  fs::path candidate;
  if (requested.is_absolute()) {
    candidate = requested;
  } else if (requested.parent_path().empty() || requested.parent_path() == ".") {
    candidate = output_directory / requested.filename();
  } else {
    std::error_code absolute_error;
    candidate = fs::absolute(requested, absolute_error);
    if (absolute_error) {
      throw std::runtime_error("cannot resolve --manifest: " + absolute_error.message());
    }
  }
  candidate = candidate.lexically_normal();
  if (candidate.filename().empty() || candidate.filename() == "." ||
      candidate.filename() == "..") {
    throw std::runtime_error("--manifest must name a regular file");
  }

  std::error_code error;
  fs::path parent = fs::weakly_canonical(candidate.parent_path(), error);
  if (error || parent != output_directory) {
    throw std::runtime_error("--manifest must be a direct child of --output-dir");
  }
  return output_directory / candidate.filename();
}

void require_absent(const fs::path& path) {
  std::error_code error;
  const bool exists = fs::exists(path, error);
  if (error) {
    throw std::runtime_error(
        "cannot inspect output path " + path_text(path) + ": " + error.message());
  }
  if (exists) {
    throw std::runtime_error("refusing to overwrite existing output: " + path_text(path));
  }
}

template <typename T, std::size_t Size>
std::array<T, Size> read_exact_array(const fs::path& path, const char* label) {
  constexpr std::size_t expected_bytes = sizeof(T) * Size;
  std::error_code error;
  const std::uintmax_t file_bytes = fs::file_size(path, error);
  if (error) {
    throw std::runtime_error(
        std::string("cannot read ") + label + " input size: " + error.message());
  }
  if (file_bytes != expected_bytes) {
    std::ostringstream message;
    message << label << " input must contain exactly " << expected_bytes
            << " bytes, got " << file_bytes;
    throw std::runtime_error(message.str());
  }

  std::array<T, Size> result{};
  std::ifstream stream(path, std::ios::binary);
  if (!stream) {
    throw std::runtime_error(std::string("cannot open ") + label + " input");
  }
  stream.read(
      reinterpret_cast<char*>(result.data()),
      static_cast<std::streamsize>(expected_bytes));
  if (stream.gcount() != static_cast<std::streamsize>(expected_bytes) ||
      stream.peek() != std::char_traits<char>::eof()) {
    throw std::runtime_error(std::string(label) + " input changed while being read");
  }
  return result;
}

void remove_noexcept(const fs::path& path) noexcept {
  std::error_code ignored;
  fs::remove(path, ignored);
}

void write_exclusive(
    const fs::path& path,
    const void* data,
    std::size_t size) {
  if (size > 0 && data == nullptr) {
    throw std::runtime_error("cannot write a null buffer to " + path_text(path));
  }

#if defined(_WIN32)
  HANDLE handle = CreateFileW(
      path.c_str(),
      GENERIC_WRITE,
      0,
      nullptr,
      CREATE_NEW,
      FILE_ATTRIBUTE_NORMAL,
      nullptr);
  if (handle == INVALID_HANDLE_VALUE) {
    throw std::runtime_error(
        "exclusive create failed for " + path_text(path) +
        " (Win32 error " + std::to_string(GetLastError()) + ")");
  }
  try {
    const auto* bytes = static_cast<const std::uint8_t*>(data);
    std::size_t written = 0;
    while (written < size) {
      const std::size_t remaining = size - written;
      const DWORD chunk = static_cast<DWORD>(std::min<std::size_t>(
          remaining,
          (std::numeric_limits<DWORD>::max)()));
      DWORD completed = 0;
      if (!WriteFile(handle, bytes + written, chunk, &completed, nullptr) ||
          completed != chunk) {
        throw std::runtime_error(
            "write failed for " + path_text(path) +
            " (Win32 error " + std::to_string(GetLastError()) + ")");
      }
      written += completed;
    }
    if (!FlushFileBuffers(handle)) {
      throw std::runtime_error(
          "flush failed for " + path_text(path) +
          " (Win32 error " + std::to_string(GetLastError()) + ")");
    }
    CloseHandle(handle);
  } catch (...) {
    CloseHandle(handle);
    remove_noexcept(path);
    throw;
  }
#else
  const int descriptor = ::open(
      path.c_str(),
      O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC,
      0600);
  if (descriptor < 0) {
    throw std::runtime_error(
        "exclusive create failed for " + path_text(path) + ": " +
        std::strerror(errno));
  }
  try {
    const auto* bytes = static_cast<const std::uint8_t*>(data);
    std::size_t written = 0;
    while (written < size) {
      const ssize_t completed = ::write(descriptor, bytes + written, size - written);
      if (completed <= 0) {
        throw std::runtime_error(
            "write failed for " + path_text(path) + ": " + std::strerror(errno));
      }
      written += static_cast<std::size_t>(completed);
    }
    if (::fsync(descriptor) != 0) {
      throw std::runtime_error(
          "flush failed for " + path_text(path) + ": " + std::strerror(errno));
    }
    ::close(descriptor);
  } catch (...) {
    ::close(descriptor);
    remove_noexcept(path);
    throw;
  }
#endif
}

DtypeSpec dtype_spec(ONNXTensorElementDataType type) {
  switch (type) {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT: return {"float32", sizeof(float)};
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE: return {"float64", sizeof(double)};
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64: return {"int64", sizeof(std::int64_t)};
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32: return {"int32", sizeof(std::int32_t)};
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL: return {"bool", sizeof(bool)};
    default: throw std::runtime_error("unsupported ONNX tensor element type");
  }
}

std::size_t checked_tensor_bytes(
    const std::vector<std::int64_t>& shape,
    std::size_t element_size) {
  std::size_t element_count = 1;
  for (const std::int64_t dimension : shape) {
    if (dimension < 0) {
      throw std::runtime_error("runtime output contains a dynamic negative shape");
    }
    const auto value = static_cast<std::uint64_t>(dimension);
    if (value > (std::numeric_limits<std::size_t>::max)()) {
      throw std::runtime_error("runtime output shape does not fit size_t");
    }
    const std::size_t size_dimension = static_cast<std::size_t>(value);
    if (size_dimension != 0 &&
        element_count > (std::numeric_limits<std::size_t>::max)() / size_dimension) {
      throw std::runtime_error("runtime output element count overflows size_t");
    }
    element_count *= size_dimension;
  }
  if (element_size != 0 &&
      element_count > (std::numeric_limits<std::size_t>::max)() / element_size) {
    throw std::runtime_error("runtime output byte size overflows size_t");
  }
  return element_count * element_size;
}

void validate_input_contract(Ort::Session& session) {
  struct ExpectedInput {
    const char* name;
    ONNXTensorElementDataType type;
    std::array<std::int64_t, 2> shape;
  };
  const std::array<ExpectedInput, 3> expected = {{
      {"byte_seq", ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, {1, 8192}},
      {"pe_features", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, {1, 256}},
      {"stat_features", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, {1, 49}},
  }};

  if (session.GetInputCount() != expected.size()) {
    throw std::runtime_error("ONNX graph must expose exactly three inputs");
  }
  Ort::AllocatorWithDefaultOptions allocator;
  std::set<std::string> seen;
  for (std::size_t index = 0; index < session.GetInputCount(); ++index) {
    auto allocated_name = session.GetInputNameAllocated(index, allocator);
    const std::string name = allocated_name ? allocated_name.get() : "";
    if (name.empty() || !seen.insert(name).second) {
      throw std::runtime_error("ONNX graph contains an empty or duplicate input name");
    }
    const auto match = std::find_if(
        expected.begin(),
        expected.end(),
        [&](const ExpectedInput& item) { return name == item.name; });
    if (match == expected.end()) {
      throw std::runtime_error("unexpected ONNX input name: " + name);
    }
    const Ort::TypeInfo type_info = session.GetInputTypeInfo(index);
    if (type_info.GetONNXType() != ONNX_TYPE_TENSOR) {
      throw std::runtime_error("ONNX input is not a tensor: " + name);
    }
    const auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
    if (tensor_info.GetElementType() != match->type) {
      throw std::runtime_error("ONNX input has the wrong dtype: " + name);
    }
    const std::vector<std::int64_t> declared_shape = tensor_info.GetShape();
    if (declared_shape.size() != match->shape.size()) {
      throw std::runtime_error("ONNX input has the wrong rank: " + name);
    }
    for (std::size_t dimension = 0; dimension < declared_shape.size(); ++dimension) {
      if (declared_shape[dimension] >= 0 &&
          declared_shape[dimension] != match->shape[dimension]) {
        throw std::runtime_error("ONNX input has an incompatible shape: " + name);
      }
    }
  }
}

std::vector<std::string> enumerate_outputs(Ort::Session& session) {
  const std::size_t count = session.GetOutputCount();
  if (count == 0) {
    throw std::runtime_error("ONNX graph exposes no outputs");
  }
  Ort::AllocatorWithDefaultOptions allocator;
  std::set<std::string> seen;
  std::vector<std::string> names;
  names.reserve(count);
  for (std::size_t index = 0; index < count; ++index) {
    auto allocated_name = session.GetOutputNameAllocated(index, allocator);
    std::string name = allocated_name ? allocated_name.get() : "";
    if (name.empty() || !seen.insert(name).second) {
      throw std::runtime_error("ONNX graph contains an empty or duplicate output name");
    }
    const Ort::TypeInfo type_info = session.GetOutputTypeInfo(index);
    if (type_info.GetONNXType() != ONNX_TYPE_TENSOR) {
      throw std::runtime_error("ONNX output is not a tensor: " + name);
    }
    static_cast<void>(dtype_spec(type_info.GetTensorTypeAndShapeInfo().GetElementType()));
    names.push_back(std::move(name));
  }
  return names;
}

std::string output_filename(std::size_t run_index, std::size_t output_index) {
  std::ostringstream name;
  name << "run" << std::setw(2) << std::setfill('0') << run_index
       << "_output" << std::setw(4) << std::setfill('0') << output_index
       << ".bin";
  return name.str();
}

std::string build_manifest_json(
    std::size_t repeat,
    const std::vector<RunRecord>& runs) {
  std::ostringstream json;
  json.imbue(std::locale::classic());
  json << "{\n"
       << "  \"schema\": \"axon_onnx_fidelity_probe_output_v1\",\n"
       << "  \"repeat\": " << repeat << ",\n"
       << "  \"runs\": [\n";
  for (std::size_t run_index = 0; run_index < runs.size(); ++run_index) {
    const RunRecord& run = runs[run_index];
    json << "    {\"index\": " << run.index << ", \"outputs\": [\n";
    for (std::size_t output_index = 0; output_index < run.outputs.size(); ++output_index) {
      const OutputRecord& output = run.outputs[output_index];
      json << "      {\"name\": \"" << json_escape(output.name)
           << "\", \"dtype\": \"" << output.dtype << "\", \"shape\": [";
      for (std::size_t dimension = 0; dimension < output.shape.size(); ++dimension) {
        if (dimension != 0) {
          json << ", ";
        }
        json << output.shape[dimension];
      }
      json << "], \"file\": \"" << json_escape(output.file)
           << "\", \"nbytes\": " << output.nbytes << "}";
      if (output_index + 1 != run.outputs.size()) {
        json << ',';
      }
      json << '\n';
    }
    json << "    ]}";
    if (run_index + 1 != runs.size()) {
      json << ',';
    }
    json << '\n';
  }
  json << "  ]\n}\n";
  return json.str();
}

int execute_probe(const Arguments& arguments) {
  const fs::path output_directory = prepare_output_directory(arguments.output_dir);
  const fs::path manifest_path =
      confine_manifest_path(output_directory, arguments.manifest);
  require_absent(manifest_path);

  const auto byte_values =
      read_exact_array<std::int64_t, kByteElementCount>(arguments.byte, "byte");
  const auto pe_values =
      read_exact_array<float, kPeElementCount>(arguments.pe, "pe");
  const auto stat_values =
      read_exact_array<float, kStatElementCount>(arguments.stat, "stat");

  Ort::Env environment(ORT_LOGGING_LEVEL_WARNING, "AxonOnnxFidelityProbe");
  Ort::SessionOptions session_options;
  session_options.SetIntraOpNumThreads(1);
  session_options.SetInterOpNumThreads(1);
  session_options.SetExecutionMode(ORT_SEQUENTIAL);
  session_options.SetGraphOptimizationLevel(ORT_DISABLE_ALL);
  session_options.SetDeterministicCompute(true);

#if defined(_WIN32)
  Ort::Session session(environment, arguments.onnx.c_str(), session_options);
#else
  Ort::Session session(environment, arguments.onnx.string().c_str(), session_options);
#endif
  validate_input_contract(session);
  const std::vector<std::string> output_names = enumerate_outputs(session);
  if (arguments.repeat >
      (std::numeric_limits<std::size_t>::max)() / output_names.size()) {
    throw std::runtime_error("repeat multiplied by output count overflows size_t");
  }

  std::vector<const char*> output_name_pointers;
  output_name_pointers.reserve(output_names.size());
  for (const std::string& name : output_names) {
    output_name_pointers.push_back(name.c_str());
  }

  const std::array<const char*, 3> input_names = {
      "byte_seq", "pe_features", "stat_features"};
  const std::array<std::int64_t, 2> byte_shape = {1, 8192};
  const std::array<std::int64_t, 2> pe_shape = {1, 256};
  const std::array<std::int64_t, 2> stat_shape = {1, 49};
  auto memory = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  std::array<Ort::Value, 3> input_tensors = {
      Ort::Value::CreateTensor<std::int64_t>(
          memory,
          const_cast<std::int64_t*>(byte_values.data()),
          byte_values.size(),
          byte_shape.data(),
          byte_shape.size()),
      Ort::Value::CreateTensor<float>(
          memory,
          const_cast<float*>(pe_values.data()),
          pe_values.size(),
          pe_shape.data(),
          pe_shape.size()),
      Ort::Value::CreateTensor<float>(
          memory,
          const_cast<float*>(stat_values.data()),
          stat_values.size(),
          stat_shape.data(),
          stat_shape.size()),
  };

  // 所有产物使用固定文件名并在推理前做一次冲突预检；真正写入仍使用原子排他创建。
  for (std::size_t run_index = 0; run_index < arguments.repeat; ++run_index) {
    for (std::size_t output_index = 0; output_index < output_names.size(); ++output_index) {
      const fs::path output_path =
          output_directory / output_filename(run_index, output_index);
      if (output_path == manifest_path) {
        throw std::runtime_error("--manifest collides with a tensor output filename");
      }
      require_absent(output_path);
    }
  }

  std::vector<fs::path> created_outputs;
  std::vector<RunRecord> run_records;
  run_records.reserve(arguments.repeat);
  try {
    for (std::size_t run_index = 0; run_index < arguments.repeat; ++run_index) {
      std::vector<Ort::Value> outputs = session.Run(
          Ort::RunOptions{nullptr},
          input_names.data(),
          input_tensors.data(),
          input_tensors.size(),
          output_name_pointers.data(),
          output_name_pointers.size());
      if (outputs.size() != output_names.size()) {
        throw std::runtime_error("ONNX Runtime returned an unexpected output count");
      }

      RunRecord run;
      run.index = run_index;
      run.outputs.reserve(outputs.size());
      for (std::size_t output_index = 0; output_index < outputs.size(); ++output_index) {
        const Ort::Value& value = outputs[output_index];
        if (!value.IsTensor()) {
          throw std::runtime_error("ONNX Runtime returned a non-tensor output");
        }
        const auto tensor_info = value.GetTensorTypeAndShapeInfo();
        const DtypeSpec spec = dtype_spec(tensor_info.GetElementType());
        const std::vector<std::int64_t> shape = tensor_info.GetShape();
        const std::size_t nbytes = checked_tensor_bytes(shape, spec.element_size);
        if (tensor_info.GetElementCount() != nbytes / spec.element_size) {
          throw std::runtime_error("ONNX Runtime output element count is inconsistent");
        }
        const void* raw_data = value.GetTensorRawData();
        if (nbytes > 0 && raw_data == nullptr) {
          throw std::runtime_error("ONNX Runtime returned a null tensor buffer");
        }

        const std::string filename = output_filename(run_index, output_index);
        const fs::path output_path = output_directory / filename;
        write_exclusive(output_path, raw_data, nbytes);
        created_outputs.push_back(output_path);
        run.outputs.push_back(OutputRecord{
            output_names[output_index],
            spec.name,
            shape,
            filename,
            nbytes,
        });
      }
      run_records.push_back(std::move(run));
    }

    const std::string manifest = build_manifest_json(arguments.repeat, run_records);
    write_exclusive(manifest_path, manifest.data(), manifest.size());
  } catch (...) {
    for (auto path = created_outputs.rbegin(); path != created_outputs.rend(); ++path) {
      remove_noexcept(*path);
    }
    throw;
  }

  std::cout << "wrote " << run_records.size() << " runs and "
            << output_names.size() << " outputs per run to "
            << path_text(output_directory) << '\n';
  return 0;
}

int probe_main(int argc, NativeChar** argv) {
  try {
    const Arguments arguments = parse_arguments(argc, argv);
    if (arguments.help) {
      print_usage();
      return 0;
    }
    return execute_probe(arguments);
  } catch (const Ort::Exception& error) {
    std::cerr << "ONNX Runtime error: " << error.what() << '\n';
    return 2;
  } catch (const std::exception& error) {
    std::cerr << "Error: " << error.what() << '\n';
    return 2;
  } catch (...) {
    std::cerr << "Error: unknown failure\n";
    return 2;
  }
}

}  // namespace

#if defined(_WIN32)
int wmain(int argc, wchar_t** argv) {
  return probe_main(argc, argv);
}
#else
int main(int argc, char** argv) {
  return probe_main(argc, argv);
}
#endif
