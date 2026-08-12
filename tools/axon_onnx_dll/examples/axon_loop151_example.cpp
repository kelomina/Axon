#include "axon_onnx_predict.h"

#if !defined(_WIN32)
#error "The Loop151 DLL example requires Windows."
#endif

#include <windows.h>

#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

namespace {

using create_fn = kvd_handle* (KVD_CALL*)(const kvd_config*);
using destroy_fn = void (KVD_CALL*)(kvd_handle*);
using scan_path_fn = int (KVD_CALL*)(kvd_handle*, const char*, char**, size_t*);
using scan_bytes_fn = int (KVD_CALL*)(kvd_handle*, const unsigned char*, size_t, char**, size_t*);
using free_fn = void (KVD_CALL*)(char*);
using validate_fn = int (KVD_CALL*)(const kvd_config*, char**, size_t*);

std::string wide_to_utf8(const std::wstring& value) {
  if (value.empty()) {
    return {};
  }
  const int required_size = WideCharToMultiByte(
      CP_UTF8,
      WC_ERR_INVALID_CHARS,
      value.data(),
      static_cast<int>(value.size()),
      nullptr,
      0,
      nullptr,
      nullptr);
  if (required_size <= 0) {
    return {};
  }
  std::string result(static_cast<std::size_t>(required_size), '\0');
  const int written = WideCharToMultiByte(
      CP_UTF8,
      WC_ERR_INVALID_CHARS,
      value.data(),
      static_cast<int>(value.size()),
      result.data(),
      required_size,
      nullptr,
      nullptr);
  if (written != required_size) {
    return {};
  }
  return result;
}

std::wstring get_argument(int argc, wchar_t** argv, const std::wstring& key) {
  for (int index = 1; index + 1 < argc; ++index) {
    if (argv[index] == key) {
      return argv[index + 1];
    }
  }
  return {};
}

bool has_flag(int argc, wchar_t** argv, const std::wstring& flag) {
  for (int index = 1; index < argc; ++index) {
    if (argv[index] == flag) {
      return true;
    }
  }
  return false;
}

bool parse_u32(const std::wstring& value, unsigned int& result) {
  if (value.empty()) {
    return false;
  }
  wchar_t* end = nullptr;
  errno = 0;
  const unsigned long long parsed = std::wcstoull(value.c_str(), &end, 10);
  if (!end || *end != L'\0' || errno == ERANGE ||
      parsed > (std::numeric_limits<unsigned int>::max)()) {
    return false;
  }
  result = static_cast<unsigned int>(parsed);
  return true;
}

bool read_bytes(const std::wstring& path, std::vector<unsigned char>& bytes) {
  FILE* file = nullptr;
  if (_wfopen_s(&file, path.c_str(), L"rb") != 0 || !file) {
    return false;
  }
  bool success = false;
  do {
    if (_fseeki64(file, 0, SEEK_END) != 0) {
      break;
    }
    const long long file_size = _ftelli64(file);
    if (file_size < 0 || static_cast<unsigned long long>(file_size) >
                             std::numeric_limits<std::size_t>::max()) {
      break;
    }
    if (_fseeki64(file, 0, SEEK_SET) != 0) {
      break;
    }
    bytes.resize(static_cast<std::size_t>(file_size));
    const std::size_t read_size =
        bytes.empty() ? 0 : std::fread(bytes.data(), 1, bytes.size(), file);
    success = read_size == bytes.size();
  } while (false);
  std::fclose(file);
  return success;
}

int usage() {
  std::wcerr
      << L"Usage:\n"
      << L"  axon_loop151_example.exe --dll <axon_loop151_champion.dll>\n"
      << L"      --runtime-config <runtime/loop151_native_runtime.json> --target <sample.exe>\n"
      << L"      [--allowed-root <directory>] [--max-file-size <bytes>]\n";
  return 2;
}

template <typename Function>
Function load_function(HMODULE module, const char* name) {
  return reinterpret_cast<Function>(GetProcAddress(module, name));
}

int run(int argc, wchar_t** argv) {
  if (argc < 7 || has_flag(argc, argv, L"--help")) {
    return usage();
  }

  const std::wstring dll_path = get_argument(argc, argv, L"--dll");
  const std::wstring runtime_path = get_argument(argc, argv, L"--runtime-config");
  const std::wstring target_path = get_argument(argc, argv, L"--target");
  const std::wstring allowed_root_path = get_argument(argc, argv, L"--allowed-root");
  const std::wstring max_file_size_text = get_argument(argc, argv, L"--max-file-size");
  if (dll_path.empty() || runtime_path.empty() || target_path.empty()) {
    return usage();
  }

  const std::string runtime_utf8 = wide_to_utf8(runtime_path);
  const std::string target_utf8 = wide_to_utf8(target_path);
  const std::string allowed_root_utf8 = wide_to_utf8(allowed_root_path);
  if (runtime_utf8.empty() || target_utf8.empty() ||
      (!allowed_root_path.empty() && allowed_root_utf8.empty())) {
    std::wcerr << L"Path conversion to UTF-8 failed.\n";
    return 3;
  }

  HMODULE module = LoadLibraryW(dll_path.c_str());
  if (!module) {
    std::wcerr << L"LoadLibraryW failed for: " << dll_path << L"\n";
    return 4;
  }

  const create_fn create = load_function<create_fn>(module, "kvd_create");
  const destroy_fn destroy = load_function<destroy_fn>(module, "kvd_destroy");
  const scan_path_fn scan_path = load_function<scan_path_fn>(module, "kvd_scan_path");
  const scan_bytes_fn scan_bytes = load_function<scan_bytes_fn>(module, "kvd_scan_bytes");
  const free_fn free_string = load_function<free_fn>(module, "kvd_free");
  const validate_fn validate = load_function<validate_fn>(module, "kvd_validate_models");
  if (!create || !destroy || !scan_path || !scan_bytes || !free_string || !validate) {
    std::cerr << "The Loop151 DLL is missing a required KVD export.\n";
    FreeLibrary(module);
    return 5;
  }

  kvd_config config{};
  config.stage2_model_json_path = runtime_utf8.c_str();
  config.allowed_scan_root = allowed_root_path.empty() ? nullptr : allowed_root_utf8.c_str();
  config.prediction_threshold = 0.5f;
  unsigned int max_file_size = 0;
  if (!max_file_size_text.empty() && !parse_u32(max_file_size_text, max_file_size)) {
    std::wcerr << L"Invalid --max-file-size value.\n";
    FreeLibrary(module);
    return 6;
  }
  config.max_file_size = max_file_size;

  char* validation_json = nullptr;
  size_t validation_length = 0;
  const int validation_code = validate(&config, &validation_json, &validation_length);
  if (validation_json) {
    std::cout << "validate_rc=" << validation_code << " validate_json="
              << std::string(validation_json, validation_length) << "\n";
    free_string(validation_json);
  } else {
    std::cout << "validate_rc=" << validation_code << " validate_json=null\n";
  }
  if (validation_code != KVD_MODEL_OK) {
    FreeLibrary(module);
    return 7;
  }

  kvd_handle* handle = create(&config);
  if (!handle) {
    std::cerr << "kvd_create failed; verify that --runtime-config points to loop151_native_runtime.json.\n";
    FreeLibrary(module);
    return 8;
  }

  std::vector<unsigned char> sample_bytes;
  if (!read_bytes(target_path, sample_bytes)) {
    std::cerr << "Cannot read the sample for kvd_scan_bytes.\n";
    destroy(handle);
    FreeLibrary(module);
    return 9;
  }

  char* path_json = nullptr;
  size_t path_length = 0;
  const int path_code = scan_path(handle, target_utf8.c_str(), &path_json, &path_length);
  if (path_json) {
    std::cout << "path_result=" << std::string(path_json, path_length) << "\n";
    free_string(path_json);
  }

  char* bytes_json = nullptr;
  size_t bytes_length = 0;
  const int bytes_code = scan_bytes(
      handle,
      sample_bytes.data(),
      sample_bytes.size(),
      &bytes_json,
      &bytes_length);
  if (bytes_json) {
    std::cout << "bytes_result=" << std::string(bytes_json, bytes_length) << "\n";
    free_string(bytes_json);
  }

  destroy(handle);
  FreeLibrary(module);
  return path_code == 0 && bytes_code == 0 ? 0 : 10;
}

}  // namespace

int wmain(int argc, wchar_t** argv) {
  return run(argc, argv);
}
