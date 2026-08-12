#include "axon_onnx_predict.h"
#include "axon_loop151_native_model.h"

#if !defined(_WIN32)
#error "The Loop151 KVD bridge is Windows-only."
#endif

#include <windows.h>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

struct BridgeConfig {
  std::filesystem::path python_executable;
  std::filesystem::path runner_script;
  std::shared_ptr<axon_loop151_native::NativeStackModel> native_model;
  bool native_mode = false;
  std::filesystem::path allowed_root;
  unsigned int max_file_size = 0;
};

struct kvd_handle {
  BridgeConfig config;
};

namespace {

std::string json_escape(const std::string& value) {
  std::string out;
  for (unsigned char character : value) {
    switch (character) {
      case '\\': out += "\\\\"; break;
      case '"': out += "\\\""; break;
      case '\n': out += "\\n"; break;
      case '\r': out += "\\r"; break;
      case '\t': out += "\\t"; break;
      default:
        if (character < 0x20) {
          char encoded[7];
          std::snprintf(encoded, sizeof(encoded), "\\u%04x", character);
          out += encoded;
        } else {
          out += static_cast<char>(character);
        }
    }
  }
  return out;
}

int write_string(const std::string& value, char** out_json, size_t* out_len) {
  if (!out_json || !out_len) {
    return -1;
  }
  *out_json = nullptr;
  *out_len = 0;
  char* buffer = static_cast<char*>(std::malloc(value.size() + 1));
  if (!buffer) {
    return -100;
  }
  std::memcpy(buffer, value.data(), value.size());
  buffer[value.size()] = '\0';
  *out_json = buffer;
  *out_len = value.size();
  return 0;
}

int write_error(const char* code, const char* message, char** out_json, size_t* out_len) {
  return write_string(
      std::string("{\"ok\":false,\"error_code\":\"") + json_escape(code) +
          "\",\"error\":\"" + json_escape(message) + "\"}",
      out_json,
      out_len);
}

std::optional<std::string> json_string(const std::string& source, const std::string& key) {
  const std::string marker = "\"" + key + "\"";
  std::size_t position = source.find(marker);
  if (position == std::string::npos) {
    return std::nullopt;
  }
  position = source.find(':', position + marker.size());
  if (position == std::string::npos) {
    return std::nullopt;
  }
  position = source.find('"', position + 1);
  if (position == std::string::npos) {
    return std::nullopt;
  }
  ++position;
  std::string result;
  bool escaped = false;
  for (; position < source.size(); ++position) {
    char character = source[position];
    if (escaped) {
      if (character == 'n') result += '\n';
      else if (character == 'r') result += '\r';
      else if (character == 't') result += '\t';
      else result += character;
      escaped = false;
    } else if (character == '\\') {
      escaped = true;
    } else if (character == '"') {
      return result;
    } else {
      result += character;
    }
  }
  return std::nullopt;
}

std::optional<double> json_number(const std::string& source, const std::string& key) {
  const std::string marker = "\"" + key + "\"";
  std::size_t position = source.find(marker);
  if (position == std::string::npos) return std::nullopt;
  position = source.find(':', position + marker.size());
  if (position == std::string::npos) return std::nullopt;
  const char* begin = source.c_str() + position + 1;
  char* end = nullptr;
  double value = std::strtod(begin, &end);
  if (end == begin || !std::isfinite(value)) return std::nullopt;
  return value;
}

std::filesystem::path resolve_relative(const std::filesystem::path& base, const std::string& value) {
  std::filesystem::path path = std::filesystem::u8path(value);
  return path.is_absolute() ? path : (base / path).lexically_normal();
}

bool is_regular_input_file(const std::filesystem::path& path) {
  std::error_code error;
  return std::filesystem::is_regular_file(path, error) && !error;
}

bool path_allowed(const std::filesystem::path& path, const std::filesystem::path& root) {
  if (root.empty()) return true;
  std::error_code error;
  const auto canonical_path = std::filesystem::weakly_canonical(path, error);
  if (error) return false;
  const auto canonical_root = std::filesystem::weakly_canonical(root, error);
  if (error) return false;
  auto path_it = canonical_path.begin();
  for (auto root_it = canonical_root.begin(); root_it != canonical_root.end(); ++root_it, ++path_it) {
    if (path_it == canonical_path.end() || *path_it != *root_it) return false;
  }
  return true;
}

std::wstring quote_command_arg(const std::wstring& value) {
  std::wstring out = L"\"";
  for (wchar_t character : value) {
    if (character == L'"') out += L"\\\"";
    else out += character;
  }
  out += L"\"";
  return out;
}

bool run_python(const BridgeConfig& config, const std::filesystem::path& target, std::string& output, std::string& error) {
  SECURITY_ATTRIBUTES attributes{};
  attributes.nLength = sizeof(attributes);
  attributes.bInheritHandle = TRUE;
  HANDLE read_pipe = nullptr;
  HANDLE write_pipe = nullptr;
  if (!CreatePipe(&read_pipe, &write_pipe, &attributes, 0)) {
    error = "Cannot create runtime output pipe.";
    return false;
  }
  SetHandleInformation(read_pipe, HANDLE_FLAG_INHERIT, 0);
  std::wstring command = quote_command_arg(config.python_executable.wstring()) + L" " +
      quote_command_arg(config.runner_script.wstring()) + L" --file " + quote_command_arg(target.wstring()) + L" --device cpu";
  std::vector<wchar_t> command_buffer(command.begin(), command.end());
  command_buffer.push_back(L'\0');
  STARTUPINFOW startup{};
  startup.cb = sizeof(startup);
  startup.dwFlags = STARTF_USESHOWWINDOW | STARTF_USESTDHANDLES;
  startup.wShowWindow = SW_HIDE;
  startup.hStdOutput = write_pipe;
  startup.hStdError = write_pipe;
  startup.hStdInput = GetStdHandle(STD_INPUT_HANDLE);
  PROCESS_INFORMATION process{};
  const BOOL created = CreateProcessW(
      nullptr, command_buffer.data(), nullptr, nullptr, TRUE, CREATE_NO_WINDOW, nullptr, nullptr, &startup, &process);
  CloseHandle(write_pipe);
  if (!created) {
    CloseHandle(read_pipe);
    error = "Cannot start the Loop151 Python runtime.";
    return false;
  }
  std::string captured;
  char buffer[4096];
  DWORD read = 0;
  while (ReadFile(read_pipe, buffer, sizeof(buffer), &read, nullptr) && read > 0) {
    captured.append(buffer, read);
    if (captured.size() > 1024 * 1024) {
      TerminateProcess(process.hProcess, 1);
      error = "Loop151 runtime output exceeded the safety limit.";
      break;
    }
  }
  CloseHandle(read_pipe);
  const DWORD wait = WaitForSingleObject(process.hProcess, 120000);
  DWORD exit_code = 1;
  GetExitCodeProcess(process.hProcess, &exit_code);
  CloseHandle(process.hThread);
  CloseHandle(process.hProcess);
  if (wait != WAIT_OBJECT_0 || exit_code != 0) {
    error = captured.empty() ? "Loop151 runtime did not complete successfully." : captured;
    return false;
  }
  const std::size_t json_start = captured.rfind('{');
  if (json_start == std::string::npos) {
    error = "Loop151 runtime did not return a JSON object.";
    return false;
  }
  output = captured.substr(json_start);
  return true;
}

bool load_bridge_config(const char* path_text, const kvd_config* config, BridgeConfig& out) {
  if (!path_text || path_text[0] == '\0') return false;
  const std::filesystem::path config_path = std::filesystem::u8path(path_text);
  std::ifstream stream(config_path, std::ios::binary);
  if (!stream) return false;
  const std::string document((std::istreambuf_iterator<char>(stream)), std::istreambuf_iterator<char>());
  const auto schema = json_string(document, "schema");
  if (schema && *schema == "axon_loop151_native_bundle_v1") {
    const auto model_path = json_string(document, "model_path");
    const std::filesystem::path native_path = model_path
        ? resolve_relative(config_path.parent_path(), *model_path)
        : config_path;
    std::string native_error;
    auto native_model = axon_loop151_native::NativeStackModel::load_file(native_path.u8string(), native_error);
    if (!native_model) return false;
    out.native_model = std::shared_ptr<axon_loop151_native::NativeStackModel>(std::move(native_model));
    out.native_mode = true;
  } else {
    if (!schema || *schema != "axon_loop151_kvd_bridge_v1") return false;
    const auto python = json_string(document, "python_executable");
    const auto runner = json_string(document, "runner_script");
    if (!python || !runner) return false;
    out.python_executable = resolve_relative(config_path.parent_path(), *python);
    out.runner_script = resolve_relative(config_path.parent_path(), *runner);
    if (!is_regular_input_file(out.python_executable) || !is_regular_input_file(out.runner_script)) return false;
  }
  if (config && config->allowed_scan_root && config->allowed_scan_root[0] != '\0') {
    out.allowed_root = std::filesystem::u8path(config->allowed_scan_root);
  }
  out.max_file_size = config ? config->max_file_size : 0;
  return true;
}

int scan_path(const BridgeConfig& config, const std::filesystem::path& path, char** out_json, size_t* out_len, bool enforce_root) {
  if (!is_regular_input_file(path)) return write_error("file_read_failed", "Input path is not a readable file.", out_json, out_len);
  if (enforce_root && !path_allowed(path, config.allowed_root)) {
    return write_error("path_not_allowed", "Input path is outside allowed_scan_root.", out_json, out_len);
  }
  std::error_code error;
  const auto size = std::filesystem::file_size(path, error);
  if (error) return write_error("file_read_failed", "Cannot determine input size.", out_json, out_len);
  if (config.max_file_size > 0 && size > config.max_file_size) {
    return write_error("file_too_large", "Input file exceeds max_file_size.", out_json, out_len);
  }
  if (config.native_mode) {
    return write_error(
        "loop151_native_feature_adapter_missing",
        "The native Loop151 model is loaded, but the raw feature adapter is not connected yet.",
        out_json,
        out_len);
  }
  std::string runner_json;
  std::string runner_error;
  if (!run_python(config, path, runner_json, runner_error)) {
    return write_error("loop151_runtime_failed", runner_error.c_str(), out_json, out_len);
  }
  const auto prediction = json_number(runner_json, "prediction");
  const auto probability = json_number(runner_json, "probability");
  if (!prediction || !probability || (*prediction != 0.0 && *prediction != 1.0)) {
    return write_error("loop151_runtime_invalid", "Loop151 runtime returned an invalid prediction.", out_json, out_len);
  }
  const bool malware = *prediction > 0.5;
  const double confidence = malware ? *probability : 1.0 - *probability;
  std::ostringstream response;
  response << "{\"ok\":true,\"mode\":\"single_pe\",\"loop_id\":\"Loop151\",\"file\":\""
           << json_escape(path.u8string()) << "\",\"is_malware\":" << (malware ? "true" : "false")
           << ",\"prediction\":" << (malware ? 1 : 0) << ",\"confidence\":" << confidence
           << ",\"loop151\":" << runner_json << "}";
  return write_string(response.str(), out_json, out_len);
}

std::filesystem::path bytes_temp_path() {
  wchar_t directory[MAX_PATH]{};
  wchar_t filename[MAX_PATH]{};
  GetTempPathW(MAX_PATH, directory);
  GetTempFileNameW(directory, L"ax1", 0, filename);
  return filename;
}

}  // namespace

extern "C" {

KVD_API kvd_handle* KVD_CALL kvd_create(const kvd_config* config) {
  if (!config) return nullptr;
  auto handle = std::make_unique<kvd_handle>();
  if (!load_bridge_config(config->stage2_model_json_path, config, handle->config)) return nullptr;
  return handle.release();
}

KVD_API void KVD_CALL kvd_destroy(kvd_handle* handle) { delete handle; }

KVD_API int KVD_CALL kvd_scan_path(kvd_handle* handle, const char* path, char** out_json, size_t* out_len) {
  if (!handle || !path) return write_error("invalid_argument", "handle and path are required.", out_json, out_len);
  return scan_path(handle->config, std::filesystem::u8path(path), out_json, out_len, true);
}

KVD_API int KVD_CALL kvd_scan_bytes(kvd_handle* handle, const unsigned char* bytes, size_t len, char** out_json, size_t* out_len) {
  if (!handle || (!bytes && len > 0)) return write_error("invalid_argument", "handle and bytes are required.", out_json, out_len);
  if (handle->config.max_file_size > 0 && len > handle->config.max_file_size) {
    return write_error("file_too_large", "Input byte buffer exceeds max_file_size.", out_json, out_len);
  }
  const auto temporary = bytes_temp_path();
  std::ofstream stream(temporary, std::ios::binary | std::ios::trunc);
  if (!stream) return write_error("file_write_failed", "Cannot create private byte scan file.", out_json, out_len);
  if (len > 0) stream.write(reinterpret_cast<const char*>(bytes), static_cast<std::streamsize>(len));
  stream.close();
  const int result = scan_path(handle->config, temporary, out_json, out_len, false);
  std::error_code error;
  std::filesystem::remove(temporary, error);
  return result;
}

KVD_API int KVD_CALL kvd_scan_paths(kvd_handle* handle, const char** paths, size_t count, char** out_json, size_t* out_len) {
  if (!handle || (!paths && count > 0)) return write_error("invalid_argument", "handle and paths are required.", out_json, out_len);
  std::ostringstream response;
  response << "[";
  for (size_t index = 0; index < count; ++index) {
    if (index) response << ",";
    char* item = nullptr;
    size_t item_len = 0;
    const int code = kvd_scan_path(handle, paths[index], &item, &item_len);
    if (code == 0 && item) response.write(item, static_cast<std::streamsize>(item_len));
    else response << "{\"ok\":false,\"error_code\":\"scan_failed\"}";
    kvd_free(item);
  }
  response << "]";
  return write_string(response.str(), out_json, out_len);
}

KVD_API int KVD_CALL kvd_parity_diagnostics_path_v1(kvd_handle*, const char*, const kvd_parity_diagnostics_options_v1*, char** out_json, size_t* out_len) {
  return write_error("unsupported_operation", "Loop151 bridge does not expose Loop28 parity diagnostics.", out_json, out_len);
}

KVD_API int KVD_CALL kvd_train_path(kvd_handle*, const char*, int, char** out_json, size_t* out_len) { return write_error("unsupported_operation", "Loop151 DLL is inference-only.", out_json, out_len); }
KVD_API int KVD_CALL kvd_train_paths(kvd_handle*, const char**, size_t, int, char** out_json, size_t* out_len) { return write_error("unsupported_operation", "Loop151 DLL is inference-only.", out_json, out_len); }
KVD_API int KVD_CALL kvd_train_from_path(kvd_handle* handle, const char* path, int label, char** out_json, size_t* out_len) { return kvd_train_path(handle, path, label, out_json, out_len); }
KVD_API void KVD_CALL kvd_signature_flush(kvd_handle*) {}
KVD_API void KVD_CALL kvd_free(char* pointer) { std::free(pointer); }

KVD_API int KVD_CALL kvd_validate_models(const kvd_config* config, char** out_error, size_t* out_len) {
  BridgeConfig bridge;
  if (!config || !load_bridge_config(config->stage2_model_json_path, config, bridge)) {
    if (out_error && out_len) write_string("loop151_bridge_config_invalid", out_error, out_len);
    return KVD_MODEL_ERR_INVALID_ARGUMENT;
  }
  if (out_error && out_len) write_string("ok", out_error, out_len);
  return KVD_MODEL_OK;
}

KVD_API int KVD_CALL kvd_extract_pe_features(const char*, float*, size_t) { return -3; }
KVD_API int KVD_CALL kvd_extract_pe_features_batch(const char**, size_t, float*, size_t, int*, unsigned int) { return -3; }
KVD_API size_t KVD_CALL kvd_get_pe_feature_dimension(void) { return 256; }

KVD_API char* KVD_CALL axon_predict_json(const char*) {
  char* output = nullptr;
  size_t length = 0;
  write_error("unsupported_operation", "Use kvd_create and kvd_scan_path or kvd_scan_bytes.", &output, &length);
  return output;
}
KVD_API void KVD_CALL axon_string_free(char* pointer) { kvd_free(pointer); }
KVD_API const char* KVD_CALL axon_version(void) { return "axon-loop151-kvd-bridge-v1"; }

}  // extern "C"
