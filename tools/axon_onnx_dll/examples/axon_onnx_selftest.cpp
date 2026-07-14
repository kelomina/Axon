#include "axon_onnx_predict.h"

#if defined(_WIN32)
#include <windows.h>
#endif

#include <algorithm>
#include <array>
#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

static std::string get_arg_value(int argc, char** argv, const std::string& key) {
  for (int i = 1; i + 1 < argc; ++i) {
    if (argv[i] == key) {
      return std::string(argv[i + 1]);
    }
  }
  return {};
}

static bool has_flag(int argc, char** argv, const std::string& flag) {
  for (int i = 1; i < argc; ++i) {
    if (argv[i] == flag) {
      return true;
    }
  }
  return false;
}

static int usage() {
  std::cerr
      << "Usage:\n"
      << "  axon_onnx_selftest --dll <axon_onnx_predict.dll> --onnx <model.onnx> --target <file>\n"
      << "                    [--threshold <0..1>] [--allowed_root <dir>] [--max_file_size <bytes>]\n"
      << "                    [--family <family_classifier.json>] [--stage2 <loop28_stage2_hgb.json>]\n"
      << "                    [--archive_scanner <axon-archive-scanner.exe>] [--nested]\n"
      << "                    [--parity_diagnostics]\n"
      << "                    [--diagnostic_component <component> --block_elements <1..256>]\n"
      << "                    (omit drilldown options for whole-digest-only diagnostics)\n"
      << "                    [--expect_scan_error <error_code>]\n";
  return 2;
}

static bool parse_u32(const std::string& s, unsigned int& out) {
  if (s.empty()) {
    return false;
  }
  char* end = nullptr;
  errno = 0;
  unsigned long long v = std::strtoull(s.c_str(), &end, 10);
  if (!end || *end != '\0' || errno == ERANGE ||
      v > (std::numeric_limits<unsigned int>::max)()) {
    return false;
  }
  out = static_cast<unsigned int>(v);
  return true;
}

static bool parse_f32(const std::string& s, float& out) {
  if (s.empty()) {
    return false;
  }
  char* end = nullptr;
  float v = std::strtof(s.c_str(), &end);
  if (!end || *end != '\0') {
    return false;
  }
  out = v;
  return true;
}

static bool diagnostic_component_bit(const std::string& name, std::uint64_t& out) {
  if (name == "byte_seq") {
    out = KVD_PARITY_DIAGNOSTICS_COMPONENT_BYTE_SEQ_V1;
  } else if (name == "pe_features") {
    out = KVD_PARITY_DIAGNOSTICS_COMPONENT_PE_FEATURES_V1;
  } else if (name == "stat_features") {
    out = KVD_PARITY_DIAGNOSTICS_COMPONENT_STAT_FEATURES_V1;
  } else if (name == "base_logits") {
    out = KVD_PARITY_DIAGNOSTICS_COMPONENT_BASE_LOGITS_V1;
  } else if (name == "base_probabilities") {
    out = KVD_PARITY_DIAGNOSTICS_COMPONENT_BASE_PROBABILITIES_V1;
  } else if (name == "stage2_features") {
    out = KVD_PARITY_DIAGNOSTICS_COMPONENT_STAGE2_FEATURES_V1;
  } else {
    return false;
  }
  return true;
}

static int hex_nibble(char value) {
  if (value >= '0' && value <= '9') {
    return value - '0';
  }
  if (value >= 'a' && value <= 'f') {
    return value - 'a' + 10;
  }
  if (value >= 'A' && value <= 'F') {
    return value - 'A' + 10;
  }
  return -1;
}

struct SensitiveHmacKey {
  std::array<unsigned char, 32> bytes{};

  ~SensitiveHmacKey() {
#if defined(_WIN32)
    SecureZeroMemory(bytes.data(), bytes.size());
#else
    std::fill(bytes.begin(), bytes.end(), 0);
#endif
  }
};

static bool read_hmac_key_hex(SensitiveHmacKey& key) {
#if defined(_WIN32)
  // Disable interactive echo so the key never enters argv or output.
  HANDLE input_handle = GetStdHandle(STD_INPUT_HANDLE);
  DWORD original_mode = 0;
  bool restore_mode =
      input_handle != INVALID_HANDLE_VALUE && input_handle != nullptr &&
      GetConsoleMode(input_handle, &original_mode) != 0;
  if (restore_mode &&
      SetConsoleMode(input_handle, original_mode & ~ENABLE_ECHO_INPUT) == 0) {
    return false;
  }
#endif

  std::string hex;
  bool read_ok = static_cast<bool>(std::getline(std::cin, hex));

#if defined(_WIN32)
  bool restore_ok = !restore_mode || SetConsoleMode(input_handle, original_mode) != 0;
#endif

  if (!hex.empty() && hex.back() == '\r') {
    hex.pop_back();
  }
  bool valid = read_ok && hex.size() == key.bytes.size() * 2;
  if (valid) {
    for (std::size_t index = 0; index < key.bytes.size(); ++index) {
      int high = hex_nibble(hex[index * 2]);
      int low = hex_nibble(hex[index * 2 + 1]);
      if (high < 0 || low < 0) {
        valid = false;
        break;
      }
      key.bytes[index] = static_cast<unsigned char>((high << 4) | low);
    }
  }
#if defined(_WIN32)
  if (!hex.empty()) {
    SecureZeroMemory(hex.data(), hex.size());
  }
#else
  std::fill(hex.begin(), hex.end(), '\0');
#endif
#if defined(_WIN32)
  return valid && restore_ok;
#else
  return valid;
#endif
}

static int run_main(int argc, char** argv) {
  if (argc < 7 || has_flag(argc, argv, "--help")) {
    return usage();
  }

  std::string dll_path = get_arg_value(argc, argv, "--dll");
  std::string onnx_path = get_arg_value(argc, argv, "--onnx");
  std::string target = get_arg_value(argc, argv, "--target");
  std::string threshold_s = get_arg_value(argc, argv, "--threshold");
  std::string family_path = get_arg_value(argc, argv, "--family");
  std::string stage2_path = get_arg_value(argc, argv, "--stage2");
  std::string archive_scanner_path = get_arg_value(argc, argv, "--archive_scanner");
  std::string allowed_root = get_arg_value(argc, argv, "--allowed_root");
  std::string max_file_size_s = get_arg_value(argc, argv, "--max_file_size");
  std::string expected_scan_error = get_arg_value(argc, argv, "--expect_scan_error");
  bool parity_diagnostics = has_flag(argc, argv, "--parity_diagnostics");
  std::string diagnostic_component = get_arg_value(argc, argv, "--diagnostic_component");
  std::string block_elements_s = get_arg_value(argc, argv, "--block_elements");
  if (dll_path.empty() || onnx_path.empty() || target.empty()) {
    return usage();
  }
  std::uint64_t drilldown_component = 0;
  unsigned int block_elements = 0;
  if (parity_diagnostics) {
    const bool has_component = !diagnostic_component.empty();
    const bool has_block_elements = !block_elements_s.empty();
    if (has_component != has_block_elements ||
        (has_component &&
         (!diagnostic_component_bit(diagnostic_component, drilldown_component) ||
          !parse_u32(block_elements_s, block_elements) ||
          block_elements < 1 || block_elements > 256))) {
      return usage();
    }
  }

#if !defined(_WIN32)
  std::cerr << "This example is Windows-only because it uses LoadLibrary/GetProcAddress.\n";
  return 1;
#else
  HMODULE mod = LoadLibraryA(dll_path.c_str());
  if (!mod) {
    std::cerr << "LoadLibrary failed: " << dll_path << "\n";
    return 1;
  }

  auto get = [&](const char* name) -> FARPROC {
    FARPROC p = GetProcAddress(mod, name);
    if (!p) {
      std::cerr << "GetProcAddress failed: " << name << "\n";
    }
    return p;
  };

  using create_fn = kvd_handle* (KVD_CALL*)(const kvd_config*);
  using destroy_fn = void (KVD_CALL*)(kvd_handle*);
  using scan_path_fn = int (KVD_CALL*)(kvd_handle*, const char*, char**, size_t*);
  using parity_diagnostics_path_fn = int (KVD_CALL*)(
      kvd_handle*,
      const char*,
      const kvd_parity_diagnostics_options_v1*,
      char**,
      size_t*);
  using free_fn = void (KVD_CALL*)(char*);
  using validate_fn = int (KVD_CALL*)(const kvd_config*, char**, size_t*);

  auto create_p = reinterpret_cast<create_fn>(get("kvd_create"));
  auto destroy_p = reinterpret_cast<destroy_fn>(get("kvd_destroy"));
  auto scan_path_p = reinterpret_cast<scan_path_fn>(get("kvd_scan_path"));
  parity_diagnostics_path_fn parity_diagnostics_path_p = nullptr;
  if (parity_diagnostics) {
    parity_diagnostics_path_p = reinterpret_cast<parity_diagnostics_path_fn>(
        get("kvd_parity_diagnostics_path_v1"));
  }
  auto free_p = reinterpret_cast<free_fn>(get("kvd_free"));
  auto validate_p = reinterpret_cast<validate_fn>(get("kvd_validate_models"));
  if (!create_p || !destroy_p || !scan_path_p || !free_p || !validate_p ||
      (parity_diagnostics && !parity_diagnostics_path_p)) {
    return 1;
  }

  kvd_config cfg{};
  cfg.onnx_model_path = onnx_path.c_str();
  if (!family_path.empty()) {
    cfg.family_classifier_json_path = family_path.c_str();
  }
  if (!stage2_path.empty()) {
    cfg.stage2_model_json_path = stage2_path.c_str();
  }
  if (!archive_scanner_path.empty()) {
    cfg.archive_scanner_path = archive_scanner_path.c_str();
  }
  cfg.scan_nested = has_flag(argc, argv, "--nested") ? 1 : 0;
  if (!allowed_root.empty()) {
    cfg.allowed_scan_root = allowed_root.c_str();
  }
  unsigned int max_file_size = 0;
  if (parse_u32(max_file_size_s, max_file_size)) {
    cfg.max_file_size = max_file_size;
  }
  float threshold = 0.0f;
  if (parse_f32(threshold_s, threshold)) {
    cfg.prediction_threshold = threshold;
  }

  char* validate_msg = nullptr;
  size_t validate_len = 0;
  int validate_rc = validate_p(&cfg, &validate_msg, &validate_len);
  std::cout << "{\"validate_rc\":" << validate_rc << ",\"validate_msg\":\"";
  if (validate_msg) {
    std::cout.write(validate_msg, static_cast<std::streamsize>(validate_len));
    free_p(validate_msg);
  }
  std::cout << "\"}\n";
  if (validate_rc != 0) {
    return 2;
  }

  kvd_handle* handle = create_p(&cfg);
  if (!handle) {
    std::cerr << "kvd_create failed\n";
    return 3;
  }

  char* out_json = nullptr;
  size_t out_len = 0;
  int rc = 0;
  if (parity_diagnostics) {
    SensitiveHmacKey hmac_key;
    if (!read_hmac_key_hex(hmac_key)) {
      std::cerr << "Parity diagnostics require one stdin line containing exactly 64 hex characters.\n";
      destroy_p(handle);
      return 4;
    }
    kvd_parity_diagnostics_options_v1 options{};
    options.struct_size = sizeof(options);
    options.abi_version = KVD_PARITY_DIAGNOSTICS_ABI_VERSION_V1;
    options.component_mask =
        drilldown_component == 0 ? KVD_PARITY_DIAGNOSTICS_COMPONENT_ALL_V1 : drilldown_component;
    options.drilldown_component = drilldown_component;
    options.block_elements = block_elements;
    options.hmac_key = hmac_key.bytes.data();
    options.hmac_key_len = hmac_key.bytes.size();
    rc = parity_diagnostics_path_p(handle, target.c_str(), &options, &out_json, &out_len);
  } else {
    rc = scan_path_p(handle, target.c_str(), &out_json, &out_len);
  }
  std::string scan_json;
  if (out_json) {
    scan_json.assign(out_json, out_len);
    std::cout.write(out_json, static_cast<std::streamsize>(out_len));
    std::cout << "\n";
    free_p(out_json);
  }
  if (!expected_scan_error.empty() &&
      scan_json.find("\"error_code\":\"" + expected_scan_error + "\"") != std::string::npos) {
    destroy_p(handle);
    return 0;
  }
  if (rc != 0) {
    std::cerr << (parity_diagnostics ? "kvd_parity_diagnostics_path_v1" : "kvd_scan_path")
              << " failed: " << rc << "\n";
    destroy_p(handle);
    return 4;
  }
  if (!expected_scan_error.empty()) {
    std::cerr << "Expected kvd_scan_path error: " << expected_scan_error << "\n";
    destroy_p(handle);
    return 5;
  }
  destroy_p(handle);
  return 0;
#endif
}

#if defined(_WIN32)
static std::string wide_to_utf8(const wchar_t* value) {
  if (!value) {
    return {};
  }
  int required_size = WideCharToMultiByte(CP_UTF8, 0, value, -1, nullptr, 0, nullptr, nullptr);
  if (required_size <= 1) {
    return {};
  }
  std::string utf8(static_cast<size_t>(required_size), '\0');
  WideCharToMultiByte(CP_UTF8, 0, value, -1, utf8.data(), required_size, nullptr, nullptr);
  utf8.pop_back();
  return utf8;
}

int wmain(int argc, wchar_t** wide_argv) {
  // Preserve non-ASCII Windows paths by converting UTF-16 arguments to the DLL's UTF-8 ABI.
  std::vector<std::string> utf8_arguments;
  utf8_arguments.reserve(static_cast<size_t>(argc));
  for (int index = 0; index < argc; ++index) {
    utf8_arguments.push_back(wide_to_utf8(wide_argv[index]));
  }
  std::vector<char*> argv;
  argv.reserve(utf8_arguments.size());
  for (std::string& argument : utf8_arguments) {
    argv.push_back(argument.data());
  }
  return run_main(argc, argv.data());
}
#else
int main(int argc, char** argv) {
  return run_main(argc, argv);
}
#endif
