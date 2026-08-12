#define kvd_create axon_loop151_legacy_kvd_create
#define kvd_destroy axon_loop151_legacy_kvd_destroy
#define kvd_scan_path axon_loop151_legacy_kvd_scan_path
#define kvd_scan_bytes axon_loop151_legacy_kvd_scan_bytes
#define kvd_scan_paths axon_loop151_legacy_kvd_scan_paths
#define kvd_parity_diagnostics_path_v1 axon_loop151_legacy_kvd_parity_diagnostics_path_v1
#define kvd_train_path axon_loop151_legacy_kvd_train_path
#define kvd_train_paths axon_loop151_legacy_kvd_train_paths
#define kvd_train_from_path axon_loop151_legacy_kvd_train_from_path
#define kvd_signature_flush axon_loop151_legacy_kvd_signature_flush
#define kvd_free axon_loop151_legacy_kvd_free
#define kvd_validate_models axon_loop151_legacy_kvd_validate_models
#define kvd_extract_pe_features axon_loop151_legacy_kvd_extract_pe_features
#define kvd_extract_pe_features_batch axon_loop151_legacy_kvd_extract_pe_features_batch
#define kvd_get_pe_feature_dimension axon_loop151_legacy_kvd_get_pe_feature_dimension
#define axon_predict_json axon_loop151_legacy_axon_predict_json
#define axon_string_free axon_loop151_legacy_axon_string_free
#define axon_version axon_loop151_legacy_axon_version
#define KVD_NO_EXPORTS 1
#include "axon_onnx_predict.cpp"
#undef KVD_NO_EXPORTS
#undef kvd_create
#undef kvd_destroy
#undef kvd_scan_path
#undef kvd_scan_bytes
#undef kvd_scan_paths
#undef kvd_parity_diagnostics_path_v1
#undef kvd_train_path
#undef kvd_train_paths
#undef kvd_train_from_path
#undef kvd_signature_flush
#undef kvd_free
#undef kvd_validate_models
#undef kvd_extract_pe_features
#undef kvd_extract_pe_features_batch
#undef kvd_get_pe_feature_dimension
#undef axon_predict_json
#undef axon_string_free
#undef axon_version

#include "axon_loop151_content_features.h"
#include "axon_loop151_native_model.h"

#if defined(_WIN32)
#include <windows.h>
#include <wincrypt.h>
#include <softpub.h>
#include <wintrust.h>
#endif

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

namespace {

using axon_loop151_native::NativeScoreModel;
using axon_loop151_native::NativeStackModel;

// Per-scan stage timings. A steady-state scan costs ~820 ms and almost none of
// that is file I/O, so the breakdown decides where optimisation is worth doing:
// the ONNX byte model scales with max_byte_length, the stage-2 stacks scale with
// tree count, and the Authenticode check is an OS call that does neither.
//
// Emitting the breakdown is opt-in through the AXON_LOOP151_TIMING environment
// variable so the ABI, the config struct and every existing caller stay
// untouched. Measuring is always on: steady_clock::now() costs tens of
// nanoseconds against stages measured in milliseconds.
struct StageTiming {
  double input = 0.0;
  double base_onnx = 0.0;
  double stage2_features = 0.0;
  double primary = 0.0;
  double conservative = 0.0;
  double content_features = 0.0;
  double content_pe_v1 = 0.0;
  double content_pe_v2 = 0.0;
  double content_strings = 0.0;
  double content_cross = 0.0;
  double noise = 0.0;
  double selector = 0.0;
  double signer = 0.0;
  double total = 0.0;
};

using TimingClock = std::chrono::steady_clock;

double elapsed_ms(const TimingClock::time_point& start) {
  return std::chrono::duration<double, std::milli>(TimingClock::now() - start).count();
}

bool env_flag(const char* name) {
#if defined(_WIN32)
  char* value = nullptr;
  std::size_t length = 0;
  if (_dupenv_s(&value, &length, name) != 0 || !value) return false;
  const bool on = length > 0 && value[0] != '0';
  std::free(value);
  return on;
#else
  const char* value = std::getenv(name);
  return value && value[0] != '0';
#endif
}

bool timing_enabled() {
  static const bool enabled = env_flag("AXON_LOOP151_TIMING");
  return enabled;
}

// Emits a second, ONNX-free decision alongside the real one so both can be
// scored against ground truth from a single scan. Off by default: production
// output stays byte-identical.
bool no_onnx_shadow_enabled() {
  static const bool enabled = env_flag("AXON_LOOP151_NO_ONNX_SHADOW");
  return enabled;
}

// One Loop151 decision outcome, produced by the shared rule/selector path.
struct Loop151Decision {
  int loop130 = 0;
  int loop136 = 0;
  float selected_probability = 0.0f;
  float selector_score = 0.0f;
  bool selector_used = false;
};

void append_timing_json(std::ostringstream& output, const StageTiming& timing) {
  output << ",\"timing_ms\":{"
         << "\"input_features\":" << timing.input
         << ",\"base_onnx\":" << timing.base_onnx
         << ",\"stage2_features\":" << timing.stage2_features
         << ",\"primary_stack\":" << timing.primary
         << ",\"conservative_stack\":" << timing.conservative
         << ",\"content_features\":" << timing.content_features
         << ",\"content_pe_v1\":" << timing.content_pe_v1
         << ",\"content_pe_v2\":" << timing.content_pe_v2
         << ",\"content_strings\":" << timing.content_strings
         << ",\"content_cross\":" << timing.content_cross
         << ",\"noise_stack\":" << timing.noise
         << ",\"selector\":" << timing.selector
         << ",\"authenticode\":" << timing.signer
         << ",\"total\":" << timing.total
         << '}';
}

constexpr std::size_t kLoop151ContentPeV2Dim = axon_loop151_native::kContentPeV2FeatureDim;
constexpr std::size_t kLoop151ContentStringDim = axon_loop151_native::kContentStringFeatureDim;
constexpr float kLoop151PrimaryThreshold = 0.31f;
constexpr float kLoop151ConservativeThreshold = 0.415f;
constexpr float kLoop151ContentCrossThreshold = 0.4f;
constexpr float kLoop151NoiseThreshold = 0.39f;
constexpr float kLoop151SelectorThreshold = 0.79f;

struct Loop151RuntimeConfig {
  std::filesystem::path config_path;
  std::filesystem::path base_onnx_path;
  std::filesystem::path primary_path;
  std::filesystem::path conservative_path;
  std::filesystem::path content_cross_path;
  std::filesystem::path noise_path;
  std::filesystem::path selector_path;
  bool base_onnx_enabled = true;
};

struct Loop151Handle {
  AxonConfig config;
  bool base_onnx_enabled = true;
  std::shared_ptr<AxonOnnxModel> base_model;
  std::unique_ptr<NativeStackModel> primary;
  std::unique_ptr<NativeStackModel> conservative;
  std::unique_ptr<NativeScoreModel> content_cross;
  std::unique_ptr<NativeStackModel> noise;
  std::unique_ptr<NativeScoreModel> selector;
};

struct Loop151StageScores {
  float primary = 0.0f;
  float conservative = 0.0f;
  float content_cross = 0.0f;
  float noise = 0.0f;
};

std::filesystem::path resolve_loop151_path(
    const std::filesystem::path& base,
    const std::string& value) {
  const auto path = path_from_utf8(value);
  return path.is_absolute() ? path.lexically_normal() : (base / path).lexically_normal();
}

bool regular_file(const std::filesystem::path& path) {
  std::error_code error;
  return std::filesystem::is_regular_file(path, error) && !error;
}

bool read_text_file(const std::filesystem::path& path, std::string& output) {
  std::ifstream input(path, std::ios::binary);
  if (!input) return false;
  std::ostringstream buffer;
  buffer << input.rdbuf();
  output = buffer.str();
  return true;
}

bool load_loop151_runtime_config(
    const kvd_config* api_config,
    Loop151RuntimeConfig& output,
    std::string& error) {
  if (!api_config || !api_config->stage2_model_json_path ||
      api_config->stage2_model_json_path[0] == '\0') {
    error = "stage2_model_json_path must point to the Loop151 native runtime config";
    return false;
  }
  output.config_path = path_from_utf8(api_config->stage2_model_json_path);
  std::string document;
  if (!read_text_file(output.config_path, document)) {
    error = "Loop151 native runtime config cannot be opened";
    return false;
  }
  std::string schema;
  if (!json_string_field(document, "schema", schema) ||
      schema != "axon_loop151_native_runtime_v1") {
    error = "Loop151 native runtime config schema is invalid";
    return false;
  }
  auto required_path = [&](const char* key, std::filesystem::path& target) {
    std::string value;
    if (!json_string_field(document, key, value) || value.empty()) {
      error = std::string("Loop151 native runtime config is missing ") + key;
      return false;
    }
    target = resolve_loop151_path(output.config_path.parent_path(), value);
    if (!regular_file(target)) {
      error = std::string("Loop151 native runtime asset is missing: ") + target.u8string();
      return false;
    }
    return true;
  };
  if (!required_path("primary_model_path", output.primary_path) ||
      !required_path("conservative_model_path", output.conservative_path) ||
      !required_path("content_cross_model_path", output.content_cross_path) ||
      !required_path("noise_model_path", output.noise_path) ||
      !required_path("selector_model_path", output.selector_path)) {
    return false;
  }
  // The base ONNX model produces a probability whose six derived columns are
  // erased again by the primary/conservative/noise stacks before scoring
  // (drop_base_prob_features), leaving content_cross as its only consumer.
  // Measured on 1,990 balanced test rows, disabling it changed 0 decisions
  // while removing 87% of scan time, so it is opt-out through the runtime
  // config. Absent field means enabled: existing configs are unaffected.
  bool base_onnx_enabled = true;
  json_bool_field(document, "base_onnx_enabled", base_onnx_enabled);
  output.base_onnx_enabled = base_onnx_enabled;

  std::string base_path;
  if (api_config->onnx_model_path && api_config->onnx_model_path[0] != '\0') {
    output.base_onnx_path = path_from_utf8(api_config->onnx_model_path);
  } else if (json_string_field(document, "base_onnx_path", base_path) && !base_path.empty()) {
    output.base_onnx_path = resolve_loop151_path(output.config_path.parent_path(), base_path);
  }
  if (output.base_onnx_enabled && !regular_file(output.base_onnx_path)) {
    error = "Loop151 native runtime requires a base ONNX model";
    return false;
  }
  return true;
}

std::vector<float> loop151_stage2_features(
    const std::vector<std::uint8_t>& bytes,
    const InferenceInput& input,
    float base_probability,
    bool include_v2,
    bool include_string) {
  std::vector<float> features = make_stage2_features(bytes, input, base_probability);
  if (include_v2) {
    const auto v2 = axon_loop151_native::content_pe_v2_features(bytes);
    features.insert(features.end(), v2.begin(), v2.end());
  }
  if (include_string) {
    const auto strings = axon_loop151_native::content_string_features(bytes);
    features.insert(features.end(), strings.begin(), strings.end());
  }
  return features;
}

float v1(const std::vector<float>& values, std::size_t index) {
  return index < values.size() ? values[index] : 0.0f;
}

float v2(const std::vector<float>& values, std::size_t index) {
  return index < values.size() ? values[index] : 0.0f;
}

float present(float value) {
  return value > 0.0f ? 1.0f : 0.0f;
}

float inverse_present(float value) {
  return value <= 0.0f ? 1.0f : 0.0f;
}

std::vector<float> loop151_content_cross(
    const std::vector<float>& pe1,
    const std::vector<float>& pe2) {
  const float is_dll = std::clamp(v1(pe1, 21), 0.0f, 1.0f);
  const float security_present = present(v1(pe1, 40));
  const float security_log = v1(pe1, 41);
  const float overlay_present = present(v1(pe1, 81));
  const float overlay_log = v1(pe1, 82);
  const float overlay_entropy = v1(pe1, 84);
  const float overlay_ratio = v1(pe1, 83);
  const float export_log = v1(pe1, 74);
  const float export_name_ratio = v1(pe1, 75);
  const float exception_present = present(v1(pe1, 37));
  const float debug_present = present(v1(pe1, 46));
  const float tls_present = present(v1(pe1, 49));
  const float large_address = std::clamp(v1(pe1, 24), 0.0f, 1.0f);
  const float subsystem = v1(pe1, 19);
  const float high_entropy_section_ratio = v1(pe1, 94);
  const float rwx_ratio = v1(pe1, 87);
  const float rw_ratio = v1(pe1, 86);
  const float zero_raw_ratio = v1(pe1, 96);
  const float raw_virtual_mismatch = v1(pe1, 95);
  const float packer_ratio = v1(pe1, 99);
  const float system_dll_ratio = v1(pe1, 65);
  const float import_api_log = v1(pe1, 62);
  const float network_ratio = v1(pe1, 68);
  const float filesystem_ratio = v1(pe1, 70);
  const float registry_ratio = v1(pe1, 71);
  const float injection_ratio = v1(pe1, 73);
  const float crypto_ratio = v1(pe1, 72);
  const float resource_log = v1(pe1, 76);
  const float resource_type_log = v1(pe1, 77);
  const float driver_api = present(v2(pe2, 67));
  const float service_api = present(v2(pe2, 64));
  const float process_api = present(v2(pe2, 85)) || present(v2(pe2, 76));
  const float crypto_cert_api = present(v2(pe2, 100));
  const float resource_api = present(v2(pe2, 103));
  const float driver_count = v2(pe2, 68);
  const float service_count = v2(pe2, 65);
  const float export_service_pattern = present(v2(pe2, 122));
  const float export_plugin_pattern = present(v2(pe2, 123));
  const float resource_icon = present(v2(pe2, 135));
  const float resource_version = present(v2(pe2, 149));
  const float resource_manifest = present(v2(pe2, 151));
  const float resource_max_entropy = v2(pe2, 130);
  const float exec_write_log = v2(pe2, 156);
  const float exec_high_entropy = v2(pe2, 157);
  const float write_high_entropy = v2(pe2, 158);
  const float zero_raw_exec = v2(pe2, 159);
  const float zero_raw_write = v2(pe2, 160);
  const float max_raw_virtual_delta = v2(pe2, 161);
  const float max_virtual_raw_log = v2(pe2, 163);
  const float ep_write = present(v2(pe2, 165));
  const float ep_entropy = v2(pe2, 166);
  const float ep_raw_virtual_delta = v2(pe2, 167);
  const float last_section_entropy = v2(pe2, 171);
  const float resource_weak = inverse_present(resource_log) * inverse_present(resource_type_log);
  const float native_subsystem_like = subsystem <= 0.02f || driver_api > 0.0f ? 1.0f : 0.0f;
  const float export_like = present(export_log) || export_service_pattern || export_plugin_pattern;
  std::vector<float> features = {
      is_dll * export_log, is_dll * security_present, is_dll * security_log,
      is_dll * overlay_present, is_dll * overlay_entropy, is_dll * exception_present,
      is_dll * debug_present, is_dll * tls_present, is_dll * large_address,
      is_dll * driver_api, is_dll * service_api, is_dll * driver_api * export_like,
      is_dll * service_api * export_service_pattern, is_dll * native_subsystem_like,
      security_present * overlay_present, security_present * overlay_log,
      security_present * overlay_entropy, security_present * export_log,
      security_present * exception_present, security_present * debug_present,
      (1.0f - security_present) * overlay_log, (1.0f - security_present) * overlay_entropy,
      (1.0f - security_present) * high_entropy_section_ratio,
      overlay_entropy * overlay_ratio, overlay_entropy * last_section_entropy,
      overlay_entropy * ep_entropy, overlay_present * resource_weak,
      exec_write_log * std::max(exec_high_entropy, write_high_entropy),
      zero_raw_exec * high_entropy_section_ratio, zero_raw_write * high_entropy_section_ratio,
      ep_write * ep_entropy, ep_entropy * overlay_entropy,
      ep_raw_virtual_delta * overlay_present, last_section_entropy * overlay_entropy,
      packer_ratio * rwx_ratio, packer_ratio * zero_raw_ratio, packer_ratio * overlay_present,
      raw_virtual_mismatch * overlay_present, max_virtual_raw_log * overlay_present,
      system_dll_ratio * import_api_log, system_dll_ratio * network_ratio,
      system_dll_ratio * filesystem_ratio, system_dll_ratio * registry_ratio,
      system_dll_ratio * injection_ratio, network_ratio * overlay_entropy,
      filesystem_ratio * overlay_entropy, registry_ratio * overlay_entropy,
      injection_ratio * exec_write_log, process_api * exec_write_log,
      driver_count * export_log, driver_api * security_present, driver_api * overlay_present,
      service_count * export_log, service_api * security_present, service_api * resource_weak,
      crypto_cert_api * security_present, crypto_ratio * overlay_present,
      export_log * resource_weak, export_name_ratio * overlay_present,
      export_name_ratio * security_present, resource_weak * security_present,
      resource_weak * overlay_present, resource_max_entropy * resource_api,
      resource_icon * overlay_present, resource_version * security_present,
      resource_manifest * security_present * overlay_present};
  if (features.size() != 66) features.resize(66, 0.0f);
  return features;
}

std::vector<float> loop151_selector_features(
    float primary,
    float noise,
    int loop130_prediction,
    int noise_prediction,
    const std::vector<float>& pe1,
    const std::vector<float>& pe2,
    const std::vector<float>& strings) {
  std::vector<float> values = {
      primary, noise, noise - primary, std::fabs(primary - 0.5f),
      std::fabs(noise - 0.5f), std::fabs(noise - 0.5f) - std::fabs(primary - 0.5f),
      static_cast<float>(loop130_prediction), static_cast<float>(noise_prediction),
      static_cast<float>(loop130_prediction == 0 && noise_prediction == 1),
      static_cast<float>(loop130_prediction == 1 && noise_prediction == 0)};
  values.insert(values.end(), {
      v1(pe1, 21), v1(pe1, 74), v1(pe1, 41), v1(pe1, 82), v1(pe1, 76),
      v1(pe1, 77), v1(pe1, 36), v1(pe1, 84), v1(pe1, 62), v1(pe1, 66),
      v2(pe2, 124), v2(pe2, 136), v2(pe2, 140), v2(pe2, 171), v2(pe2, 163),
      v2(pe2, 94), v2(pe2, 99), v2(pe2, 23), strings.size() > 39 ? strings[39] : 0.0f,
      strings.size() > 41 ? strings[41] : 0.0f,
      strings.size() > 19 ? strings[19] : 0.0f,
      strings.size() > 20 ? strings[20] : 0.0f});
  return values;
}

bool native_model_score(
    const NativeScoreModel& model,
    const std::vector<float>& features,
    float& score,
    std::string& error) {
  error.clear();
  score = model.predict_probability(features, &error);
  return error.empty() && std::isfinite(score);
}

bool native_stack_score(
    const NativeStackModel& model,
    const std::vector<float>& features,
    float& score,
    std::string& error) {
  error.clear();
  score = model.predict_probability(features, &error);
  return error.empty() && std::isfinite(score);
}

#if defined(_WIN32)
std::string wide_to_utf8(const std::wstring& value) {
  if (value.empty()) return {};
  const int required = WideCharToMultiByte(CP_UTF8, WC_ERR_INVALID_CHARS, value.data(), static_cast<int>(value.size()), nullptr, 0, nullptr, nullptr);
  if (required <= 0) return {};
  std::string output(static_cast<std::size_t>(required), '\0');
  if (WideCharToMultiByte(CP_UTF8, WC_ERR_INVALID_CHARS, value.data(), static_cast<int>(value.size()), output.data(), required, nullptr, nullptr) != required) return {};
  return output;
}

std::pair<std::string, std::string> query_authenticode(const std::filesystem::path& path) {
  const std::wstring path_text = path.wstring();
  WINTRUST_FILE_INFO file_info{};
  file_info.cbStruct = sizeof(file_info);
  file_info.pcwszFilePath = path_text.c_str();
  GUID policy = WINTRUST_ACTION_GENERIC_VERIFY_V2;
  WINTRUST_DATA trust{};
  trust.cbStruct = sizeof(trust);
  trust.dwUIChoice = WTD_UI_NONE;
  trust.fdwRevocationChecks = WTD_REVOKE_NONE;
  trust.dwUnionChoice = WTD_CHOICE_FILE;
  trust.pFile = &file_info;
  trust.dwStateAction = WTD_STATEACTION_VERIFY;
  trust.dwProvFlags = WTD_CACHE_ONLY_URL_RETRIEVAL;
  const LONG status = WinVerifyTrust(nullptr, &policy, &trust);
  std::string subject;
  if (status == ERROR_SUCCESS && trust.hWVTStateData) {
    CRYPT_PROVIDER_DATA* provider = WTHelperProvDataFromStateData(trust.hWVTStateData);
    if (provider) {
      CRYPT_PROVIDER_SGNR* signer = WTHelperGetProvSignerFromChain(provider, 0, FALSE, 0);
      if (signer && signer->csCertChain > 0 && signer->pasCertChain[0].pCert) {
        PCCERT_CONTEXT cert = signer->pasCertChain[0].pCert;
        DWORD required = CertGetNameStringW(cert, CERT_NAME_SIMPLE_DISPLAY_TYPE, 0, nullptr, nullptr, 0);
        if (required > 1) {
          std::wstring name(required, L'\0');
          CertGetNameStringW(cert, CERT_NAME_SIMPLE_DISPLAY_TYPE, 0, nullptr, name.data(), required);
          name.resize(required - 1);
          subject = wide_to_utf8(name);
        }
      }
    }
  }
  trust.dwStateAction = WTD_STATEACTION_CLOSE;
  WinVerifyTrust(nullptr, &policy, &trust);
  return {status == ERROR_SUCCESS ? "Valid" : "Unavailable", subject};
}
#else
std::pair<std::string, std::string> query_authenticode(const std::filesystem::path&) {
  return {"Unavailable", {}};
}
#endif

bool trusted_signer_downgrade(
    int loop136_prediction,
    const std::filesystem::path& path,
    std::vector<std::string>& matched_terms) {
  if (loop136_prediction != 1) return false;
  const auto [status, subject] = query_authenticode(path);
  if (status != "Valid") return false;
  const std::string lower_subject = [&]() {
    std::string value = subject;
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return value;
  }();
  static const std::array<const char*, 12> terms = {
      "Microsoft Corporation", "Microsoft Windows", "Seagate Technology", "FinalWire",
      "NetEase", "Beijing Sogou", "Beijing Kingsoft", "Beijing Qihu", "Wondershare",
      "IObit", "Yozosoft", "Huya"};
  for (const char* term : terms) {
    std::string lower_term = term;
    std::transform(lower_term.begin(), lower_term.end(), lower_term.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    if (lower_subject.find(lower_term) != std::string::npos) matched_terms.emplace_back(term);
  }
  return !matched_terms.empty();
}

std::string loop151_prediction_json(
    const std::string& path,
    const Loop151StageScores& scores,
    int loop130_prediction,
    int loop136_prediction,
    float selected_probability,
    float selector_score,
    bool selector_used,
    int final_prediction,
    bool signer_downgrade,
    const std::vector<std::string>& signer_terms,
    const StageTiming& timing,
    const Loop151Decision* no_onnx) {
  const float probability = selected_probability;
  const float confidence = final_prediction == 1 ? probability : 1.0f - probability;
  std::ostringstream output;
  output.setf(std::ios::fixed);
  output.precision(8);
  output << "{\"ok\":true,\"loop_id\":\"Loop151\",\"mode\":\"single_pe\",\"file\":\""
         << json_escape(path) << "\",\"prediction\":" << final_prediction
         << ",\"is_malware\":" << (final_prediction ? "true" : "false")
         << ",\"label\":\"" << (final_prediction ? "malicious" : "benign")
         << "\",\"probability\":" << probability << ",\"confidence\":" << confidence
         << ",\"prob_benign\":" << (1.0f - probability) << ",\"prob_malicious\":" << probability
         << ",\"primary_probability\":" << scores.primary
         << ",\"conservative_probability\":" << scores.conservative
         << ",\"content_cross_probability\":" << scores.content_cross
         << ",\"loop130_prediction\":" << loop130_prediction
         << ",\"loop134_probability\":" << scores.noise
         << ",\"loop136_prediction\":" << loop136_prediction
         << ",\"selector_score\":";
  if (selector_used) output << selector_score; else output << "null";
  output << ",\"trusted_signer_downgrade\":" << (signer_downgrade ? "true" : "false")
         << ",\"trusted_signer_terms\":[";
  for (std::size_t index = 0; index < signer_terms.size(); ++index) {
    if (index) output << ',';
    output << '\"' << json_escape(signer_terms[index]) << '\"';
  }
  output << ']';
  if (timing_enabled()) append_timing_json(output, timing);
  if (no_onnx) {
    // Signer downgrade is orthogonal to the ONNX question, so it is applied to
    // the shadow decision too; otherwise the comparison would credit the
    // ONNX-free path with false positives the guard would have removed anyway.
    const int shadow_final = signer_downgrade ? 0 : no_onnx->loop136;
    output << ",\"no_onnx\":{\"prediction\":" << shadow_final
           << ",\"loop130_prediction\":" << no_onnx->loop130
           << ",\"loop136_prediction\":" << no_onnx->loop136
           << ",\"probability\":" << no_onnx->selected_probability
           << ",\"selector_score\":";
    if (no_onnx->selector_used) output << no_onnx->selector_score; else output << "null";
    output << '}';
  }
  output << '}';
  return output.str();
}

int write_loop151_error(const std::string& code, const std::string& message, char** out_json, size_t* out_len) {
  return write_string_out(error_json(code, message), out_json, out_len);
}

Loop151Handle* native_handle(kvd_handle* handle) {
  return reinterpret_cast<Loop151Handle*>(handle);
}

const Loop151Handle* native_handle(const kvd_handle* handle) {
  return reinterpret_cast<const Loop151Handle*>(handle);
}

bool load_loop151_models(
    const kvd_config* api_config,
    Loop151RuntimeConfig& runtime,
    Loop151Handle& handle,
    std::string& error) {
  if (!load_loop151_runtime_config(api_config, runtime, error)) return false;
  handle.base_onnx_enabled = runtime.base_onnx_enabled;
  try {
    // Skipping the load, not just the inference: the base model plus its
    // external .data file is ~42 MB of the package and a matching share of
    // init time and resident memory.
    if (runtime.base_onnx_enabled) {
      handle.base_model = std::make_shared<AxonOnnxModel>(runtime.base_onnx_path.u8string());
    }
    handle.primary = NativeStackModel::load_file(runtime.primary_path.u8string(), error);
    if (!handle.primary) return false;
    handle.conservative = NativeStackModel::load_file(runtime.conservative_path.u8string(), error);
    if (!handle.conservative) return false;
    handle.content_cross = NativeScoreModel::load_file(runtime.content_cross_path.u8string(), error);
    if (!handle.content_cross) return false;
    handle.noise = NativeStackModel::load_file(runtime.noise_path.u8string(), error);
    if (!handle.noise) return false;
    handle.selector = NativeScoreModel::load_file(runtime.selector_path.u8string(), error);
    if (!handle.selector) return false;
  } catch (const std::exception& exception) {
    error = exception.what();
    return false;
  } catch (...) {
    error = "Loop151 native model initialization failed";
    return false;
  }
  return true;
}

bool loop151_predict_bytes(
    Loop151Handle& handle,
    const std::vector<std::uint8_t>& bytes,
    std::string& response,
    const std::filesystem::path* signer_path) {
  StageTiming timing;
  const auto scan_start = TimingClock::now();

  auto mark = TimingClock::now();
  InferenceInput input = make_inference_input(bytes);
  timing.input = elapsed_ms(mark);

  float base_probability = 0.0f;
  if (handle.base_onnx_enabled) {
    std::array<float, 2> ignored_logits{};
    mark = TimingClock::now();
    Prediction base_prediction = handle.base_model->predict(input, 0.5f, &ignored_logits);
    timing.base_onnx = elapsed_ms(mark);
    if (!base_prediction.ok) {
      response = error_json(base_prediction.error_code, base_prediction.error);
      return false;
    }
    base_probability = base_prediction.prob_malicious;
  }
  std::string model_error;
  Loop151StageScores scores;

  // Every content feature block is a pure function of the file bytes, but the
  // original flow recomputed them: make_stage2_features runs three times with
  // identical arguments (each redoing lightweight, byte-summary and pe_v1),
  // pe_v2 was built three times and the string block twice. Computing each
  // once and concatenating produces bit-identical vectors.
  mark = TimingClock::now();
  auto sub_mark = TimingClock::now();
  const auto pe_v1 = content_pe_v1_features(bytes);
  timing.content_pe_v1 = elapsed_ms(sub_mark);
  sub_mark = TimingClock::now();
  const auto pe_v2 = axon_loop151_native::content_pe_v2_features(bytes);
  timing.content_pe_v2 = elapsed_ms(sub_mark);
  sub_mark = TimingClock::now();
  const auto strings = axon_loop151_native::content_string_features(bytes);
  timing.content_strings = elapsed_ms(sub_mark);
  timing.content_features = elapsed_ms(mark);

  mark = TimingClock::now();
  const auto stage2_base = make_stage2_features(bytes, input, base_probability);
  auto with_content = [&](bool include_v2, bool include_string) {
    std::vector<float> features = stage2_base;
    if (include_v2) features.insert(features.end(), pe_v2.begin(), pe_v2.end());
    if (include_string) features.insert(features.end(), strings.begin(), strings.end());
    return features;
  };
  const auto primary_features = with_content(true, false);
  const auto& conservative_features = primary_features;
  const auto noise_features = with_content(true, true);
  timing.stage2_features = elapsed_ms(mark);

  mark = TimingClock::now();
  const bool primary_ok = native_stack_score(*handle.primary, primary_features, scores.primary, model_error);
  timing.primary = elapsed_ms(mark);
  mark = TimingClock::now();
  const bool conservative_ok = primary_ok &&
      native_stack_score(*handle.conservative, conservative_features, scores.conservative, model_error);
  timing.conservative = elapsed_ms(mark);
  if (!primary_ok || !conservative_ok) {
    response = error_json("loop151_model_failed", model_error);
    return false;
  }

  // content_cross is the base model's only consumer, so it goes away with it.
  if (handle.base_onnx_enabled) {
    auto cross_features = stage2_base;
    const auto cross_extra = loop151_content_cross(pe_v1, pe_v2);
    cross_features.insert(cross_features.end(), cross_extra.begin(), cross_extra.end());
    mark = TimingClock::now();
    const bool cross_ok =
        native_model_score(*handle.content_cross, cross_features, scores.content_cross, model_error);
    timing.content_cross = elapsed_ms(mark);
    if (!cross_ok) {
      response = error_json("loop151_model_failed", model_error);
      return false;
    }
  }
  mark = TimingClock::now();
  const bool noise_ok = native_stack_score(*handle.noise, noise_features, scores.noise, model_error);
  timing.noise = elapsed_ms(mark);
  if (!noise_ok) {
    response = error_json("loop151_model_failed", model_error);
    return false;
  }
  const int primary_prediction = scores.primary >= kLoop151PrimaryThreshold ? 1 : 0;
  const int conservative_prediction = scores.conservative >= kLoop151ConservativeThreshold ? 1 : 0;
  const int cross_prediction = scores.content_cross >= kLoop151ContentCrossThreshold ? 1 : 0;
  const int noise_prediction = scores.noise >= kLoop151NoiseThreshold ? 1 : 0;

  // The r4/r5 rules and the selector stage are shared by both decision paths, so
  // they live in one lambda: the ONNX-free variant must not drift from the real
  // one as the rules change. Only `possible` differs between the two.
  bool selector_failed = false;
  auto decide = [&](bool possible, Loop151Decision& decision) {
    const bool r4 = possible && scores.primary <= 0.65f &&
        v2(pe_v2, 124) >= 2.0f && v2(pe_v2, 136) >= 1.5f && v1(pe_v1, 36) >= 0.001f;
    const bool r5_flip = r4 || (possible && !r4 && strings.size() > 39 && strings[39] >= 3.0f);
    decision.loop130 = r5_flip ? 0 : primary_prediction;
    decision.loop136 = decision.loop130;
    decision.selected_probability = scores.primary;
    if (decision.loop130 != noise_prediction) {
      const auto selector_features = loop151_selector_features(
          scores.primary, scores.noise, decision.loop130, noise_prediction, pe_v1, pe_v2, strings);
      if (!native_model_score(*handle.selector, selector_features, decision.selector_score, model_error)) {
        selector_failed = true;
        return;
      }
      decision.selector_used = true;
      if (decision.selector_score >= kLoop151SelectorThreshold) {
        decision.loop136 = noise_prediction;
        decision.selected_probability = scores.noise;
      }
    }
  };

  // Without the base model there is no cross term; dropping it from the OR makes
  // `possible` strictly harder to satisfy, so fewer malicious->benign rule flips.
  const bool possible = handle.base_onnx_enabled
      ? (primary_prediction == 1 && (conservative_prediction == 0 || cross_prediction == 0))
      : (primary_prediction == 1 && conservative_prediction == 0);
  Loop151Decision decision;
  mark = TimingClock::now();
  decide(possible, decision);
  timing.selector = elapsed_ms(mark);
  if (selector_failed) {
    response = error_json("loop151_selector_failed", model_error);
    return false;
  }

  // ONNX-free variant: content_cross is the only consumer of the base model's
  // probability (primary/conservative/noise erase those six derived columns
  // before scoring), so dropping the ONNX run means dropping the cross term.
  // Removing it from the OR makes `possible` strictly harder to satisfy, which
  // is the conservative degradation: fewer malicious->benign rule flips.
  Loop151Decision no_onnx_decision;
  if (no_onnx_shadow_enabled()) {
    const bool possible_no_onnx = primary_prediction == 1 && conservative_prediction == 0;
    decide(possible_no_onnx, no_onnx_decision);
    if (selector_failed) {
      response = error_json("loop151_selector_failed", model_error);
      return false;
    }
  }

  std::vector<std::string> signer_terms;
  mark = TimingClock::now();
  const bool downgraded = signer_path && trusted_signer_downgrade(decision.loop136, *signer_path, signer_terms);
  timing.signer = elapsed_ms(mark);
  const int final_prediction = downgraded ? 0 : decision.loop136;
  timing.total = elapsed_ms(scan_start);
  response = loop151_prediction_json(
      signer_path ? signer_path->u8string() : "<bytes>", scores, decision.loop130,
      decision.loop136, decision.selected_probability, decision.selector_score,
      decision.selector_used, final_prediction, downgraded, signer_terms, timing,
      no_onnx_shadow_enabled() ? &no_onnx_decision : nullptr);
  return true;
}

bool is_loop151_runtime(const kvd_config* config) {
  if (!config || !config->stage2_model_json_path) return false;
  std::string document;
  if (!read_text_file(path_from_utf8(config->stage2_model_json_path), document)) return false;
  return document.find("axon_loop151_native_runtime_v1") != std::string::npos;
}

}  // namespace

extern "C" {

KVD_API kvd_handle* KVD_CALL kvd_create(const kvd_config* config) {
  if (!is_loop151_runtime(config)) return nullptr;
  auto handle = std::make_unique<Loop151Handle>();
  handle->config = config_from_api(config);
  Loop151RuntimeConfig runtime;
  std::string error;
  if (!load_loop151_models(config, runtime, *handle, error)) return nullptr;
  return reinterpret_cast<kvd_handle*>(handle.release());
}

KVD_API void KVD_CALL kvd_destroy(kvd_handle* handle) {
  delete native_handle(handle);
}

KVD_API int KVD_CALL kvd_scan_path(kvd_handle* api_handle, const char* path, char** out_json, size_t* out_len) {
  if (!api_handle || !path) return write_loop151_error("invalid_argument", "handle and path are required.", out_json, out_len);
  Loop151Handle* handle = native_handle(api_handle);
  const std::string path_text(path);
  if (!path_allowed(path_text, handle->config.allowed_scan_root)) return write_loop151_error("path_not_allowed", "Input path is outside allowed_scan_root.", out_json, out_len);
  std::vector<std::uint8_t> bytes;
  std::string error;
  if (!read_file_bytes_limited(path_text, bytes, error, handle->config.max_file_size)) return write_loop151_error(error == "file_too_large" ? "file_too_large" : "file_read_failed", error, out_json, out_len);
  std::string response;
  const std::filesystem::path signer_path = path_from_utf8(path_text);
  loop151_predict_bytes(*handle, bytes, response, &signer_path);
  return write_string_out(response, out_json, out_len);
}

KVD_API int KVD_CALL kvd_scan_bytes(kvd_handle* api_handle, const unsigned char* bytes, size_t len, char** out_json, size_t* out_len) {
  if (!api_handle || (!bytes && len > 0)) return write_loop151_error("invalid_argument", "handle and bytes are required.", out_json, out_len);
  Loop151Handle* handle = native_handle(api_handle);
  if (handle->config.max_file_size > 0 && len > handle->config.max_file_size) return write_loop151_error("file_too_large", "Input byte buffer exceeds max_file_size.", out_json, out_len);
  std::vector<std::uint8_t> buffer;
  if (len > 0) buffer.assign(bytes, bytes + len);
  std::string response;
  loop151_predict_bytes(*handle, buffer, response, nullptr);
  return write_string_out(response, out_json, out_len);
}

KVD_API int KVD_CALL kvd_parity_diagnostics_path_v1(
    kvd_handle*,
    const char*,
    const kvd_parity_diagnostics_options_v1*,
    char** out_json,
    size_t* out_len) {
  return write_loop151_error(
      "unsupported_operation",
      "Loop151 native runtime does not expose Loop28 parity diagnostics.",
      out_json,
      out_len);
}

KVD_API int KVD_CALL kvd_scan_paths(kvd_handle* handle, const char** paths, size_t count, char** out_json, size_t* out_len) {
  if (!handle || (!paths && count > 0)) return write_loop151_error("invalid_argument", "handle and paths are required.", out_json, out_len);
  std::ostringstream output;
  output << '[';
  for (size_t index = 0; index < count; ++index) {
    if (index) output << ',';
    char* item = nullptr;
    size_t item_len = 0;
    kvd_scan_path(handle, paths[index], &item, &item_len);
    if (item) {
      output.write(item, static_cast<std::streamsize>(item_len));
      std::free(item);
    } else {
      output << error_json("scan_failed", "Failed to scan path.");
    }
  }
  output << ']';
  return write_string_out(output.str(), out_json, out_len);
}

KVD_API void KVD_CALL kvd_free(char* pointer) {
  std::free(pointer);
}

KVD_API int KVD_CALL kvd_validate_models(const kvd_config* config, char** out_error, size_t* out_len) {
  if (!config) return KVD_MODEL_ERR_INVALID_ARGUMENT;
  if (!is_loop151_runtime(config)) {
    return write_loop151_error("loop151_runtime_required", "stage2_model_json_path must identify a Loop151 native runtime config.", out_error, out_len);
  }
  Loop151Handle handle;
  Loop151RuntimeConfig runtime;
  std::string error;
  if (!load_loop151_models(config, runtime, handle, error)) {
    write_string_out(error, out_error, out_len);
    return KVD_MODEL_ERR_ONNX_MAIN_INVALID;
  }
  write_string_out("ok", out_error, out_len);
  return KVD_MODEL_OK;
}

KVD_API int KVD_CALL kvd_train_path(kvd_handle*, const char*, int, char** out_json, size_t* out_len) {
  return write_loop151_error("unsupported_operation", "Loop151 DLL is inference-only.", out_json, out_len);
}

KVD_API int KVD_CALL kvd_train_paths(kvd_handle*, const char**, size_t, int, char** out_json, size_t* out_len) {
  return write_loop151_error("unsupported_operation", "Loop151 DLL is inference-only.", out_json, out_len);
}

KVD_API int KVD_CALL kvd_train_from_path(kvd_handle* handle, const char* path, int label, char** out_json, size_t* out_len) {
  return kvd_train_path(handle, path, label, out_json, out_len);
}

KVD_API void KVD_CALL kvd_signature_flush(kvd_handle*) {}

KVD_API int KVD_CALL kvd_extract_pe_features(const char* path, float* out_features, size_t out_len) {
  if (!path || !out_features || out_len < kAxonPeFeatureDim) return -1;
  std::vector<std::uint8_t> bytes;
  std::string error;
  if (!read_file_bytes(path, bytes, error)) return -2;
  const auto features = fixed_v2_pe_features(bytes);
  std::copy_n(features.data(), kAxonPeFeatureDim, out_features);
  return 0;
}

KVD_API int KVD_CALL kvd_extract_pe_features_batch(const char** paths, size_t count, float* out_features, size_t feature_dim, int* out_status, unsigned int) {
  if ((!paths && count > 0) || !out_features || !out_status || feature_dim < kAxonPeFeatureDim) return -1;
  for (size_t index = 0; index < count; ++index) {
    out_status[index] = kvd_extract_pe_features(paths[index], out_features + index * feature_dim, feature_dim);
  }
  return 0;
}

KVD_API size_t KVD_CALL kvd_get_pe_feature_dimension(void) {
  return kAxonPeFeatureDim;
}

KVD_API char* KVD_CALL axon_predict_json(const char*) {
  const std::string response = error_json("unsupported_operation", "Use kvd_create plus kvd_scan_path or kvd_scan_bytes for Loop151.");
  char* output = nullptr;
  size_t length = 0;
  write_string_out(response, &output, &length);
  return output;
}

KVD_API void KVD_CALL axon_string_free(char* pointer) {
  std::free(pointer);
}

KVD_API const char* KVD_CALL axon_version(void) {
  return "axon_loop151_native/1.0.0";
}

}  // extern "C"
