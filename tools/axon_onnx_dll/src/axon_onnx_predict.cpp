#include "axon_onnx_predict.h"

#include <onnxruntime_cxx_api.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cwctype>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <mutex>
#include <numeric>
#include <optional>
#include <set>
#include <sstream>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#if defined(_WIN32)
#include <windows.h>
#include <bcrypt.h>
#endif

namespace {

constexpr std::size_t kAxonByteLength = 8192;
constexpr std::size_t kAxonPeFeatureDim = 256;
constexpr std::size_t kAxonStatFeatureDim = 49;
constexpr std::size_t kAxonLightweightFeatureDim = 256;
constexpr std::size_t kContentPeFeatureDim = 100;
constexpr std::size_t kStage2FeatureDim = 1520;
constexpr std::size_t kStage2PrefixLen = 256;
constexpr std::size_t kStage2ChunkCount = 16;
constexpr std::size_t kAxonFixedSectionSlots = 32;
constexpr int kStatChunkCount = 10;
constexpr int kStatChunkDiffCount = kStatChunkCount - 1;
constexpr std::size_t kProcessOutputLimitBytes = 16 * 1024;
constexpr std::uint32_t kProcessTimeoutMs = 300000;
constexpr std::size_t kNestedArchiveReportResponseLimit = 16 * 1024;
constexpr std::size_t kNestedPredictionResponseLimit = 256;
constexpr const char* kAxonVersion = "axon_onnx_predict/0.2.0-native-loop28";

struct InferenceInput {
  std::array<std::int64_t, kAxonByteLength> byte_seq{};
  std::array<float, kAxonPeFeatureDim> pe_features{};
  std::array<float, kAxonStatFeatureDim> stat_features{};
  std::uint64_t original_length = 0;
};

struct Prediction {
  bool ok = false;
  std::string error_code;
  std::string error;
  int prediction = 0;
  float confidence = 0.0f;
  float prob_benign = 0.0f;
  float prob_malicious = 0.0f;
  int base_prediction = 0;
  float base_confidence = 0.0f;
  float base_prob_benign = 0.0f;
  float base_prob_malicious = 0.0f;
  bool stage2_enabled = false;
  float stage2_threshold = 0.5f;
  std::size_t stage2_feature_dim = 0;
  std::uint64_t original_length = 0;
};

struct PredictionCapture {
  InferenceInput input;
  std::array<float, 2> base_logits{};
  std::array<float, 2> base_probabilities{};
  std::vector<float> stage2_features;
};

struct AxonConfig {
  std::string onnx_model_path;
  std::string stage2_model_json_path;
  std::string family_classifier_json_path;
  std::string archive_scanner_path;
  std::string allowed_scan_root;
  std::uint64_t max_file_size = 0;
  float threshold = 0.5f;
  bool scan_nested = false;
};

struct FamilyPrediction {
  int cluster_id = -1;
  std::string family_name;
  bool is_new_family = false;
  float distance = 0.0f;
  float threshold = 0.0f;
};

class FamilyClassifier {
 public:
  static std::optional<FamilyClassifier> load_from_json(const std::string& path);

  std::optional<FamilyPrediction> predict(const std::vector<float>& features) const;

  bool ok() const {
    return !centroids_.empty();
  }

 private:
  std::vector<int> cluster_ids_;
  std::vector<std::vector<float>> centroids_;
  std::vector<float> thresholds_;
  std::vector<std::string> family_names_;
  std::vector<float> scaler_mean_;
  std::vector<float> scaler_scale_;
};

class Stage2HgbModel {
 public:
  static std::optional<Stage2HgbModel> load_from_json(const std::string& path);

  float predict_probability(const std::vector<float>& features) const;

  bool ok() const {
    return n_features_ > 0 && !tree_offsets_.empty() && !node_values_.empty();
  }

  float threshold() const {
    return threshold_;
  }

  std::size_t n_features() const {
    return n_features_;
  }

 private:
  std::size_t n_features_ = 0;
  double baseline_prediction_ = 0.0;
  float threshold_ = 0.5f;
  std::vector<int> tree_offsets_;
  std::vector<double> node_values_;
  std::vector<int> node_feature_idx_;
  std::vector<double> node_num_thresholds_;
  std::vector<int> node_missing_go_to_left_;
  std::vector<int> node_left_;
  std::vector<int> node_right_;
  std::vector<int> node_is_leaf_;
};

class AxonOnnxModel {
 public:
  explicit AxonOnnxModel(const std::string& model_path)
      : env_(ORT_LOGGING_LEVEL_WARNING, "AxonOnnxPredict"), session_options_() {
    session_options_.SetIntraOpNumThreads(1);
    session_options_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_DISABLE_ALL);
#if defined(_WIN32)
    std::wstring wpath = utf8_to_wide(model_path);
    session_ = std::make_unique<Ort::Session>(env_, wpath.c_str(), session_options_);
#else
    session_ = std::make_unique<Ort::Session>(env_, model_path.c_str(), session_options_);
#endif
    inspect_model();
  }

  Prediction predict(
      const InferenceInput& input,
      float threshold,
      std::array<float, 2>* out_logits = nullptr) const {
    Prediction result;
    result.original_length = input.original_length;
    if (!session_) {
      result.error_code = "model_not_loaded";
      result.error = "ONNX session is not initialized.";
      return result;
    }

    try {
      std::array<std::int64_t, 2> byte_shape{
          1, static_cast<std::int64_t>(kAxonByteLength)};
      std::array<std::int64_t, 2> pe_shape{
          1, static_cast<std::int64_t>(kAxonPeFeatureDim)};
      std::array<std::int64_t, 2> stat_shape{
          1, static_cast<std::int64_t>(kAxonStatFeatureDim)};

      auto memory = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
      Ort::Value byte_tensor = Ort::Value::CreateTensor<std::int64_t>(
          memory,
          const_cast<std::int64_t*>(input.byte_seq.data()),
          input.byte_seq.size(),
          byte_shape.data(),
          byte_shape.size());
      Ort::Value pe_tensor = Ort::Value::CreateTensor<float>(
          memory,
          const_cast<float*>(input.pe_features.data()),
          input.pe_features.size(),
          pe_shape.data(),
          pe_shape.size());
      Ort::Value stat_tensor = Ort::Value::CreateTensor<float>(
          memory,
          const_cast<float*>(input.stat_features.data()),
          input.stat_features.size(),
          stat_shape.data(),
          stat_shape.size());

      std::array<Ort::Value, 3> inputs = {
          std::move(byte_tensor), std::move(pe_tensor), std::move(stat_tensor)};

      auto outputs = session_->Run(
          Ort::RunOptions{nullptr},
          input_name_ptrs_.data(),
          inputs.data(),
          inputs.size(),
          output_name_ptrs_.data(),
          output_name_ptrs_.size());

      if (outputs.empty() || !outputs[0].IsTensor()) {
        result.error_code = "invalid_model_output";
        result.error = "ONNX model returned no tensor output.";
        return result;
      }

      auto info = outputs[0].GetTensorTypeAndShapeInfo();
      std::size_t count = info.GetElementCount();
      const float* data = outputs[0].GetTensorData<float>();
      if (!data || count < 2) {
        result.error_code = "invalid_model_output";
        result.error = "ONNX output must contain at least two logits.";
        return result;
      }

      float logit0 = data[0];
      float logit1 = data[1];
      if (out_logits) {
        *out_logits = {logit0, logit1};
      }
      float max_logit = std::max(logit0, logit1);
      float e0 = std::exp(logit0 - max_logit);
      float e1 = std::exp(logit1 - max_logit);
      float denom = e0 + e1;
      if (!(denom > 0.0f) || !std::isfinite(denom)) {
        result.error_code = "invalid_probability";
        result.error = "ONNX logits could not be converted to probabilities.";
        return result;
      }

      result.prob_benign = e0 / denom;
      result.prob_malicious = e1 / denom;
      result.prediction = result.prob_malicious >= threshold ? 1 : 0;
      result.confidence = result.prediction == 1 ? result.prob_malicious : result.prob_benign;
      result.ok = true;
      return result;
    } catch (const std::exception& e) {
      result.error_code = "onnx_run_failed";
      result.error = e.what();
      return result;
    } catch (...) {
      result.error_code = "onnx_run_failed";
      result.error = "Unknown ONNX Runtime error.";
      return result;
    }
  }

 private:
#if defined(_WIN32)
  static std::wstring utf8_to_wide(const std::string& text) {
    if (text.empty()) {
      return {};
    }
    int needed = MultiByteToWideChar(CP_UTF8, 0, text.data(), static_cast<int>(text.size()), nullptr, 0);
    if (needed <= 0) {
      return std::wstring(text.begin(), text.end());
    }
    std::wstring wide(static_cast<std::size_t>(needed), L'\0');
    MultiByteToWideChar(CP_UTF8, 0, text.data(), static_cast<int>(text.size()), wide.data(), needed);
    return wide;
  }
#endif

  void inspect_model() {
    Ort::AllocatorWithDefaultOptions allocator;
    std::size_t input_count = session_->GetInputCount();
    std::size_t output_count = session_->GetOutputCount();
    if (input_count != 3 || output_count < 1) {
      throw std::runtime_error("Axon ONNX model must expose three inputs and at least one output.");
    }

    for (std::size_t i = 0; i < input_count; ++i) {
      auto name = session_->GetInputNameAllocated(i, allocator);
      input_names_.emplace_back(name.get());
    }
    for (std::size_t i = 0; i < output_count; ++i) {
      auto name = session_->GetOutputNameAllocated(i, allocator);
      output_names_.emplace_back(name.get());
    }
    for (const auto& name : input_names_) {
      input_name_ptrs_.push_back(name.c_str());
    }
    for (const auto& name : output_names_) {
      output_name_ptrs_.push_back(name.c_str());
    }
  }

  Ort::Env env_;
  Ort::SessionOptions session_options_;
  std::unique_ptr<Ort::Session> session_;
  std::vector<std::string> input_names_;
  std::vector<std::string> output_names_;
  std::vector<const char*> input_name_ptrs_;
  std::vector<const char*> output_name_ptrs_;
};

}  // namespace

struct kvd_handle {
  AxonConfig config;
  std::shared_ptr<AxonOnnxModel> model;
  std::shared_ptr<Stage2HgbModel> stage2_model;
  std::shared_ptr<FamilyClassifier> family_classifier;
};

namespace {

template <typename T>
T read_le(const std::vector<std::uint8_t>& data, std::size_t offset) {
  if (offset + sizeof(T) > data.size()) {
    return 0;
  }
  T value = 0;
  std::memcpy(&value, data.data() + offset, sizeof(T));
  return value;
}

std::string json_escape(const std::string& text) {
  std::string out;
  out.reserve(text.size() + 8);
  for (unsigned char ch : text) {
    switch (ch) {
      case '\\': out += "\\\\"; break;
      case '"': out += "\\\""; break;
      case '\b': out += "\\b"; break;
      case '\f': out += "\\f"; break;
      case '\n': out += "\\n"; break;
      case '\r': out += "\\r"; break;
      case '\t': out += "\\t"; break;
      default:
        if (ch < 0x20) {
          char buf[7];
          std::snprintf(buf, sizeof(buf), "\\u%04x", ch);
          out += buf;
        } else {
          out += static_cast<char>(ch);
        }
    }
  }
  return out;
}

class JsonLite {
 public:
  explicit JsonLite(std::string text) : text_(std::move(text)) {}

  bool number_value(const std::string& key, float& out) const {
    std::size_t pos = find_key(key);
    if (pos == std::string::npos) {
      return false;
    }
    pos = text_.find(':', pos);
    if (pos == std::string::npos) {
      return false;
    }
    ++pos;
    skip_ws(pos);
    const char* begin = text_.c_str() + pos;
    char* end = nullptr;
    float value = std::strtof(begin, &end);
    if (end == begin) {
      return false;
    }
    out = value;
    return true;
  }

  bool number_value(const std::string& key, double& out) const {
    std::size_t pos = find_key(key);
    if (pos == std::string::npos) {
      return false;
    }
    pos = text_.find(':', pos);
    if (pos == std::string::npos) {
      return false;
    }
    ++pos;
    skip_ws(pos);
    const char* begin = text_.c_str() + pos;
    char* end = nullptr;
    double value = std::strtod(begin, &end);
    if (end == begin || !std::isfinite(value)) {
      return false;
    }
    out = value;
    return true;
  }

  bool int_value(const std::string& key, int& out) const {
    float value = 0.0f;
    if (!number_value(key, value)) {
      return false;
    }
    out = static_cast<int>(value);
    return true;
  }

  bool number_array(const std::string& key, std::vector<float>& out) const {
    std::size_t pos = find_key(key);
    if (pos == std::string::npos) {
      return false;
    }
    pos = text_.find('[', pos);
    if (pos == std::string::npos) {
      return false;
    }
    return parse_number_array(pos, out);
  }

  bool number_array(const std::string& key, std::vector<double>& out) const {
    std::size_t pos = find_key(key);
    if (pos == std::string::npos) {
      return false;
    }
    pos = text_.find('[', pos);
    if (pos == std::string::npos) {
      return false;
    }
    return parse_number_array(pos, out);
  }

  bool int_array(const std::string& key, std::vector<int>& out) const {
    std::vector<float> values;
    if (!number_array(key, values)) {
      return false;
    }
    out.clear();
    out.reserve(values.size());
    for (float value : values) {
      out.push_back(static_cast<int>(value));
    }
    return true;
  }

  bool string_array(const std::string& key, std::vector<std::string>& out) const {
    std::size_t pos = find_key(key);
    if (pos == std::string::npos) {
      return false;
    }
    pos = text_.find('[', pos);
    if (pos == std::string::npos) {
      return false;
    }
    return parse_string_array(pos, out);
  }

  bool number_matrix(const std::string& key, std::vector<std::vector<float>>& out) const {
    std::size_t pos = find_key(key);
    if (pos == std::string::npos) {
      return false;
    }
    pos = text_.find('[', pos);
    if (pos == std::string::npos) {
      return false;
    }
    ++pos;
    out.clear();
    for (;;) {
      skip_ws(pos);
      if (pos >= text_.size()) {
        return false;
      }
      if (text_[pos] == ']') {
        ++pos;
        return true;
      }
      if (text_[pos] != '[') {
        return false;
      }
      std::vector<float> row;
      if (!parse_number_array(pos, row)) {
        return false;
      }
      out.push_back(std::move(row));
      skip_ws(pos);
      if (pos < text_.size() && text_[pos] == ',') {
        ++pos;
      }
    }
  }

 private:
  std::size_t find_key(const std::string& key) const {
    return text_.find("\"" + key + "\"");
  }

  void skip_ws(std::size_t& pos) const {
    while (pos < text_.size() && std::isspace(static_cast<unsigned char>(text_[pos]))) {
      ++pos;
    }
  }

  bool parse_number_array(std::size_t& pos, std::vector<float>& out) const {
    skip_ws(pos);
    if (pos >= text_.size() || text_[pos] != '[') {
      return false;
    }
    ++pos;
    out.clear();
    for (;;) {
      skip_ws(pos);
      if (pos >= text_.size()) {
        return false;
      }
      if (text_[pos] == ']') {
        ++pos;
        return true;
      }
      const char* begin = text_.c_str() + pos;
      char* end = nullptr;
      float value = std::strtof(begin, &end);
      if (end == begin) {
        return false;
      }
      out.push_back(value);
      pos = static_cast<std::size_t>(end - text_.c_str());
      skip_ws(pos);
      if (pos < text_.size() && text_[pos] == ',') {
        ++pos;
      }
    }
  }

  bool parse_number_array(std::size_t& pos, std::vector<double>& out) const {
    skip_ws(pos);
    if (pos >= text_.size() || text_[pos] != '[') {
      return false;
    }
    ++pos;
    out.clear();
    for (;;) {
      skip_ws(pos);
      if (pos >= text_.size()) {
        return false;
      }
      if (text_[pos] == ']') {
        ++pos;
        return true;
      }
      const char* begin = text_.c_str() + pos;
      char* end = nullptr;
      double value = std::strtod(begin, &end);
      if (end == begin || !std::isfinite(value)) {
        return false;
      }
      out.push_back(value);
      pos = static_cast<std::size_t>(end - text_.c_str());
      skip_ws(pos);
      if (pos < text_.size() && text_[pos] == ',') {
        ++pos;
      }
    }
  }

  bool parse_string_array(std::size_t& pos, std::vector<std::string>& out) const {
    skip_ws(pos);
    if (pos >= text_.size() || text_[pos] != '[') {
      return false;
    }
    ++pos;
    out.clear();
    for (;;) {
      skip_ws(pos);
      if (pos >= text_.size()) {
        return false;
      }
      if (text_[pos] == ']') {
        ++pos;
        return true;
      }
      if (text_[pos] != '"') {
        return false;
      }
      ++pos;
      std::string value;
      while (pos < text_.size()) {
        char ch = text_[pos++];
        if (ch == '"') {
          break;
        }
        if (ch == '\\' && pos < text_.size()) {
          char escaped = text_[pos++];
          switch (escaped) {
            case '"': value.push_back('"'); break;
            case '\\': value.push_back('\\'); break;
            case '/': value.push_back('/'); break;
            case 'b': value.push_back('\b'); break;
            case 'f': value.push_back('\f'); break;
            case 'n': value.push_back('\n'); break;
            case 'r': value.push_back('\r'); break;
            case 't': value.push_back('\t'); break;
            default: value.push_back(escaped); break;
          }
        } else {
          value.push_back(ch);
        }
      }
      out.push_back(std::move(value));
      skip_ws(pos);
      if (pos < text_.size() && text_[pos] == ',') {
        ++pos;
      }
    }
  }

  std::string text_;
};

std::filesystem::path path_from_utf8(const std::string& text) {
#if defined(_WIN32)
  if (text.empty()) {
    return {};
  }
  int needed = MultiByteToWideChar(CP_UTF8, 0, text.data(), static_cast<int>(text.size()), nullptr, 0);
  if (needed <= 0) {
    return std::filesystem::path(text);
  }
  std::wstring wide(static_cast<std::size_t>(needed), L'\0');
  MultiByteToWideChar(CP_UTF8, 0, text.data(), static_cast<int>(text.size()), wide.data(), needed);
  return std::filesystem::path(wide);
#else
  return std::filesystem::path(text);
#endif
}

std::optional<FamilyClassifier> FamilyClassifier::load_from_json(const std::string& path) {
  std::ifstream input(path_from_utf8(path), std::ios::binary);
  if (!input) {
    return std::nullopt;
  }
  std::ostringstream buffer;
  buffer << input.rdbuf();
  JsonLite json(buffer.str());

  FamilyClassifier classifier;
  if (!json.int_array("cluster_ids", classifier.cluster_ids_)) {
    return std::nullopt;
  }
  if (!json.number_matrix("centroids", classifier.centroids_)) {
    return std::nullopt;
  }
  if (!json.number_array("thresholds", classifier.thresholds_)) {
    return std::nullopt;
  }
  if (!json.string_array("family_names", classifier.family_names_)) {
    return std::nullopt;
  }
  if (!json.number_array("scaler_mean", classifier.scaler_mean_)) {
    return std::nullopt;
  }
  if (!json.number_array("scaler_scale", classifier.scaler_scale_)) {
    return std::nullopt;
  }

  if (classifier.centroids_.empty()) {
    return std::nullopt;
  }
  std::size_t dim = classifier.centroids_[0].size();
  if (dim == 0 || classifier.scaler_mean_.size() != dim || classifier.scaler_scale_.size() != dim) {
    return std::nullopt;
  }
  for (const auto& row : classifier.centroids_) {
    if (row.size() != dim) {
      return std::nullopt;
    }
  }
  if (classifier.cluster_ids_.size() != classifier.centroids_.size() ||
      classifier.thresholds_.size() != classifier.centroids_.size() ||
      classifier.family_names_.size() != classifier.centroids_.size()) {
    return std::nullopt;
  }
  return classifier;
}

std::optional<FamilyPrediction> FamilyClassifier::predict(const std::vector<float>& features) const {
  if (centroids_.empty() || features.size() != scaler_mean_.size()) {
    return std::nullopt;
  }

  std::vector<float> scaled(features.size(), 0.0f);
  for (std::size_t i = 0; i < features.size(); ++i) {
    float denom = scaler_scale_[i];
    if (std::fabs(denom) < 1e-12f) {
      denom = 1.0f;
    }
    scaled[i] = (features[i] - scaler_mean_[i]) / denom;
  }

  std::size_t best_i = 0;
  float best_d2 = std::numeric_limits<float>::infinity();
  for (std::size_t i = 0; i < centroids_.size(); ++i) {
    float d2 = 0.0f;
    for (std::size_t j = 0; j < scaled.size(); ++j) {
      float delta = scaled[j] - centroids_[i][j];
      d2 += delta * delta;
    }
    if (d2 < best_d2) {
      best_d2 = d2;
      best_i = i;
    }
  }

  FamilyPrediction result;
  result.cluster_id = cluster_ids_[best_i];
  result.family_name = family_names_[best_i];
  result.distance = std::sqrt(best_d2);
  result.threshold = thresholds_[best_i];
  result.is_new_family = result.distance > result.threshold;
  return result;
}

std::optional<Stage2HgbModel> Stage2HgbModel::load_from_json(const std::string& path) {
  std::ifstream input(path_from_utf8(path), std::ios::binary);
  if (!input) {
    return std::nullopt;
  }
  std::ostringstream buffer;
  buffer << input.rdbuf();
  JsonLite json(buffer.str());

  Stage2HgbModel model;
  int n_features = 0;
  if (!json.int_value("n_features", n_features) || n_features <= 0) {
    return std::nullopt;
  }
  model.n_features_ = static_cast<std::size_t>(n_features);
  if (!json.number_value("baseline_prediction", model.baseline_prediction_)) {
    return std::nullopt;
  }
  float threshold = 0.5f;
  if (json.number_value("threshold", threshold) && threshold > 0.0f && threshold < 1.0f) {
    model.threshold_ = threshold;
  }
  if (!json.int_array("tree_node_offsets", model.tree_offsets_) ||
      !json.number_array("node_values", model.node_values_) ||
      !json.int_array("node_feature_idx", model.node_feature_idx_) ||
      !json.number_array("node_num_thresholds", model.node_num_thresholds_) ||
      !json.int_array("node_missing_go_to_left", model.node_missing_go_to_left_) ||
      !json.int_array("node_left", model.node_left_) ||
      !json.int_array("node_right", model.node_right_) ||
      !json.int_array("node_is_leaf", model.node_is_leaf_)) {
    return std::nullopt;
  }
  std::size_t node_count = model.node_values_.size();
  if (model.tree_offsets_.size() < 2 || node_count == 0) {
    return std::nullopt;
  }
  if (model.node_feature_idx_.size() != node_count ||
      model.node_num_thresholds_.size() != node_count ||
      model.node_missing_go_to_left_.size() != node_count ||
      model.node_left_.size() != node_count ||
      model.node_right_.size() != node_count ||
      model.node_is_leaf_.size() != node_count) {
    return std::nullopt;
  }
  if (model.tree_offsets_.front() != 0 ||
      model.tree_offsets_.back() != static_cast<int>(node_count)) {
    return std::nullopt;
  }
  for (std::size_t i = 1; i < model.tree_offsets_.size(); ++i) {
    if (model.tree_offsets_[i] < model.tree_offsets_[i - 1]) {
      return std::nullopt;
    }
  }
  return model;
}

float Stage2HgbModel::predict_probability(const std::vector<float>& features) const {
  if (features.size() != n_features_) {
    return 0.0f;
  }
  double score = baseline_prediction_;
  for (std::size_t tree_index = 0; tree_index + 1 < tree_offsets_.size(); ++tree_index) {
    int start = tree_offsets_[tree_index];
    int stop = tree_offsets_[tree_index + 1];
    if (start < 0 || stop <= start || static_cast<std::size_t>(stop) > node_values_.size()) {
      continue;
    }
    int node = start;
    for (int guard = 0; guard < 4096; ++guard) {
      if (node < start || node >= stop) {
        break;
      }
      std::size_t idx = static_cast<std::size_t>(node);
      if (node_is_leaf_[idx]) {
        score += node_values_[idx];
        break;
      }
      int feature_idx = node_feature_idx_[idx];
      if (feature_idx < 0 || static_cast<std::size_t>(feature_idx) >= features.size()) {
        break;
      }
      float value = features[static_cast<std::size_t>(feature_idx)];
      bool go_left = false;
      if (std::isnan(value)) {
        go_left = node_missing_go_to_left_[idx] != 0;
      } else {
        go_left = static_cast<double>(value) <= node_num_thresholds_[idx];
      }
      int child_relative = go_left ? node_left_[idx] : node_right_[idx];
      node = start + child_relative;
    }
  }
  score = std::max(-50.0, std::min(50.0, score));
  return static_cast<float>(1.0 / (1.0 + std::exp(-score)));
}

int write_string_out(const std::string& text, char** out_json, size_t* out_len) {
  if (!out_json || !out_len) {
    return -1;
  }
  char* buffer = static_cast<char*>(std::malloc(text.size() + 1));
  if (!buffer) {
    return -2;
  }
  std::memcpy(buffer, text.data(), text.size());
  buffer[text.size()] = '\0';
  *out_json = buffer;
  *out_len = text.size();
  return 0;
}

int write_literal_out_noexcept(
    const char* text,
    std::size_t text_len,
    char** out_json,
    size_t* out_len) noexcept {
  if (!text || !out_json || !out_len) {
    return -1;
  }
  char* buffer = static_cast<char*>(std::malloc(text_len + 1));
  if (!buffer) {
    return -2;
  }
  std::memcpy(buffer, text, text_len);
  buffer[text_len] = '\0';
  *out_json = buffer;
  *out_len = text_len;
  return 0;
}

std::string error_json(const std::string& code, const std::string& message) {
  std::ostringstream out;
  out << "{\"ok\":false,\"error_code\":\"" << json_escape(code)
      << "\",\"error\":\"" << json_escape(message) << "\"}";
  return out.str();
}

int write_error(const std::string& code, const std::string& message, char** out_json, size_t* out_len) {
  return write_string_out(error_json(code, message), out_json, out_len);
}

void append_u32_le(std::vector<std::uint8_t>& out, std::uint32_t value) {
  for (unsigned int shift = 0; shift < 32; shift += 8) {
    out.push_back(static_cast<std::uint8_t>((value >> shift) & 0xffu));
  }
}

void append_u64_le(std::vector<std::uint8_t>& out, std::uint64_t value) {
  for (unsigned int shift = 0; shift < 64; shift += 8) {
    out.push_back(static_cast<std::uint8_t>((value >> shift) & 0xffu));
  }
}

class LocalHmacSha256 {
 public:
  LocalHmacSha256(const unsigned char* key, std::size_t key_len)
      : key_(key), key_len_(key_len) {
#if defined(_WIN32)
    if (!key_ || key_len_ != 32 || key_len_ > std::numeric_limits<ULONG>::max()) {
      return;
    }
    if (!BCRYPT_SUCCESS(BCryptOpenAlgorithmProvider(
            &algorithm_, BCRYPT_SHA256_ALGORITHM, nullptr, BCRYPT_ALG_HANDLE_HMAC_FLAG))) {
      return;
    }
    ULONG copied = 0;
    ULONG hash_length = 0;
    if (!BCRYPT_SUCCESS(BCryptGetProperty(
            algorithm_,
            BCRYPT_OBJECT_LENGTH,
            reinterpret_cast<PUCHAR>(&object_length_),
            sizeof(object_length_),
            &copied,
            0)) ||
        copied != sizeof(object_length_) ||
        !BCRYPT_SUCCESS(BCryptGetProperty(
            algorithm_,
            BCRYPT_HASH_LENGTH,
            reinterpret_cast<PUCHAR>(&hash_length),
            sizeof(hash_length),
            &copied,
            0)) ||
        copied != sizeof(hash_length) || hash_length != 32) {
      BCryptCloseAlgorithmProvider(algorithm_, 0);
      algorithm_ = nullptr;
      return;
    }
    valid_ = true;
#endif
  }

  ~LocalHmacSha256() {
#if defined(_WIN32)
    if (algorithm_) {
      BCryptCloseAlgorithmProvider(algorithm_, 0);
    }
#endif
  }

  LocalHmacSha256(const LocalHmacSha256&) = delete;
  LocalHmacSha256& operator=(const LocalHmacSha256&) = delete;

  bool valid() const {
    return valid_;
  }

  bool digest(
      const std::vector<std::uint8_t>& message,
      std::array<std::uint8_t, 32>& out) const {
#if defined(_WIN32)
    if (!valid_ || message.size() > std::numeric_limits<ULONG>::max()) {
      return false;
    }
    std::vector<unsigned char> hash_object(object_length_);
    BCRYPT_HASH_HANDLE hash = nullptr;
    NTSTATUS status = BCryptCreateHash(
        algorithm_,
        &hash,
        hash_object.empty() ? nullptr : hash_object.data(),
        object_length_,
        const_cast<PUCHAR>(key_),
        static_cast<ULONG>(key_len_),
        0);
    if (BCRYPT_SUCCESS(status)) {
      status = BCryptHashData(
          hash,
          message.empty() ? nullptr : const_cast<PUCHAR>(message.data()),
          static_cast<ULONG>(message.size()),
          0);
    }
    if (BCRYPT_SUCCESS(status)) {
      status = BCryptFinishHash(hash, out.data(), static_cast<ULONG>(out.size()), 0);
    }
    if (hash) {
      BCryptDestroyHash(hash);
    }
    if (!hash_object.empty()) {
      SecureZeroMemory(hash_object.data(), hash_object.size());
    }
    return BCRYPT_SUCCESS(status);
#else
    (void)message;
    (void)out;
    return false;
#endif
  }

 private:
  const unsigned char* key_ = nullptr;
  std::size_t key_len_ = 0;
  bool valid_ = false;
#if defined(_WIN32)
  BCRYPT_ALG_HANDLE algorithm_ = nullptr;
  ULONG object_length_ = 0;
#endif
};

struct DiagnosticTensor {
  std::uint64_t component;
  const char* name;
  const char* dtype;
  std::size_t element_count;
  const void* data;
  bool is_i64;
};

std::string hex_digest(const std::array<std::uint8_t, 32>& digest) {
  static constexpr char kHex[] = "0123456789abcdef";
  std::string out;
  out.resize(digest.size() * 2);
  for (std::size_t index = 0; index < digest.size(); ++index) {
    out[index * 2] = kHex[digest[index] >> 4];
    out[index * 2 + 1] = kHex[digest[index] & 0x0f];
  }
  return out;
}

bool tensor_hmac_digest(
    const LocalHmacSha256& hmac,
    const DiagnosticTensor& tensor,
    std::size_t start,
    std::size_t count,
    std::string& out_digest) {
  if (start > tensor.element_count || count > tensor.element_count - start) {
    return false;
  }
  static constexpr char kCanonicalPrefix[] = "axon_tensor_le_v1";
  const std::size_t element_size = tensor.is_i64 ? sizeof(std::int64_t) : sizeof(float);
  std::vector<std::uint8_t> message;
  // Build the cross-language message explicitly in little-endian order.
  message.reserve(
      sizeof(kCanonicalPrefix) + std::strlen(tensor.name) + std::strlen(tensor.dtype) + 2 +
      3 * sizeof(std::uint64_t) + count * element_size);
  message.insert(
      message.end(),
      reinterpret_cast<const std::uint8_t*>(kCanonicalPrefix),
      reinterpret_cast<const std::uint8_t*>(kCanonicalPrefix) + sizeof(kCanonicalPrefix));
  message.insert(message.end(), tensor.name, tensor.name + std::strlen(tensor.name));
  message.push_back(0);
  message.insert(message.end(), tensor.dtype, tensor.dtype + std::strlen(tensor.dtype));
  message.push_back(0);
  append_u64_le(message, static_cast<std::uint64_t>(tensor.element_count));
  append_u64_le(message, static_cast<std::uint64_t>(start));
  append_u64_le(message, static_cast<std::uint64_t>(count));
  if (tensor.is_i64) {
    const auto* values = static_cast<const std::int64_t*>(tensor.data);
    for (std::size_t index = start; index < start + count; ++index) {
      append_u64_le(message, static_cast<std::uint64_t>(values[index]));
    }
  } else {
    static_assert(sizeof(float) == sizeof(std::uint32_t), "parity diagnostics require binary32 floats");
    const auto* values = static_cast<const float*>(tensor.data);
    for (std::size_t index = start; index < start + count; ++index) {
      std::uint32_t bits = 0;
      std::memcpy(&bits, &values[index], sizeof(bits));
      append_u32_le(message, bits);
    }
  }
  std::array<std::uint8_t, 32> digest{};
  if (!hmac.digest(message, digest)) {
    return false;
  }
  out_digest = hex_digest(digest);
  return true;
}

std::string prediction_json(
    const Prediction& p,
    const std::string& path,
    float threshold,
    const std::optional<FamilyPrediction>& family = std::nullopt) {
  if (!p.ok) {
    return error_json(p.error_code, p.error);
  }
  std::ostringstream out;
  out.setf(std::ios::fixed);
  out.precision(8);
  out << "{\"ok\":true"
      << ",\"mode\":\"single_pe\""
      << ",\"file\":\"" << json_escape(path) << "\""
      << ",\"is_malware\":" << (p.prediction == 1 ? "true" : "false")
      << ",\"confidence\":" << p.confidence
      << ",\"axon_malware\":" << (p.prediction == 1 ? "true" : "false")
      << ",\"axon_score\":" << p.prob_malicious
      << ",\"prediction\":" << p.prediction
      << ",\"label\":\"" << (p.prediction == 1 ? "malicious" : "benign") << "\""
      << ",\"prob_benign\":" << p.prob_benign
      << ",\"prob_malicious\":" << p.prob_malicious
      << ",\"threshold\":" << threshold
      << ",\"original_length\":" << p.original_length;
  if (p.stage2_enabled) {
    out << ",\"base_model\":{"
        << "\"prediction\":" << p.base_prediction
        << ",\"confidence\":" << p.base_confidence
        << ",\"prob_benign\":" << p.base_prob_benign
        << ",\"prob_malicious\":" << p.base_prob_malicious
        << "}"
        << ",\"stage2\":{"
        << "\"enabled\":true"
        << ",\"threshold\":" << p.stage2_threshold
        << ",\"feature_dim\":" << p.stage2_feature_dim
        << ",\"prob_malicious\":" << p.prob_malicious
        << "}";
  }
  if (family) {
    out << ",\"malware_family\":{"
        << "\"family_name\":\"" << json_escape(family->family_name) << "\""
        << ",\"cluster_id\":" << family->cluster_id
        << ",\"is_new_family\":" << (family->is_new_family ? "true" : "false")
        << ",\"distance\":" << family->distance
        << ",\"threshold\":" << family->threshold
        << "}";
  }
  out
      << "}";
  return out.str();
}

std::string bytes_prediction_json(const Prediction& p, float threshold) {
  return prediction_json(p, "<bytes>", threshold);
}

struct ArchivePeTarget {
  std::string logical_path;
  std::string extracted_path;
  std::string sha256;
};

bool read_file_bytes(const std::string& path, std::vector<std::uint8_t>& out, std::string& error);
bool read_file_bytes_limited(
    const std::string& path,
    std::vector<std::uint8_t>& out,
    std::string& error,
    std::uint64_t max_size);
InferenceInput make_inference_input(const std::vector<std::uint8_t>& file_bytes);

std::string nested_archive_prediction_json(
    kvd_handle* handle,
    const std::string& path_text,
    const std::string& report_json,
    const std::vector<std::pair<ArchivePeTarget, Prediction>>& predictions) {
  int malicious_count = 0;
  for (const auto& item : predictions) {
    if (item.second.ok && item.second.prediction == 1) {
      ++malicious_count;
    }
  }
  std::ostringstream out;
  out.setf(std::ios::fixed);
  out.precision(8);
  out << "{\"ok\":true"
      << ",\"mode\":\"nested_archive\""
      << ",\"file\":\"" << json_escape(path_text) << "\""
      << ",\"parent_verdict\":\"" << (malicious_count > 0 ? "malicious" : "benign_or_no_malicious_inner_pe") << "\""
      << ",\"runtime_rule\":\"any malicious inner PE triggers parent alert\""
      << ",\"training_label_policy\":\"unknown_training_label: do not inherit parent archive/MSI label\""
      << ",\"pe_prediction_count\":" << predictions.size()
      << ",\"malicious_inner_count\":" << malicious_count
      << ",\"predictions_truncated\":"
      << (predictions.size() > kNestedPredictionResponseLimit ? "true" : "false")
      << ",\"archive_report\":";
  if (report_json.empty()) {
    out << "null";
  } else if (report_json.size() > kNestedArchiveReportResponseLimit) {
    out << "{\"truncated\":true"
        << ",\"original_chars\":" << report_json.size()
        << ",\"limit_chars\":" << kNestedArchiveReportResponseLimit
        << "}";
  } else {
    out << report_json;
  }
  out << ",\"predictions\":[";
  const std::size_t visible_count = std::min<std::size_t>(predictions.size(), kNestedPredictionResponseLimit);
  for (std::size_t i = 0; i < visible_count; ++i) {
    if (i > 0) {
      out << ",";
    }
    const auto& target = predictions[i].first;
    const auto& p = predictions[i].second;
    out << "{\"logical_path\":\"" << json_escape(target.logical_path) << "\""
        << ",\"sha256\":\"" << json_escape(target.sha256) << "\"";
    if (!p.ok) {
      out << ",\"status\":\"prediction_failed\""
          << ",\"error_code\":\"" << json_escape(p.error_code) << "\""
          << ",\"error\":\"" << json_escape(p.error) << "\"";
    } else {
      out << ",\"status\":\"predicted\""
          << ",\"prediction\":" << p.prediction
          << ",\"label\":\"" << (p.prediction == 1 ? "malicious" : "benign") << "\""
          << ",\"confidence\":" << p.confidence
          << ",\"prob_benign\":" << p.prob_benign
          << ",\"prob_malicious\":" << p.prob_malicious
          << ",\"original_length\":" << p.original_length;
      if (p.stage2_enabled) {
        out << ",\"base_model\":{"
            << "\"prediction\":" << p.base_prediction
            << ",\"confidence\":" << p.base_confidence
            << ",\"prob_benign\":" << p.base_prob_benign
            << ",\"prob_malicious\":" << p.base_prob_malicious
            << "}"
            << ",\"stage2\":{"
            << "\"enabled\":true"
            << ",\"threshold\":" << p.stage2_threshold
            << ",\"feature_dim\":" << p.stage2_feature_dim
            << ",\"prob_malicious\":" << p.prob_malicious
            << "}";
      }
      if (p.prediction == 1 && handle->family_classifier) {
        std::vector<std::uint8_t> bytes;
        std::string error;
        if (read_file_bytes_limited(target.extracted_path, bytes, error, handle->config.max_file_size)) {
          InferenceInput input = make_inference_input(bytes);
          std::vector<float> family_features;
          family_features.reserve(kAxonPeFeatureDim + kAxonStatFeatureDim);
          family_features.insert(family_features.end(), input.pe_features.begin(), input.pe_features.end());
          family_features.insert(family_features.end(), input.stat_features.begin(), input.stat_features.end());
          auto family = handle->family_classifier->predict(family_features);
          if (family) {
            out << ",\"malware_family\":{"
                << "\"family_name\":\"" << json_escape(family->family_name) << "\""
                << ",\"cluster_id\":" << family->cluster_id
                << ",\"is_new_family\":" << (family->is_new_family ? "true" : "false")
                << ",\"distance\":" << family->distance
                << ",\"threshold\":" << family->threshold
                << "}";
          }
        }
      }
    }
    out << "}";
  }
  out << "]}";
  return out.str();
}

double safe_ratio_f64(double numerator, double denominator) {
  return numerator / std::max(denominator, 1.0);
}

float safe_ratio(double numerator, double denominator) {
  return static_cast<float>(safe_ratio_f64(numerator, denominator));
}

float safe_log1p_norm(double value, double denom) {
  return static_cast<float>(std::log1p(std::max(value, 0.0)) / denom);
}

bool read_file_bytes_limited(
    const std::string& path,
    std::vector<std::uint8_t>& out,
    std::string& error,
    std::uint64_t max_size) {
  std::ifstream input(path_from_utf8(path), std::ios::binary);
  if (!input) {
    error = "Failed to open file.";
    return false;
  }
  input.seekg(0, std::ios::end);
  std::streamoff size = input.tellg();
  if (size < 0) {
    error = "Failed to get file size.";
    return false;
  }
  auto file_size = static_cast<std::uint64_t>(size);
  if (max_size > 0 && file_size > max_size) {
    error = "file_too_large";
    return false;
  }
  input.seekg(0, std::ios::beg);
  out.resize(static_cast<std::size_t>(file_size));
  if (!out.empty()) {
    input.read(reinterpret_cast<char*>(out.data()), static_cast<std::streamsize>(file_size));
    if (!input) {
      error = "Failed to read full file.";
      return false;
    }
  }
  return true;
}

bool read_file_bytes(const std::string& path, std::vector<std::uint8_t>& out, std::string& error) {
  return read_file_bytes_limited(path, out, error, 0);
}

std::string quote_cmd_arg(const std::string& text) {
  std::string out = "\"";
  for (char ch : text) {
    if (ch == '"') {
      out += "\\\"";
    } else {
      out += ch;
    }
  }
  out += "\"";
  return out;
}

std::string normalize_path_string(const std::filesystem::path& path);
bool json_string_field(const std::string& text, const std::string& key, std::string& out);

bool path_components_equal(
    const std::filesystem::path& left,
    const std::filesystem::path& right) {
#if defined(_WIN32)
  const std::wstring left_text = left.native();
  const std::wstring right_text = right.native();
  return CompareStringOrdinal(
             left_text.c_str(),
             -1,
             right_text.c_str(),
             -1,
             TRUE) == CSTR_EQUAL;
#else
  return left == right;
#endif
}

bool path_inside_normalized_root(
    const std::filesystem::path& path,
    const std::filesystem::path& root) {
  auto path_component = path.begin();
  for (auto root_component = root.begin(); root_component != root.end(); ++root_component) {
    if (path_component == path.end() ||
        !path_components_equal(*path_component, *root_component)) {
      return false;
    }
    ++path_component;
  }
  return true;
}

std::optional<std::filesystem::path> absolute_lexically_normal(
    const std::filesystem::path& path) {
  std::error_code ec;
  auto absolute_path = std::filesystem::absolute(path, ec);
  if (ec) {
    return std::nullopt;
  }
  return absolute_path.lexically_normal();
}

std::string create_archive_scanner_temp_root(std::string& error) {
  std::error_code ec;
  std::filesystem::path base = std::filesystem::temp_directory_path(ec);
  if (ec) {
    error = "failed to resolve temp directory";
    return {};
  }
#if defined(_WIN32)
  unsigned long pid = static_cast<unsigned long>(GetCurrentProcessId());
#else
  unsigned long pid = 0;
#endif
  auto ticks = std::chrono::steady_clock::now().time_since_epoch().count();
  for (int attempt = 0; attempt < 16; ++attempt) {
    std::filesystem::path root = base / (
        "axon-archive-scanner-root-" + std::to_string(pid) + "-" +
        std::to_string(ticks) + "-" + std::to_string(attempt));
    if (std::filesystem::create_directory(root, ec)) {
      return normalize_path_string(root);
    }
    if (ec) {
      ec.clear();
    }
  }
  error = "failed to create archive scanner temp root";
  return {};
}

bool path_inside_root(const std::filesystem::path& path, const std::filesystem::path& root) {
  std::error_code ec;
  auto resolved_path = std::filesystem::weakly_canonical(path, ec);
  if (ec) {
    return false;
  }
  auto resolved_root = std::filesystem::weakly_canonical(root, ec);
  if (ec) {
    return false;
  }
  return path_inside_normalized_root(resolved_path, resolved_root);
}

void cleanup_archive_scan_temp(const std::string& report_json, const std::string& scanner_temp_root) {
  std::error_code cleanup_ec;
  std::filesystem::path root = path_from_utf8(scanner_temp_root);
  std::string temp_dir;
  if (json_string_field(report_json, "temp_dir", temp_dir) && !temp_dir.empty()) {
    std::filesystem::path temp_path = path_from_utf8(temp_dir);
    std::string name = temp_path.filename().string();
    if (name.rfind("axon-archive-scanner-", 0) == 0 && path_inside_root(temp_path, root)) {
      std::filesystem::remove_all(temp_path, cleanup_ec);
    }
  }
  if (!scanner_temp_root.empty()) {
    std::filesystem::remove_all(root, cleanup_ec);
  }
}

bool run_process_capture_stdout(const std::string& command, std::string& stdout_text, std::string& error) {
#if defined(_WIN32)
  auto close_if_valid = [](HANDLE handle) {
    if (handle && handle != INVALID_HANDLE_VALUE) {
      CloseHandle(handle);
    }
  };
  SECURITY_ATTRIBUTES sa{};
  sa.nLength = sizeof(sa);
  sa.bInheritHandle = TRUE;
  sa.lpSecurityDescriptor = nullptr;

  HANDLE read_pipe = nullptr;
  HANDLE write_pipe = nullptr;
  if (!CreatePipe(&read_pipe, &write_pipe, &sa, 0)) {
    error = "failed to create stdout pipe";
    return false;
  }
  SetHandleInformation(read_pipe, HANDLE_FLAG_INHERIT, 0);

  std::wstring wide_command;
  int needed = MultiByteToWideChar(CP_UTF8, 0, command.data(), static_cast<int>(command.size()), nullptr, 0);
  if (needed > 0) {
    wide_command.resize(static_cast<std::size_t>(needed));
    MultiByteToWideChar(CP_UTF8, 0, command.data(), static_cast<int>(command.size()), wide_command.data(), needed);
  } else {
    wide_command.assign(command.begin(), command.end());
  }

  STARTUPINFOW si{};
  si.cb = sizeof(si);
  si.dwFlags = STARTF_USESTDHANDLES;
  si.hStdOutput = write_pipe;
  si.hStdError = write_pipe;
  si.hStdInput = GetStdHandle(STD_INPUT_HANDLE);
  PROCESS_INFORMATION pi{};

  BOOL ok = CreateProcessW(
      nullptr,
      wide_command.data(),
      nullptr,
      nullptr,
      TRUE,
      CREATE_NO_WINDOW,
      nullptr,
      nullptr,
      &si,
      &pi);
  CloseHandle(write_pipe);
  if (!ok) {
    CloseHandle(read_pipe);
    error = "failed to start process";
    return false;
  }

  auto fail_process = [&](const std::string& message) {
    TerminateProcess(pi.hProcess, 1);
    WaitForSingleObject(pi.hProcess, 5000);
    close_if_valid(read_pipe);
    close_if_valid(pi.hThread);
    close_if_valid(pi.hProcess);
    error = message;
    return false;
  };

  char buffer[4096];
  ULONGLONG started = GetTickCount64();
  bool process_done = false;
  while (!process_done) {
    DWORD available = 0;
    if (!PeekNamedPipe(read_pipe, nullptr, 0, nullptr, &available, nullptr)) {
      break;
    }
    while (available > 0) {
      DWORD to_read = std::min<DWORD>(available, static_cast<DWORD>(sizeof(buffer)));
      DWORD read = 0;
      if (!ReadFile(read_pipe, buffer, to_read, &read, nullptr) || read == 0) {
        break;
      }
      if (stdout_text.size() + read > kProcessOutputLimitBytes) {
        return fail_process("process output exceeded capture limit");
      }
      stdout_text.append(buffer, buffer + read);
      available -= read;
    }
    DWORD wait_rc = WaitForSingleObject(pi.hProcess, 25);
    if (wait_rc == WAIT_OBJECT_0) {
      process_done = true;
    } else if (wait_rc == WAIT_TIMEOUT) {
      ULONGLONG elapsed = GetTickCount64() - started;
      if (elapsed > static_cast<ULONGLONG>(kProcessTimeoutMs)) {
        return fail_process("process timed out");
      }
    } else {
      return fail_process("failed while waiting for process");
    }
  }
  for (;;) {
    DWORD available = 0;
    if (!PeekNamedPipe(read_pipe, nullptr, 0, nullptr, &available, nullptr) || available == 0) {
      break;
    }
    DWORD to_read = std::min<DWORD>(available, static_cast<DWORD>(sizeof(buffer)));
    DWORD read = 0;
    if (!ReadFile(read_pipe, buffer, to_read, &read, nullptr) || read == 0) {
      break;
    }
    if (stdout_text.size() + read > kProcessOutputLimitBytes) {
      return fail_process("process output exceeded capture limit");
    }
    stdout_text.append(buffer, buffer + read);
  }
  CloseHandle(read_pipe);
  DWORD exit_code = 1;
  GetExitCodeProcess(pi.hProcess, &exit_code);
  CloseHandle(pi.hThread);
  CloseHandle(pi.hProcess);
  if (exit_code != 0) {
    error = "process returned non-zero exit code: " + stdout_text;
    return false;
  }
  return true;
#else
  FILE* pipe = popen(command.c_str(), "r");
  if (!pipe) {
    error = "failed to start process";
    return false;
  }
  char buffer[4096];
  while (std::fgets(buffer, sizeof(buffer), pipe)) {
    if (stdout_text.size() + std::strlen(buffer) > kProcessOutputLimitBytes) {
#if defined(_WIN32)
      _pclose(pipe);
#else
      pclose(pipe);
#endif
      error = "process output exceeded capture limit";
      return false;
    }
    stdout_text += buffer;
  }
#if defined(_WIN32)
  int rc = _pclose(pipe);
#else
  int rc = pclose(pipe);
#endif
  if (rc != 0) {
    error = "process returned non-zero exit code";
    return false;
  }
  return true;
#endif
}

bool run_archive_scan_json(
    const std::string& scanner_path,
    const std::string& input_path,
    std::string& report_json,
    std::string& scanner_temp_root,
    std::string& error) {
  if (scanner_path.empty()) {
    error = "archive_scanner_path is required when scan_nested is enabled";
    return false;
  }
  std::error_code ec;
  if (!std::filesystem::exists(path_from_utf8(scanner_path), ec) || ec) {
    error = "archive scanner executable was not found";
    return false;
  }
  scanner_temp_root = create_archive_scanner_temp_root(error);
  if (scanner_temp_root.empty()) {
    return false;
  }
  std::ostringstream command;
  command << quote_cmd_arg(scanner_path)
          << " --input " << quote_cmd_arg(input_path)
          << " --output json"
          << " --max-depth 4"
          << " --max-files 4096"
          << " --max-total-bytes " << (512ull * 1024ull * 1024ull)
          << " --max-file-bytes " << (128ull * 1024ull * 1024ull)
          << " --temp-root " << quote_cmd_arg(scanner_temp_root)
          << " --keep-temp";
  std::string captured;
  if (!run_process_capture_stdout(command.str(), captured, error)) {
    cleanup_archive_scan_temp("", scanner_temp_root);
    return false;
  }
  std::size_t start = captured.find('{');
  if (start == std::string::npos) {
    error = "archive scanner returned no JSON object";
    cleanup_archive_scan_temp("", scanner_temp_root);
    return false;
  }
  bool in_string = false;
  bool escape = false;
  int depth = 0;
  for (std::size_t i = start; i < captured.size(); ++i) {
    char ch = captured[i];
    if (in_string) {
      if (escape) {
        escape = false;
      } else if (ch == '\\') {
        escape = true;
      } else if (ch == '"') {
        in_string = false;
      }
      continue;
    }
    if (ch == '"') {
      in_string = true;
      continue;
    }
    if (ch == '{') {
      ++depth;
    } else if (ch == '}') {
      --depth;
      if (depth == 0) {
        report_json = captured.substr(start, i - start + 1);
        return true;
      }
    }
  }
  error = "archive scanner returned incomplete JSON";
  cleanup_archive_scan_temp("", scanner_temp_root);
  return false;
}

bool json_string_field(const std::string& text, const std::string& key, std::string& out) {
  std::size_t pos = text.find("\"" + key + "\"");
  if (pos == std::string::npos) {
    return false;
  }
  pos = text.find(':', pos);
  if (pos == std::string::npos) {
    return false;
  }
  ++pos;
  while (pos < text.size() && std::isspace(static_cast<unsigned char>(text[pos]))) {
    ++pos;
  }
  if (pos >= text.size()) {
    return false;
  }
  if (text.compare(pos, 4, "null") == 0) {
    out.clear();
    return true;
  }
  if (text[pos] != '"') {
    return false;
  }
  ++pos;
  std::string value;
  while (pos < text.size()) {
    char ch = text[pos++];
    if (ch == '"') {
      out = std::move(value);
      return true;
    }
    if (ch == '\\' && pos < text.size()) {
      char escaped = text[pos++];
      switch (escaped) {
        case '"': value.push_back('"'); break;
        case '\\': value.push_back('\\'); break;
        case '/': value.push_back('/'); break;
        case 'b': value.push_back('\b'); break;
        case 'f': value.push_back('\f'); break;
        case 'n': value.push_back('\n'); break;
        case 'r': value.push_back('\r'); break;
        case 't': value.push_back('\t'); break;
        default: value.push_back(escaped); break;
      }
    } else {
      value.push_back(ch);
    }
  }
  return false;
}

bool json_bool_field(const std::string& text, const std::string& key, bool& out) {
  std::size_t pos = text.find("\"" + key + "\"");
  if (pos == std::string::npos) {
    return false;
  }
  pos = text.find(':', pos);
  if (pos == std::string::npos) {
    return false;
  }
  ++pos;
  while (pos < text.size() && std::isspace(static_cast<unsigned char>(text[pos]))) {
    ++pos;
  }
  if (text.compare(pos, 4, "true") == 0) {
    out = true;
    return true;
  }
  if (text.compare(pos, 5, "false") == 0) {
    out = false;
    return true;
  }
  return false;
}

std::vector<std::string> json_entry_objects(const std::string& report_json) {
  std::vector<std::string> objects;
  std::size_t pos = report_json.find("\"entries\"");
  if (pos == std::string::npos) {
    return objects;
  }
  pos = report_json.find('[', pos);
  if (pos == std::string::npos) {
    return objects;
  }
  bool in_string = false;
  bool escape = false;
  int depth = 0;
  std::size_t object_start = std::string::npos;
  for (; pos < report_json.size(); ++pos) {
    char ch = report_json[pos];
    if (in_string) {
      if (escape) {
        escape = false;
      } else if (ch == '\\') {
        escape = true;
      } else if (ch == '"') {
        in_string = false;
      }
      continue;
    }
    if (ch == '"') {
      in_string = true;
      continue;
    }
    if (ch == '{') {
      if (depth == 0) {
        object_start = pos;
      }
      ++depth;
    } else if (ch == '}') {
      --depth;
      if (depth == 0 && object_start != std::string::npos) {
        objects.push_back(report_json.substr(object_start, pos - object_start + 1));
        object_start = std::string::npos;
      }
    } else if (ch == ']' && depth == 0) {
      break;
    }
  }
  return objects;
}

std::vector<ArchivePeTarget> archive_pe_targets(const std::string& report_json) {
  std::vector<ArchivePeTarget> targets;
  for (const auto& object : json_entry_objects(report_json)) {
    std::string kind;
    std::string status;
    std::string extracted_path;
    std::string logical_path;
    std::string sha256;
    bool candidate = false;
    json_string_field(object, "kind", kind);
    json_string_field(object, "status", status);
    json_string_field(object, "extracted_path", extracted_path);
    json_string_field(object, "logical_path", logical_path);
    json_string_field(object, "sha256", sha256);
    json_bool_field(object, "candidate_for_axon", candidate);
    if (kind == "pe" && candidate && !extracted_path.empty() &&
        (status == "candidate" || status == "scanned")) {
      targets.push_back({logical_path, extracted_path, sha256});
    }
  }
  return targets;
}

std::string normalize_path_string(const std::filesystem::path& path) {
  auto u8 = path.u8string();
  return std::string(reinterpret_cast<const char*>(u8.data()), u8.size());
}

bool path_allowed(const std::string& path, const std::string& allowed_root) {
  if (allowed_root.empty()) {
    return true;
  }
  auto input_path = path_from_utf8(path);
  auto input_root = path_from_utf8(allowed_root);
  auto lexical_path = absolute_lexically_normal(input_path);
  auto lexical_root = absolute_lexically_normal(input_root);
  if (!lexical_path || !lexical_root) {
    return false;
  }
  std::error_code ec;
  auto full_path = std::filesystem::weakly_canonical(*lexical_path, ec);
  if (ec) {
    return false;
  }
  auto root_path = std::filesystem::weakly_canonical(*lexical_root, ec);
  if (ec) {
    return false;
  }
  return path_inside_normalized_root(full_path, root_path);
}

float numpy_pairwise_sum_f32(const float* values, std::size_t size) {
  if (size < 8) {
    float result = -0.0f;
    for (std::size_t index = 0; index < size; ++index) {
      result += values[index];
    }
    return result;
  }
  if (size <= 128) {
    float partial[8] = {
        values[0], values[1], values[2], values[3],
        values[4], values[5], values[6], values[7]};
    std::size_t index = 8;
    const std::size_t unrolled_end = size - (size % 8);
    for (; index < unrolled_end; index += 8) {
      for (std::size_t lane = 0; lane < 8; ++lane) {
        partial[lane] += values[index + lane];
      }
    }
    float result = ((partial[0] + partial[1]) + (partial[2] + partial[3])) +
                   ((partial[4] + partial[5]) + (partial[6] + partial[7]));
    for (; index < size; ++index) {
      result += values[index];
    }
    return result;
  }
  std::size_t left_size = size / 2;
  left_size -= left_size % 8;
  return numpy_pairwise_sum_f32(values, left_size) +
         numpy_pairwise_sum_f32(values + left_size, size - left_size);
}

double numpy_pairwise_sum_f64(const double* values, std::size_t size) {
  if (size < 8) {
    double result = -0.0;
    for (std::size_t index = 0; index < size; ++index) {
      result += values[index];
    }
    return result;
  }
  if (size <= 128) {
    double partial[8] = {
        values[0], values[1], values[2], values[3],
        values[4], values[5], values[6], values[7]};
    std::size_t index = 8;
    const std::size_t unrolled_end = size - (size % 8);
    for (; index < unrolled_end; index += 8) {
      for (std::size_t lane = 0; lane < 8; ++lane) {
        partial[lane] += values[index + lane];
      }
    }
    double result = ((partial[0] + partial[1]) + (partial[2] + partial[3])) +
                    ((partial[4] + partial[5]) + (partial[6] + partial[7]));
    for (; index < size; ++index) {
      result += values[index];
    }
    return result;
  }
  std::size_t left_size = size / 2;
  left_size -= left_size % 8;
  return numpy_pairwise_sum_f64(values, left_size) +
         numpy_pairwise_sum_f64(values + left_size, size - left_size);
}

double numpy_mean_f64(const double* values, std::size_t size) {
  return size == 0 ? 0.0 : numpy_pairwise_sum_f64(values, size) / static_cast<double>(size);
}

double numpy_std_f64(const double* values, std::size_t size) {
  if (size == 0) {
    return 0.0;
  }
  double mean = numpy_mean_f64(values, size);
  std::vector<double> squared(size);
  for (std::size_t index = 0; index < size; ++index) {
    double difference = values[index] - mean;
    squared[index] = difference * difference;
  }
  double variance = numpy_pairwise_sum_f64(squared.data(), squared.size()) /
                    static_cast<double>(size);
  return std::sqrt(variance);
}

std::pair<double, double> numpy_mean_std_u8(
    const std::uint8_t* values,
    std::size_t size) {
  if (!values || size == 0) {
    return {0.0, 0.0};
  }
  std::vector<double> converted(size);
  for (std::size_t index = 0; index < size; ++index) {
    converted[index] = values[index];
  }
  const double mean = numpy_mean_f64(converted.data(), converted.size());
  for (double& value : converted) {
    const double difference = value - mean;
    value = difference * difference;
  }
  const double variance =
      numpy_pairwise_sum_f64(converted.data(), converted.size()) /
      static_cast<double>(size);
  return {mean, std::sqrt(variance)};
}

float numpy_mean_f32(const float* values, std::size_t size) {
  return size == 0 ? 0.0f : numpy_pairwise_sum_f32(values, size) / static_cast<float>(size);
}

float numpy_std_f32(const float* values, std::size_t size) {
  if (size == 0) {
    return 0.0f;
  }
  float mean = numpy_mean_f32(values, size);
  std::vector<float> squared(size);
  for (std::size_t index = 0; index < size; ++index) {
    float difference = values[index] - mean;
    squared[index] = difference * difference;
  }
  float variance = numpy_pairwise_sum_f32(squared.data(), squared.size()) /
                   static_cast<float>(size);
  return std::sqrt(variance);
}

float numpy_entropy_from_f32_counts(const std::array<float, 256>& counts) {
  float total = numpy_pairwise_sum_f32(counts.data(), counts.size());
  if (total <= 0.0f) {
    return 0.0f;
  }
  std::vector<float> terms;
  terms.reserve(counts.size());
  for (float count : counts) {
    if (count > 0.0f) {
      float probability = count / total;
      terms.push_back(probability * std::log2(probability));
    }
  }
  return -numpy_pairwise_sum_f32(terms.data(), terms.size()) / 8.0f;
}

double entropy_normalized_f64(const std::uint8_t* data, std::size_t size) {
  if (!data || size == 0) {
    return 0.0;
  }
  std::array<std::uint64_t, 256> counts{};
  for (std::size_t i = 0; i < size; ++i) {
    counts[data[i]] += 1;
  }
  double denom = static_cast<double>(size);
  std::vector<double> terms;
  terms.reserve(counts.size());
  for (auto count : counts) {
    if (count == 0) {
      continue;
    }
    double p = static_cast<double>(count) / denom;
    terms.push_back(p * std::log2(p));
  }
  return -numpy_pairwise_sum_f64(terms.data(), terms.size()) / 8.0;
}

float entropy_normalized(const std::uint8_t* data, std::size_t size) {
  return static_cast<float>(entropy_normalized_f64(data, size));
}

double entropy_normalized_f64(
    const std::vector<std::uint8_t>& data,
    std::size_t offset,
    std::size_t size) {
  if (offset >= data.size()) {
    return 0.0;
  }
  std::size_t available = std::min(size, data.size() - offset);
  return entropy_normalized_f64(data.data() + offset, available);
}

float entropy_normalized(const std::vector<std::uint8_t>& data, std::size_t offset, std::size_t size) {
  if (offset >= data.size()) {
    return 0.0f;
  }
  std::size_t available = std::min(size, data.size() - offset);
  return static_cast<float>(entropy_normalized_f64(data.data() + offset, available));
}

struct ParsedSection {
  std::string name;
  std::uint32_t virtual_address = 0;
  std::uint32_t raw_ptr = 0;
  std::uint32_t raw_size = 0;
  std::uint32_t virtual_size = 0;
  std::uint32_t characteristics = 0;
};

struct ParsedPeInfo {
  bool valid = false;
  bool is_pe64 = false;
  std::uint16_t number_of_sections = 0;
  std::uint16_t size_of_optional_header = 0;
  std::uint16_t characteristics = 0;
  std::uint16_t machine = 0;
  std::uint32_t timestamp = 0;
  std::uint16_t optional_magic = 0;
  std::uint8_t major_linker = 0;
  std::uint8_t minor_linker = 0;
  std::uint32_t size_of_code = 0;
  std::uint32_t size_of_initialized_data = 0;
  std::uint32_t size_of_uninitialized_data = 0;
  std::uint32_t address_of_entry_point = 0;
  std::uint64_t image_base = 0;
  std::uint32_t section_alignment = 0;
  std::uint32_t file_alignment = 0;
  std::uint32_t size_of_image = 0;
  std::uint16_t subsystem = 0;
  std::uint16_t dll_characteristics = 0;
  std::uint32_t checksum = 0;
  std::uint32_t size_of_headers = 0;
  std::uint32_t number_of_rva_and_sizes = 0;
  std::size_t optional_header_offset = 0;
  std::array<std::uint32_t, 16> data_directory_rva{};
  std::array<std::uint32_t, 16> data_directory_size{};
  std::uint32_t import_table_rva = 0;
  std::uint32_t export_table_rva = 0;
  std::uint32_t debug_rva = 0;
  std::uint32_t reloc_rva = 0;
  std::uint32_t tls_rva = 0;
  std::uint32_t exception_rva = 0;
  std::uint32_t security_virtual_address = 0;
  std::vector<ParsedSection> sections;
};

std::optional<std::size_t> parsed_rva_to_offset(const ParsedPeInfo& pe, std::uint32_t rva) {
  if (rva == 0) {
    return std::nullopt;
  }
  if (rva < pe.size_of_headers) {
    return static_cast<std::size_t>(rva);
  }
  const std::uint64_t rva64 = rva;
  for (const auto& section : pe.sections) {
    const std::uint64_t section_start = section.virtual_address;
    const std::uint64_t span = std::max(section.virtual_size, section.raw_size);
    if (span == 0) {
      continue;
    }
    const std::uint64_t section_end = section_start + span;
    if (rva64 >= section_start && rva64 < section_end) {
      const std::uint64_t offset =
          static_cast<std::uint64_t>(section.raw_ptr) + (rva64 - section_start);
      if (offset > (std::numeric_limits<std::size_t>::max)()) {
        return std::nullopt;
      }
      return static_cast<std::size_t>(offset);
    }
  }
  return std::nullopt;
}

std::string read_c_string_at(const std::vector<std::uint8_t>& data, std::size_t offset, std::size_t max_len = 512) {
  if (offset >= data.size()) {
    return {};
  }
  std::string out;
  for (std::size_t i = offset; i < data.size() && out.size() < max_len; ++i) {
    if (data[i] == 0) {
      break;
    }
    char ch = static_cast<char>(data[i]);
    if (ch >= 'A' && ch <= 'Z') {
      ch = static_cast<char>(ch - 'A' + 'a');
    }
    out.push_back(ch);
  }
  return out;
}

ParsedPeInfo parse_pe(const std::vector<std::uint8_t>& data) {
  ParsedPeInfo pe;
  if (data.size() < 0x40 || read_le<std::uint16_t>(data, 0) != 0x5A4D) {
    return pe;
  }
  std::uint32_t pe_offset = read_le<std::uint32_t>(data, 0x3C);
  if (pe_offset > data.size() || pe_offset + 24 > data.size()) {
    return pe;
  }
  if (read_le<std::uint32_t>(data, pe_offset) != 0x00004550) {
    return pe;
  }

  std::size_t file_header = pe_offset + 4;
  pe.machine = read_le<std::uint16_t>(data, file_header);
  pe.number_of_sections = read_le<std::uint16_t>(data, file_header + 2);
  pe.timestamp = read_le<std::uint32_t>(data, file_header + 4);
  pe.characteristics = read_le<std::uint16_t>(data, file_header + 18);
  pe.size_of_optional_header = read_le<std::uint16_t>(data, file_header + 16);
  std::size_t optional = file_header + 20;
  if (optional + pe.size_of_optional_header > data.size() || pe.size_of_optional_header < 96) {
    return pe;
  }

  std::uint16_t magic = read_le<std::uint16_t>(data, optional);
  pe.optional_magic = magic;
  pe.is_pe64 = magic == 0x20B;
  if (magic != 0x10B && magic != 0x20B) {
    return pe;
  }
  const std::size_t minimum_optional_size = pe.is_pe64 ? 112 : 96;
  if (pe.size_of_optional_header < minimum_optional_size) {
    return pe;
  }
  pe.optional_header_offset = optional;

  pe.major_linker = data.size() > optional + 2 ? data[optional + 2] : 0;
  pe.minor_linker = data.size() > optional + 3 ? data[optional + 3] : 0;
  pe.size_of_code = read_le<std::uint32_t>(data, optional + 4);
  pe.size_of_initialized_data = read_le<std::uint32_t>(data, optional + 8);
  pe.size_of_uninitialized_data = read_le<std::uint32_t>(data, optional + 12);
  pe.address_of_entry_point = read_le<std::uint32_t>(data, optional + 16);
  pe.image_base = pe.is_pe64 ? read_le<std::uint64_t>(data, optional + 24) : read_le<std::uint32_t>(data, optional + 28);
  pe.section_alignment = read_le<std::uint32_t>(data, optional + 32);
  pe.file_alignment = read_le<std::uint32_t>(data, optional + 36);
  pe.size_of_image = read_le<std::uint32_t>(data, optional + 56);
  pe.checksum = read_le<std::uint32_t>(data, optional + 64);
  pe.subsystem = read_le<std::uint16_t>(data, optional + 68);
  pe.dll_characteristics = read_le<std::uint16_t>(data, optional + 70);
  pe.size_of_headers = read_le<std::uint32_t>(data, optional + 60);
  pe.number_of_rva_and_sizes = read_le<std::uint32_t>(data, optional + (pe.is_pe64 ? 108 : 92));

  std::size_t data_dir = optional + (pe.is_pe64 ? 112 : 96);
  auto read_dir = [&](std::size_t index) {
    std::size_t off = data_dir + index * 8;
    if (off + 8 > optional + pe.size_of_optional_header || off + 8 > data.size()) {
      return;
    }
    pe.data_directory_rva[index] = read_le<std::uint32_t>(data, off);
    pe.data_directory_size[index] = read_le<std::uint32_t>(data, off + 4);
  };
  const std::size_t directory_count = std::min<std::size_t>(
      pe.number_of_rva_and_sizes,
      pe.data_directory_rva.size());
  for (std::size_t index = 0; index < directory_count; ++index) {
    read_dir(index);
  }
  pe.export_table_rva = pe.data_directory_rva[0];
  pe.import_table_rva = pe.data_directory_rva[1];
  pe.security_virtual_address = pe.data_directory_rva[4];
  pe.reloc_rva = pe.data_directory_rva[5];
  pe.debug_rva = pe.data_directory_rva[6];
  pe.tls_rva = pe.data_directory_rva[9];
  pe.exception_rva = pe.data_directory_rva[3];

  std::size_t section_table = optional + pe.size_of_optional_header;
  for (std::uint16_t i = 0; i < pe.number_of_sections; ++i) {
    std::size_t off = section_table + static_cast<std::size_t>(i) * 40;
    if (off + 40 > data.size()) {
      break;
    }
    ParsedSection section;
    char name_buf[9] = {};
    std::memcpy(name_buf, data.data() + off, 8);
    section.name = name_buf;
    section.name.erase(std::find(section.name.begin(), section.name.end(), '\0'), section.name.end());
    std::transform(section.name.begin(), section.name.end(), section.name.begin(), [](unsigned char c) {
      return static_cast<char>(std::tolower(c));
    });
    section.virtual_size = read_le<std::uint32_t>(data, off + 8);
    section.virtual_address = read_le<std::uint32_t>(data, off + 12);
    section.raw_size = read_le<std::uint32_t>(data, off + 16);
    section.raw_ptr = read_le<std::uint32_t>(data, off + 20);
    section.characteristics = read_le<std::uint32_t>(data, off + 36);
    pe.sections.push_back(section);
  }

  pe.valid = true;
  return pe;
}

struct ResourceDirectoryStats {
  std::uint64_t entry_count = 0;
  std::set<std::uint32_t> ids;
  bool has_named_id = false;
  bool parsed = false;
};

ResourceDirectoryStats collect_resource_directory_stats(
    const std::vector<std::uint8_t>& data,
    const ParsedPeInfo& pe) {
  ResourceDirectoryStats stats;
  constexpr std::size_t kMaxEntriesPerDirectory = 4096;
  constexpr std::size_t kMaxTotalEntries = 0x8000;
  constexpr std::size_t kMaxDepth = 32;
  const std::uint32_t resource_rva = pe.data_directory_rva[2];
  if (resource_rva == 0) {
    return stats;
  }
  const auto base_offset_result = parsed_rva_to_offset(pe, resource_rva);
  if (!base_offset_result) {
    return stats;
  }
  const std::size_t base_offset = *base_offset_result;
  if (base_offset >= data.size() || data.size() - base_offset < 16) {
    return stats;
  }

  struct PendingDirectory {
    std::uint32_t relative_offset;
    std::size_t depth;
    std::vector<std::uint32_t> ancestors;
  };
  std::vector<PendingDirectory> pending = {{0, 0, {0}}};
  stats.parsed = true;

  while (!pending.empty()) {
    PendingDirectory current = std::move(pending.back());
    pending.pop_back();
    if (current.depth > kMaxDepth || current.relative_offset > data.size() - base_offset) {
      continue;
    }
    const std::size_t directory_offset = base_offset + current.relative_offset;
    if (directory_offset > data.size() || data.size() - directory_offset < 16) {
      continue;
    }
    const std::size_t named_count = read_le<std::uint16_t>(data, directory_offset + 12);
    const std::size_t id_count = read_le<std::uint16_t>(data, directory_offset + 14);
    const std::size_t entry_count = named_count + id_count;
    if (entry_count > kMaxEntriesPerDirectory ||
        stats.entry_count + entry_count > kMaxTotalEntries) {
      continue;
    }
    const std::size_t entries_offset = directory_offset + 16;
    if (entries_offset > data.size() || entry_count > (data.size() - entries_offset) / 8) {
      continue;
    }

    for (std::size_t index = 0; index < entry_count; ++index) {
      const std::size_t entry_offset = entries_offset + index * 8;
      const std::uint32_t name = read_le<std::uint32_t>(data, entry_offset);
      const std::uint32_t target = read_le<std::uint32_t>(data, entry_offset + 4);
      const bool is_directory = (target & 0x80000000u) != 0;
      const std::uint32_t relative_target = target & 0x7fffffffu;
      bool valid_target = false;
      if (relative_target <= data.size() - base_offset) {
        const std::size_t target_offset = base_offset + relative_target;
        const std::size_t required_size = is_directory ? 16 : 16;
        valid_target = target_offset <= data.size() &&
                       data.size() - target_offset >= required_size;
      }
      if (!valid_target) {
        break;
      }
      if (is_directory &&
          std::find(
              current.ancestors.begin(),
              current.ancestors.end(),
              relative_target) != current.ancestors.end()) {
        break;
      }

      stats.entry_count += 1;
      if ((name & 0x80000000u) == 0) {
        stats.ids.insert(name);
      } else {
        stats.has_named_id = true;
      }
      if (is_directory) {
        std::vector<std::uint32_t> ancestors = current.ancestors;
        ancestors.push_back(relative_target);
        pending.push_back(
            {relative_target, current.depth + 1, std::move(ancestors)});
      }
    }
  }
  return stats;
}

struct RelocationDirectoryStats {
  std::uint64_t block_count = 0;
  std::uint64_t entry_count = 0;
  bool parsed = false;
};

RelocationDirectoryStats collect_relocation_directory_stats(
    const std::vector<std::uint8_t>& data,
    const ParsedPeInfo& pe) {
  RelocationDirectoryStats stats;
  constexpr std::uint64_t kMaxRelocationBlocks = 65536;
  constexpr std::uint64_t kMaxRelocationWorkItems = 1u << 20;
  const std::uint32_t relocation_rva = pe.data_directory_rva[5];
  const std::uint32_t declared_size = pe.data_directory_size[5];
  if (relocation_rva == 0 || declared_size == 0) {
    return stats;
  }
  const auto start_result = parsed_rva_to_offset(pe, relocation_rva);
  if (!start_result) {
    return stats;
  }
  const std::size_t start = *start_result;
  if (start >= data.size() || declared_size > data.size() - start) {
    return stats;
  }
  const std::size_t available = declared_size;
  std::size_t consumed = 0;
  std::uint64_t work_items = 0;
  while (consumed < available && available - consumed >= 8) {
    const std::size_t block_offset = start + consumed;
    const std::uint32_t virtual_address = read_le<std::uint32_t>(data, block_offset);
    const std::uint32_t block_size = read_le<std::uint32_t>(data, block_offset + 4);
    if (virtual_address > pe.size_of_image || block_size > pe.size_of_image ||
        block_size < 8 || block_size > available - consumed ||
        ((block_size - 8) % 2) != 0 ||
        stats.block_count >= kMaxRelocationBlocks) {
      return {};
    }
    stats.block_count += 1;
    std::set<std::uint16_t> offsets_and_types;
    const std::size_t candidate_entries = (block_size - 8) / 2;
    if (candidate_entries > kMaxRelocationWorkItems - work_items) {
      return {};
    }
    for (std::size_t index = 0; index < candidate_entries; ++index) {
      work_items += 1;
      const std::uint16_t entry = read_le<std::uint16_t>(data, block_offset + 8 + index * 2);
      if (!offsets_and_types.insert(entry).second) {
        break;
      }
      stats.entry_count += 1;
    }
    consumed += block_size;
  }
  if (consumed != available) {
    return {};
  }
  stats.parsed = true;
  return stats;
}

struct TlsDirectoryStats {
  bool parsed = false;
  bool has_callbacks = false;
};

TlsDirectoryStats collect_tls_directory_stats(
    const std::vector<std::uint8_t>& data,
    const ParsedPeInfo& pe) {
  TlsDirectoryStats stats;
  const std::uint32_t tls_rva = pe.data_directory_rva[9];
  if (tls_rva == 0) {
    return stats;
  }
  const auto offset_result = parsed_rva_to_offset(pe, tls_rva);
  if (!offset_result) {
    return stats;
  }
  const std::size_t offset = *offset_result;
  const std::size_t structure_size = pe.is_pe64 ? 40 : 24;
  const std::size_t callback_offset = pe.is_pe64 ? 24 : 12;
  if (offset >= data.size() || data.size() - offset < structure_size) {
    return stats;
  }
  stats.parsed = true;
  const std::uint64_t callback_address = pe.is_pe64
      ? read_le<std::uint64_t>(data, offset + callback_offset)
      : read_le<std::uint32_t>(data, offset + callback_offset);
  stats.has_callbacks = callback_address != 0;
  return stats;
}

bool has_parsed_debug_directory(
    const std::vector<std::uint8_t>& data,
    const ParsedPeInfo& pe) {
  const std::uint32_t rva = pe.data_directory_rva[6];
  const std::uint32_t size = pe.data_directory_size[6];
  if (rva == 0 || size < 28) {
    return false;
  }
  const auto offset_result = parsed_rva_to_offset(pe, rva);
  if (!offset_result) {
    return false;
  }
  const std::size_t offset = *offset_result;
  return offset < data.size() && data.size() - offset >= 28;
}

bool has_parsed_exception_directory(
    const std::vector<std::uint8_t>& data,
    const ParsedPeInfo& pe) {
  constexpr std::uint16_t kMachineAmd64 = 0x8664;
  constexpr std::uint16_t kMachineIa64 = 0x0200;
  if (pe.machine != kMachineAmd64 && pe.machine != kMachineIa64) {
    return false;
  }
  const std::uint32_t rva = pe.data_directory_rva[3];
  const std::uint32_t size = pe.data_directory_size[3];
  if (rva == 0 || size < 12) {
    return false;
  }
  const auto offset_result = parsed_rva_to_offset(pe, rva);
  if (!offset_result) {
    return false;
  }
  const std::size_t offset = *offset_result;
  return offset < data.size() && data.size() - offset >= 12;
}

bool contains_keyword(const std::string& text, const std::vector<std::string>& keywords, bool prefix_only = false) {
  for (const auto& keyword : keywords) {
    if (prefix_only) {
      if (text.rfind(keyword, 0) == 0) {
        return true;
      }
    } else if (text.find(keyword) != std::string::npos) {
      return true;
    }
  }
  return false;
}

void count_import_categories(
    const std::vector<std::uint8_t>& data,
    const ParsedPeInfo& pe,
    std::array<std::uint32_t, 6>& category_counts,
    std::uint32_t& total_apis) {
  category_counts.fill(0);
  total_apis = 0;
  const auto import_offset_result = parsed_rva_to_offset(pe, pe.import_table_rva);
  if (!import_offset_result || *import_offset_result >= data.size()) {
    return;
  }
  const std::size_t import_offset = *import_offset_result;

  const std::vector<std::string> network = {"internet", "http", "socket", "connect", "recv", "send", "url", "download", "upload", "proxy", "wsa", "ftp", "smtp"};
  const std::vector<std::string> process = {"createprocess", "openprocess", "virtualalloc", "virtualprotect", "writeprocessmemory", "readprocessmemory", "createremotethread", "shellexecute", "winexec", "loadlibrary", "getprocaddress"};
  const std::vector<std::string> filesystem = {"createfile", "readfile", "writefile", "deletefile", "movefile", "copyfile", "getfilesize", "setfilepointer", "findfirstfile", "findnextfile", "gettemppath"};
  const std::vector<std::string> registry = {"regopenkey", "regsetvalue", "regcreatekey", "regdeletekey", "regqueryvalue", "regclosekey", "savekey", "restorekey"};
  const std::vector<std::string> crypto = {"cryptencrypt", "cryptdecrypt", "cryptderivekey", "cryptgenkey", "cryptcreatehash", "crypthashdata", "cryptsignhash", "cryptverify"};
  const std::vector<std::string> injection = {"createremotethread", "virtualallocex", "writeprocessmemory", "readprocessmemory", "queueuserapc", "setwindowshookex", "rtlcreateuserthread", "ntcreatethreadex"};
  auto matches_category = [](const std::string& api, const std::vector<std::string>& keywords) {
    for (const auto& keyword : keywords) {
      const bool prefix_only = keyword == "connect" || keyword == "send" || keyword == "recv";
      if ((prefix_only && api.rfind(keyword, 0) == 0) ||
          (!prefix_only && api.find(keyword) != std::string::npos)) {
        return true;
      }
    }
    return false;
  };

  for (std::size_t desc = import_offset, guard = 0; desc + 20 <= data.size() && guard < 4096; desc += 20, ++guard) {
    std::uint32_t original_first_thunk = read_le<std::uint32_t>(data, desc);
    std::uint32_t name_rva = read_le<std::uint32_t>(data, desc + 12);
    std::uint32_t first_thunk = read_le<std::uint32_t>(data, desc + 16);
    if (original_first_thunk == 0 && name_rva == 0 && first_thunk == 0) {
      break;
    }
    std::uint32_t thunk_rva = original_first_thunk ? original_first_thunk : first_thunk;
    const auto thunk_offset_result = parsed_rva_to_offset(pe, thunk_rva);
    if (!thunk_offset_result || *thunk_offset_result >= data.size()) {
      continue;
    }
    const std::size_t thunk_offset = *thunk_offset_result;

    std::size_t thunk_size = pe.is_pe64 ? 8 : 4;
    std::uint64_t ordinal_mask = pe.is_pe64 ? 0x8000000000000000ull : 0x80000000ull;
    for (std::size_t thunk = thunk_offset, thunk_guard = 0; thunk + thunk_size <= data.size() && thunk_guard < 8192; thunk += thunk_size, ++thunk_guard) {
      std::uint64_t value = pe.is_pe64 ? read_le<std::uint64_t>(data, thunk) : read_le<std::uint32_t>(data, thunk);
      if (value == 0) {
        break;
      }
      if ((value & ordinal_mask) != 0) {
        continue;
      }
      std::uint32_t hint_name_rva = static_cast<std::uint32_t>(value & 0xFFFFFFFFu);
      const auto hint_name_offset_result = parsed_rva_to_offset(pe, hint_name_rva);
      if (!hint_name_offset_result || *hint_name_offset_result >= data.size() ||
          data.size() - *hint_name_offset_result <= 2) {
        continue;
      }
      std::string api = read_c_string_at(data, *hint_name_offset_result + 2);
      if (api.empty()) {
        continue;
      }
      total_apis += 1;
      if (matches_category(api, network)) category_counts[0] += 1;
      if (matches_category(api, process)) category_counts[1] += 1;
      if (matches_category(api, filesystem)) category_counts[2] += 1;
      if (matches_category(api, registry)) category_counts[3] += 1;
      if (matches_category(api, crypto)) category_counts[4] += 1;
      if (matches_category(api, injection)) category_counts[5] += 1;
    }
  }
}

struct ImportExportStats {
  std::set<std::string> import_dlls;
  std::vector<std::string> import_names;
  std::vector<int> imports_per_dll;
  std::uint32_t ordinal_imports = 0;
  std::array<std::uint32_t, 6> category_counts{};
  std::uint32_t export_count = 0;
  std::uint32_t export_name_count = 0;
};

ImportExportStats collect_import_export_stats(const std::vector<std::uint8_t>& data, const ParsedPeInfo& pe) {
  ImportExportStats stats;
  const auto import_offset_result = parsed_rva_to_offset(pe, pe.import_table_rva);
  const std::size_t import_offset = import_offset_result.value_or(0);
  const std::array<std::vector<std::string>, 6> category_keywords = {{
      {"internet", "http", "socket", "connect", "recv", "send", "url", "download", "upload", "wsa"},
      {"createprocess", "openprocess", "virtualalloc", "virtualprotect", "writeprocessmemory", "readprocessmemory", "createremotethread", "shellexecute", "winexec", "loadlibrary", "getprocaddress"},
      {"createfile", "readfile", "writefile", "deletefile", "movefile", "copyfile", "findfirstfile"},
      {"regopenkey", "regsetvalue", "regcreatekey", "regdeletekey", "regqueryvalue"},
      {"cryptencrypt", "cryptdecrypt", "cryptderivekey", "cryptgenkey", "cryptcreatehash", "crypthashdata"},
      {"createremotethread", "virtualallocex", "writeprocessmemory", "queueuserapc", "setwindowshookex"},
  }};

  if (import_offset != 0 && import_offset < data.size()) {
    for (std::size_t desc = import_offset, guard = 0; desc + 20 <= data.size() && guard < 4096; desc += 20, ++guard) {
      std::uint32_t original_first_thunk = read_le<std::uint32_t>(data, desc);
      std::uint32_t name_rva = read_le<std::uint32_t>(data, desc + 12);
      std::uint32_t first_thunk = read_le<std::uint32_t>(data, desc + 16);
      if (original_first_thunk == 0 && name_rva == 0 && first_thunk == 0) {
        break;
      }
      const auto name_offset = parsed_rva_to_offset(pe, name_rva);
      std::string dll_name = name_offset ? read_c_string_at(data, *name_offset) : std::string{};
      if (!dll_name.empty()) {
        stats.import_dlls.insert(dll_name);
      }
      std::uint32_t thunk_rva = original_first_thunk ? original_first_thunk : first_thunk;
      const auto thunk_offset_result = parsed_rva_to_offset(pe, thunk_rva);
      const std::size_t thunk_offset = thunk_offset_result.value_or(0);
      int dll_import_count = 0;
      if (thunk_offset != 0 && thunk_offset < data.size()) {
        std::size_t thunk_size = pe.is_pe64 ? 8 : 4;
        std::uint64_t ordinal_mask = pe.is_pe64 ? 0x8000000000000000ull : 0x80000000ull;
        for (std::size_t thunk = thunk_offset, thunk_guard = 0; thunk + thunk_size <= data.size() && thunk_guard < 8192; thunk += thunk_size, ++thunk_guard) {
          std::uint64_t value = pe.is_pe64 ? read_le<std::uint64_t>(data, thunk) : read_le<std::uint32_t>(data, thunk);
          if (value == 0) {
            break;
          }
          ++dll_import_count;
          if ((value & ordinal_mask) != 0) {
            ++stats.ordinal_imports;
            continue;
          }
          std::uint32_t hint_name_rva = static_cast<std::uint32_t>(value & 0xFFFFFFFFu);
          const auto hint_name_offset_result = parsed_rva_to_offset(pe, hint_name_rva);
          if (!hint_name_offset_result || *hint_name_offset_result >= data.size() ||
              data.size() - *hint_name_offset_result <= 2) {
            continue;
          }
          std::string api = read_c_string_at(data, *hint_name_offset_result + 2);
          if (api.empty()) {
            continue;
          }
          stats.import_names.push_back(api);
          for (std::size_t i = 0; i < category_keywords.size(); ++i) {
            if (contains_keyword(api, category_keywords[i])) {
              stats.category_counts[i] += 1;
            }
          }
        }
      }
      stats.imports_per_dll.push_back(dll_import_count);
    }
  }

  const auto export_offset_result = parsed_rva_to_offset(pe, pe.export_table_rva);
  if (export_offset_result && *export_offset_result <= data.size() &&
      data.size() - *export_offset_result >= 40) {
    const std::size_t export_offset = *export_offset_result;
    std::uint32_t number_of_functions = read_le<std::uint32_t>(data, export_offset + 20);
    std::uint32_t number_of_names = read_le<std::uint32_t>(data, export_offset + 24);
    stats.export_count = number_of_functions;
    stats.export_name_count = number_of_names;
  }
  return stats;
}

std::vector<float> lightweight_features(const std::vector<std::uint8_t>& data) {
  std::vector<float> features(kAxonLightweightFeatureDim, 0.0f);
  std::string lower;
  std::size_t read_size = std::min<std::size_t>(data.size(), 65536);
  lower.reserve(read_size);
  for (std::size_t i = 0; i < read_size; ++i) {
    lower.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(data[i]))));
  }
  const std::vector<std::pair<std::string, int>> dll_patterns = {
      {"kernel32.dll", 83}, {"user32.dll", 16}, {"ntdll.dll", 23}, {"advapi32.dll", 57},
      {"ws2_32.dll", 1}, {"wininet.dll", 99}, {"ole32.dll", 4}, {"shell32.dll", 9},
      {"msvcrt.dll", 115}, {"msvcrtd.dll", 25}, {"vcruntime.dll", 60}, {"ucrtbase.dll", 72}};
  const std::vector<std::pair<std::string, int>> api_patterns = {
      {"virtualalloc", 174}, {"virtualprotect", 195}, {"createremotethread", 186},
      {"writeprocessmemory", 181}, {"readprocessmemory", 224}, {"winexec", 217},
      {"shellexecute", 146}, {"loadlibrary", 254}, {"getprocaddress", 197},
      {"createprocess", 174}, {"internetopen", 242}, {"internetreadfile", 186},
      {"urldownloadtofile", 170}, {"createfile", 180}, {"regopenkey", 160}};
  const std::vector<std::pair<std::string, int>> section_patterns = {
      {".text", 244}, {".data", 229}, {".rdata", 229}, {".rsrc", 250}, {".reloc", 237},
      {".code", 226}, {".idata", 231}, {".edata", 229}, {".tls", 225}, {".bss", 251}};
  for (const auto& [pattern, index] : dll_patterns) {
    if (lower.find(pattern) != std::string::npos && index >= 0 && index < static_cast<int>(features.size())) {
      features[static_cast<std::size_t>(index)] = 1.0f;
    }
  }
  for (const auto& [pattern, index] : api_patterns) {
    if (lower.find(pattern) != std::string::npos && index >= 0 && index < static_cast<int>(features.size())) {
      features[static_cast<std::size_t>(index)] = 1.0f;
    }
  }
  for (const auto& [pattern, index] : section_patterns) {
    if (lower.find(pattern) != std::string::npos && index >= 0 && index < static_cast<int>(features.size())) {
      features[static_cast<std::size_t>(index)] = 1.0f;
    }
  }
  float norm = 0.0f;
  for (float value : features) {
    norm += value * value;
  }
  norm = std::sqrt(norm);
  if (norm > 0.0f) {
    for (float& value : features) {
      value /= norm;
    }
  }
  return features;
}

std::vector<float> byte_summary_features(const std::vector<std::uint8_t>& byte_seq) {
  std::vector<float> out;
  out.reserve(256 + 256 + kStage2PrefixLen + kStage2ChunkCount * 5 + 5);
  std::array<std::uint64_t, 256> counts{};
  for (auto byte : byte_seq) {
    counts[byte] += 1;
  }
  std::array<float, 256> counts_f32{};
  for (std::size_t index = 0; index < counts.size(); ++index) {
    counts_f32[index] = static_cast<float>(counts[index]);
  }
  float length = static_cast<float>(std::max<std::size_t>(byte_seq.size(), 1));
  for (auto count : counts) {
    out.push_back(static_cast<float>(count) / length);
  }
  double log_denom = std::log1p(static_cast<double>(length));
  for (auto count : counts) {
    float log_count = std::log1p(static_cast<float>(count));
    out.push_back(static_cast<float>(static_cast<double>(log_count) / log_denom));
  }
  for (std::size_t i = 0; i < kStage2PrefixLen; ++i) {
    out.push_back(i < byte_seq.size() ? static_cast<float>(byte_seq[i]) / 255.0f : 0.0f);
  }
  for (std::size_t chunk_index = 0; chunk_index < kStage2ChunkCount; ++chunk_index) {
    std::size_t start = chunk_index * byte_seq.size() / kStage2ChunkCount;
    std::size_t end = (chunk_index + 1) * byte_seq.size() / kStage2ChunkCount;
    if (end <= start) {
      out.insert(out.end(), {0.0f, 0.0f, 0.0f, 0.0f, 0.0f});
      continue;
    }
    const auto [mean, stddev] = numpy_mean_std_u8(
        byte_seq.data() + start,
        end - start);
    std::array<float, 256> chunk_counts{};
    std::uint64_t nonzero = 0;
    for (std::size_t i = start; i < end; ++i) {
      chunk_counts[byte_seq[i]] += 1.0f;
      if (byte_seq[i] != 0) {
        nonzero += 1;
      }
    }
    float max_count = 0.0f;
    for (auto count : chunk_counts) {
      max_count = std::max(max_count, count);
    }
    out.push_back(static_cast<float>(mean / 255.0));
    out.push_back(static_cast<float>(stddev / 255.0));
    out.push_back(numpy_entropy_from_f32_counts(chunk_counts));
    out.push_back(static_cast<float>(nonzero) / static_cast<float>(end - start));
    out.push_back(static_cast<float>(max_count) / static_cast<float>(end - start));
  }
  out.push_back(numpy_entropy_from_f32_counts(counts_f32));
  std::uint64_t nonzero = 0;
  for (auto byte : byte_seq) {
    if (byte != 0) ++nonzero;
  }
  const auto [mean, stddev] = numpy_mean_std_u8(byte_seq.data(), byte_seq.size());
  std::uint64_t max_count = 0;
  for (auto count : counts) {
    max_count = std::max(max_count, count);
  }
  out.push_back(static_cast<float>(nonzero) / length);
  out.push_back(static_cast<float>(mean / 255.0));
  out.push_back(static_cast<float>(stddev / 255.0));
  out.push_back(static_cast<float>(max_count) / length);
  return out;
}

std::vector<float> content_pe_v1_features(const std::vector<std::uint8_t>& data) {
  std::vector<float> features;
  features.reserve(kContentPeFeatureDim);
  ParsedPeInfo pe = parse_pe(data);
  if (!pe.valid) {
    return std::vector<float>(kContentPeFeatureDim, 0.0f);
  }

  double file_size = static_cast<double>(data.size());
  double timestamp_year = pe.timestamp > 0 ? 1970.0 + static_cast<double>(pe.timestamp) / 31557600.0 : 0.0;
  float timestamp_valid = (timestamp_year >= 1970.0 && timestamp_year <= 2099.0) ? 1.0f : 0.0f;
  float timestamp_year_norm = timestamp_valid ? static_cast<float>((std::min(std::max(timestamp_year, 1970.0), 2099.0) - 1970.0) / 129.0) : 0.0f;

  auto push = [&](double value) {
    features.push_back(static_cast<float>(std::isfinite(value) ? value : 0.0));
  };

  push(std::log1p(file_size));
  push(std::min(file_size, 100.0 * 1024.0 * 1024.0) / (100.0 * 1024.0 * 1024.0));
  push(static_cast<double>(pe.machine) / 65535.0);
  push(static_cast<double>(pe.characteristics) / 65535.0);
  push(std::min(static_cast<double>(pe.number_of_sections), 64.0) / 64.0);
  push(timestamp_valid);
  push(timestamp_year_norm);
  push(static_cast<double>(pe.optional_magic) / 65535.0);
  push(std::min(static_cast<double>(pe.major_linker), 255.0) / 255.0);
  push(std::min(static_cast<double>(pe.minor_linker), 255.0) / 255.0);
  push(safe_ratio_f64(pe.size_of_code, file_size));
  push(safe_ratio_f64(pe.size_of_initialized_data, file_size));
  push(safe_ratio_f64(pe.size_of_uninitialized_data, file_size));
  push(safe_ratio_f64(pe.address_of_entry_point, std::max({static_cast<double>(pe.size_of_image), file_size, 1.0})));
  push(std::log1p(static_cast<double>(pe.image_base)) / 64.0);
  push(std::log1p(static_cast<double>(pe.section_alignment)) / 16.0);
  push(std::log1p(static_cast<double>(pe.file_alignment)) / 16.0);
  push(safe_ratio_f64(pe.size_of_image, file_size));
  push(safe_ratio_f64(pe.size_of_headers, file_size));
  push(static_cast<double>(pe.subsystem) / 32.0);
  push(static_cast<double>(pe.dll_characteristics) / 65535.0);
  push((pe.characteristics & 0x2000) ? 1.0 : 0.0);
  push((pe.characteristics & 0x0002) ? 1.0 : 0.0);
  push((pe.characteristics & 0x1000) ? 1.0 : 0.0);
  push((pe.characteristics & 0x0020) ? 1.0 : 0.0);
  push((pe.characteristics & 0x0100) ? 1.0 : 0.0);
  push((pe.characteristics & 0x0001) ? 1.0 : 0.0);
  push((pe.characteristics & 0x0200) ? 1.0 : 0.0);

  const std::array<int, 11> dir_indexes = {0, 1, 2, 3, 4, 5, 6, 9, 12, 13, 14};
  for (int index : dir_indexes) {
    double size = static_cast<double>(pe.data_directory_size[static_cast<std::size_t>(index)]);
    double rva = static_cast<double>(pe.data_directory_rva[static_cast<std::size_t>(index)]);
    push((size > 0.0 || rva > 0.0) ? 1.0 : 0.0);
    push(std::log1p(std::max(size, 0.0)));
    push(safe_ratio_f64(size, file_size));
  }

  ImportExportStats stats = collect_import_export_stats(data, pe);
  std::set<std::string> unique_imports(stats.import_names.begin(), stats.import_names.end());
  const std::set<std::string> system_dlls = {
      "kernel32.dll", "user32.dll", "advapi32.dll", "shell32.dll", "ole32.dll", "oleaut32.dll",
      "msvcrt.dll", "ntdll.dll", "ws2_32.dll", "wininet.dll", "urlmon.dll", "crypt32.dll",
      "secur32.dll", "netapi32.dll", "dnsapi.dll", "iphlpapi.dll", "gdi32.dll", "comdlg32.dll",
      "comctl32.dll", "shlwapi.dll", "version.dll", "setupapi.dll", "imm32.dll"};
  int system_dll_count = 0;
  for (const auto& dll : stats.import_dlls) {
    if (system_dlls.count(dll) > 0) {
      ++system_dll_count;
    }
  }
  int total_imports = static_cast<int>(stats.import_names.size() + stats.ordinal_imports);
  int max_imports_per_dll = 0;
  int total_per_dll = 0;
  for (int value : stats.imports_per_dll) {
    max_imports_per_dll = std::max(max_imports_per_dll, value);
    total_per_dll += value;
  }
  push(std::log1p(static_cast<double>(stats.import_dlls.size())));
  push(std::log1p(static_cast<double>(total_imports)));
  push(std::log1p(static_cast<double>(unique_imports.size())));
  push(safe_ratio_f64(stats.ordinal_imports, total_imports));
  push(safe_ratio_f64(system_dll_count, stats.import_dlls.size()));
  push(safe_ratio_f64(total_imports, stats.imports_per_dll.size()));
  push(safe_ratio_f64(max_imports_per_dll, 512.0));
  for (std::uint32_t count : stats.category_counts) {
    push(safe_ratio_f64(count, total_imports));
  }

  push(std::log1p(static_cast<double>(stats.export_count)));
  push(safe_ratio_f64(stats.export_name_count, stats.export_count));

  ResourceDirectoryStats resource_stats = collect_resource_directory_stats(data, pe);
  TlsDirectoryStats tls_stats = collect_tls_directory_stats(data, pe);
  RelocationDirectoryStats relocation_stats = collect_relocation_directory_stats(data, pe);
  push(std::log1p(static_cast<double>(resource_stats.entry_count)));
  const std::size_t resource_unique_ids =
      resource_stats.ids.size() + (resource_stats.has_named_id ? 1u : 0u);
  push(std::log1p(static_cast<double>(resource_unique_ids)));
  push(tls_stats.parsed && tls_stats.has_callbacks ? std::log1p(1.0) : 0.0);
  push(std::log1p(static_cast<double>(relocation_stats.block_count)));
  push(std::log1p(static_cast<double>(relocation_stats.entry_count)));

  std::size_t overlay_offset = 0;
  auto update_overlay_boundary = [&](std::size_t offset, std::uint64_t size) {
    const std::uint64_t end = static_cast<std::uint64_t>(offset) + size;
    if (end <= data.size() && end > overlay_offset) {
      overlay_offset = static_cast<std::size_t>(end);
    }
  };
  update_overlay_boundary(pe.optional_header_offset, pe.size_of_optional_header);
  for (const auto& section : pe.sections) {
    update_overlay_boundary(section.raw_ptr, section.raw_size);
  }
  const std::size_t directory_count = std::min<std::size_t>(
      pe.number_of_rva_and_sizes,
      pe.data_directory_rva.size());
  for (std::size_t index = 0; index < directory_count; ++index) {
    if (index == 4) {
      continue;
    }
    const std::uint32_t rva = pe.data_directory_rva[index];
    const auto directory_offset = parsed_rva_to_offset(pe, rva);
    if (directory_offset) {
      update_overlay_boundary(*directory_offset, pe.data_directory_size[index]);
    } else if (rva == 0) {
      update_overlay_boundary(0, pe.data_directory_size[index]);
    }
  }
  std::size_t overlay_size = overlay_offset < data.size() ? data.size() - overlay_offset : 0;
  push(overlay_size > 0 ? 1.0 : 0.0);
  push(std::log1p(static_cast<double>(overlay_size)));
  push(safe_ratio_f64(overlay_size, file_size));
  push(overlay_size > 0 ? entropy_normalized_f64(data, overlay_offset, std::min<std::size_t>(overlay_size, 65536)) : 0.0);

  const std::set<std::string> common_sections = {".text", ".data", ".rdata", ".rsrc", ".idata", ".edata", ".bss", ".reloc", ".tls"};
  const std::vector<std::string> packer_keywords = {"upx", "aspack", "themida", "vmprotect", "enigma", "packed", "nspack", "upack"};
  std::array<int, 8> combo_counts{};
  int nonstandard_names = 0;
  int raw_virtual_mismatch = 0;
  int zero_raw = 0;
  int packer_hits = 0;
  std::vector<double> section_entropies;
  for (const auto& section : pe.sections) {
    bool is_exec = (section.characteristics & 0x20000000u) != 0;
    bool is_write = (section.characteristics & 0x80000000u) != 0;
    bool is_read = (section.characteristics & 0x40000000u) != 0;
    int combo = 7;
    if (is_exec && is_read && is_write) combo = 2;
    else if (is_exec && is_write) combo = 3;
    else if (is_exec && is_read) combo = 0;
    else if (is_read && is_write) combo = 1;
    else if (is_exec) combo = 4;
    else if (is_read) combo = 5;
    else if (is_write) combo = 6;
    combo_counts[combo] += 1;
    if (!section.name.empty() && common_sections.count(section.name) == 0) {
      ++nonstandard_names;
    }
    if (contains_keyword(section.name, packer_keywords)) {
      ++packer_hits;
    }
    if (section.raw_size == 0) {
      ++zero_raw;
    }
    double max_size = std::max(section.raw_size, section.virtual_size);
    if (max_size > 0.0 && std::fabs(static_cast<double>(section.raw_size) - static_cast<double>(section.virtual_size)) / max_size > 0.50) {
      ++raw_virtual_mismatch;
    }
    if (section.raw_size > 0 && section.raw_ptr < data.size()) {
      section_entropies.push_back(entropy_normalized_f64(data, section.raw_ptr, std::min<std::size_t>({4096, static_cast<std::size_t>(section.raw_size), data.size() - section.raw_ptr})));
    }
  }
  double section_count = std::max<std::size_t>(pe.sections.size(), 1);
  for (int count : combo_counts) {
    push(static_cast<double>(count) / section_count);
  }
  int high_entropy = 0;
  double entropy_max = 0.0;
  for (double value : section_entropies) {
    if (value >= 0.80) ++high_entropy;
    entropy_max = std::max<double>(entropy_max, value);
  }
  push(safe_ratio_f64(nonstandard_names, section_count));
  push(safe_ratio_f64(high_entropy, section_entropies.size()));
  push(safe_ratio_f64(raw_virtual_mismatch, section_count));
  push(safe_ratio_f64(zero_raw, section_count));
  push(section_entropies.empty()
      ? 0.0
      : numpy_mean_f64(section_entropies.data(), section_entropies.size()));
  push(entropy_max);
  push(safe_ratio_f64(packer_hits, section_count));

  if (features.size() < kContentPeFeatureDim) {
    features.resize(kContentPeFeatureDim, 0.0f);
  }
  if (features.size() > kContentPeFeatureDim) {
    features.resize(kContentPeFeatureDim);
  }
  return features;
}

std::vector<float> fixed_v2_pe_features(const std::vector<std::uint8_t>& data) {
  std::vector<float> features(kAxonPeFeatureDim, 0.0f);
  ParsedPeInfo pe = parse_pe(data);
  if (!pe.valid) {
    return features;
  }

  std::size_t idx = 0;
  float file_size = static_cast<float>(data.size());
  features[idx++] = file_size;
  features[idx++] = std::log1p(file_size);
  features[idx++] = static_cast<float>(pe.size_of_optional_header);
  features[idx++] = static_cast<float>(pe.size_of_optional_header + 24) / std::max(file_size, 1.0f);
  features[idx++] = static_cast<float>(pe.subsystem);
  features[idx++] = static_cast<float>(pe.dll_characteristics);
  features[idx++] = static_cast<float>(pe.checksum);
  features[idx++] = pe.checksum == 0 ? 1.0f : 0.0f;
  features[idx++] = (pe.dll_characteristics & 0x0040) ? 1.0f : 0.0f;
  features[idx++] = (pe.dll_characteristics & 0x0080) ? 1.0f : 0.0f;
  features[idx++] = (pe.dll_characteristics & 0x4000) ? 1.0f : 0.0f;
  features[idx++] = (pe.characteristics & 0x0004) ? 1.0f : 0.0f;
  RelocationDirectoryStats relocation_stats = collect_relocation_directory_stats(data, pe);
  TlsDirectoryStats tls_stats = collect_tls_directory_stats(data, pe);
  features[idx++] = has_parsed_debug_directory(data, pe) ? 1.0f : 0.0f;
  features[idx++] = relocation_stats.block_count > 0 ? 1.0f : 0.0f;
  features[idx++] = tls_stats.parsed ? 1.0f : 0.0f;
  features[idx++] = has_parsed_exception_directory(data, pe) ? 1.0f : 0.0f;
  features[idx++] = 0.0f;
  features[idx++] = static_cast<float>(pe.number_of_sections);

  std::vector<std::uint32_t> section_sizes;
  std::vector<std::uint32_t> section_vsizes;
  std::vector<double> section_entropies;
  std::vector<std::string> section_names;
  section_sizes.reserve(pe.sections.size());
  section_vsizes.reserve(pe.sections.size());
  section_names.reserve(pe.sections.size());

  for (std::size_t slot = 0; slot < kAxonFixedSectionSlots; ++slot) {
    if (slot < pe.sections.size()) {
      const auto& section = pe.sections[slot];
      bool is_exec = (section.characteristics & 0x20000000u) != 0;
      bool is_write = (section.characteristics & 0x80000000u) != 0;
      bool is_read = (section.characteristics & 0x40000000u) != 0;
      features[idx++] = is_exec ? 1.0f : 0.0f;
      features[idx++] = is_write ? 1.0f : 0.0f;
      features[idx++] = is_read ? 1.0f : 0.0f;
    } else {
      idx += 3;
    }
  }

  for (const auto& section : pe.sections) {
    section_sizes.push_back(section.raw_size);
    section_vsizes.push_back(section.virtual_size);
    section_names.push_back(section.name);
    if (section.raw_size > 0 && section.raw_size < 10 * 1024 * 1024 && section.raw_ptr < data.size()) {
      std::size_t sample = std::min<std::size_t>(256, std::min<std::size_t>(section.raw_size, data.size() - section.raw_ptr));
      if (sample > 0) {
        section_entropies.push_back(entropy_normalized_f64(data, section.raw_ptr, sample));
      }
    }
  }

  if (!section_entropies.empty()) {
    auto [min_it, max_it] = std::minmax_element(section_entropies.begin(), section_entropies.end());
    double mean = numpy_mean_f64(section_entropies.data(), section_entropies.size());
    int high_entropy = 0;
    for (double value : section_entropies) {
      if (value > 0.8) {
        high_entropy += 1;
      }
    }
    features[idx++] = static_cast<float>(*max_it);
    features[idx++] = static_cast<float>(*min_it);
    features[idx++] = static_cast<float>(mean);
    features[idx++] = static_cast<float>(numpy_std_f64(section_entropies.data(), section_entropies.size()));
    features[idx++] = static_cast<float>(high_entropy) / static_cast<float>(section_entropies.size());
  } else {
    idx += 5;
  }

  double avg_raw = 0.0;
  if (!section_sizes.empty()) {
    std::uint64_t total_raw = 0;
    std::uint64_t total_vsize = 0;
    for (std::size_t i = 0; i < section_sizes.size(); ++i) {
      total_raw += section_sizes[i];
      total_vsize += section_vsizes[i];
    }
    avg_raw = static_cast<double>(total_raw) / static_cast<double>(section_sizes.size());
    double avg_vsize = static_cast<double>(total_vsize) / static_cast<double>(section_vsizes.size());
    auto [min_it, max_it] = std::minmax_element(section_sizes.begin(), section_sizes.end());
    std::vector<double> squared_size_differences;
    squared_size_differences.reserve(section_sizes.size());
    for (std::uint32_t value : section_sizes) {
      double difference = static_cast<double>(value) - avg_raw;
      squared_size_differences.push_back(difference * difference);
    }
    double variance = numpy_pairwise_sum_f64(
        squared_size_differences.data(), squared_size_differences.size()) /
        static_cast<double>(section_sizes.size());
    double std_raw = std::sqrt(variance);
    features[idx++] = static_cast<float>(total_raw);
    features[idx++] = static_cast<float>(total_vsize);
    features[idx++] = static_cast<float>(avg_raw);
    features[idx++] = static_cast<float>(avg_vsize);
    features[idx++] = static_cast<float>(*min_it);
    features[idx++] = static_cast<float>(*max_it);
    features[idx++] = static_cast<float>(std_raw);
    features[idx++] = static_cast<float>(std_raw / std::max(avg_raw, 1.0));
  } else {
    idx += 8;
  }

  std::vector<int> name_lengths;
  for (const auto& name : section_names) {
    if (!name.empty()) {
      name_lengths.push_back(static_cast<int>(name.size()));
    }
  }
  features[idx++] = static_cast<float>(name_lengths.size());
  if (!name_lengths.empty()) {
    auto [min_it, max_it] = std::minmax_element(name_lengths.begin(), name_lengths.end());
    float sum = static_cast<float>(std::accumulate(name_lengths.begin(), name_lengths.end(), 0));
    features[idx++] = sum / static_cast<float>(name_lengths.size());
    features[idx++] = static_cast<float>(*max_it);
    features[idx++] = static_cast<float>(*min_it);
  } else {
    idx += 3;
  }

  if (!section_sizes.empty() && avg_raw > 0.0) {
    int long_count = 0;
    int short_count = 0;
    for (auto size : section_sizes) {
      if (static_cast<double>(size) > 2.0 * avg_raw) {
        long_count += 1;
      }
      if (static_cast<double>(size) < 0.5 * avg_raw) {
        short_count += 1;
      }
    }
    features[idx++] = static_cast<float>(long_count);
    features[idx++] = static_cast<float>(long_count) / static_cast<float>(section_sizes.size());
    features[idx++] = static_cast<float>(short_count);
    features[idx++] = static_cast<float>(short_count) / static_cast<float>(section_sizes.size());
  } else {
    idx += 4;
  }

  std::array<std::uint32_t, 6> category_counts{};
  std::uint32_t total_apis = 0;
  count_import_categories(data, pe, category_counts, total_apis);
  for (std::uint32_t count : category_counts) {
    features[idx++] = static_cast<float>(count) / static_cast<float>(std::max<std::uint32_t>(total_apis, 1));
  }

  const std::vector<std::string> packer_keywords = {
      "upx", "themida", "vmprotect", "aspack", "mpress", "pecompact", "obsidium", "enigma", "packed"};
  int packer_hits = 0;
  for (const auto& name : section_names) {
    if (contains_keyword(name, packer_keywords)) {
      packer_hits += 1;
    }
  }
  features[idx++] = static_cast<float>(packer_hits);
  features[idx++] = static_cast<float>(packer_hits) / static_cast<float>(std::max<std::uint16_t>(pe.number_of_sections, 1));
  return features;
}

std::vector<float> statistical_features(const std::vector<std::uint8_t>& data) {
  std::vector<float> features;
  features.reserve(kAxonStatFeatureDim);
  std::array<std::uint64_t, 256> counts{};
  for (auto byte : data) {
    counts[byte] += 1;
  }
  std::size_t length = data.size();

  double mean = 0.0;
  double std_val = 0.0;
  double min_val = 0.0;
  double max_val = 0.0;
  double median = 0.0;
  double q25 = 0.0;
  double q75 = 0.0;
  if (length > 0) {
    double weighted = 0.0;
    double weighted_sq = 0.0;
    for (int i = 0; i < 256; ++i) {
      weighted += static_cast<double>(counts[i]) * static_cast<double>(i);
      weighted_sq += static_cast<double>(counts[i]) * static_cast<double>(i * i);
    }
    mean = weighted / static_cast<double>(length);
    double var = std::max(0.0, weighted_sq / static_cast<double>(length) - mean * mean);
    std_val = std::sqrt(var);
    for (int i = 0; i < 256; ++i) {
      if (counts[i] > 0) {
        min_val = i;
        break;
      }
    }
    for (int i = 255; i >= 0; --i) {
      if (counts[i] > 0) {
        max_val = i;
        break;
      }
    }
    auto quantile = [&](double q) -> double {
      std::uint64_t target = static_cast<std::uint64_t>(std::ceil(q * static_cast<double>(length)));
      std::uint64_t cumulative = 0;
      for (int i = 0; i < 256; ++i) {
        cumulative += counts[i];
        if (cumulative >= target) {
          return static_cast<double>(i);
        }
      }
      return 0.0;
    };
    median = quantile(0.50);
    q25 = quantile(0.25);
    q75 = quantile(0.75);
  }

  auto push = [&](double value) {
    features.push_back(static_cast<float>(value));
  };
  push(mean);
  push(std_val);
  push(min_val);
  push(max_val);
  push(median);
  push(q25);
  push(q75);
  push(counts[0]);
  push(counts[255]);
  push(counts[0x90]);
  std::uint64_t printable = 0;
  for (int i = 32; i < 127; ++i) {
    printable += counts[i];
  }
  push(printable);
  push(entropy_normalized(data.data(), data.size()));

  constexpr int segment_count = 3;
  for (int seg = 0; seg < segment_count; ++seg) {
    std::size_t start = 0;
    std::size_t end = 0;
    if (length >= segment_count) {
      std::size_t seg_len = length / segment_count;
      start = static_cast<std::size_t>(seg) * seg_len;
      end = seg == segment_count - 1 ? length : start + seg_len;
    } else {
      start = 0;
      end = length;
    }
    if (end <= start) {
      push(0.0);
      push(0.0);
      push(0.0);
      continue;
    }
    const auto [seg_mean, seg_stddev] = numpy_mean_std_u8(
        data.data() + start,
        end - start);
    push(seg_mean);
    push(seg_stddev);
    push(entropy_normalized(data, start, end - start));
  }

  std::array<float, kStatChunkCount> chunk_means{};
  std::array<float, kStatChunkCount> chunk_stds{};
  std::size_t chunk_size = std::max<std::size_t>(1, length / kStatChunkCount);
  for (int chunk_idx = 0; chunk_idx < kStatChunkCount; ++chunk_idx) {
    std::size_t start = static_cast<std::size_t>(chunk_idx) * chunk_size;
    std::size_t end = chunk_idx == kStatChunkCount - 1 ? length : start + chunk_size;
    if (start >= length || end <= start) {
      chunk_means[chunk_idx] = 0.0f;
      chunk_stds[chunk_idx] = 0.0f;
      continue;
    }
    end = std::min(end, length);
    const auto [c_mean, c_stddev] = numpy_mean_std_u8(
        data.data() + start,
        end - start);
    chunk_means[chunk_idx] = static_cast<float>(c_mean);
    chunk_stds[chunk_idx] = static_cast<float>(c_stddev);
  }
  for (float value : chunk_means) {
    push(value);
  }
  for (float value : chunk_stds) {
    push(value);
  }

  auto diff_stats = [&](const std::array<float, kStatChunkCount>& values) {
    std::array<float, kStatChunkDiffCount> diffs{};
    std::array<float, kStatChunkDiffCount> absolute_diffs{};
    for (int i = 0; i < kStatChunkDiffCount; ++i) {
      diffs[i] = values[i + 1] - values[i];
      absolute_diffs[i] = std::fabs(diffs[i]);
    }
    auto [min_it, max_it] = std::minmax_element(diffs.begin(), diffs.end());
    push(numpy_mean_f32(absolute_diffs.data(), absolute_diffs.size()));
    push(numpy_std_f32(diffs.data(), diffs.size()));
    push(*max_it);
    push(*min_it);
  };
  diff_stats(chunk_means);
  diff_stats(chunk_stds);

  if (features.size() < kAxonStatFeatureDim) {
    features.resize(kAxonStatFeatureDim, 0.0f);
  }
  if (features.size() > kAxonStatFeatureDim) {
    features.resize(kAxonStatFeatureDim);
  }
  return features;
}

InferenceInput make_inference_input(const std::vector<std::uint8_t>& file_bytes) {
  InferenceInput input;
  input.original_length = static_cast<std::uint64_t>(file_bytes.size());
  std::size_t copy_len = std::min<std::size_t>(file_bytes.size(), kAxonByteLength);
  for (std::size_t i = 0; i < copy_len; ++i) {
    input.byte_seq[i] = static_cast<std::int64_t>(file_bytes[i]);
  }

  std::vector<float> pe = fixed_v2_pe_features(file_bytes);
  std::copy_n(pe.data(), kAxonPeFeatureDim, input.pe_features.data());

  std::vector<std::uint8_t> stat_source(
      file_bytes.begin(),
      file_bytes.begin() + static_cast<std::ptrdiff_t>(copy_len));
  std::vector<float> stat = statistical_features(stat_source);
  std::copy_n(stat.data(), kAxonStatFeatureDim, input.stat_features.data());
  return input;
}

std::vector<float> make_stage2_features(
    const std::vector<std::uint8_t>& file_bytes,
    const InferenceInput& input,
    float base_prob_malicious) {
  std::vector<float> features;
  features.reserve(kStage2FeatureDim);
  double prob = static_cast<double>(base_prob_malicious);
  double clipped_prob = std::min(std::max(prob, 1.0e-6), 1.0 - 1.0e-6);
  features.push_back(static_cast<float>(prob));
  features.push_back(static_cast<float>(prob * prob));
  features.push_back(static_cast<float>(std::fabs(prob - 0.5)));
  features.push_back(static_cast<float>(std::log(std::max(prob, 1.0e-6))));
  features.push_back(static_cast<float>(std::log(std::max(1.0 - prob, 1.0e-6))));
  features.push_back(static_cast<float>(std::log(clipped_prob / (1.0 - clipped_prob))));
  features.insert(features.end(), input.stat_features.begin(), input.stat_features.end());
  features.insert(features.end(), input.pe_features.begin(), input.pe_features.end());
  std::vector<float> light = lightweight_features(file_bytes);
  features.insert(features.end(), light.begin(), light.end());
  std::vector<std::uint8_t> byte_seq(kAxonByteLength, 0);
  std::size_t copy_len = std::min<std::size_t>(file_bytes.size(), kAxonByteLength);
  std::copy_n(file_bytes.begin(), copy_len, byte_seq.begin());
  std::vector<float> byte_summary = byte_summary_features(byte_seq);
  features.insert(features.end(), byte_summary.begin(), byte_summary.end());
  std::vector<float> content = content_pe_v1_features(file_bytes);
  features.insert(features.end(), content.begin(), content.end());
  if (features.size() < kStage2FeatureDim) {
    features.resize(kStage2FeatureDim, 0.0f);
  }
  if (features.size() > kStage2FeatureDim) {
    features.resize(kStage2FeatureDim);
  }
  return features;
}

Prediction predict_bytes_native(
    kvd_handle* handle,
    const std::vector<std::uint8_t>& bytes,
    InferenceInput* out_input = nullptr,
    PredictionCapture* out_capture = nullptr) {
  // Capture diagnostics from the same feature extraction and prediction pass.
  InferenceInput input = make_inference_input(bytes);
  if (out_input) {
    *out_input = input;
  }
  std::array<float, 2> base_logits{};
  Prediction prediction = handle->model->predict(
      input,
      handle->config.threshold,
      out_capture ? &base_logits : nullptr);
  if (!prediction.ok) {
    return prediction;
  }
  prediction.base_prediction = prediction.prediction;
  prediction.base_confidence = prediction.confidence;
  prediction.base_prob_benign = prediction.prob_benign;
  prediction.base_prob_malicious = prediction.prob_malicious;
  if (out_capture) {
    out_capture->input = input;
    out_capture->base_logits = base_logits;
    out_capture->base_probabilities = {
        prediction.base_prob_benign,
        prediction.base_prob_malicious};
  }
  if (handle->stage2_model && handle->stage2_model->ok()) {
    std::vector<float> stage2_features = make_stage2_features(bytes, input, prediction.base_prob_malicious);
    float stage2_prob = handle->stage2_model->predict_probability(stage2_features);
    float threshold = handle->stage2_model->threshold();
    prediction.stage2_enabled = true;
    prediction.stage2_threshold = threshold;
    prediction.stage2_feature_dim = stage2_features.size();
    prediction.prob_malicious = stage2_prob;
    prediction.prob_benign = 1.0f - stage2_prob;
    prediction.prediction = stage2_prob >= threshold ? 1 : 0;
    prediction.confidence = prediction.prediction == 1 ? prediction.prob_malicious : prediction.prob_benign;
    if (out_capture) {
      out_capture->stage2_features = std::move(stage2_features);
    }
  }
  return prediction;
}

bool validate_parity_diagnostics_options(
    const kvd_parity_diagnostics_options_v1* options,
    std::string& error_code,
    std::string& error_message) {
  if (!options) {
    error_code = "invalid_diagnostics_options";
    error_message = "Parity diagnostics options are required.";
    return false;
  }
  if (options->struct_size < sizeof(kvd_parity_diagnostics_options_v1)) {
    error_code = "invalid_diagnostics_options";
    error_message = "Parity diagnostics options struct_size is too small.";
    return false;
  }
  if (options->abi_version != KVD_PARITY_DIAGNOSTICS_ABI_VERSION_V1) {
    error_code = "unsupported_diagnostics_abi";
    error_message = "Parity diagnostics ABI version is not supported.";
    return false;
  }
  const std::uint64_t all_components = KVD_PARITY_DIAGNOSTICS_COMPONENT_ALL_V1;
  if (options->component_mask == 0 || (options->component_mask & ~all_components) != 0) {
    error_code = "invalid_component_mask";
    error_message = "component_mask must contain only supported parity components.";
    return false;
  }
  const std::uint64_t drilldown = options->drilldown_component;
  if ((drilldown == 0 && options->block_elements != 0) ||
      (drilldown != 0 &&
       ((drilldown & (drilldown - 1)) != 0 ||
        (drilldown & all_components) == 0 ||
        (drilldown & options->component_mask) == 0))) {
    error_code = "invalid_drilldown_component";
    error_message = "drilldown_component and block_elements must both be zero or select one component.";
    return false;
  }
  if (drilldown != 0 && (options->block_elements < 1 || options->block_elements > 256)) {
    error_code = "invalid_block_elements";
    error_message = "block_elements must be between 1 and 256.";
    return false;
  }
  if (!options->hmac_key || options->hmac_key_len != 32) {
    error_code = "invalid_hmac_key";
    error_message = "Parity diagnostics require exactly 32 HMAC key bytes.";
    return false;
  }
  return true;
}

bool parity_diagnostics_json(
    const Prediction& prediction,
    const PredictionCapture& capture,
    const kvd_parity_diagnostics_options_v1& options,
    std::string& out_json,
    std::string& error_code,
    std::string& error_message) {
  if ((options.component_mask & KVD_PARITY_DIAGNOSTICS_COMPONENT_STAGE2_FEATURES_V1) != 0 &&
      capture.stage2_features.size() != kStage2FeatureDim) {
    error_code = "stage2_features_unavailable";
    error_message = "Stage-2 features were requested but no valid Stage-2 prediction was captured.";
    return false;
  }

  LocalHmacSha256 hmac(options.hmac_key, options.hmac_key_len);
  if (!hmac.valid()) {
    error_code = "hmac_unavailable";
    error_message = "Windows BCrypt HMAC-SHA256 initialization failed.";
    return false;
  }

  const std::array<DiagnosticTensor, 6> tensors{{
      {
          KVD_PARITY_DIAGNOSTICS_COMPONENT_BYTE_SEQ_V1,
          "byte_seq",
          "i64le",
          capture.input.byte_seq.size(),
          capture.input.byte_seq.data(),
          true,
      },
      {
          KVD_PARITY_DIAGNOSTICS_COMPONENT_PE_FEATURES_V1,
          "pe_features",
          "f32le",
          capture.input.pe_features.size(),
          capture.input.pe_features.data(),
          false,
      },
      {
          KVD_PARITY_DIAGNOSTICS_COMPONENT_STAT_FEATURES_V1,
          "stat_features",
          "f32le",
          capture.input.stat_features.size(),
          capture.input.stat_features.data(),
          false,
      },
      {
          KVD_PARITY_DIAGNOSTICS_COMPONENT_BASE_LOGITS_V1,
          "base_logits",
          "f32le",
          capture.base_logits.size(),
          capture.base_logits.data(),
          false,
      },
      {
          KVD_PARITY_DIAGNOSTICS_COMPONENT_BASE_PROBABILITIES_V1,
          "base_probabilities",
          "f32le",
          capture.base_probabilities.size(),
          capture.base_probabilities.data(),
          false,
      },
      {
          KVD_PARITY_DIAGNOSTICS_COMPONENT_STAGE2_FEATURES_V1,
          "stage2_features",
          "f32le",
          capture.stage2_features.size(),
          capture.stage2_features.data(),
          false,
      },
  }};

  std::ostringstream out;
  out.setf(std::ios::fixed);
  out.precision(8);
  out << "{\"ok\":true"
      << ",\"prediction\":" << prediction.prediction
      << ",\"prob_benign\":" << prediction.prob_benign
      << ",\"prob_malicious\":" << prediction.prob_malicious
      << ",\"base_model\":{"
      << "\"prediction\":" << prediction.base_prediction
      << ",\"prob_benign\":" << prediction.base_prob_benign
      << ",\"prob_malicious\":" << prediction.base_prob_malicious << "}"
      << ",\"stage2\":{\"enabled\":" << (prediction.stage2_enabled ? "true" : "false")
      << ",\"prob_malicious\":";
  if (prediction.stage2_enabled) {
    out << prediction.prob_malicious;
  } else {
    out << "null";
  }
  out << "}"
      << ",\"diagnostics\":{"
      << "\"schema\":\"axon_parity_diagnostics_v1\""
      << ",\"encoding\":\"axon_tensor_le_v1\""
      << ",\"digest\":\"hmac-sha256\""
      << ",\"components\":{";

  bool first_component = true;
  for (const auto& tensor : tensors) {
    if ((options.component_mask & tensor.component) == 0) {
      continue;
    }
    std::string whole_digest;
    if (!tensor_hmac_digest(hmac, tensor, 0, tensor.element_count, whole_digest)) {
      error_code = "diagnostic_hmac_failed";
      error_message = "BCrypt HMAC-SHA256 failed while hashing a parity component.";
      return false;
    }
    if (!first_component) {
      out << ",";
    }
    first_component = false;
    out << "\"" << tensor.name << "\":{"
        << "\"dtype\":\"" << tensor.dtype << "\""
        << ",\"shape\":[" << tensor.element_count << "]"
        << ",\"digest\":\"" << whole_digest << "\"";
    if (options.drilldown_component == tensor.component) {
      out << ",\"blocks\":[";
      bool first_block = true;
      for (std::size_t start = 0; start < tensor.element_count; start += options.block_elements) {
        const std::size_t count = std::min<std::size_t>(
            options.block_elements,
            tensor.element_count - start);
        std::string block_digest;
        if (!tensor_hmac_digest(hmac, tensor, start, count, block_digest)) {
          error_code = "diagnostic_hmac_failed";
          error_message = "BCrypt HMAC-SHA256 failed while hashing a parity block.";
          return false;
        }
        if (!first_block) {
          out << ",";
        }
        first_block = false;
        out << "{\"start\":" << start
            << ",\"count\":" << count
            << ",\"digest\":\"" << block_digest << "\"}";
      }
      out << "]";
    }
    out << "}";
  }
  out << "}}}";
  out_json = out.str();
  return true;
}

AxonConfig config_from_api(const kvd_config* cfg) {
  AxonConfig out;
  if (!cfg) {
    return out;
  }
  if (cfg->onnx_model_path && cfg->onnx_model_path[0] != '\0') {
    out.onnx_model_path = cfg->onnx_model_path;
  } else if (cfg->model_path && cfg->model_path[0] != '\0') {
    out.onnx_model_path = cfg->model_path;
  }
  if (cfg->family_classifier_json_path && cfg->family_classifier_json_path[0] != '\0') {
    out.family_classifier_json_path = cfg->family_classifier_json_path;
  }
  if (cfg->stage2_model_json_path && cfg->stage2_model_json_path[0] != '\0') {
    out.stage2_model_json_path = cfg->stage2_model_json_path;
  }
  if (cfg->archive_scanner_path && cfg->archive_scanner_path[0] != '\0') {
    out.archive_scanner_path = cfg->archive_scanner_path;
  }
  if (cfg->allowed_scan_root) {
    out.allowed_scan_root = cfg->allowed_scan_root;
  }
  out.max_file_size = cfg->max_file_size;
  if (cfg->prediction_threshold > 0.0f && cfg->prediction_threshold < 1.0f) {
    out.threshold = cfg->prediction_threshold;
  }
  out.scan_nested = cfg->scan_nested != 0;
  return out;
}

int unsupported_train(char** out_json, size_t* out_len) {
  return write_error("unsupported_operation", "Axon ONNX DLL is inference-only and does not train models.", out_json, out_len);
}

}  // namespace

extern "C" {

KVD_API kvd_handle* KVD_CALL kvd_create(const kvd_config* config) {
  if (!config) {
    return nullptr;
  }
  AxonConfig cfg = config_from_api(config);
  if (cfg.onnx_model_path.empty()) {
    return nullptr;
  }
  try {
    auto handle = std::make_unique<kvd_handle>();
    handle->config = std::move(cfg);
    handle->model = std::make_shared<AxonOnnxModel>(handle->config.onnx_model_path);
    if (!handle->config.stage2_model_json_path.empty()) {
      auto stage2 = Stage2HgbModel::load_from_json(handle->config.stage2_model_json_path);
      if (!stage2 || !stage2->ok() || stage2->n_features() != kStage2FeatureDim) {
        return nullptr;
      }
      handle->stage2_model = std::make_shared<Stage2HgbModel>(std::move(*stage2));
    }
    if (!handle->config.family_classifier_json_path.empty()) {
      auto family = FamilyClassifier::load_from_json(handle->config.family_classifier_json_path);
      if (!family || !family->ok()) {
        return nullptr;
      }
      handle->family_classifier = std::make_shared<FamilyClassifier>(std::move(*family));
    }
    return handle.release();
  } catch (...) {
    return nullptr;
  }
}

KVD_API void KVD_CALL kvd_destroy(kvd_handle* handle) {
  delete handle;
}

KVD_API int KVD_CALL kvd_scan_path(kvd_handle* handle, const char* path, char** out_json, size_t* out_len) {
  if (!handle || !handle->model || !path) {
    return write_error("invalid_argument", "handle and path are required.", out_json, out_len);
  }
  std::string path_text(path);
  if (!path_allowed(path_text, handle->config.allowed_scan_root)) {
    return write_error("path_not_allowed", "Input path is outside allowed_scan_root.", out_json, out_len);
  }
  if (handle->config.scan_nested) {
    std::string report_json;
    std::string scanner_temp_root;
    std::string scan_error;
    if (!run_archive_scan_json(handle->config.archive_scanner_path, path_text, report_json, scanner_temp_root, scan_error)) {
      return write_error("archive_scan_failed", scan_error, out_json, out_len);
    }
    std::vector<ArchivePeTarget> targets = archive_pe_targets(report_json);
    std::vector<std::pair<ArchivePeTarget, Prediction>> predictions;
    predictions.reserve(targets.size());
    for (const auto& target : targets) {
      std::vector<std::uint8_t> inner_bytes;
      std::string error;
      Prediction prediction;
      if (!read_file_bytes_limited(target.extracted_path, inner_bytes, error, handle->config.max_file_size)) {
        if (error == "file_too_large") {
          prediction.error_code = "file_too_large";
          prediction.error = "Inner PE exceeds max_file_size.";
        } else {
          prediction.error_code = "file_read_failed";
          prediction.error = error;
        }
      } else {
        prediction = predict_bytes_native(handle, inner_bytes, nullptr);
      }
      predictions.push_back({target, prediction});
    }
    std::string response = nested_archive_prediction_json(handle, path_text, report_json, predictions);
    cleanup_archive_scan_temp(report_json, scanner_temp_root);
    return write_string_out(response, out_json, out_len);
  }
  std::vector<std::uint8_t> bytes;
  std::string error;
  if (!read_file_bytes_limited(path_text, bytes, error, handle->config.max_file_size)) {
    if (error == "file_too_large") {
      return write_error("file_too_large", "Input file exceeds max_file_size.", out_json, out_len);
    }
    return write_error("file_read_failed", error, out_json, out_len);
  }
  InferenceInput input;
  Prediction prediction = predict_bytes_native(handle, bytes, &input);
  std::optional<FamilyPrediction> family;
  if (prediction.ok && prediction.prediction == 1 && handle->family_classifier) {
    std::vector<float> family_features;
    family_features.reserve(kAxonPeFeatureDim + kAxonStatFeatureDim);
    family_features.insert(family_features.end(), input.pe_features.begin(), input.pe_features.end());
    family_features.insert(family_features.end(), input.stat_features.begin(), input.stat_features.end());
    family = handle->family_classifier->predict(family_features);
  }
  return write_string_out(
      prediction_json(prediction, path_text, handle->config.threshold, family),
      out_json,
      out_len);
}

KVD_API int KVD_CALL kvd_parity_diagnostics_path_v1(
    kvd_handle* handle,
    const char* path,
    const kvd_parity_diagnostics_options_v1* options,
    char** out_json,
    size_t* out_len) {
  if (!out_json || !out_len) {
    return -1;
  }
  *out_json = nullptr;
  *out_len = 0;
  try {
    if (!handle || !handle->model || !path) {
      return write_error(
          "invalid_argument",
          "handle and path are required for parity diagnostics.",
          out_json,
          out_len);
    }
    std::string error_code;
    std::string error_message;
    if (!validate_parity_diagnostics_options(options, error_code, error_message)) {
      return write_error(error_code, error_message, out_json, out_len);
    }
    if ((options->component_mask & KVD_PARITY_DIAGNOSTICS_COMPONENT_STAGE2_FEATURES_V1) != 0 &&
        (!handle->stage2_model || !handle->stage2_model->ok())) {
      return write_error(
          "stage2_features_unavailable",
          "Stage-2 features require a valid Stage-2 model.",
          out_json,
          out_len);
    }

    const std::string path_text(path);
    if (!path_allowed(path_text, handle->config.allowed_scan_root)) {
      return write_error(
          "path_not_allowed",
          "Input path is outside allowed_scan_root.",
          out_json,
          out_len);
    }
    std::vector<std::uint8_t> bytes;
    std::string read_error;
    if (!read_file_bytes_limited(path_text, bytes, read_error, handle->config.max_file_size)) {
      if (read_error == "file_too_large") {
        return write_error(
            "file_too_large",
            "Input file exceeds max_file_size.",
            out_json,
            out_len);
      }
      return write_error(
          "file_read_failed",
          "Failed to read input for parity diagnostics.",
          out_json,
          out_len);
    }

    PredictionCapture capture;
    Prediction prediction = predict_bytes_native(handle, bytes, nullptr, &capture);
    if (!prediction.ok) {
      return write_error(
          "prediction_failed",
          "Native prediction failed during parity capture.",
          out_json,
          out_len);
    }
    std::string response;
    if (!parity_diagnostics_json(
            prediction,
            capture,
            *options,
            response,
            error_code,
            error_message)) {
      return write_error(error_code, error_message, out_json, out_len);
    }
    return write_string_out(response, out_json, out_len);
  } catch (...) {
    static constexpr char kUnexpectedError[] =
        "{\"ok\":false,\"error_code\":\"diagnostics_internal_error\","
        "\"error\":\"Parity diagnostics failed unexpectedly.\"}";
    return write_literal_out_noexcept(
        kUnexpectedError,
        sizeof(kUnexpectedError) - 1,
        out_json,
        out_len);
  }
}

KVD_API int KVD_CALL kvd_scan_bytes(kvd_handle* handle, const unsigned char* bytes, size_t len, char** out_json, size_t* out_len) {
  if (!handle || !handle->model || (!bytes && len > 0)) {
    return write_error("invalid_argument", "handle and bytes are required.", out_json, out_len);
  }
  if (handle->config.max_file_size > 0 && len > handle->config.max_file_size) {
    return write_error("file_too_large", "Input byte buffer exceeds max_file_size.", out_json, out_len);
  }
  std::vector<std::uint8_t> buffer;
  if (len > 0) {
    buffer.assign(bytes, bytes + len);
  }
  InferenceInput input;
  Prediction prediction = predict_bytes_native(handle, buffer, &input);
  std::optional<FamilyPrediction> family;
  if (prediction.ok && prediction.prediction == 1 && handle->family_classifier) {
    std::vector<float> family_features;
    family_features.reserve(kAxonPeFeatureDim + kAxonStatFeatureDim);
    family_features.insert(family_features.end(), input.pe_features.begin(), input.pe_features.end());
    family_features.insert(family_features.end(), input.stat_features.begin(), input.stat_features.end());
    family = handle->family_classifier->predict(family_features);
  }
  return write_string_out(
      prediction_json(prediction, "<bytes>", handle->config.threshold, family),
      out_json,
      out_len);
}

KVD_API int KVD_CALL kvd_scan_paths(kvd_handle* handle, const char** paths, size_t count, char** out_json, size_t* out_len) {
  if (!handle || (!paths && count > 0)) {
    return write_error("invalid_argument", "handle and paths are required.", out_json, out_len);
  }
  std::ostringstream arr;
  arr << "[";
  for (size_t i = 0; i < count; ++i) {
    if (i > 0) {
      arr << ",";
    }
    char* item_json = nullptr;
    size_t item_len = 0;
    int rc = kvd_scan_path(handle, paths[i], &item_json, &item_len);
    if (rc == 0 && item_json) {
      arr.write(item_json, static_cast<std::streamsize>(item_len));
      kvd_free(item_json);
    } else {
      arr << error_json("scan_failed", "Failed to scan path.");
    }
  }
  arr << "]";
  return write_string_out(arr.str(), out_json, out_len);
}

KVD_API int KVD_CALL kvd_train_path(kvd_handle*, const char*, int, char** out_json, size_t* out_len) {
  return unsupported_train(out_json, out_len);
}

KVD_API int KVD_CALL kvd_train_paths(kvd_handle*, const char**, size_t, int, char** out_json, size_t* out_len) {
  return unsupported_train(out_json, out_len);
}

KVD_API int KVD_CALL kvd_train_from_path(kvd_handle* handle, const char* path, int label, char** out_json, size_t* out_len) {
  return kvd_train_path(handle, path, label, out_json, out_len);
}

KVD_API void KVD_CALL kvd_signature_flush(kvd_handle*) {}

KVD_API void KVD_CALL kvd_free(char* p) {
  std::free(p);
}

KVD_API int KVD_CALL kvd_validate_models(const kvd_config* config, char** out_error, size_t* out_len) {
  if (!config) {
    return KVD_MODEL_ERR_INVALID_ARGUMENT;
  }
  AxonConfig cfg = config_from_api(config);
  if (cfg.onnx_model_path.empty()) {
    if (out_error && out_len) {
      write_string_out("onnx_model_main_missing", out_error, out_len);
    }
    return KVD_MODEL_ERR_ONNX_MAIN_MISSING;
  }
  std::error_code ec;
  if (!std::filesystem::exists(path_from_utf8(cfg.onnx_model_path), ec) || ec) {
    if (out_error && out_len) {
      write_string_out("onnx_model_main_missing", out_error, out_len);
    }
    return KVD_MODEL_ERR_ONNX_MAIN_MISSING;
  }
  try {
    AxonOnnxModel model(cfg.onnx_model_path);
  } catch (...) {
    if (out_error && out_len) {
      write_string_out("onnx_model_main_invalid", out_error, out_len);
    }
    return KVD_MODEL_ERR_ONNX_MAIN_INVALID;
  }
  if (!cfg.stage2_model_json_path.empty()) {
    if (!std::filesystem::exists(path_from_utf8(cfg.stage2_model_json_path), ec) || ec) {
      if (out_error && out_len) {
        write_string_out("stage2_model_missing", out_error, out_len);
      }
      return KVD_MODEL_ERR_INVALID_ARGUMENT;
    }
    auto stage2 = Stage2HgbModel::load_from_json(cfg.stage2_model_json_path);
    if (!stage2 || !stage2->ok() || stage2->n_features() != kStage2FeatureDim) {
      if (out_error && out_len) {
        write_string_out("stage2_model_invalid", out_error, out_len);
      }
      return KVD_MODEL_ERR_INVALID_ARGUMENT;
    }
  }
  if (!cfg.family_classifier_json_path.empty()) {
    if (!std::filesystem::exists(path_from_utf8(cfg.family_classifier_json_path), ec) || ec) {
      if (out_error && out_len) {
        write_string_out("family_classifier_missing", out_error, out_len);
      }
      return KVD_MODEL_ERR_FAMILY_MISSING;
    }
    auto family = FamilyClassifier::load_from_json(cfg.family_classifier_json_path);
    if (!family || !family->ok()) {
      if (out_error && out_len) {
        write_string_out("family_classifier_invalid", out_error, out_len);
      }
      return KVD_MODEL_ERR_FAMILY_INVALID;
    }
  }
  if (out_error && out_len) {
    write_string_out("ok", out_error, out_len);
  }
  return KVD_MODEL_OK;
}

KVD_API int KVD_CALL kvd_extract_pe_features(const char* path, float* out_features, size_t out_len) {
  if (!path || !out_features || out_len < kAxonPeFeatureDim) {
    return -1;
  }
  std::vector<std::uint8_t> bytes;
  std::string error;
  if (!read_file_bytes(path, bytes, error)) {
    return -2;
  }
  std::vector<float> features = fixed_v2_pe_features(bytes);
  std::copy_n(features.data(), kAxonPeFeatureDim, out_features);
  return 0;
}

KVD_API int KVD_CALL kvd_extract_pe_features_batch(
    const char** paths,
    size_t count,
    float* out_features,
    size_t feature_dim,
    int* out_status,
    unsigned int thread_count) {
  if (!paths || !out_features || !out_status || feature_dim < kAxonPeFeatureDim) {
    return -1;
  }
  if (count == 0) {
    return 0;
  }
  std::size_t workers = thread_count > 0 ? thread_count : std::thread::hardware_concurrency();
  if (workers == 0) {
    workers = 1;
  }
  workers = std::min<std::size_t>(workers, count);
  workers = std::min<std::size_t>(workers, 16);
  std::atomic<std::size_t> next{0};
  std::vector<std::thread> pool;
  pool.reserve(workers);
  for (std::size_t t = 0; t < workers; ++t) {
    pool.emplace_back([&]() {
      for (;;) {
        std::size_t i = next.fetch_add(1);
        if (i >= count) {
          return;
        }
        float* dst = out_features + i * feature_dim;
        std::fill_n(dst, feature_dim, 0.0f);
        if (!paths[i]) {
          out_status[i] = -1;
          continue;
        }
        int rc = kvd_extract_pe_features(paths[i], dst, feature_dim);
        out_status[i] = rc;
      }
    });
  }
  for (auto& worker : pool) {
    worker.join();
  }
  return 0;
}

KVD_API size_t KVD_CALL kvd_get_pe_feature_dimension(void) {
  return kAxonPeFeatureDim;
}

KVD_API char* KVD_CALL axon_predict_json(const char* request_json) {
  if (!request_json) {
    std::string text = error_json("invalid_argument", "request_json is null.");
    char* out = nullptr;
    size_t len = 0;
    write_string_out(text, &out, &len);
    return out;
  }
  std::string message =
      "axon_predict_json is available only as a compatibility symbol in the C++ ONNX DLL. "
      "Use kvd_create plus kvd_scan_path/kvd_scan_bytes, or the Rust tools/predict_dll wrapper for JSON checkpoint requests.";
  char* out = nullptr;
  size_t len = 0;
  write_string_out(error_json("unsupported_operation", message), &out, &len);
  return out;
}

KVD_API void KVD_CALL axon_string_free(char* ptr) {
  kvd_free(ptr);
}

KVD_API const char* KVD_CALL axon_version(void) {
  return kAxonVersion;
}

}  // extern "C"
