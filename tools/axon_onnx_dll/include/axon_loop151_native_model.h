#pragma once

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

namespace axon_loop151_native {

namespace detail {
// Parsed JSON node. Defined in the implementation file and named here only so
// the loaders can build a model from an already-parsed document. Without it,
// nested models had to be re-serialised with json_encode and parsed a second
// time, which dominated model load time.
class JsonValue;
}  // namespace detail

class NativeScoreModel {
 public:
  NativeScoreModel();
  NativeScoreModel(NativeScoreModel&&) noexcept;
  NativeScoreModel& operator=(NativeScoreModel&&) noexcept;
  NativeScoreModel(const NativeScoreModel&) = delete;
  NativeScoreModel& operator=(const NativeScoreModel&) = delete;
  ~NativeScoreModel();

  static std::unique_ptr<NativeScoreModel> load_file(
      const std::string& path,
      std::string& error);

  static std::unique_ptr<NativeScoreModel> load_document(
      const std::string& document,
      const std::string& source_name,
      std::string& error);

  // Internal: build from an already-parsed node, skipping a serialise/re-parse
  // round trip. Prefer load_file or load_document from outside this library.
  static std::unique_ptr<NativeScoreModel> load_parsed(
      const detail::JsonValue& root,
      const std::string& source_name,
      std::string& error);

  float predict_probability(
      const std::vector<float>& features,
      std::string* error = nullptr) const;

  std::size_t feature_count() const;
  const std::string& model_type() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

class NativeStackModel {
 public:
  NativeStackModel();
  NativeStackModel(NativeStackModel&&) noexcept;
  NativeStackModel& operator=(NativeStackModel&&) noexcept;
  NativeStackModel(const NativeStackModel&) = delete;
  NativeStackModel& operator=(const NativeStackModel&) = delete;
  ~NativeStackModel();

  static std::unique_ptr<NativeStackModel> load_file(
      const std::string& path,
      std::string& error);

  static std::unique_ptr<NativeStackModel> load_document(
      const std::string& document,
      const std::string& source_name,
      std::string& error);

  float predict_probability(
      const std::vector<float>& features,
      std::string* error = nullptr) const;

  std::size_t feature_count() const;
  std::size_t base_model_count() const;
  float threshold() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace axon_loop151_native
