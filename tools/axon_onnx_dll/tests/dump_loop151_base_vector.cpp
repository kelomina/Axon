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
#include "../src/axon_onnx_predict.cpp"
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

#include <iomanip>
#include <iostream>

int main(int argc, char** argv) {
  if (argc != 2) return 2;
  std::vector<std::uint8_t> bytes;
  std::string error;
  if (!read_file_bytes(argv[1], bytes, error)) return 3;
  const InferenceInput input = make_inference_input(bytes);
  const auto import_stats = collect_import_export_stats(bytes, parse_pe(bytes));
  std::cerr << "imports=" << import_stats.import_names.size()
            << " ordinals=" << import_stats.ordinal_imports << " names=";
  for (const auto& name : import_stats.import_names) std::cerr << name << ',';
  std::cerr << '\n';
  const auto prediction = input;
  (void)prediction;
  const auto base = make_stage2_features(bytes, input, 0.9315964579582214f);
  std::cout << std::setprecision(9) << "{\"base\":[";
  for (std::size_t index = 0; index < base.size(); ++index) {
    if (index) std::cout << ',';
    std::cout << base[index];
  }
  std::cout << "]}\n";
  return 0;
}
