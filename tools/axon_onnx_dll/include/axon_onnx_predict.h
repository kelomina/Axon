#pragma once

#include <stddef.h>
#include <stdint.h>

#if defined(_WIN32)
  #if defined(KVD_BUILD_DLL)
    #define KVD_API __declspec(dllexport)
  #else
    #define KVD_API __declspec(dllimport)
  #endif
  #define KVD_CALL __cdecl
#else
  #define KVD_API
  #define KVD_CALL
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef struct kvd_handle kvd_handle;

/*
 * This struct intentionally mirrors KoloVirusDetector's predictor DLL config.
 *
 * Axon ONNX DLL uses:
 *   - onnx_model_path: path to the Axon ONNX model; keep its .onnx.data file in the same directory
 *   - prediction_threshold: malware threshold, default 0.5 when <= 0
 *   - allowed_scan_root: optional UTF-8 canonical directory guard for kvd_scan_path;
 *     configure the physical target root when scanning through a directory link
 *   - max_file_size: optional file size guard; 0 means no explicit guard
 *   - stage2_model_json_path: optional Loop28 Stage-2 HGB JSON model
 *   - archive_scanner_path: optional native helper exe launched internally by the DLL for MSI/ZIP/7z/CAB
 *   - scan_nested: non-zero makes kvd_scan_path inspect inner PE files through that internal helper
 *
 * LightGBM model fields are kept for ABI compatibility and ignored by Axon.
 * family_classifier_json_path is optional; when present, malware results include malware_family.
 */
typedef struct kvd_config {
  const char* model_path;
  const char* model_normal_path;
  const char* model_packed_path;
  const char* family_classifier_json_path;
  const char* allowed_scan_root;
  unsigned int max_file_size;
  float prediction_threshold;
  const char* onnx_model_path;
  const char* onnx_model_normal_path;
  const char* onnx_model_packed_path;
  const char* stage2_model_json_path;
  const char* archive_scanner_path;
  int scan_nested;
} kvd_config;

typedef enum kvd_model_check_result {
  KVD_MODEL_OK = 0,
  KVD_MODEL_ERR_INVALID_ARGUMENT = -1,
  KVD_MODEL_ERR_FAMILY_MISSING = -17,
  KVD_MODEL_ERR_FAMILY_INVALID = -18,
  KVD_MODEL_ERR_ONNX_MAIN_MISSING = -30,
  KVD_MODEL_ERR_ONNX_MAIN_INVALID = -31,
  KVD_MODEL_ERR_OOM = -100
} kvd_model_check_result;

/*
 * Versioned, append-only parity diagnostics ABI. Callers must zero-initialize
 * the structure, set struct_size to sizeof(kvd_parity_diagnostics_options_v1),
 * and provide a private 32-byte HMAC key. Future versions may append fields.
 */
typedef enum kvd_parity_diagnostics_component_v1 {
  KVD_PARITY_DIAGNOSTICS_COMPONENT_BYTE_SEQ_V1 = 1u << 0,
  KVD_PARITY_DIAGNOSTICS_COMPONENT_PE_FEATURES_V1 = 1u << 1,
  KVD_PARITY_DIAGNOSTICS_COMPONENT_STAT_FEATURES_V1 = 1u << 2,
  KVD_PARITY_DIAGNOSTICS_COMPONENT_BASE_LOGITS_V1 = 1u << 3,
  KVD_PARITY_DIAGNOSTICS_COMPONENT_BASE_PROBABILITIES_V1 = 1u << 4,
  KVD_PARITY_DIAGNOSTICS_COMPONENT_STAGE2_FEATURES_V1 = 1u << 5,
  KVD_PARITY_DIAGNOSTICS_COMPONENT_ALL_V1 = (1u << 6) - 1u
} kvd_parity_diagnostics_component_v1;

enum { KVD_PARITY_DIAGNOSTICS_ABI_VERSION_V1 = 1 };

typedef struct kvd_parity_diagnostics_options_v1 {
  size_t struct_size;
  uint32_t abi_version;
  uint64_t component_mask;
  uint64_t drilldown_component;
  uint32_t block_elements;
  const unsigned char* hmac_key;
  size_t hmac_key_len;
} kvd_parity_diagnostics_options_v1;

KVD_API kvd_handle* KVD_CALL kvd_create(const kvd_config* config);
KVD_API void KVD_CALL kvd_destroy(kvd_handle* handle);

KVD_API int KVD_CALL kvd_scan_path(kvd_handle* handle, const char* path, char** out_json, size_t* out_len);
KVD_API int KVD_CALL kvd_scan_bytes(kvd_handle* handle, const unsigned char* bytes, size_t len, char** out_json, size_t* out_len);
KVD_API int KVD_CALL kvd_scan_paths(kvd_handle* handle, const char** paths, size_t count, char** out_json, size_t* out_len);

KVD_API int KVD_CALL kvd_parity_diagnostics_path_v1(
    kvd_handle* handle,
    const char* path,
    const kvd_parity_diagnostics_options_v1* options,
    char** out_json,
    size_t* out_len);

/* Training/signature APIs are exported for caller compatibility, but Axon ONNX DLL is inference-only. */
KVD_API int KVD_CALL kvd_train_path(kvd_handle* handle, const char* path, int label, char** out_json, size_t* out_len);
KVD_API int KVD_CALL kvd_train_paths(kvd_handle* handle, const char** paths, size_t count, int label, char** out_json, size_t* out_len);
KVD_API int KVD_CALL kvd_train_from_path(kvd_handle* handle, const char* path, int label, char** out_json, size_t* out_len);
KVD_API void KVD_CALL kvd_signature_flush(kvd_handle* handle);

KVD_API void KVD_CALL kvd_free(char* p);
KVD_API int KVD_CALL kvd_validate_models(const kvd_config* config, char** out_error, size_t* out_len);

KVD_API int KVD_CALL kvd_extract_pe_features(const char* path, float* out_features, size_t out_len);
KVD_API int KVD_CALL kvd_extract_pe_features_batch(
    const char** paths,
    size_t count,
    float* out_features,
    size_t feature_dim,
    int* out_status,
    unsigned int thread_count);
KVD_API size_t KVD_CALL kvd_get_pe_feature_dimension(void);

/* Legacy Axon JSON entry points kept for simple dynamic loading. */
KVD_API char* KVD_CALL axon_predict_json(const char* request_json);
KVD_API void KVD_CALL axon_string_free(char* ptr);
KVD_API const char* KVD_CALL axon_version(void);

#ifdef __cplusplus
}
#endif
