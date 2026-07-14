#ifndef AXON_PREDICT_H
#define AXON_PREDICT_H

#ifdef _WIN32
#define AXON_API __declspec(dllimport)
#else
#define AXON_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

// request_json: UTF-8 JSON object. Returns a heap-allocated UTF-8 JSON string.
// Release the returned pointer with axon_string_free.
AXON_API char *axon_predict_json(const char *request_json);

// Frees strings returned by axon_predict_json.
AXON_API void axon_string_free(char *ptr);

// Static version string. Do not free.
AXON_API const char *axon_version(void);

#ifdef __cplusplus
}
#endif

#endif
