// Steady-state scan benchmark for the packaged Loop151 native champion DLL.
//
// The one-shot example reports a single wall time that is dominated by model
// loading, which hides the number that actually decides real-time feasibility.
// This harness separates the three costs an AV integration cares about:
//
//   1. init      -- LoadLibrary + kvd_validate_models + kvd_create, paid once
//                   when the scanning service starts
//   2. footprint -- resident set after init and again after warm scans, so
//                   load-time parsing spikes can be told apart from the memory
//                   the engine actually holds while running
//   3. per-file  -- latency percentiles over many scans, after warmup
//
// kvd_scan_path is the realistic path (the DLL reads the file itself);
// kvd_scan_bytes is measured too so file I/O can be subtracted from compute.
//
// Usage:
//   axon_loop151_bench.exe --dll <axon_loop151_champion.dll>
//       --runtime-config <runtime/loop151_native_runtime.json>
//       --dir <sample directory> [--count 500] [--warmup 20]
//       [--allowed-root <dir>] [--max-file-size <bytes>] [--csv <out.csv>]

#include "axon_onnx_predict.h"

#if !defined(_WIN32)
#error "The Loop151 benchmark requires Windows."
#endif

#include <windows.h>
#include <psapi.h>

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <string>
#include <vector>

namespace {

using create_fn = kvd_handle* (KVD_CALL*)(const kvd_config*);
using destroy_fn = void (KVD_CALL*)(kvd_handle*);
using scan_path_fn = int (KVD_CALL*)(kvd_handle*, const char*, char**, size_t*);
using scan_bytes_fn = int (KVD_CALL*)(kvd_handle*, const unsigned char*, size_t, char**, size_t*);
using free_fn = void (KVD_CALL*)(char*);
using validate_fn = int (KVD_CALL*)(const kvd_config*, char**, size_t*);

using Clock = std::chrono::steady_clock;

double ms_since(const Clock::time_point& start) {
  return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

std::string wide_to_utf8(const std::wstring& value) {
  if (value.empty()) {
    return {};
  }
  const int required = WideCharToMultiByte(
      CP_UTF8, 0, value.data(), static_cast<int>(value.size()), nullptr, 0, nullptr, nullptr);
  if (required <= 0) {
    return {};
  }
  std::string result(static_cast<std::size_t>(required), '\0');
  const int written = WideCharToMultiByte(
      CP_UTF8, 0, value.data(), static_cast<int>(value.size()),
      result.data(), required, nullptr, nullptr);
  if (written != required) {
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

unsigned int parse_u32_or(const std::wstring& value, unsigned int fallback) {
  if (value.empty()) {
    return fallback;
  }
  wchar_t* end = nullptr;
  errno = 0;
  const unsigned long long parsed = std::wcstoull(value.c_str(), &end, 10);
  if (!end || *end != L'\0' || errno == ERANGE ||
      parsed > (std::numeric_limits<unsigned int>::max)()) {
    return fallback;
  }
  return static_cast<unsigned int>(parsed);
}

struct Footprint {
  double working_set_mib = 0.0;
  double peak_working_set_mib = 0.0;
  double private_mib = 0.0;
};

Footprint footprint() {
  Footprint result;
  PROCESS_MEMORY_COUNTERS_EX counters{};
  counters.cb = sizeof(counters);
  if (GetProcessMemoryInfo(
          GetCurrentProcess(),
          reinterpret_cast<PROCESS_MEMORY_COUNTERS*>(&counters),
          sizeof(counters))) {
    const double mib = 1024.0 * 1024.0;
    result.working_set_mib = static_cast<double>(counters.WorkingSetSize) / mib;
    result.peak_working_set_mib = static_cast<double>(counters.PeakWorkingSetSize) / mib;
    result.private_mib = static_cast<double>(counters.PrivateUsage) / mib;
  }
  return result;
}

void print_footprint(const char* label, const Footprint& value) {
  std::cout << "  " << std::left << std::setw(30) << label << std::right
            << std::fixed << std::setprecision(1)
            << std::setw(9) << value.working_set_mib << " MiB working set, "
            << std::setw(9) << value.private_mib << " MiB private\n";
}

struct LabelledSample {
  std::wstring path;
  int label = 0;
};

// Reads the 7:1:2 split csv (sample_index,sha256,label,split,date,source_path).
// Columns are located by header name so a reordered or extended csv still works.
std::vector<LabelledSample> read_split_csv(
    const std::wstring& csv_path, const std::string& wanted_split, std::size_t limit) {
  std::vector<LabelledSample> samples;
  std::ifstream input(csv_path);
  if (!input) {
    return samples;
  }
  std::string line;
  if (!std::getline(input, line)) {
    return samples;
  }
  if (line.size() >= 3 && static_cast<unsigned char>(line[0]) == 0xEF) {
    line.erase(0, 3);  // UTF-8 BOM
  }

  auto split_fields = [](const std::string& text) {
    std::vector<std::string> fields;
    std::string current;
    for (const char character : text) {
      if (character == ',') {
        fields.push_back(current);
        current.clear();
      } else if (character != '\r') {
        current.push_back(character);
      }
    }
    fields.push_back(current);
    return fields;
  };

  const std::vector<std::string> header = split_fields(line);
  std::size_t label_column = header.size();
  std::size_t path_column = header.size();
  std::size_t split_column = header.size();
  for (std::size_t index = 0; index < header.size(); ++index) {
    if (header[index] == "label") label_column = index;
    else if (header[index] == "source_path") path_column = index;
    else if (header[index] == "split") split_column = index;
  }
  if (label_column == header.size() || path_column == header.size()) {
    return samples;
  }

  // The split csv writes every benign row of a split before its malicious rows,
  // so taking the first N would yield a single-class sample and an undefined F1.
  // Read the whole split, then stride across it: the two class blocks are equal
  // sized, so an even stride is balanced and stays deterministic.
  std::vector<LabelledSample> all;
  while (std::getline(input, line)) {
    const std::vector<std::string> fields = split_fields(line);
    if (fields.size() <= (std::max)(label_column, path_column)) {
      continue;
    }
    if (!wanted_split.empty() && split_column < fields.size() &&
        fields[split_column] != wanted_split) {
      continue;
    }
    const std::string& utf8_path = fields[path_column];
    const int required = MultiByteToWideChar(
        CP_UTF8, 0, utf8_path.c_str(), static_cast<int>(utf8_path.size()), nullptr, 0);
    if (required <= 0) {
      continue;
    }
    std::wstring wide(static_cast<std::size_t>(required), L'\0');
    MultiByteToWideChar(
        CP_UTF8, 0, utf8_path.c_str(), static_cast<int>(utf8_path.size()),
        wide.data(), required);
    all.push_back({wide, std::atoi(fields[label_column].c_str())});
  }

  if (all.size() <= limit) {
    return all;
  }
  const double stride = static_cast<double>(all.size()) / static_cast<double>(limit);
  samples.reserve(limit);
  for (std::size_t index = 0; index < limit; ++index) {
    const std::size_t position =
        (std::min)(static_cast<std::size_t>(index * stride), all.size() - 1);
    samples.push_back(all[position]);
  }
  return samples;
}

struct Confusion {
  long long tp = 0, fp = 0, tn = 0, fn = 0;

  void add(int predicted, int label) {
    if (predicted == 1 && label == 1) ++tp;
    else if (predicted == 1 && label == 0) ++fp;
    else if (predicted == 0 && label == 0) ++tn;
    else ++fn;
  }
  long long total() const { return tp + fp + tn + fn; }
  double f1() const {
    const double denominator = 2.0 * tp + fp + fn;
    return denominator > 0.0 ? 2.0 * tp / denominator : 0.0;
  }
  double accuracy() const {
    return total() > 0 ? static_cast<double>(tp + tn) / total() : 0.0;
  }
};

void report_confusion(const char* label, const Confusion& matrix) {
  std::cout << "  " << std::left << std::setw(22) << label << std::right
            << std::fixed << std::setprecision(6)
            << " F1 " << matrix.f1()
            << "   acc " << matrix.accuracy()
            << std::setprecision(0)
            << "   errors " << std::setw(6) << (matrix.fp + matrix.fn)
            << "   FP " << std::setw(5) << matrix.fp
            << "   FN " << std::setw(5) << matrix.fn
            << "   n " << matrix.total() << "\n";
}

std::vector<std::wstring> list_files(const std::wstring& directory, std::size_t limit) {
  std::vector<std::wstring> files;
  std::wstring pattern = directory;
  if (!pattern.empty() && pattern.back() != L'\\' && pattern.back() != L'/') {
    pattern += L'\\';
  }
  const std::wstring prefix = pattern;
  pattern += L'*';

  WIN32_FIND_DATAW entry{};
  HANDLE find = FindFirstFileW(pattern.c_str(), &entry);
  if (find == INVALID_HANDLE_VALUE) {
    return files;
  }
  do {
    if (entry.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) {
      continue;
    }
    files.push_back(prefix + entry.cFileName);
  } while (files.size() < limit && FindNextFileW(find, &entry));
  FindClose(find);
  return files;
}

bool read_bytes(const std::wstring& path, std::vector<unsigned char>& bytes) {
  FILE* file = nullptr;
  if (_wfopen_s(&file, path.c_str(), L"rb") != 0 || !file) {
    return false;
  }
  bool ok = false;
  do {
    if (_fseeki64(file, 0, SEEK_END) != 0) break;
    const long long size = _ftelli64(file);
    if (size < 0) break;
    if (_fseeki64(file, 0, SEEK_SET) != 0) break;
    bytes.resize(static_cast<std::size_t>(size));
    const std::size_t read = bytes.empty() ? 0 : std::fread(bytes.data(), 1, bytes.size(), file);
    ok = read == bytes.size();
  } while (false);
  std::fclose(file);
  return ok;
}

// Pull `"key":<number>` out of the result JSON. The DLL only emits the
// timing_ms block when AXON_LOOP151_TIMING is set, so a miss is normal.
bool extract_number(const std::string& json, const std::string& key, double& value) {
  const std::string needle = "\"" + key + "\":";
  const std::size_t position = json.find(needle);
  if (position == std::string::npos) {
    return false;
  }
  const char* begin = json.c_str() + position + needle.size();
  char* end = nullptr;
  const double parsed = std::strtod(begin, &end);
  if (end == begin) {
    return false;
  }
  value = parsed;
  return true;
}

const char* const kStageKeys[] = {
    "input_features", "base_onnx", "stage2_features", "primary_stack",
    "conservative_stack", "content_features", "content_pe_v1", "content_pe_v2",
    "content_strings", "content_cross", "noise_stack", "selector",
    "authenticode", "total",
};

double percentile(std::vector<double> values, double fraction) {
  if (values.empty()) {
    return 0.0;
  }
  std::sort(values.begin(), values.end());
  const std::size_t index = static_cast<std::size_t>(fraction * (values.size() - 1) + 0.5);
  return values[(std::min)(index, values.size() - 1)];
}

void report_latency(const char* label, const std::vector<double>& samples) {
  if (samples.empty()) {
    std::cout << "  " << label << ": no samples\n";
    return;
  }
  double sum = 0.0;
  for (const double value : samples) {
    sum += value;
  }
  const double mean = sum / static_cast<double>(samples.size());
  const double p50 = percentile(samples, 0.50);
  std::cout << std::fixed << std::setprecision(3)
            << "  " << std::left << std::setw(18) << label << std::right
            << "  n=" << std::setw(6) << samples.size()
            << "  p50=" << std::setw(9) << p50
            << "  p90=" << std::setw(9) << percentile(samples, 0.90)
            << "  p99=" << std::setw(9) << percentile(samples, 0.99)
            << "  max=" << std::setw(9) << percentile(samples, 1.0)
            << "  mean=" << std::setw(9) << mean << " ms\n";
  std::cout << "  " << std::setw(18) << " " << "  throughput at p50: "
            << std::setprecision(1) << (p50 > 0.0 ? 1000.0 / p50 : 0.0) << " files/s\n";
}

int usage() {
  std::wcerr
      << L"Usage:\n"
      << L"  axon_loop151_bench.exe --dll <axon_loop151_champion.dll>\n"
      << L"      --runtime-config <runtime/loop151_native_runtime.json>\n"
      << L"      (--dir <sample directory> | --split-csv <split_712.csv> [--split test])\n"
      << L"      [--count 500] [--warmup 20]\n"
      << L"      [--allowed-root <dir>] [--max-file-size <bytes>] [--csv <out.csv>]\n"
      << L"\n"
      << L"  --split-csv supplies ground-truth labels, which enables the accuracy\n"
      << L"  comparison. Set AXON_LOOP151_TIMING=1 for the stage breakdown and\n"
      << L"  AXON_LOOP151_NO_ONNX_SHADOW=1 to score the ONNX-free chain alongside.\n";
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
  const std::wstring dir_path = get_argument(argc, argv, L"--dir");
  const std::wstring split_csv = get_argument(argc, argv, L"--split-csv");
  std::wstring split_name = get_argument(argc, argv, L"--split");
  if (split_csv.empty() && dir_path.empty()) {
    return usage();
  }
  if (!split_csv.empty() && split_name.empty()) {
    split_name = L"test";
  }
  const std::wstring allowed_root = get_argument(argc, argv, L"--allowed-root");
  const std::wstring csv_path = get_argument(argc, argv, L"--csv");
  const unsigned int count = parse_u32_or(get_argument(argc, argv, L"--count"), 500);
  const unsigned int warmup = parse_u32_or(get_argument(argc, argv, L"--warmup"), 20);
  const unsigned int max_file_size =
      parse_u32_or(get_argument(argc, argv, L"--max-file-size"), 0);
  if (dll_path.empty() || runtime_path.empty()) {
    return usage();
  }

  const std::string runtime_utf8 = wide_to_utf8(runtime_path);
  const std::string allowed_root_utf8 = wide_to_utf8(allowed_root);
  if (runtime_utf8.empty()) {
    std::wcerr << L"Path conversion to UTF-8 failed.\n";
    return 3;
  }

  const Footprint baseline = footprint();

  const Clock::time_point load_start = Clock::now();
  HMODULE module = LoadLibraryW(dll_path.c_str());
  const double load_ms = ms_since(load_start);
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
  config.allowed_scan_root = allowed_root.empty() ? nullptr : allowed_root_utf8.c_str();
  config.prediction_threshold = 0.5f;
  config.max_file_size = max_file_size;

  char* validation_json = nullptr;
  size_t validation_length = 0;
  const Clock::time_point validate_start = Clock::now();
  const int validation_code = validate(&config, &validation_json, &validation_length);
  const double validate_ms = ms_since(validate_start);
  if (validation_json) {
    free_string(validation_json);
  }
  if (validation_code != KVD_MODEL_OK) {
    std::cerr << "kvd_validate_models failed with code " << validation_code << "\n";
    FreeLibrary(module);
    return 7;
  }

  const Clock::time_point create_start = Clock::now();
  kvd_handle* handle = create(&config);
  const double create_ms = ms_since(create_start);
  if (!handle) {
    std::cerr << "kvd_create failed.\n";
    FreeLibrary(module);
    return 8;
  }
  const Footprint after_init = footprint();

  std::vector<std::wstring> files;
  std::vector<int> labels;
  if (!split_csv.empty()) {
    const std::vector<LabelledSample> samples =
        read_split_csv(split_csv, wide_to_utf8(split_name), count + warmup);
    for (const LabelledSample& sample : samples) {
      files.push_back(sample.path);
      labels.push_back(sample.label);
    }
    if (files.empty()) {
      std::wcerr << L"No rows read from: " << split_csv << L"\n";
      destroy(handle);
      FreeLibrary(module);
      return 9;
    }
  } else {
    files = list_files(dir_path, count + warmup);
  }
  if (files.empty()) {
    std::wcerr << L"No files found under: " << dir_path << L"\n";
    destroy(handle);
    FreeLibrary(module);
    return 9;
  }

  std::cout << "=== Loop151 native champion, steady-state scan benchmark ===\n";
  std::cout << "  files discovered   : " << files.size() << "\n";
  std::cout << "  warmup / measured  : " << warmup << " / "
            << (files.size() > warmup ? files.size() - warmup : 0) << "\n\n";

  std::cout << "=== 1. one-time init cost ===\n";
  std::cout << std::fixed << std::setprecision(1)
            << "  LoadLibrary            " << std::setw(10) << load_ms << " ms\n"
            << "  kvd_validate_models    " << std::setw(10) << validate_ms << " ms\n"
            << "  kvd_create             " << std::setw(10) << create_ms << " ms\n"
            << "  init total             " << std::setw(10)
            << (load_ms + validate_ms + create_ms) << " ms\n\n";

  std::cout << "=== 2. memory footprint ===\n";
  print_footprint("before loading the DLL", baseline);
  print_footprint("after init", after_init);

  std::vector<double> path_ms;
  std::vector<double> bytes_ms;
  std::vector<std::wstring> measured_files;
  std::map<std::string, std::vector<double>> stage_samples;
  Confusion full_chain;
  Confusion no_onnx_chain;
  bool have_labels = false;
  long long disagreements = 0;
  path_ms.reserve(files.size());
  bytes_ms.reserve(files.size());

  Footprint after_warmup{};
  std::size_t failures = 0;

  for (std::size_t index = 0; index < files.size(); ++index) {
    const std::string target_utf8 = wide_to_utf8(files[index]);
    if (target_utf8.empty()) {
      continue;
    }
    const bool measuring = index >= warmup;

    char* json = nullptr;
    size_t length = 0;
    const Clock::time_point scan_start = Clock::now();
    const int code = scan_path(handle, target_utf8.c_str(), &json, &length);
    const double elapsed = ms_since(scan_start);
    std::string result_json;
    if (json) {
      result_json.assign(json, length);
      free_string(json);
    }
    if (code != 0) {
      ++failures;
      continue;
    }
    if (measuring) {
      path_ms.push_back(elapsed);
      measured_files.push_back(files[index]);
      for (const char* key : kStageKeys) {
        double value = 0.0;
        if (extract_number(result_json, key, value)) {
          stage_samples[key].push_back(value);
        }
      }
      if (index < labels.size()) {
        double predicted = 0.0;
        if (extract_number(result_json, "prediction", predicted)) {
          full_chain.add(static_cast<int>(predicted), labels[index]);
          have_labels = true;
        }
        // The shadow block repeats the "prediction" key, so parse from the
        // "no_onnx" object onward rather than from the start of the document.
        const std::size_t shadow = result_json.find("\"no_onnx\":");
        if (shadow != std::string::npos) {
          double shadow_predicted = 0.0;
          if (extract_number(result_json.substr(shadow), "prediction", shadow_predicted)) {
            no_onnx_chain.add(static_cast<int>(shadow_predicted), labels[index]);
            if (static_cast<int>(shadow_predicted) != static_cast<int>(predicted)) {
              ++disagreements;
            }
          }
        }
      }
    }

    std::vector<unsigned char> raw;
    if (read_bytes(files[index], raw) && !raw.empty()) {
      char* bytes_json = nullptr;
      size_t bytes_length = 0;
      const Clock::time_point bytes_start = Clock::now();
      const int bytes_code = scan_bytes(handle, raw.data(), raw.size(), &bytes_json, &bytes_length);
      const double bytes_elapsed = ms_since(bytes_start);
      if (bytes_json) {
        free_string(bytes_json);
      }
      if (bytes_code == 0 && measuring) {
        bytes_ms.push_back(bytes_elapsed);
      }
    }

    if (index + 1 == warmup) {
      after_warmup = footprint();
    }
  }

  const Footprint final_state = footprint();
  if (warmup > 0 && after_warmup.working_set_mib > 0.0) {
    print_footprint("after warmup", after_warmup);
  }
  print_footprint("after all scans (steady)", final_state);
  std::cout << "  " << std::left << std::setw(30) << "peak working set" << std::right
            << std::fixed << std::setprecision(1)
            << std::setw(9) << final_state.peak_working_set_mib << " MiB\n";
  std::cout << "  " << std::left << std::setw(30) << "growth across scans" << std::right
            << std::setw(9) << (final_state.working_set_mib - after_init.working_set_mib)
            << " MiB   (a rising number means a per-scan leak)\n\n";

  std::cout << "=== 3. per-file latency, steady state ===\n";
  report_latency("kvd_scan_path", path_ms);
  report_latency("kvd_scan_bytes", bytes_ms);
  if (!path_ms.empty() && !bytes_ms.empty()) {
    const double delta = percentile(path_ms, 0.5) - percentile(bytes_ms, 0.5);
    std::cout << std::setprecision(3)
              << "  file I/O share (path p50 - bytes p50): " << delta << " ms\n";
  }
  if (failures) {
    std::cout << "  scans that returned an error: " << failures << "\n";
  }

  if (!stage_samples.empty()) {
    double stage_total = 0.0;
    const auto total_it = stage_samples.find("total");
    if (total_it != stage_samples.end()) {
      stage_total = percentile(total_it->second, 0.50);
    }
    std::cout << "\n=== 4. stage breakdown (p50 per scan, AXON_LOOP151_TIMING) ===\n";
    std::cout << "  " << std::left << std::setw(22) << "stage" << std::right
              << std::setw(11) << "p50 ms" << std::setw(11) << "p90 ms"
              << std::setw(9) << "share" << "\n";
    std::cout << "  " << std::string(52, '-') << "\n";
    for (const char* key : kStageKeys) {
      const auto found = stage_samples.find(key);
      if (found == stage_samples.end() || found->second.empty()) {
        continue;
      }
      const double p50 = percentile(found->second, 0.50);
      const double share = stage_total > 0.0 ? p50 / stage_total * 100.0 : 0.0;
      std::cout << "  " << std::left << std::setw(22) << key << std::right
                << std::fixed << std::setprecision(3)
                << std::setw(11) << p50
                << std::setw(11) << percentile(found->second, 0.90)
                << std::setprecision(1) << std::setw(8) << share << "%\n";
    }
  } else {
    std::cout << "\n  (no stage breakdown: set AXON_LOOP151_TIMING=1 and use an "
                 "instrumented DLL)\n";
  }

  if (have_labels) {
    std::cout << "\n=== 5. accuracy against ground truth ===\n";
    report_confusion("full chain", full_chain);
    if (no_onnx_chain.total() > 0) {
      report_confusion("no-ONNX chain", no_onnx_chain);
      std::cout << std::setprecision(6)
                << "  delta F1 (no-ONNX - full): "
                << (no_onnx_chain.f1() - full_chain.f1()) << "\n"
                << "  delta errors             : "
                << ((no_onnx_chain.fp + no_onnx_chain.fn) - (full_chain.fp + full_chain.fn))
                << "\n  decisions that disagree  : " << disagreements
                << " of " << full_chain.total() << "\n";
    } else {
      std::cout << "  (no shadow decision: set AXON_LOOP151_NO_ONNX_SHADOW=1)\n";
    }
  }

  if (!csv_path.empty() && !path_ms.empty()) {
    std::ofstream csv(csv_path);
    if (csv) {
      csv << "index,scan_path_ms,scan_bytes_ms\n";
      for (std::size_t index = 0; index < path_ms.size(); ++index) {
        csv << index << ',' << path_ms[index] << ',';
        if (index < bytes_ms.size()) {
          csv << bytes_ms[index];
        }
        csv << '\n';
      }
      std::wcout << L"\n  per-scan samples written to " << csv_path << L"\n";
    }
  }

  destroy(handle);
  FreeLibrary(module);
  return 0;
}

}  // namespace

int wmain(int argc, wchar_t** argv) {
  return run(argc, argv);
}
