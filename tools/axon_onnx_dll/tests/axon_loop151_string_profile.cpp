// Profiler for the internals of content_string_features.
//
// That block is roughly half of a scan and all of its p90 tail, but the stage
// timers in the DLL only see it as one number. Two rounds of optimisation aimed
// at plausible-looking hotspots (locale calls, redundant passes) moved it barely
// at all, so this measures each component directly instead of inferring.
//
// The implementation file is included rather than linked so the anonymous
// namespace helpers are reachable.
//
// Usage: axon_loop151_string_profile.exe <file-or-directory> [max files]

#include "../src/axon_loop151_content_features.cpp"

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <string>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

double ms_since(const Clock::time_point& start) {
  return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

std::vector<std::uint8_t> read_file(const std::filesystem::path& path) {
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    return {};
  }
  return std::vector<std::uint8_t>(
      std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>());
}

struct Stat {
  std::vector<double> samples;

  void add(double value) { samples.push_back(value); }

  double percentile(double fraction) {
    if (samples.empty()) return 0.0;
    std::sort(samples.begin(), samples.end());
    const std::size_t index =
        static_cast<std::size_t>(fraction * (samples.size() - 1) + 0.5);
    return samples[(std::min)(index, samples.size() - 1)];
  }
  double sum() const {
    double total = 0.0;
    for (const double value : samples) total += value;
    return total;
  }
};

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cerr << "usage: axon_loop151_string_profile <file-or-directory> [max files]\n";
    return 2;
  }
  const std::size_t limit = argc >= 3 ? std::strtoul(argv[2], nullptr, 10) : 200;

  std::vector<std::filesystem::path> paths;
  const std::filesystem::path root(argv[1]);
  if (std::filesystem::is_directory(root)) {
    for (const auto& entry : std::filesystem::directory_iterator(root)) {
      if (entry.is_regular_file()) {
        paths.push_back(entry.path());
        if (paths.size() >= limit) break;
      }
    }
  } else {
    paths.push_back(root);
  }
  std::cout << "profiling " << paths.size() << " files\n\n";

  std::map<std::string, Stat> stats;
  Stat total_stat;
  std::uint64_t total_bytes = 0;
  std::uint64_t sampled_bytes = 0;
  // The implementation is included, so every helper is inlinable and any result
  // that is not observed can be optimised away entirely -- which silently turned
  // a 104 MiB scan into "0.031 ms" on the first attempt. Every component result
  // is folded into this checksum and printed, so none of them can be elided.
  double checksum = 0.0;

  for (const std::filesystem::path& path : paths) {
    const std::vector<std::uint8_t> input = read_file(path);
    if (input.empty()) continue;
    total_bytes += input.size();

    const Clock::time_point whole = Clock::now();

    Clock::time_point mark = Clock::now();
    const std::vector<std::uint8_t> data = axon_loop151_native::string_sample(input);
    stats["string_sample (copy)"].add(ms_since(mark));
    sampled_bytes += data.size();
    if (data.empty()) continue;
    checksum += static_cast<double>(data.back());

    mark = Clock::now();
    const std::array<std::uint8_t, 256>& lower = axon_loop151_native::lowercase_table();
    std::vector<std::uint8_t> lowered(data.size());
    for (std::size_t index = 0; index < data.size(); ++index) {
      lowered[index] = lower[data[index]];
    }
    stats["lowercase (copy)"].add(ms_since(mark));
    checksum += static_cast<double>(lowered.back());

    mark = Clock::now();
    std::array<std::uint64_t, 256> byte_counts{};
    double ascii_length_sum = 0.0;
    std::size_t ascii_max = 0;
    const std::size_t ascii_runs = axon_loop151_native::ascii_run_count(
        data, &ascii_length_sum, &ascii_max, &byte_counts);
    stats["ascii_run + histogram"].add(ms_since(mark));
    checksum += static_cast<double>(byte_counts[0]);
    checksum += static_cast<double>(ascii_runs) + ascii_length_sum +
        static_cast<double>(ascii_max);

    mark = Clock::now();
    const std::size_t utf16_runs = axon_loop151_native::utf16_ascii_run_count(data);
    stats["utf16_ascii_run_count"].add(ms_since(mark));
    checksum += static_cast<double>(utf16_runs);

    mark = Clock::now();
    const std::size_t urls = axon_loop151_native::url_regex_count(lowered);
    stats["url_regex_count"].add(ms_since(mark));
    checksum += static_cast<double>(urls);

    mark = Clock::now();
    const std::size_t ipv4s = axon_loop151_native::ipv4_regex_count(lowered);
    stats["ipv4_regex_count"].add(ms_since(mark));
    checksum += static_cast<double>(ipv4s);

    mark = Clock::now();
    const double entropy = axon_loop151_native::entropy_from_counts(byte_counts, data.size());
    stats["entropy"].add(ms_since(mark));
    checksum += entropy;

    // What the DLL actually runs: all 89 patterns in a single pass.
    mark = Clock::now();
    const std::vector<std::size_t> pattern_counts =
        axon_loop151_native::count_all_patterns(
            lowered, axon_loop151_native::multi_pattern_matcher());
    stats["89 patterns (1 pass)"].add(ms_since(mark));
    for (const std::size_t value : pattern_counts) checksum += static_cast<double>(value);

    // The per-pattern path it replaced, kept for comparison.
    mark = Clock::now();
    std::size_t legacy_total = 0;
    for (std::string_view pattern : {"\\software\\", "\\registry\\", "hkey_",
                                     "c:\\", "\\windows\\", "\\system32\\"}) {
      legacy_total += axon_loop151_native::count_substring(lowered, pattern);
    }
    for (const auto& patterns : axon_loop151_native::string_patterns()) {
      for (std::string_view pattern : patterns) {
        legacy_total += axon_loop151_native::count_substring(lowered, pattern);
      }
    }
    stats["[legacy] 89 separate"].add(ms_since(mark));
    checksum += static_cast<double>(legacy_total);

    total_stat.add(ms_since(whole));
  }

  const double grand_total = total_stat.sum();
  std::cout << std::left << std::setw(26) << "component" << std::right
            << std::setw(11) << "p50 ms" << std::setw(11) << "p90 ms"
            << std::setw(11) << "total ms" << std::setw(9) << "share" << "\n";
  std::cout << std::string(68, '-') << "\n";
  for (auto& entry : stats) {
    const double sum = entry.second.sum();
    std::cout << std::left << std::setw(26) << entry.first << std::right
              << std::fixed << std::setprecision(3)
              << std::setw(11) << entry.second.percentile(0.50)
              << std::setw(11) << entry.second.percentile(0.90)
              << std::setw(11) << sum
              << std::setprecision(1) << std::setw(8)
              << (grand_total > 0.0 ? sum / grand_total * 100.0 : 0.0) << "%\n";
  }
  std::cout << std::string(68, '-') << "\n";
  std::cout << std::left << std::setw(26) << "TOTAL" << std::right
            << std::fixed << std::setprecision(3)
            << std::setw(11) << total_stat.percentile(0.50)
            << std::setw(11) << total_stat.percentile(0.90)
            << std::setw(11) << grand_total << "\n\n";
  std::cout << "input bytes: " << total_bytes / (1024 * 1024) << " MiB, "
            << "after string_sample: " << sampled_bytes / (1024 * 1024) << " MiB\n";
  // Printed purely so no measured component can be optimised away.
  std::cout << "checksum: " << std::setprecision(6) << checksum << "\n";
  return 0;
}
