#include "axon_loop151_native_model.h"

#include <cctype>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

std::vector<float> read_array(const std::string& text, const std::string& key) {
  const std::string marker = "\"" + key + "\"";
  const std::size_t key_pos = text.find(marker);
  if (key_pos == std::string::npos) return {};
  const std::size_t begin = text.find('[', key_pos + marker.size());
  if (begin == std::string::npos) return {};
  std::size_t position = begin + 1;
  std::vector<float> values;
  while (position < text.size()) {
    while (position < text.size() && (std::isspace(static_cast<unsigned char>(text[position])) || text[position] == ',')) ++position;
    if (position >= text.size() || text[position] == ']') return values;
    char* end = nullptr;
    const float value = std::strtof(text.c_str() + position, &end);
    if (end == text.c_str() + position) return {};
    values.push_back(value);
    position = static_cast<std::size_t>(end - text.c_str());
  }
  return {};
}

std::string read_file(const char* path) {
  std::ifstream input(path, std::ios::binary);
  std::ostringstream buffer;
  buffer << input.rdbuf();
  return buffer.str();
}

int main(int argc, char** argv) {
  if (argc != 6) return 2;
  const std::string vectors = read_file(argv[1]);
  std::string error;
  auto primary = axon_loop151_native::NativeStackModel::load_file(argv[2], error);
  if (!primary) { std::cerr << error << '\n'; return 3; }
  auto conservative = axon_loop151_native::NativeStackModel::load_file(argv[3], error);
  if (!conservative) { std::cerr << error << '\n'; return 4; }
  auto cross = axon_loop151_native::NativeScoreModel::load_file(argv[4], error);
  if (!cross) { std::cerr << error << '\n'; return 5; }
  auto noise = axon_loop151_native::NativeStackModel::load_file(argv[5], error);
  if (!noise) { std::cerr << error << '\n'; return 6; }
  const auto primary_features = read_array(vectors, "primary_vector");
  const auto conservative_features = read_array(vectors, "conservative_vector");
  const auto cross_features = read_array(vectors, "cross_vector");
  const auto noise_features = read_array(vectors, "noise_vector");
  float primary_score = primary->predict_probability(primary_features, &error);
  if (!error.empty()) { std::cerr << error << '\n'; return 7; }
  float conservative_score = conservative->predict_probability(conservative_features, &error);
  if (!error.empty()) { std::cerr << error << '\n'; return 8; }
  float cross_score = cross->predict_probability(cross_features, &error);
  if (!error.empty()) { std::cerr << error << '\n'; return 9; }
  float noise_score = noise->predict_probability(noise_features, &error);
  if (!error.empty()) { std::cerr << error << '\n'; return 10; }
  std::cout << std::setprecision(10) << "{\"primary\":" << primary_score
            << ",\"conservative\":" << conservative_score
            << ",\"cross\":" << cross_score << ",\"noise\":" << noise_score << "}\n";
  return 0;
}
