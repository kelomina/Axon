#include "axon_loop151_content_features.h"

#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <vector>

int main(int argc, char** argv) {
  if (argc != 2) return 2;
  std::ifstream input(argv[1], std::ios::binary);
  if (!input) return 3;
  const std::vector<std::uint8_t> bytes{
      std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
  const auto v2 = axon_loop151_native::content_pe_v2_features(bytes);
  const auto strings = axon_loop151_native::content_string_features(bytes);
  std::cout << std::setprecision(9);
  std::cout << "{\"v2\":[";
  for (std::size_t index = 0; index < v2.size(); ++index) {
    if (index) std::cout << ',';
    std::cout << v2[index];
  }
  std::cout << "],\"strings\":[";
  for (std::size_t index = 0; index < strings.size(); ++index) {
    if (index) std::cout << ',';
    std::cout << strings[index];
  }
  std::cout << "]}\n";
  return 0;
}
