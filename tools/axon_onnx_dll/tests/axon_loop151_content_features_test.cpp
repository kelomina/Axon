#include "../src/axon_loop151_content_features.h"

#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

namespace {

void write_u16(std::vector<std::uint8_t>& data, std::size_t offset, std::uint16_t value) {
  data[offset] = static_cast<std::uint8_t>(value & 0xffu);
  data[offset + 1] = static_cast<std::uint8_t>((value >> 8) & 0xffu);
}

void write_u32(std::vector<std::uint8_t>& data, std::size_t offset, std::uint32_t value) {
  for (std::size_t index = 0; index < 4; ++index) {
    data[offset + index] = static_cast<std::uint8_t>((value >> (index * 8)) & 0xffu);
  }
}

std::vector<std::uint8_t> minimal_pe32() {
  std::vector<std::uint8_t> data(0x400, 0);
  data[0] = 'M';
  data[1] = 'Z';
  write_u32(data, 0x3c, 0x80);
  std::memcpy(data.data() + 0x80, "PE\0\0", 4);
  const std::size_t file_header = 0x84;
  write_u16(data, file_header, 0x14c);
  write_u16(data, file_header + 2, 1);
  write_u16(data, file_header + 16, 0xe0);
  write_u16(data, file_header + 18, 0x010f);
  const std::size_t optional = file_header + 20;
  write_u16(data, optional, 0x10b);
  data[optional + 2] = 14;
  write_u32(data, optional + 4, 0x200);
  write_u32(data, optional + 8, 0x200);
  write_u32(data, optional + 16, 0x1000);
  write_u32(data, optional + 28, 0x400000);
  write_u32(data, optional + 32, 0x1000);
  write_u32(data, optional + 36, 0x200);
  write_u32(data, optional + 56, 0x3000);
  write_u32(data, optional + 60, 0x200);
  write_u16(data, optional + 68, 3);
  write_u16(data, optional + 70, 0x8140);
  write_u32(data, optional + 92, 16);
  const std::size_t section = optional + 0xe0;
  std::memcpy(data.data() + section, ".text\0\0\0", 8);
  write_u32(data, section + 8, 0x1000);
  write_u32(data, section + 12, 0x1000);
  write_u32(data, section + 16, 0x200);
  write_u32(data, section + 20, 0x200);
  write_u32(data, section + 36, 0x60000020);
  std::fill(data.begin() + 0x200, data.end(), 0x90);
  return data;
}

}  // namespace

int main() {
  const auto pe = minimal_pe32();
  const auto pe_features = axon_loop151_native::content_pe_v2_features(pe);
  assert(pe_features.size() == axon_loop151_native::kContentPeV2FeatureDim);
  for (float value : pe_features) {
    assert(std::isfinite(value));
  }
  assert(pe_features[64 + 48 + 3 + 9 + 29] > 0.0f);
  assert(pe_features[64 + 48 + 3 + 9 + 29 + 11] == 1.0f);
  assert(pe_features.back() >= 0.0f);

  const std::vector<std::uint8_t> strings = {
      'M', 'Z', 'h', 't', 't', 'p', ':', '/', '/', 'e', 'x', 'a', 'm', 'p', 'l', 'e', '.', 'c', 'o', 'm', 0,
      'M', 'i', 'c', 'r', 'o', 's', 'o', 'f', 't', ' ', 'C', 'o', 'r', 'p', 'o', 'r', 'a', 't', 'i', 'o', 'n', 0,
      'V', 'i', 'r', 't', 'u', 'a', 'l', 'A', 'l', 'l', 'o', 'c', 0,
  };
  const auto string_features = axon_loop151_native::content_string_features(strings);
  assert(string_features.size() == axon_loop151_native::kContentStringFeatureDim);
  for (float value : string_features) {
    assert(std::isfinite(value));
  }
  assert(string_features[10] > 0.0f);
  assert(string_features[39] > 0.0f);

  const auto invalid_features = axon_loop151_native::content_pe_v2_features({'M', 'Z', 0});
  assert(invalid_features.size() == axon_loop151_native::kContentPeV2FeatureDim);
  for (float value : invalid_features) {
    assert(value == 0.0f);
  }
  return 0;
}
