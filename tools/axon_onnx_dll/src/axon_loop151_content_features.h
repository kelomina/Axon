#pragma once

#include <cstdint>
#include <vector>

namespace axon_loop151_native {

constexpr std::size_t kContentPeV2FeatureDim = 182;
constexpr std::size_t kContentStringFeatureDim = 43;

std::vector<float> content_pe_v2_features(const std::vector<std::uint8_t>& data);

std::vector<float> content_string_features(const std::vector<std::uint8_t>& data);

}  // namespace axon_loop151_native
