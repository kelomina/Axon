// Verifies that the std::from_chars number parser reproduces std::strtod
// bit-for-bit on the actual stage-2 assets, which is where the substitution has
// to hold: these doubles become tree thresholds and leaf values.
//
// Every numeric token in each supplied JSON file is parsed both ways and the
// raw bit patterns compared, so a one-ulp difference cannot slip through.
// Synthetic edge cases (subnormals, 17-significant-digit values, exponents at
// the representable limits) are checked as well.
//
// Usage: axon_loop151_number_parse_test.exe <model.json> [<model.json> ...]
// With no arguments only the synthetic cases run.

#include <charconv>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <system_error>
#include <vector>

namespace {

bool parse_strtod(const char* document, std::size_t position, std::size_t size,
                  double& out, std::size_t& next) {
  const char* begin = document + position;
  char* end = nullptr;
  const double parsed = std::strtod(begin, &end);
  if (end == begin || !std::isfinite(parsed)) {
    return false;
  }
  (void)size;
  out = parsed;
  next = static_cast<std::size_t>(end - document);
  return true;
}

bool parse_from_chars(const char* document, std::size_t position, std::size_t size,
                      double& out, std::size_t& next) {
  const char* const document_end = document + size;
  const char* parse_from = document + position;
  if (parse_from != document_end && *parse_from == '+') {
    ++parse_from;
  }
  double parsed = 0.0;
  const std::from_chars_result result = std::from_chars(parse_from, document_end, parsed);
  if (result.ec != std::errc{} || result.ptr == parse_from || !std::isfinite(parsed)) {
    return false;
  }
  out = parsed;
  next = static_cast<std::size_t>(result.ptr - document);
  return true;
}

bool same_bits(double left, double right) {
  std::uint64_t left_bits = 0;
  std::uint64_t right_bits = 0;
  std::memcpy(&left_bits, &left, sizeof(left_bits));
  std::memcpy(&right_bits, &right, sizeof(right_bits));
  return left_bits == right_bits;
}

bool starts_number(char character) {
  return character == '-' || character == '+' ||
      (character >= '0' && character <= '9');
}

long long check_document(const std::string& text, const std::string& label, long long& mismatches) {
  const char* document = text.data();
  const std::size_t size = text.size();
  long long checked = 0;
  std::size_t position = 0;
  bool in_string = false;
  while (position < size) {
    const char character = text[position];
    if (in_string) {
      if (character == '\\') {
        position += 2;
        continue;
      }
      if (character == '"') {
        in_string = false;
      }
      ++position;
      continue;
    }
    if (character == '"') {
      in_string = true;
      ++position;
      continue;
    }
    if (!starts_number(character)) {
      ++position;
      continue;
    }
    double legacy = 0.0;
    double modern = 0.0;
    std::size_t legacy_next = position;
    std::size_t modern_next = position;
    const bool legacy_ok = parse_strtod(document, position, size, legacy, legacy_next);
    const bool modern_ok = parse_from_chars(document, position, size, modern, modern_next);
    ++checked;
    if (legacy_ok != modern_ok || (legacy_ok && (!same_bits(legacy, modern) || legacy_next != modern_next))) {
      ++mismatches;
      if (mismatches <= 5) {
        std::cout << "MISMATCH in " << label << " at offset " << position
                  << ": strtod ok=" << legacy_ok << " value=" << legacy << " next=" << legacy_next
                  << " | from_chars ok=" << modern_ok << " value=" << modern << " next=" << modern_next
                  << "\n";
      }
    }
    position = legacy_ok ? legacy_next : position + 1;
  }
  return checked;
}

}  // namespace

int main(int argc, char** argv) {
  long long mismatches = 0;
  long long checked = 0;

  const std::vector<std::string> synthetic = {
      "0", "-0", "1", "-1", "3.14159265358979", "1e308", "-1e308", "1e-308",
      "4.9406564584124654e-324", "2.2250738585072014e-308", "1.7976931348623157e308",
      "0.1", "0.2", "0.30000000000000004", "123456789012345678901234567890",
      "1.0000000000000002", "9007199254740993", "-0.0", "1e-320", "5e-324",
      "0.000000000000000000000000001", "1E5", "1e+5", "+1.5", "1.5e-0",
  };
  for (const std::string& token : synthetic) {
    double legacy = 0.0;
    double modern = 0.0;
    std::size_t legacy_next = 0;
    std::size_t modern_next = 0;
    const bool legacy_ok = parse_strtod(token.c_str(), 0, token.size(), legacy, legacy_next);
    const bool modern_ok = parse_from_chars(token.c_str(), 0, token.size(), modern, modern_next);
    ++checked;
    if (legacy_ok != modern_ok || (legacy_ok && (!same_bits(legacy, modern) || legacy_next != modern_next))) {
      ++mismatches;
      std::cout << "MISMATCH synthetic \"" << token << "\": strtod ok=" << legacy_ok
                << " value=" << legacy << " next=" << legacy_next << " | from_chars ok=" << modern_ok
                << " value=" << modern << " next=" << modern_next << "\n";
    }
  }
  std::cout << "synthetic tokens checked: " << checked << "\n";

  for (int index = 1; index < argc; ++index) {
    std::ifstream input(argv[index], std::ios::binary);
    if (!input) {
      std::cout << "skipping unreadable file: " << argv[index] << "\n";
      continue;
    }
    std::string text((std::istreambuf_iterator<char>(input)), std::istreambuf_iterator<char>());
    const long long count = check_document(text, argv[index], mismatches);
    checked += count;
    std::cout << "  " << argv[index] << ": " << count << " numeric tokens\n";
  }

  std::cout << "total checked: " << checked << "   mismatches: " << mismatches << "\n";
  return mismatches == 0 ? 0 : 1;
}
