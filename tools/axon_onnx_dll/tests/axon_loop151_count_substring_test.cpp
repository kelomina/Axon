// Equivalence check for the memchr-accelerated count_substring rewrite.
// Both implementations are reproduced verbatim and compared over randomised
// inputs biased toward the cases that matter: repeated first bytes, overlapping
// candidates, patterns at the very start and end, and empty/short buffers.

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <random>
#include <string>
#include <string_view>
#include <vector>

std::size_t count_naive(const std::vector<std::uint8_t>& data, std::string_view pattern) {
  if (pattern.empty() || data.size() < pattern.size()) {
    return 0;
  }
  std::size_t count = 0;
  for (std::size_t index = 0; index + pattern.size() <= data.size();) {
    if (std::equal(pattern.begin(), pattern.end(), data.begin() + static_cast<std::ptrdiff_t>(index))) {
      ++count;
      index += pattern.size();
    } else {
      ++index;
    }
  }
  return count;
}

std::size_t count_memchr(const std::vector<std::uint8_t>& data, std::string_view pattern) {
  if (pattern.empty() || data.size() < pattern.size()) {
    return 0;
  }
  const std::size_t pattern_size = pattern.size();
  const std::size_t last_start = data.size() - pattern_size;
  const auto first_byte = static_cast<std::uint8_t>(pattern.front());
  std::size_t count = 0;
  std::size_t index = 0;
  while (index <= last_start) {
    const void* hit = std::memchr(data.data() + index, first_byte, last_start - index + 1);
    if (!hit) {
      break;
    }
    index = static_cast<std::size_t>(static_cast<const std::uint8_t*>(hit) - data.data());
    if (std::equal(pattern.begin(), pattern.end(), data.begin() + static_cast<std::ptrdiff_t>(index))) {
      ++count;
      index += pattern_size;
    } else {
      ++index;
    }
  }
  return count;
}

// Mirrors MultiPatternMatcher's single-pass counting so it can be compared with
// per-pattern count_substring over the same inputs.
std::vector<std::size_t> count_all_reference(
    const std::vector<std::uint8_t>& data,
    const std::vector<std::string>& patterns) {
  std::array<std::vector<std::uint16_t>, 256> buckets;
  for (std::size_t index = 0; index < patterns.size(); ++index) {
    if (patterns[index].empty()) continue;
    buckets[static_cast<std::uint8_t>(patterns[index].front())]
        .push_back(static_cast<std::uint16_t>(index));
  }
  std::vector<std::size_t> counts(patterns.size(), 0);
  std::vector<std::size_t> resume_at(patterns.size(), 0);
  for (std::size_t index = 0; index < data.size(); ++index) {
    for (const std::uint16_t pattern_index : buckets[data[index]]) {
      if (index < resume_at[pattern_index]) continue;
      const std::string& pattern = patterns[pattern_index];
      if (pattern.size() > data.size() - index) continue;
      if (std::memcmp(data.data() + index, pattern.data(), pattern.size()) != 0) continue;
      ++counts[pattern_index];
      resume_at[pattern_index] = index + pattern.size();
    }
  }
  return counts;
}

int main() {
  std::mt19937 rng(9997);
  long long checked = 0;
  long long mismatches = 0;

  const std::vector<std::string> fixed_patterns = {
      "a", "aa", "aaa", "ab", "http://", "c:\\", "\\system32\\", "hkey_",
      "zzzzzzzzzzzzzzzzzzzzzzzzz",
  };

  auto compare = [&](const std::vector<std::uint8_t>& data, std::string_view pattern) {
    const std::size_t naive = count_naive(data, pattern);
    const std::size_t fast = count_memchr(data, pattern);
    ++checked;
    if (naive != fast) {
      ++mismatches;
      if (mismatches <= 5) {
        std::cout << "MISMATCH pattern=\"" << pattern << "\" size=" << data.size()
                  << " naive=" << naive << " memchr=" << fast << "\n";
      }
    }
  };

  // Tiny alphabets maximise repeated first bytes and overlapping candidates.
  for (int alphabet : {1, 2, 3, 8, 256}) {
    std::uniform_int_distribution<int> byte_dist(0, alphabet - 1);
    for (std::size_t size : {0u, 1u, 2u, 3u, 7u, 8u, 33u, 256u, 4096u}) {
      for (int trial = 0; trial < 40; ++trial) {
        std::vector<std::uint8_t> data(size);
        for (std::uint8_t& value : data) {
          value = static_cast<std::uint8_t>(byte_dist(rng));
        }
        for (const std::string& pattern : fixed_patterns) {
          compare(data, pattern);
        }
        // Random patterns drawn from the same alphabet, plus a slice of the
        // buffer itself so genuine hits at arbitrary offsets are exercised.
        for (int variant = 0; variant < 6; ++variant) {
          const std::size_t pattern_size = 1 + (rng() % 6);
          std::string pattern(pattern_size, '\0');
          for (char& character : pattern) {
            character = static_cast<char>(byte_dist(rng));
          }
          compare(data, pattern);
        }
        if (data.size() >= 4) {
          const std::size_t offset = rng() % (data.size() - 3);
          const std::size_t length = 1 + (rng() % 4);
          std::string slice(reinterpret_cast<const char*>(data.data() + offset),
                            (std::min)(length, data.size() - offset));
          compare(data, slice);
          compare(data, slice + slice);
        }
      }
    }
  }

  // Explicit edge cases: pattern exactly at the start, exactly at the end, and
  // spanning the whole buffer.
  const std::vector<std::uint8_t> edge = {'a', 'b', 'c', 'a', 'b', 'c'};
  compare(edge, "abc");
  compare(edge, "abcabc");
  compare(edge, "c");
  compare(edge, "cab");
  compare(edge, "abcabcabc");

  // Single-pass multi-pattern counting must agree with per-pattern
  // count_substring for every pattern. The real 89-pattern set is used, over
  // randomised buffers built from the pattern alphabet so matches actually
  // occur, including adjacent and overlapping candidates.
  const std::vector<std::string> real_patterns = {
      "\\software\\", "\\registry\\", "hkey_", "c:\\", "\\windows\\", "\\system32\\",
      "http://", "https://", "www.", "ftp://",
      "socket", "connect", "recv", "send", "wininet", "ws2_32", "internetopen", "urldownload",
      "powershell", "cmd.exe", "wscript", "cscript", "mshta", "rundll32", "regsvr32",
      "currentversion\\run", "runonce", "\\services\\", "startup", "schtasks", "autostart",
      "createremotethread", "virtualalloc", "virtualprotect", "writeprocessmemory", "queueuserapc",
      "password", "credential", "token", "cookie", "browser", "wallet",
      "cryptencrypt", "cryptdecrypt", "bcrypt", "advapi32", "base64", "aes", "rsa",
      "isdebuggerpresent", "checkremotedebugger", "ntqueryinformationprocess", "sleep", "sandbox",
      "vmware", "virtualbox", "vbox", "qemu", "wine_get_unix_file_name",
      "upx", "themida", "vmprotect", "aspack", "enigma", "packed",
      "createfile", "writefile", "deletefile", "copyfile", "movefile", "findfirstfile",
      "regopenkey", "regsetvalue", "regcreatekey", "regdeletekey", "regqueryvalue",
      "microsoft", "windows", "google", "adobe", "intel", "nvidia", "mozilla", "oracle",
      "companyname", "productname", "filedescription", "originalfilename", "copyright",
  };
  long long multi_checked = 0;
  long long multi_mismatches = 0;
  for (int trial = 0; trial < 300; ++trial) {
    std::vector<std::uint8_t> buffer;
    const std::size_t pieces = 1 + (rng() % 60);
    for (std::size_t piece = 0; piece < pieces; ++piece) {
      if (rng() % 3 == 0) {
        // Filler drawn from a tiny alphabet so partial matches are common.
        const std::size_t filler = rng() % 12;
        for (std::size_t byte = 0; byte < filler; ++byte) {
          static const char alphabet[] = "\\wsc:hkey_./";
          buffer.push_back(static_cast<std::uint8_t>(alphabet[rng() % (sizeof(alphabet) - 1)]));
        }
      } else {
        const std::string& chosen = real_patterns[rng() % real_patterns.size()];
        // Sometimes append a truncated copy to create a near miss.
        const std::size_t take = (rng() % 4 == 0 && chosen.size() > 1)
            ? 1 + (rng() % (chosen.size() - 1))
            : chosen.size();
        for (std::size_t byte = 0; byte < take; ++byte) {
          buffer.push_back(static_cast<std::uint8_t>(chosen[byte]));
        }
      }
    }
    const std::vector<std::size_t> fast = count_all_reference(buffer, real_patterns);
    for (std::size_t index = 0; index < real_patterns.size(); ++index) {
      const std::size_t slow = count_naive(buffer, real_patterns[index]);
      ++multi_checked;
      if (slow != fast[index]) {
        ++multi_mismatches;
        if (multi_mismatches <= 5) {
          std::cout << "MULTI MISMATCH pattern=\"" << real_patterns[index]
                    << "\" size=" << buffer.size() << " per-pattern=" << slow
                    << " single-pass=" << fast[index] << "\n";
        }
      }
    }
  }
  std::cout << "multi-pattern comparisons: " << multi_checked
            << "   mismatches: " << multi_mismatches << "\n";

  std::cout << "comparisons: " << checked << "   mismatches: " << mismatches << "\n";
  return (mismatches == 0 && multi_mismatches == 0) ? 0 : 1;
}
