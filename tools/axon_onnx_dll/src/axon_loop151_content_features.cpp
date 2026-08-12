#include "axon_loop151_content_features.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <cstdint>
#include <limits>
#include <numeric>
#include <optional>
#include <set>
#include <string>
#include <string_view>
#include <unordered_set>
#include <utility>
#include <vector>

namespace axon_loop151_native {
namespace {

// std::tolower, std::isalnum and std::isspace consult the current C locale on
// every call. The string feature block applies them byte by byte across a
// buffer of up to 2.5 MB, which made those calls the largest single cost in a
// scan. Each table is filled by calling the very function it replaces, over all
// 256 byte values, so the results are equivalent by construction. The tables are
// function-local statics, so they capture the locale in effect at first use --
// the same locale the per-byte calls would have seen. This library never calls
// setlocale.
const std::array<std::uint8_t, 256>& lowercase_table() {
  static const std::array<std::uint8_t, 256> table = [] {
    std::array<std::uint8_t, 256> values{};
    for (std::size_t index = 0; index < values.size(); ++index) {
      values[index] = static_cast<std::uint8_t>(
          std::tolower(static_cast<unsigned char>(index)));
    }
    return values;
  }();
  return table;
}

const std::array<bool, 256>& alnum_table() {
  static const std::array<bool, 256> table = [] {
    std::array<bool, 256> values{};
    for (std::size_t index = 0; index < values.size(); ++index) {
      values[index] = std::isalnum(static_cast<unsigned char>(index)) != 0;
    }
    return values;
  }();
  return table;
}

const std::array<bool, 256>& space_table() {
  static const std::array<bool, 256> table = [] {
    std::array<bool, 256> values{};
    for (std::size_t index = 0; index < values.size(); ++index) {
      values[index] = std::isspace(static_cast<unsigned char>(index)) != 0;
    }
    return values;
  }();
  return table;
}

template <typename T>
T read_le(const std::vector<std::uint8_t>& data, std::size_t offset) {
  if (offset > data.size() || sizeof(T) > data.size() - offset) {
    return 0;
  }
  T value = 0;
  for (std::size_t index = 0; index < sizeof(T); ++index) {
    value |= static_cast<T>(data[offset + index]) << (index * 8);
  }
  return value;
}

double safe_ratio(double numerator, double denominator) {
  return numerator / std::max(denominator, 1.0);
}

// Shannon entropy of a byte histogram, normalised to [0, 1]. Split out so a
// caller that already built the histogram does not have to walk the buffer
// again; the arithmetic is identical either way.
double entropy_from_counts(const std::array<std::uint64_t, 256>& counts, std::size_t total) {
  if (total == 0) {
    return 0.0;
  }
  const double denominator = static_cast<double>(total);
  double result = 0.0;
  for (std::uint64_t count : counts) {
    if (count == 0) {
      continue;
    }
    const double probability = static_cast<double>(count) / denominator;
    result -= probability * std::log2(probability);
  }
  return result / 8.0;
}

double entropy_normalized(const std::vector<std::uint8_t>& data, std::size_t offset, std::size_t size) {
  if (offset >= data.size() || size == 0) {
    return 0.0;
  }
  const std::size_t available = std::min(size, data.size() - offset);
  std::array<std::uint64_t, 256> counts{};
  for (std::size_t index = 0; index < available; ++index) {
    counts[data[offset + index]] += 1;
  }
  return entropy_from_counts(counts, available);
}

std::string lower_ascii(std::string value) {
  for (char& character : value) {
    character = static_cast<char>(std::tolower(static_cast<unsigned char>(character)));
  }
  return value;
}

std::string read_string(const std::vector<std::uint8_t>& data, std::size_t offset, std::size_t limit = 512) {
  if (offset >= data.size()) {
    return {};
  }
  std::string value;
  value.reserve(std::min(limit, data.size() - offset));
  for (std::size_t index = offset; index < data.size() && value.size() < limit; ++index) {
    if (data[index] == 0) {
      break;
    }
    value.push_back(static_cast<char>(data[index]));
  }
  return lower_ascii(std::move(value));
}

struct NativeSection {
  std::string name;
  std::uint32_t virtual_address = 0;
  std::uint32_t raw_pointer = 0;
  std::uint32_t raw_size = 0;
  std::uint32_t virtual_size = 0;
  std::uint32_t characteristics = 0;
};

struct NativePe {
  bool valid = false;
  bool is_pe64 = false;
  std::uint16_t machine = 0;
  std::uint16_t number_of_sections = 0;
  std::uint16_t characteristics = 0;
  std::uint16_t optional_header_size = 0;
  std::uint16_t optional_magic = 0;
  std::uint8_t major_linker = 0;
  std::uint8_t minor_linker = 0;
  std::uint32_t timestamp = 0;
  std::uint32_t size_of_code = 0;
  std::uint32_t size_of_initialized_data = 0;
  std::uint32_t size_of_uninitialized_data = 0;
  std::uint32_t entry_point = 0;
  std::uint64_t image_base = 0;
  std::uint32_t section_alignment = 0;
  std::uint32_t file_alignment = 0;
  std::uint32_t size_of_image = 0;
  std::uint32_t size_of_headers = 0;
  std::uint16_t subsystem = 0;
  std::uint16_t dll_characteristics = 0;
  std::uint32_t checksum = 0;
  std::uint32_t number_of_rva_and_sizes = 0;
  std::array<std::uint32_t, 16> directory_rva{};
  std::array<std::uint32_t, 16> directory_size{};
  std::vector<NativeSection> sections;
};

NativePe parse_pe(const std::vector<std::uint8_t>& data) {
  NativePe pe;
  if (data.size() < 0x40 || read_le<std::uint16_t>(data, 0) != 0x5A4D) {
    return pe;
  }
  const std::uint32_t pe_offset = read_le<std::uint32_t>(data, 0x3C);
  if (pe_offset > data.size() || data.size() - pe_offset < 24 ||
      read_le<std::uint32_t>(data, pe_offset) != 0x00004550) {
    return pe;
  }
  const std::size_t file_header = static_cast<std::size_t>(pe_offset) + 4;
  pe.machine = read_le<std::uint16_t>(data, file_header);
  pe.number_of_sections = read_le<std::uint16_t>(data, file_header + 2);
  pe.timestamp = read_le<std::uint32_t>(data, file_header + 4);
  pe.optional_header_size = read_le<std::uint16_t>(data, file_header + 16);
  pe.characteristics = read_le<std::uint16_t>(data, file_header + 18);
  const std::size_t optional = file_header + 20;
  if (optional > data.size() || pe.optional_header_size > data.size() - optional ||
      pe.optional_header_size < 96) {
    return pe;
  }
  pe.optional_magic = read_le<std::uint16_t>(data, optional);
  pe.is_pe64 = pe.optional_magic == 0x20B;
  if (pe.optional_magic != 0x10B && pe.optional_magic != 0x20B) {
    return pe;
  }
  const std::size_t minimum_optional_size = pe.is_pe64 ? 112 : 96;
  if (pe.optional_header_size < minimum_optional_size) {
    return pe;
  }
  pe.major_linker = read_le<std::uint8_t>(data, optional + 2);
  pe.minor_linker = read_le<std::uint8_t>(data, optional + 3);
  pe.size_of_code = read_le<std::uint32_t>(data, optional + 4);
  pe.size_of_initialized_data = read_le<std::uint32_t>(data, optional + 8);
  pe.size_of_uninitialized_data = read_le<std::uint32_t>(data, optional + 12);
  pe.entry_point = read_le<std::uint32_t>(data, optional + 16);
  pe.image_base = pe.is_pe64 ? read_le<std::uint64_t>(data, optional + 24)
                             : read_le<std::uint32_t>(data, optional + 28);
  pe.section_alignment = read_le<std::uint32_t>(data, optional + 32);
  pe.file_alignment = read_le<std::uint32_t>(data, optional + 36);
  pe.size_of_image = read_le<std::uint32_t>(data, optional + 56);
  pe.size_of_headers = read_le<std::uint32_t>(data, optional + 60);
  pe.checksum = read_le<std::uint32_t>(data, optional + 64);
  pe.subsystem = read_le<std::uint16_t>(data, optional + 68);
  pe.dll_characteristics = read_le<std::uint16_t>(data, optional + 70);
  pe.number_of_rva_and_sizes = read_le<std::uint32_t>(data, optional + (pe.is_pe64 ? 108 : 92));
  const std::size_t directory_offset = optional + (pe.is_pe64 ? 112 : 96);
  const std::size_t directory_count = std::min<std::size_t>(pe.number_of_rva_and_sizes, 16);
  for (std::size_t index = 0; index < directory_count; ++index) {
    const std::size_t offset = directory_offset + index * 8;
    if (offset > data.size() || data.size() - offset < 8 ||
        offset > optional + pe.optional_header_size ||
        8 > optional + pe.optional_header_size - offset) {
      break;
    }
    pe.directory_rva[index] = read_le<std::uint32_t>(data, offset);
    pe.directory_size[index] = read_le<std::uint32_t>(data, offset + 4);
  }
  const std::size_t section_table = optional + pe.optional_header_size;
  for (std::uint16_t index = 0; index < pe.number_of_sections; ++index) {
    const std::size_t offset = section_table + static_cast<std::size_t>(index) * 40;
    if (offset > data.size() || data.size() - offset < 40) {
      break;
    }
    NativeSection section;
    char name[9] = {};
    for (std::size_t byte = 0; byte < 8; ++byte) {
      name[byte] = static_cast<char>(data[offset + byte]);
    }
    section.name = lower_ascii(std::string(name));
    section.virtual_size = read_le<std::uint32_t>(data, offset + 8);
    section.virtual_address = read_le<std::uint32_t>(data, offset + 12);
    section.raw_size = read_le<std::uint32_t>(data, offset + 16);
    section.raw_pointer = read_le<std::uint32_t>(data, offset + 20);
    section.characteristics = read_le<std::uint32_t>(data, offset + 36);
    pe.sections.push_back(std::move(section));
  }
  pe.valid = true;
  return pe;
}

std::optional<std::size_t> rva_to_offset(const NativePe& pe, const std::vector<std::uint8_t>& data, std::uint32_t rva) {
  if (rva == 0) {
    return std::nullopt;
  }
  if (rva < pe.size_of_headers && rva < data.size()) {
    return static_cast<std::size_t>(rva);
  }
  for (const NativeSection& section : pe.sections) {
    const std::uint64_t start = section.virtual_address;
    const std::uint64_t span = std::max(section.virtual_size, section.raw_size);
    if (span == 0 || static_cast<std::uint64_t>(rva) < start || static_cast<std::uint64_t>(rva) >= start + span) {
      continue;
    }
    const std::uint64_t offset = static_cast<std::uint64_t>(section.raw_pointer) +
                                 (static_cast<std::uint64_t>(rva) - start);
    if (offset < data.size()) {
      return static_cast<std::size_t>(offset);
    }
    return std::nullopt;
  }
  return std::nullopt;
}

std::string section_dll_name(const std::string& value) {
  const std::size_t slash = value.find_last_of("\\/");
  return slash == std::string::npos ? value : value.substr(slash + 1);
}

struct ImportStats {
  std::array<std::uint32_t, 32> dll_api_counts{};
  std::array<std::uint32_t, 16> category_counts{};
  std::uint32_t named_imports = 0;
  std::uint32_t ordinal_imports = 0;
  std::uint32_t delay_imports = 0;
  std::set<std::string> imported_dlls;
  std::set<std::string> delay_import_dlls;
};

const std::array<std::string_view, 32>& v2_import_dll_names() {
  static const std::array<std::string_view, 32> names = {
      "kernel32.dll", "ntdll.dll", "user32.dll", "advapi32.dll", "shell32.dll", "ole32.dll",
      "oleaut32.dll", "gdi32.dll", "comctl32.dll", "comdlg32.dll", "shlwapi.dll", "version.dll",
      "setupapi.dll", "msvcrt.dll", "ucrtbase.dll", "vcruntime140.dll", "ws2_32.dll", "wininet.dll",
      "winhttp.dll", "urlmon.dll", "dnsapi.dll", "iphlpapi.dll", "netapi32.dll", "crypt32.dll",
      "bcrypt.dll", "secur32.dll", "psapi.dll", "wtsapi32.dll", "dbghelp.dll", "imagehlp.dll", "mpr.dll",
      "wintrust.dll"};
  return names;
}

const std::array<std::vector<std::string_view>, 16>& v2_api_keywords() {
  static const std::array<std::vector<std::string_view>, 16> keywords = {{
      {"openscmanager", "createservice", "startservice", "controlservice", "deleteservice"},
      {"ntloaddriver", "zwloaddriver", "deviceiocontrol", "createsymboliclink", "ioctl"},
      {"adjusttokenprivileges", "openprocesstoken", "lookupprivilege", "impersonate"},
      {"isdebuggerpresent", "checkremotedebugger", "ntqueryinformationprocess", "outputdebugstring"},
      {"virtualalloc", "virtualallocex", "virtualprotect", "virtualprotectex", "heapalloc"},
      {"createthread", "createremotethread", "queueuserapc", "rtlcreateuserthread", "setthreadcontext"},
      {"loadlibrary", "getprocaddress", "ldrloaddll", "freelibrary"},
      {"createtoolhelp32snapshot", "process32first", "process32next", "enumprocesses"},
      {"regsetvalue", "regcreatekey", "createservice", "schtasks", "startup"},
      {"internetopen", "internetconnect", "httpopenrequest", "httpsendrequest", "winhttp"},
      {"socket", "connect", "bind", "listen", "accept", "wsastartup"},
      {"createfile", "writefile", "deletefile", "movefile", "copyfile", "setfileattributes"},
      {"crypt", "bcrypt", "cert", "winverifytrust"},
      {"findresource", "loadresource", "lockresource", "sizeofresource", "beginupdateresource"},
      {"msi", "setup", "install", "uninstall"},
      {"cocreateinstance", "coinitialize", "clsidfromprogid", "regsvr"},
  }};
  return keywords;
}

bool contains_any(std::string_view text, const std::vector<std::string_view>& keywords) {
  return std::any_of(keywords.begin(), keywords.end(), [text](std::string_view keyword) {
    return text.find(keyword) != std::string_view::npos;
  });
}

std::uint32_t rva_from_pointer(const NativePe& pe, std::uint64_t value) {
  if (value >= pe.image_base && value - pe.image_base <= std::numeric_limits<std::uint32_t>::max()) {
    return static_cast<std::uint32_t>(value - pe.image_base);
  }
  return static_cast<std::uint32_t>(value & 0xffffffffu);
}

void parse_import_table(
    const std::vector<std::uint8_t>& data,
    const NativePe& pe,
    std::uint32_t table_rva,
    bool delay_import,
    ImportStats& stats) {
  const auto table_offset = rva_to_offset(pe, data, table_rva);
  if (!table_offset) {
    return;
  }
  const auto& dll_names = v2_import_dll_names();
  const auto& categories = v2_api_keywords();
  const std::size_t descriptor_size = delay_import ? 32 : 20;
  const std::uint64_t ordinal_mask = pe.is_pe64 ? 0x8000000000000000ull : 0x80000000ull;
  for (std::size_t descriptor = *table_offset, guard = 0;
       descriptor <= data.size() && data.size() - descriptor >= descriptor_size && guard < 4096;
       descriptor += descriptor_size, ++guard) {
    const std::uint32_t first = read_le<std::uint32_t>(data, descriptor);
    const std::uint32_t name_rva = read_le<std::uint32_t>(data, descriptor + (delay_import ? 4 : 12));
    std::uint64_t thunk_value = delay_import ? read_le<std::uint32_t>(data, descriptor + 16)
                                             : read_le<std::uint32_t>(data, descriptor);
    const std::uint32_t second = read_le<std::uint32_t>(data, descriptor + (delay_import ? 20 : 16));
    if (delay_import) {
      if (first == 0 && name_rva == 0 && second == 0) {
        break;
      }
      thunk_value = read_le<std::uint32_t>(data, descriptor + 16);
    } else if (first == 0 && name_rva == 0 && second == 0) {
      break;
    }
    const auto name_offset = rva_to_offset(pe, data, name_rva);
    const std::string dll_name = name_offset ? section_dll_name(read_string(data, *name_offset)) : std::string{};
    if (!dll_name.empty()) {
      stats.imported_dlls.insert(dll_name);
      if (delay_import) {
        stats.delay_import_dlls.insert(dll_name);
      }
    }
    if (delay_import && (read_le<std::uint32_t>(data, descriptor) & 1u) == 0u) {
      thunk_value = rva_from_pointer(pe, thunk_value);
    }
    if (!delay_import && thunk_value == 0) {
      thunk_value = second;
    }
    const auto thunk_offset = rva_to_offset(pe, data, static_cast<std::uint32_t>(thunk_value));
    if (!thunk_offset) {
      continue;
    }
    std::uint32_t entry_count = 0;
    for (std::size_t thunk = *thunk_offset, thunk_guard = 0;
         thunk <= data.size() && data.size() - thunk >= (pe.is_pe64 ? 8u : 4u) && thunk_guard < 8192;
         thunk += pe.is_pe64 ? 8u : 4u, ++thunk_guard) {
      const std::uint64_t value = pe.is_pe64 ? read_le<std::uint64_t>(data, thunk)
                                             : read_le<std::uint32_t>(data, thunk);
      if (value == 0) {
        break;
      }
      ++entry_count;
      if (delay_import) {
        ++stats.delay_imports;
      }
      if ((value & ordinal_mask) != 0) {
        ++stats.ordinal_imports;
        continue;
      }
      const auto hint_name_offset = rva_to_offset(pe, data, rva_from_pointer(pe, value));
      if (!hint_name_offset || *hint_name_offset > data.size() || data.size() - *hint_name_offset <= 2) {
        continue;
      }
      const std::string api_name = read_string(data, *hint_name_offset + 2);
      if (api_name.empty()) {
        continue;
      }
      ++stats.named_imports;
      for (std::size_t category = 0; category < categories.size(); ++category) {
        if (contains_any(api_name, categories[category])) {
          ++stats.category_counts[category];
        }
      }
    }
    for (std::size_t index = 0; index < dll_names.size(); ++index) {
      if (dll_name == dll_names[index]) {
        stats.dll_api_counts[index] += entry_count;
        break;
      }
    }
  }
}

struct ExportStats {
  std::uint32_t export_count = 0;
  std::uint32_t export_name_count = 0;
  std::uint32_t forwarder_count = 0;
  std::vector<std::uint32_t> ordinals;
  std::vector<std::size_t> name_lengths;
  std::array<std::uint32_t, 4> pattern_hits{};
};

const std::array<std::vector<std::string_view>, 4>& v2_export_keywords() {
  static const std::array<std::vector<std::string_view>, 4> keywords = {{
      {"dllgetclassobject", "dllcanunloadnow", "dllregisterserver", "dllunregisterserver"},
      {"cplapplet"},
      {"servicemain", "handler", "startservice"},
      {"plugin", "initialize", "init", "register"},
  }};
  return keywords;
}

ExportStats collect_exports(const std::vector<std::uint8_t>& data, const NativePe& pe) {
  ExportStats stats;
  const std::uint32_t table_rva = pe.directory_rva[0];
  const auto table_offset = rva_to_offset(pe, data, table_rva);
  if (!table_offset || *table_offset > data.size() || data.size() - *table_offset < 40) {
    return stats;
  }
  const std::size_t offset = *table_offset;
  const std::uint32_t base = read_le<std::uint32_t>(data, offset + 16);
  const std::uint32_t function_count = read_le<std::uint32_t>(data, offset + 20);
  const std::uint32_t name_count = read_le<std::uint32_t>(data, offset + 24);
  const std::uint32_t functions_rva = read_le<std::uint32_t>(data, offset + 28);
  const std::uint32_t names_rva = read_le<std::uint32_t>(data, offset + 32);
  const std::uint32_t ordinals_rva = read_le<std::uint32_t>(data, offset + 36);
  const std::uint64_t table_end = static_cast<std::uint64_t>(table_rva) + pe.directory_size[0];
  const auto function_offset = rva_to_offset(pe, data, functions_rva);
  const auto names_offset = rva_to_offset(pe, data, names_rva);
  const auto ordinals_offset = rva_to_offset(pe, data, ordinals_rva);
  if (!function_offset || !names_offset || !ordinals_offset) {
    return stats;
  }
  stats.export_count = function_count;
  stats.ordinals.reserve(function_count);
  for (std::uint32_t index = 0; index < function_count; ++index) {
    stats.ordinals.push_back(base + index);
    const std::size_t function_entry = *function_offset + static_cast<std::size_t>(index) * 4;
    const std::uint32_t function_rva = read_le<std::uint32_t>(data, function_entry);
    if (function_rva >= table_rva && static_cast<std::uint64_t>(function_rva) < table_end) {
      ++stats.forwarder_count;
    }
  }
  const auto& patterns = v2_export_keywords();
  for (std::uint32_t index = 0; index < name_count; ++index) {
    const std::size_t name_entry = *names_offset + static_cast<std::size_t>(index) * 4;
    const std::size_t ordinal_entry = *ordinals_offset + static_cast<std::size_t>(index) * 2;
    if (name_entry > data.size() || data.size() - name_entry < 4 ||
        ordinal_entry > data.size() || data.size() - ordinal_entry < 2) {
      break;
    }
    const std::uint16_t function_index = read_le<std::uint16_t>(data, ordinal_entry);
    if (function_index >= function_count) {
      continue;
    }
    const auto name = rva_to_offset(pe, data, read_le<std::uint32_t>(data, name_entry));
    if (!name) {
      continue;
    }
    const std::string export_name = read_string(data, *name);
    if (export_name.empty()) {
      continue;
    }
    ++stats.export_name_count;
    stats.name_lengths.push_back(export_name.size());
    for (std::size_t pattern = 0; pattern < patterns.size(); ++pattern) {
      if (contains_any(export_name, patterns[pattern])) {
        ++stats.pattern_hits[pattern];
      }
    }
  }
  return stats;
}

struct ResourceStats {
  std::uint32_t entry_count = 0;
  std::uint32_t named_entry_count = 0;
  std::set<std::uint32_t> languages;
  std::array<std::uint32_t, 11> type_counts{};
  std::vector<std::uint32_t> data_sizes;
  std::vector<double> data_entropies;
};

const std::array<std::uint32_t, 11>& v2_resource_type_ids() {
  static const std::array<std::uint32_t, 11> ids = {1, 2, 3, 4, 5, 6, 10, 12, 14, 16, 24};
  return ids;
}

void collect_resource_entries(
    const std::vector<std::uint8_t>& data,
    const NativePe& pe,
    std::size_t base,
    std::size_t directory_offset,
    int depth,
    std::optional<std::uint32_t> root_type,
    std::set<std::size_t>& ancestors,
    ResourceStats& stats) {
  if (depth > 32 || directory_offset > data.size() || data.size() - directory_offset < 16 ||
      ancestors.count(directory_offset) != 0) {
    return;
  }
  ancestors.insert(directory_offset);
  const std::size_t named_count = read_le<std::uint16_t>(data, directory_offset + 12);
  const std::size_t id_count = read_le<std::uint16_t>(data, directory_offset + 14);
  const std::size_t count = named_count + id_count;
  if (count > 4096 || directory_offset + 16 > data.size() || count > (data.size() - directory_offset - 16) / 8) {
    ancestors.erase(directory_offset);
    return;
  }
  for (std::size_t index = 0; index < count; ++index) {
    const std::size_t entry_offset = directory_offset + 16 + index * 8;
    const std::uint32_t name = read_le<std::uint32_t>(data, entry_offset);
    const std::uint32_t target = read_le<std::uint32_t>(data, entry_offset + 4);
    const bool named = (name & 0x80000000u) != 0;
    const std::optional<std::uint32_t> numeric_id = named ? std::nullopt : std::optional<std::uint32_t>(name);
    std::optional<std::uint32_t> current_root = root_type;
    if (depth == 0 && numeric_id) {
      current_root = numeric_id;
    }
    ++stats.entry_count;
    if (named) {
      ++stats.named_entry_count;
    }
    if (depth == 2 && numeric_id) {
      stats.languages.insert(*numeric_id);
    }
    if (current_root) {
      const auto& ids = v2_resource_type_ids();
      for (std::size_t type = 0; type < ids.size(); ++type) {
        if (*current_root == ids[type]) {
          ++stats.type_counts[type];
          break;
        }
      }
    }
    const bool is_directory = (target & 0x80000000u) != 0;
    const std::uint32_t relative_target = target & 0x7fffffffu;
    if (relative_target > data.size() - base) {
      continue;
    }
    const std::size_t target_offset = base + relative_target;
    if (is_directory) {
      collect_resource_entries(data, pe, base, target_offset, depth + 1, current_root, ancestors, stats);
      continue;
    }
    if (target_offset > data.size() || data.size() - target_offset < 16) {
      continue;
    }
    const std::uint32_t data_rva = read_le<std::uint32_t>(data, target_offset);
    const std::uint32_t data_size = read_le<std::uint32_t>(data, target_offset + 4);
    if (data_size == 0) {
      continue;
    }
    stats.data_sizes.push_back(data_size);
    if (stats.data_entropies.size() < 64) {
      const auto data_offset = rva_to_offset(pe, data, data_rva);
      if (data_offset) {
        stats.data_entropies.push_back(entropy_normalized(data, *data_offset, std::min<std::size_t>(data_size, 4096)));
      }
    }
  }
  ancestors.erase(directory_offset);
}

ResourceStats collect_resources(const std::vector<std::uint8_t>& data, const NativePe& pe) {
  ResourceStats stats;
  const auto root_offset = rva_to_offset(pe, data, pe.directory_rva[2]);
  if (!root_offset) {
    return stats;
  }
  std::set<std::size_t> ancestors;
  collect_resource_entries(data, pe, *root_offset, *root_offset, 0, std::nullopt, ancestors, stats);
  return stats;
}

double section_entropy(const std::vector<std::uint8_t>& data, const NativeSection& section) {
  if (section.raw_size == 0 || section.raw_pointer >= data.size()) {
    return 0.0;
  }
  return entropy_normalized(data, section.raw_pointer, std::min<std::size_t>(section.raw_size, 4096));
}

std::vector<float> zero_features(std::size_t size) {
  return std::vector<float>(size, 0.0f);
}

void append_feature(std::vector<float>& features, double value) {
  features.push_back(std::isfinite(value) ? static_cast<float>(value) : 0.0f);
}

std::vector<float> content_pe_v2_features_impl(const std::vector<std::uint8_t>& data) {
  NativePe pe = parse_pe(data);
  if (!pe.valid) {
    return zero_features(kContentPeV2FeatureDim);
  }
  std::vector<float> features;
  features.reserve(kContentPeV2FeatureDim);
  ImportStats imports;
  parse_import_table(data, pe, pe.directory_rva[1], false, imports);
  parse_import_table(data, pe, pe.directory_rva[13], true, imports);
  const std::uint32_t total_imports = imports.named_imports + imports.ordinal_imports;
  const auto& dll_names = v2_import_dll_names();
  for (std::size_t index = 0; index < dll_names.size(); ++index) {
    const std::string dll_name(dll_names[index]);
    append_feature(features, imports.imported_dlls.count(dll_name) != 0 ? 1.0 : 0.0);
    append_feature(features, safe_ratio(imports.dll_api_counts[index], total_imports));
  }
  for (std::uint32_t count : imports.category_counts) {
    append_feature(features, count > 0 ? 1.0 : 0.0);
    append_feature(features, std::log1p(static_cast<double>(count)));
    append_feature(features, safe_ratio(count, total_imports));
  }
  append_feature(features, std::log1p(static_cast<double>(imports.delay_import_dlls.size())));
  append_feature(features, std::log1p(static_cast<double>(imports.delay_imports)));
  append_feature(features, safe_ratio(imports.delay_imports, total_imports));

  const ExportStats exports = collect_exports(data, pe);
  const double ordinal_only = static_cast<double>(exports.export_count) - exports.export_name_count;
  append_feature(features, safe_ratio(ordinal_only, exports.export_count));
  append_feature(features, safe_ratio(exports.forwarder_count, exports.export_count));
  const double mean_name_length = exports.name_lengths.empty()
      ? 0.0
      : std::accumulate(exports.name_lengths.begin(), exports.name_lengths.end(), 0.0) /
            static_cast<double>(exports.name_lengths.size());
  const std::size_t max_name_length = exports.name_lengths.empty()
      ? 0
      : *std::max_element(exports.name_lengths.begin(), exports.name_lengths.end());
  append_feature(features, safe_ratio(mean_name_length, 128.0));
  append_feature(features, safe_ratio(max_name_length, 256.0));
  const std::uint32_t ordinal_span = exports.ordinals.empty()
      ? 0
      : *std::max_element(exports.ordinals.begin(), exports.ordinals.end()) -
            *std::min_element(exports.ordinals.begin(), exports.ordinals.end()) + 1;
  append_feature(features, std::log1p(static_cast<double>(ordinal_span)));
  for (std::uint32_t count : exports.pattern_hits) {
    append_feature(features, count > 0 ? 1.0 : 0.0);
  }

  const ResourceStats resources = collect_resources(data, pe);
  const double resource_total_size = std::accumulate(resources.data_sizes.begin(), resources.data_sizes.end(), 0.0);
  const std::uint32_t max_resource_size = resources.data_sizes.empty()
      ? 0
      : *std::max_element(resources.data_sizes.begin(), resources.data_sizes.end());
  const double mean_resource_entropy = resources.data_entropies.empty()
      ? 0.0
      : std::accumulate(resources.data_entropies.begin(), resources.data_entropies.end(), 0.0) /
            static_cast<double>(resources.data_entropies.size());
  const double max_resource_entropy = resources.data_entropies.empty()
      ? 0.0
      : *std::max_element(resources.data_entropies.begin(), resources.data_entropies.end());
  append_feature(features, std::log1p(static_cast<double>(resources.data_sizes.size())));
  append_feature(features, safe_ratio(resources.named_entry_count, resources.entry_count));
  append_feature(features, std::log1p(static_cast<double>(resources.languages.size())));
  append_feature(features, std::log1p(resource_total_size));
  append_feature(features, safe_ratio(max_resource_size, data.size()));
  append_feature(features, mean_resource_entropy);
  append_feature(features, max_resource_entropy);
  for (std::uint32_t count : resources.type_counts) {
    append_feature(features, count > 0 ? 1.0 : 0.0);
    append_feature(features, std::log1p(static_cast<double>(count)));
  }

  struct SectionInfo {
    bool executable = false;
    bool writable = false;
    bool readable = false;
    bool zero_raw = false;
    bool contains_entry_point = false;
    double entropy = 0.0;
    double raw_virtual_delta = 0.0;
    double virtual_raw_ratio = 0.0;
  };
  std::vector<SectionInfo> section_infos;
  section_infos.reserve(pe.sections.size());
  std::array<std::uint32_t, 8> group_hits{};
  const std::array<std::vector<std::string_view>, 8> section_groups = {{
      {".text", "code"}, {".data", ".rdata", ".bss"}, {".rsrc"}, {".idata"},
      {".edata"}, {".reloc"}, {".tls"}, {"upx", "aspack", "themida", "vmprotect", "enigma", "packed", "nspack", "upack"},
  }};
  for (const NativeSection& section : pe.sections) {
    const bool executable = (section.characteristics & 0x20000000u) != 0;
    const bool writable = (section.characteristics & 0x80000000u) != 0;
    const bool readable = (section.characteristics & 0x40000000u) != 0;
    const double raw_size = section.raw_size;
    const double virtual_size = section.virtual_size;
    const double span = std::max({raw_size, virtual_size, 1.0});
    const std::uint64_t section_end = static_cast<std::uint64_t>(section.virtual_address) +
                                      std::max<std::uint32_t>(section.raw_size, section.virtual_size);
    const bool contains_entry_point = static_cast<std::uint64_t>(pe.entry_point) >= section.virtual_address &&
                                      static_cast<std::uint64_t>(pe.entry_point) < section_end;
    SectionInfo info;
    info.executable = executable;
    info.writable = writable;
    info.readable = readable;
    info.zero_raw = section.raw_size == 0;
    info.contains_entry_point = contains_entry_point;
    info.entropy = section_entropy(data, section);
    info.raw_virtual_delta = std::fabs(raw_size - virtual_size) / span;
    info.virtual_raw_ratio = virtual_size / std::max(raw_size, 1.0);
    section_infos.push_back(info);
    for (std::size_t group = 0; group < section_groups.size(); ++group) {
      if (contains_any(section.name, section_groups[group])) {
        ++group_hits[group];
      }
    }
  }
  const double section_count = std::max<std::size_t>(section_infos.size(), 1);
  std::vector<const SectionInfo*> executable_sections;
  std::vector<const SectionInfo*> writable_sections;
  std::vector<const SectionInfo*> readable_sections;
  std::vector<const SectionInfo*> executable_writable_sections;
  for (const SectionInfo& info : section_infos) {
    if (info.executable) executable_sections.push_back(&info);
    if (info.writable) writable_sections.push_back(&info);
    if (info.readable) readable_sections.push_back(&info);
    if (info.executable && info.writable) executable_writable_sections.push_back(&info);
  }
  auto high_entropy_ratio = [](const std::vector<const SectionInfo*>& values) {
    if (values.empty()) return 0.0;
    return static_cast<double>(std::count_if(values.begin(), values.end(), [](const SectionInfo* info) {
      return info->entropy >= 0.80;
    })) / static_cast<double>(values.size());
  };
  auto zero_raw_ratio = [](const std::vector<const SectionInfo*>& values) {
    if (values.empty()) return 0.0;
    return static_cast<double>(std::count_if(values.begin(), values.end(), [](const SectionInfo* info) {
      return info->zero_raw;
    })) / static_cast<double>(values.size());
  };
  std::vector<double> deltas;
  std::vector<double> virtual_ratios;
  deltas.reserve(section_infos.size());
  virtual_ratios.reserve(section_infos.size());
  for (const SectionInfo& info : section_infos) {
    deltas.push_back(info.raw_virtual_delta);
    virtual_ratios.push_back(info.virtual_raw_ratio);
  }
  const SectionInfo* entry_point_section = nullptr;
  if (!section_infos.empty()) {
    const auto entry_point_iterator = std::find_if(section_infos.begin(), section_infos.end(), [](const SectionInfo& info) {
      return info.contains_entry_point;
    });
    if (entry_point_iterator != section_infos.end()) {
      entry_point_section = &*entry_point_iterator;
    }
  }
  const auto* first_section = section_infos.empty() ? nullptr : &section_infos.front();
  const auto* last_section = section_infos.empty() ? nullptr : &section_infos.back();
  append_feature(features, std::log1p(static_cast<double>(executable_sections.size())));
  append_feature(features, std::log1p(static_cast<double>(writable_sections.size())));
  append_feature(features, std::log1p(static_cast<double>(readable_sections.size())));
  append_feature(features, std::log1p(static_cast<double>(executable_writable_sections.size())));
  append_feature(features, high_entropy_ratio(executable_sections));
  append_feature(features, high_entropy_ratio(writable_sections));
  append_feature(features, zero_raw_ratio(executable_sections));
  append_feature(features, zero_raw_ratio(writable_sections));
  append_feature(features, deltas.empty() ? 0.0 : *std::max_element(deltas.begin(), deltas.end()));
  append_feature(features, deltas.empty() ? 0.0 : std::accumulate(deltas.begin(), deltas.end(), 0.0) / deltas.size());
  append_feature(features, virtual_ratios.empty() ? 0.0 : std::log1p(*std::max_element(virtual_ratios.begin(), virtual_ratios.end())));
  append_feature(features, entry_point_section && entry_point_section->executable ? 1.0 : 0.0);
  append_feature(features, entry_point_section && entry_point_section->writable ? 1.0 : 0.0);
  append_feature(features, entry_point_section ? entry_point_section->entropy : 0.0);
  append_feature(features, entry_point_section ? entry_point_section->raw_virtual_delta : 0.0);
  append_feature(features, first_section ? first_section->entropy : 0.0);
  append_feature(features, first_section && first_section->executable ? 1.0 : 0.0);
  append_feature(features, first_section && first_section->writable ? 1.0 : 0.0);
  append_feature(features, last_section ? last_section->entropy : 0.0);
  append_feature(features, last_section && last_section->executable ? 1.0 : 0.0);
  append_feature(features, last_section && last_section->writable ? 1.0 : 0.0);
  for (std::uint32_t count : group_hits) {
    append_feature(features, safe_ratio(count, section_count));
  }
  if (features.size() != kContentPeV2FeatureDim) {
    return zero_features(kContentPeV2FeatureDim);
  }
  return features;
}

std::vector<std::uint8_t> string_sample(const std::vector<std::uint8_t>& data) {
  constexpr std::size_t head_size = 2 * 1024 * 1024;
  constexpr std::size_t tail_size = 512 * 1024;
  if (data.size() <= head_size + tail_size) {
    return data;
  }
  std::vector<std::uint8_t> sampled;
  sampled.reserve(head_size + tail_size);
  sampled.insert(sampled.end(), data.begin(), data.begin() + static_cast<std::ptrdiff_t>(head_size));
  sampled.insert(sampled.end(), data.end() - static_cast<std::ptrdiff_t>(tail_size), data.end());
  return sampled;
}

// Non-overlapping occurrence count. The string feature block runs this 89 times
// (83 patterns plus six explicit calls) over a buffer of up to 2.5 MB, which was
// 79.8% of a scan and the whole of its p90 tail. Advancing one byte at a time
// meant ~222 MB of comparisons per file; memchr skips to the next candidate
// first byte instead. Counting semantics are unchanged: on a match the cursor
// jumps past the pattern, otherwise it advances one byte.
std::size_t count_substring(const std::vector<std::uint8_t>& data, std::string_view pattern) {
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

// Counts printable-ASCII runs of at least four bytes. The caller only ever
// needed the mean and maximum run length, so those are accumulated here instead
// of materialising every length in a vector. `length_sum` is accumulated as a
// double in run order, matching what std::accumulate(begin, end, 0.0) produced.
std::size_t ascii_run_count(
    const std::vector<std::uint8_t>& data,
    double* length_sum = nullptr,
    std::size_t* length_max = nullptr,
    std::array<std::uint64_t, 256>* histogram = nullptr) {
  std::size_t current = 0;
  std::size_t count = 0;
  auto close_run = [&] {
    if (current >= 4) {
      ++count;
      if (length_sum) *length_sum += static_cast<double>(current);
      if (length_max && current > *length_max) *length_max = current;
    }
  };
  // The byte histogram is folded in here so the buffer is walked once instead
  // of twice; the counts are identical either way.
  for (std::uint8_t value : data) {
    if (histogram) (*histogram)[value] += 1;
    if (value >= 32 && value <= 126) {
      ++current;
    } else {
      close_run();
      current = 0;
    }
  }
  close_run();
  return count;
}

// Counts UTF-16LE printable-ASCII runs. Only the count is consumed, so no run
// lengths are collected.
std::size_t utf16_ascii_run_count(const std::vector<std::uint8_t>& data) {
  std::size_t current = 0;
  std::size_t count = 0;
  std::size_t index = 0;
  while (index + 1 < data.size()) {
    if (data[index] >= 32 && data[index] <= 126 && data[index + 1] == 0) {
      ++current;
      index += 2;
      continue;
    }
    if (current >= 4) {
      ++count;
    }
    current = 0;
    ++index;
  }
  if (current >= 4) {
    ++count;
  }
  return count;
}

std::size_t url_regex_count(const std::vector<std::uint8_t>& data) {
  std::size_t count = 0;
  for (std::size_t index = 0; index < data.size();) {
    // Both recognised prefixes start with 'h', so any position that does not
    // hold an 'h' can only fall through to ++index. Jumping straight to the
    // next 'h' skips that byte-at-a-time walk without changing the scan.
    const void* candidate =
        std::memchr(data.data() + index, 'h', data.size() - index);
    if (!candidate) {
      break;
    }
    index = static_cast<std::size_t>(
        static_cast<const std::uint8_t*>(candidate) - data.data());
    std::size_t prefix = 0;
    if (index + 7 <= data.size() && std::equal(data.begin() + static_cast<std::ptrdiff_t>(index), data.begin() + static_cast<std::ptrdiff_t>(index + 7), "http://")) {
      prefix = 7;
    } else if (index + 8 <= data.size() && std::equal(data.begin() + static_cast<std::ptrdiff_t>(index), data.begin() + static_cast<std::ptrdiff_t>(index + 8), "https://")) {
      prefix = 8;
    }
    if (prefix == 0) {
      ++index;
      continue;
    }
    std::size_t end = index + prefix;
    while (end < data.size() && data[end] != 0 && data[end] != '"' && data[end] != '\'' &&
           !space_table()[data[end]]) {
      ++end;
    }
    if (end > index + prefix) {
      ++count;
      index = end;
    } else {
      ++index;
    }
  }
  return count;
}

bool word_byte(std::uint8_t value) {
  return alnum_table()[value] || value == '_';
}

std::size_t ipv4_regex_count(const std::vector<std::uint8_t>& data) {
  std::size_t count = 0;
  for (std::size_t index = 0; index < data.size(); ++index) {
    // The first group needs at least one digit, so a position that does not
    // start with a digit always ends with valid == false. Testing that first
    // replaces a word_byte lookup plus the four-group setup with one compare,
    // and it is the same decision.
    if (data[index] < '0' || data[index] > '9') {
      continue;
    }
    if (index > 0 && word_byte(data[index - 1])) {
      continue;
    }
    std::size_t cursor = index;
    bool valid = true;
    for (int group = 0; group < 4; ++group) {
      std::size_t digits = 0;
      while (cursor < data.size() && data[cursor] >= '0' && data[cursor] <= '9' && digits < 4) {
        ++cursor;
        ++digits;
      }
      if (digits == 0 || digits > 3 || (group < 3 && (cursor >= data.size() || data[cursor] != '.'))) {
        valid = false;
        break;
      }
      if (group < 3) {
        ++cursor;
      }
    }
    if (valid && (cursor >= data.size() || !word_byte(data[cursor]))) {
      ++count;
      index = cursor == 0 ? index : cursor - 1;
    }
  }
  return count;
}

const std::array<std::vector<std::string_view>, 14>& string_patterns() {
  static const std::array<std::vector<std::string_view>, 14> patterns = {{
      {"http://", "https://", "www.", "ftp://"},
      {"socket", "connect", "recv", "send", "wininet", "ws2_32", "internetopen", "urldownload"},
      {"powershell", "cmd.exe", "wscript", "cscript", "mshta", "rundll32", "regsvr32"},
      {"currentversion\\run", "runonce", "\\services\\", "startup", "schtasks", "autostart"},
      {"createremotethread", "virtualalloc", "virtualprotect", "writeprocessmemory", "queueuserapc"},
      {"password", "credential", "token", "cookie", "browser", "wallet"},
      {"cryptencrypt", "cryptdecrypt", "bcrypt", "advapi32", "base64", "aes", "rsa"},
      {"isdebuggerpresent", "checkremotedebugger", "ntqueryinformationprocess", "sleep", "sandbox"},
      {"vmware", "virtualbox", "vbox", "qemu", "wine_get_unix_file_name"},
      {"upx", "themida", "vmprotect", "aspack", "enigma", "packed"},
      {"createfile", "writefile", "deletefile", "copyfile", "movefile", "findfirstfile"},
      {"regopenkey", "regsetvalue", "regcreatekey", "regdeletekey", "regqueryvalue"},
      {"microsoft", "windows", "google", "adobe", "intel", "nvidia", "mozilla", "oracle"},
      {"companyname", "productname", "filedescription", "originalfilename", "copyright"},
  }};
  return patterns;
}

// The string block matches 89 fixed patterns: six counted directly and 83 in
// fourteen groups. Running count_substring once per pattern meant 89 separate
// walks of a buffer up to 2.5 MB, which measured as 69% of the whole block.
//
// This matcher makes one pass instead. Patterns are bucketed by first byte, so
// a position only tests the handful of patterns that could start there. The
// per-pattern non-overlapping count is preserved exactly: count_substring jumps
// past a match and resumes after it, which is the same as visiting every
// position in order and ignoring any that falls inside the previous match of
// that pattern -- tracked here in `resume_at`.
//
// Flat layout, fixed so feature order cannot drift:
//   [0..2]  registry aggregate    [3..5]  path aggregate    [6..]  the 14 groups
struct MultiPatternMatcher {
  std::vector<std::string_view> patterns;
  // Buckets in compressed-row form rather than 256 separate vectors: the whole
  // index is two contiguous arrays, so the inner loop stays in L1 and the
  // empty-bucket test is one load from `bucket_begin`.
  std::array<std::uint16_t, 257> bucket_begin{};
  std::vector<std::uint16_t> bucket_entries;
  std::array<std::pair<std::size_t, std::size_t>, 14> group_ranges{};
};

const MultiPatternMatcher& multi_pattern_matcher() {
  static const MultiPatternMatcher matcher = [] {
    MultiPatternMatcher built;
    for (std::string_view pattern : {"\\software\\", "\\registry\\", "hkey_",
                                     "c:\\", "\\windows\\", "\\system32\\"}) {
      built.patterns.push_back(pattern);
    }
    const auto& groups = string_patterns();
    for (std::size_t group = 0; group < groups.size(); ++group) {
      const std::size_t begin = built.patterns.size();
      for (std::string_view pattern : groups[group]) {
        built.patterns.push_back(pattern);
      }
      built.group_ranges[group] = {begin, built.patterns.size()};
    }

    std::array<std::vector<std::uint16_t>, 256> staging;
    for (std::size_t index = 0; index < built.patterns.size(); ++index) {
      const std::string_view pattern = built.patterns[index];
      if (pattern.empty()) continue;
      staging[static_cast<std::uint8_t>(pattern.front())]
          .push_back(static_cast<std::uint16_t>(index));
    }
    std::uint16_t cursor = 0;
    for (std::size_t byte = 0; byte < 256; ++byte) {
      built.bucket_begin[byte] = cursor;
      for (const std::uint16_t index : staging[byte]) {
        built.bucket_entries.push_back(index);
        ++cursor;
      }
    }
    built.bucket_begin[256] = cursor;
    return built;
  }();
  return matcher;
}

std::vector<std::size_t> count_all_patterns(
    const std::vector<std::uint8_t>& data, const MultiPatternMatcher& matcher) {
  std::vector<std::size_t> counts(matcher.patterns.size(), 0);
  std::vector<std::size_t> resume_at(matcher.patterns.size(), 0);
  if (data.empty()) {
    return counts;
  }
  const std::uint8_t* const bytes = data.data();
  const std::size_t size = data.size();
  const std::uint16_t* const entries = matcher.bucket_entries.data();
  for (std::size_t index = 0; index < size; ++index) {
    const std::uint8_t byte = bytes[index];
    const std::uint16_t bucket_first = matcher.bucket_begin[byte];
    const std::uint16_t bucket_last = matcher.bucket_begin[byte + 1];
    if (bucket_first == bucket_last) {
      continue;
    }
    for (std::uint16_t slot = bucket_first; slot != bucket_last; ++slot) {
      const std::uint16_t pattern_index = entries[slot];
      if (index < resume_at[pattern_index]) {
        continue;
      }
      const std::string_view pattern = matcher.patterns[pattern_index];
      if (pattern.size() > size - index) {
        continue;
      }
      if (std::memcmp(bytes + index, pattern.data(), pattern.size()) != 0) {
        continue;
      }
      ++counts[pattern_index];
      resume_at[pattern_index] = index + pattern.size();
    }
  }
  return counts;
}

std::vector<float> content_string_features_impl(const std::vector<std::uint8_t>& input) {
  const std::vector<std::uint8_t> data = string_sample(input);
  if (data.empty()) {
    return zero_features(kContentStringFeatureDim);
  }
  const std::array<std::uint8_t, 256>& lower = lowercase_table();
  std::vector<std::uint8_t> lowered(data.size());
  for (std::size_t index = 0; index < data.size(); ++index) {
    lowered[index] = lower[data[index]];
  }
  const double length = static_cast<double>(data.size());

  // The byte histogram is produced by the ascii run scan, which already walks
  // every byte; printable, null and high-byte totals are then sums over its
  // buckets rather than three more passes.
  std::array<std::uint64_t, 256> byte_counts{};
  double ascii_length_sum = 0.0;
  std::size_t ascii_max = 0;
  const std::size_t ascii_count =
      ascii_run_count(data, &ascii_length_sum, &ascii_max, &byte_counts);
  // The utf16 run lengths were collected and then never read; only the count is
  // used, so nothing is accumulated for them.
  const std::size_t utf16_count = utf16_ascii_run_count(data);
  const double ascii_mean =
      ascii_count == 0 ? 0.0 : ascii_length_sum / static_cast<double>(ascii_count);

  std::size_t printable = 0;
  std::size_t high_bytes = 0;
  for (std::size_t value = 32; value <= 126; ++value) {
    printable += static_cast<std::size_t>(byte_counts[value]);
  }
  for (std::size_t value = 128; value < 256; ++value) {
    high_bytes += static_cast<std::size_t>(byte_counts[value]);
  }
  const std::size_t nulls = static_cast<std::size_t>(byte_counts[0]);
  std::vector<float> features;
  features.reserve(kContentStringFeatureDim);
  append_feature(features, std::log1p(length));
  append_feature(features, safe_ratio(printable, length));
  append_feature(features, safe_ratio(nulls, length));
  append_feature(features, safe_ratio(high_bytes, length));
  append_feature(features, std::log1p(static_cast<double>(ascii_count)));
  append_feature(features, safe_ratio(ascii_count, length / 1024.0));
  append_feature(features, std::min(ascii_mean, 512.0) / 512.0);
  append_feature(features, std::min(static_cast<double>(ascii_max), 4096.0) / 4096.0);
  append_feature(features, std::log1p(static_cast<double>(utf16_count)));
  append_feature(features, safe_ratio(utf16_count, length / 1024.0));
  append_feature(features, std::log1p(static_cast<double>(url_regex_count(lowered))));
  append_feature(features, std::log1p(static_cast<double>(ipv4_regex_count(lowered))));
  const MultiPatternMatcher& matcher = multi_pattern_matcher();
  const std::vector<std::size_t> pattern_counts = count_all_patterns(lowered, matcher);
  append_feature(features, std::log1p(
      pattern_counts[0] + pattern_counts[1] + pattern_counts[2]));
  append_feature(features, std::log1p(
      pattern_counts[3] + pattern_counts[4] + pattern_counts[5]));
  append_feature(features, entropy_from_counts(byte_counts, data.size()));
  for (const auto& range : matcher.group_ranges) {
    std::size_t count = 0;
    for (std::size_t index = range.first; index < range.second; ++index) {
      count += pattern_counts[index];
    }
    append_feature(features, std::log1p(static_cast<double>(count)));
    append_feature(features, count > 0 ? 1.0 : 0.0);
  }
  if (features.size() != kContentStringFeatureDim) {
    return zero_features(kContentStringFeatureDim);
  }
  return features;
}

}  // namespace

std::vector<float> content_pe_v2_features(const std::vector<std::uint8_t>& data) {
  try {
    return content_pe_v2_features_impl(data);
  } catch (...) {
    return zero_features(kContentPeV2FeatureDim);
  }
}

std::vector<float> content_string_features(const std::vector<std::uint8_t>& data) {
  try {
    return content_string_features_impl(data);
  } catch (...) {
    return zero_features(kContentStringFeatureDim);
  }
}

}  // namespace axon_loop151_native
