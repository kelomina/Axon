#include "axon_loop151_native_model.h"

#include <algorithm>
#include <cctype>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <initializer_list>
#include <limits>
#include <map>
#include <memory>
#include <numeric>
#include <filesystem>
#include <sstream>
#include <utility>

namespace axon_loop151_native {
namespace detail {

class JsonValue {
 public:
  enum class Type { Null, Boolean, Number, String, Array, Object };

  Type type = Type::Null;
  bool boolean_value = false;
  double number_value = 0.0;
  std::string string_value;
  std::vector<JsonValue> array_value;
  // A stage-2 asset is essentially a handful of very large flat numeric arrays:
  // the noise model alone holds 11.4M numbers against 2,486 strings, 178 arrays
  // and 49 objects. Giving each of those numbers its own JsonValue cost ~96
  // bytes plus vector-growth copies, which is what drove a 4.8 GiB load peak.
  // Arrays whose elements are all numbers are therefore stored unboxed here and
  // never materialised as child nodes. Mixed or non-numeric arrays fall back to
  // `array_value`, so nothing else about the DOM changes.
  std::vector<double> number_array_value;
  bool numeric_array = false;

  const JsonValue* field(const char* name) const {
    if (type != Type::Object) {
      return nullptr;
    }
    auto it = object_value.find(name);
    return it == object_value.end() ? nullptr : &it->second;
  }

  std::size_t array_size() const {
    return numeric_array ? number_array_value.size() : array_value.size();
  }

  // Materialise the unboxed numbers back into child nodes. Only used by the
  // handful of call sites that need real JsonValue elements.
  void expand_numeric_array() {
    if (!numeric_array) {
      return;
    }
    array_value.clear();
    array_value.reserve(number_array_value.size());
    for (const double number : number_array_value) {
      JsonValue item;
      item.type = Type::Number;
      item.number_value = number;
      array_value.push_back(std::move(item));
    }
    number_array_value.clear();
    number_array_value.shrink_to_fit();
    numeric_array = false;
  }

  std::map<std::string, JsonValue> object_value;
};

}  // namespace detail

namespace {

using detail::JsonValue;

class JsonParser {
 public:
  explicit JsonParser(const std::string& text) : text_(text) {}

  bool parse(JsonValue& value, std::string& error) {
    skip_space();
    if (!parse_value(value, error)) {
      return false;
    }
    skip_space();
    if (position_ != text_.size()) {
      error = "trailing characters after JSON document";
      return false;
    }
    return true;
  }

 private:
  bool parse_value(JsonValue& value, std::string& error) {
    skip_space();
    if (position_ >= text_.size()) {
      error = "unexpected end of JSON document";
      return false;
    }
    switch (text_[position_]) {
      case 'n': return parse_literal("null", JsonValue::Type::Null, value, error);
      case 't':
        value.boolean_value = true;
        return parse_literal("true", JsonValue::Type::Boolean, value, error);
      case 'f':
        value.boolean_value = false;
        return parse_literal("false", JsonValue::Type::Boolean, value, error);
      case '"':
        value.type = JsonValue::Type::String;
        return parse_string(value.string_value, error);
      case '[':
        return parse_array(value, error);
      case '{':
        return parse_object(value, error);
      default:
        if (text_[position_] == '-' || text_[position_] == '+' ||
            (text_[position_] >= '0' && text_[position_] <= '9')) {
          return parse_number(value, error);
        }
        error = "invalid JSON value";
        return false;
    }
  }

  bool parse_literal(
      const char* literal,
      JsonValue::Type type,
      JsonValue& value,
      std::string& error) {
    const std::size_t length = std::char_traits<char>::length(literal);
    if (text_.compare(position_, length, literal) != 0) {
      error = "invalid JSON literal";
      return false;
    }
    position_ += length;
    value.type = type;
    return true;
  }

  // std::strtod consults the C locale on every call, which cost roughly a
  // microsecond each; across the 19.2M numbers in the stage-2 assets that was
  // most of the model load time. std::from_chars is locale-independent and
  // produces the same correctly-rounded double.
  //
  // Two grammar details are preserved deliberately: the dispatcher accepts a
  // leading '+' that from_chars rejects, so it is consumed here. Hexadecimal
  // float literals, which strtod would have accepted, are no longer parsed --
  // JSON does not permit them and no emitter produces them.
  bool parse_number(JsonValue& value, std::string& error) {
    const char* const document = text_.data();
    const char* const document_end = document + text_.size();
    const char* parse_from = document + position_;
    if (parse_from != document_end && *parse_from == '+') {
      ++parse_from;
    }
    double parsed = 0.0;
    const std::from_chars_result result = std::from_chars(parse_from, document_end, parsed);
    if (result.ec != std::errc{} || result.ptr == parse_from || !std::isfinite(parsed)) {
      error = "invalid JSON number";
      return false;
    }
    position_ = static_cast<std::size_t>(result.ptr - document);
    value.type = JsonValue::Type::Number;
    value.number_value = parsed;
    return true;
  }

  bool parse_string(std::string& output, std::string& error) {
    if (position_ >= text_.size() || text_[position_] != '"') {
      error = "JSON string must start with a quote";
      return false;
    }
    ++position_;
    output.clear();
    while (position_ < text_.size()) {
      const unsigned char character = static_cast<unsigned char>(text_[position_++]);
      if (character == '"') {
        return true;
      }
      if (character == '\\') {
        if (!parse_escape(output, error)) {
          return false;
        }
        continue;
      }
      if (character < 0x20) {
        error = "control character in JSON string";
        return false;
      }
      output.push_back(static_cast<char>(character));
    }
    error = "unterminated JSON string";
    return false;
  }

  bool parse_escape(std::string& output, std::string& error) {
    if (position_ >= text_.size()) {
      error = "unterminated JSON escape";
      return false;
    }
    const char escaped = text_[position_++];
    switch (escaped) {
      case '"': output.push_back('"'); return true;
      case '\\': output.push_back('\\'); return true;
      case '/': output.push_back('/'); return true;
      case 'b': output.push_back('\b'); return true;
      case 'f': output.push_back('\f'); return true;
      case 'n': output.push_back('\n'); return true;
      case 'r': output.push_back('\r'); return true;
      case 't': output.push_back('\t'); return true;
      case 'u': {
        unsigned int codepoint = 0;
        for (int digit = 0; digit < 4; ++digit) {
          if (position_ >= text_.size()) {
            error = "truncated unicode escape";
            return false;
          }
          const char value = text_[position_++];
          codepoint <<= 4;
          if (value >= '0' && value <= '9') codepoint += static_cast<unsigned int>(value - '0');
          else if (value >= 'a' && value <= 'f') codepoint += static_cast<unsigned int>(value - 'a' + 10);
          else if (value >= 'A' && value <= 'F') codepoint += static_cast<unsigned int>(value - 'A' + 10);
          else {
            error = "invalid unicode escape";
            return false;
          }
        }
        append_utf8(output, codepoint);
        return true;
      }
      default:
        error = "unsupported JSON escape";
        return false;
    }
  }

  static void append_utf8(std::string& output, unsigned int codepoint) {
    if (codepoint <= 0x7f) {
      output.push_back(static_cast<char>(codepoint));
    } else if (codepoint <= 0x7ff) {
      output.push_back(static_cast<char>(0xc0 | (codepoint >> 6)));
      output.push_back(static_cast<char>(0x80 | (codepoint & 0x3f)));
    } else {
      output.push_back(static_cast<char>(0xe0 | (codepoint >> 12)));
      output.push_back(static_cast<char>(0x80 | ((codepoint >> 6) & 0x3f)));
      output.push_back(static_cast<char>(0x80 | (codepoint & 0x3f)));
    }
  }

  // Matches the number branch of parse_value, including the leading '+' it
  // accepts, so the unboxed array fast path covers exactly the same elements.
  static bool starts_number(char character) {
    return character == '-' || character == '+' ||
        (character >= '0' && character <= '9');
  }

  bool parse_array(JsonValue& value, std::string& error) {
    value.type = JsonValue::Type::Array;
    value.array_value.clear();
    value.number_array_value.clear();
    // Assume the array is all numbers and unbox as we go; the first non-number
    // element demotes it back to a normal node array.
    value.numeric_array = true;
    ++position_;
    skip_space();
    if (position_ < text_.size() && text_[position_] == ']') {
      ++position_;
      value.numeric_array = false;
      return true;
    }
    while (position_ < text_.size()) {
      if (value.numeric_array && starts_number(text_[position_])) {
        JsonValue number;
        if (!parse_number(number, error)) {
          return false;
        }
        value.number_array_value.push_back(number.number_value);
      } else {
        if (value.numeric_array) {
          value.expand_numeric_array();
        }
        JsonValue item;
        if (!parse_value(item, error)) {
          return false;
        }
        value.array_value.push_back(std::move(item));
      }
      skip_space();
      if (position_ >= text_.size()) {
        break;
      }
      if (text_[position_] == ']') {
        ++position_;
        return true;
      }
      if (text_[position_] != ',') {
        error = "expected comma in JSON array";
        return false;
      }
      ++position_;
      skip_space();
    }
    error = "unterminated JSON array";
    return false;
  }

  bool parse_object(JsonValue& value, std::string& error) {
    value.type = JsonValue::Type::Object;
    value.object_value.clear();
    ++position_;
    skip_space();
    if (position_ < text_.size() && text_[position_] == '}') {
      ++position_;
      return true;
    }
    while (position_ < text_.size()) {
      if (text_[position_] != '"') {
        error = "JSON object key must be a string";
        return false;
      }
      std::string key;
      if (!parse_string(key, error)) {
        return false;
      }
      if (value.object_value.find(key) != value.object_value.end()) {
        error = "duplicate JSON object key: " + key;
        return false;
      }
      skip_space();
      if (position_ >= text_.size() || text_[position_] != ':') {
        error = "expected colon after JSON object key";
        return false;
      }
      ++position_;
      JsonValue item;
      if (!parse_value(item, error)) {
        return false;
      }
      value.object_value.emplace(std::move(key), std::move(item));
      skip_space();
      if (position_ >= text_.size()) {
        break;
      }
      if (text_[position_] == '}') {
        ++position_;
        return true;
      }
      if (text_[position_] != ',') {
        error = "expected comma in JSON object";
        return false;
      }
      ++position_;
      skip_space();
    }
    error = "unterminated JSON object";
    return false;
  }

  void skip_space() {
    while (position_ < text_.size()) {
      const unsigned char character = static_cast<unsigned char>(text_[position_]);
      if (character != ' ' && character != '\t' && character != '\r' && character != '\n') {
        break;
      }
      ++position_;
    }
  }

  const std::string& text_;
  std::size_t position_ = 0;
};

const JsonValue* first_field(const JsonValue& value, std::initializer_list<const char*> names) {
  for (const char* name : names) {
    if (const JsonValue* field = value.field(name)) {
      return field;
    }
  }
  return nullptr;
}

bool number_value(const JsonValue* value, double& output) {
  if (!value || value->type != JsonValue::Type::Number || !std::isfinite(value->number_value)) {
    return false;
  }
  output = value->number_value;
  return true;
}

bool integer_value(const JsonValue* value, int& output) {
  double value_as_number = 0.0;
  if (!number_value(value, value_as_number) || value_as_number < static_cast<double>(std::numeric_limits<int>::min()) ||
      value_as_number > static_cast<double>(std::numeric_limits<int>::max())) {
    return false;
  }
  output = static_cast<int>(value_as_number);
  return std::fabs(value_as_number - static_cast<double>(output)) < 1.0e-9;
}

bool boolean_value(const JsonValue* value, bool& output) {
  if (!value || value->type != JsonValue::Type::Boolean) {
    return false;
  }
  output = value->boolean_value;
  return true;
}

bool string_value(const JsonValue* value, std::string& output) {
  if (!value || value->type != JsonValue::Type::String) {
    return false;
  }
  output = value->string_value;
  return true;
}

bool number_array(const JsonValue* value, std::vector<double>& output) {
  if (!value || value->type != JsonValue::Type::Array) {
    return false;
  }
  if (value->numeric_array) {
    // The parser already stored these unboxed; every element is finite by
    // construction because parse_number rejects non-finite input.
    output = value->number_array_value;
    return true;
  }
  output.clear();
  output.reserve(value->array_value.size());
  for (const JsonValue& item : value->array_value) {
    double number = 0.0;
    if (!number_value(&item, number)) {
      return false;
    }
    output.push_back(number);
  }
  return true;
}

bool integer_array(const JsonValue* value, std::vector<int>& output) {
  if (!value || value->type != JsonValue::Type::Array) {
    return false;
  }
  if (value->numeric_array) {
    output.clear();
    output.reserve(value->number_array_value.size());
    for (const double number : value->number_array_value) {
      // Mirrors integer_value exactly, including its 1e-9 tolerance rather than
      // an exact floor comparison -- a stricter test here would reject models
      // the boxed path accepts.
      if (!std::isfinite(number) || number < static_cast<double>(std::numeric_limits<int>::min()) ||
          number > static_cast<double>(std::numeric_limits<int>::max())) {
        return false;
      }
      const int converted = static_cast<int>(number);
      if (std::fabs(number - static_cast<double>(converted)) >= 1.0e-9) {
        return false;
      }
      output.push_back(converted);
    }
    return true;
  }
  output.clear();
  output.reserve(value->array_value.size());
  for (const JsonValue& item : value->array_value) {
    int number = 0;
    if (!integer_value(&item, number)) {
      return false;
    }
    output.push_back(number);
  }
  return true;
}

std::string json_model_type(const JsonValue& value) {
  std::string type;
  if (string_value(first_field(value, {"model_type", "type", "kind"}), type)) {
    std::transform(type.begin(), type.end(), type.begin(), [](unsigned char character) {
      return static_cast<char>(std::tolower(character));
    });
  }
  return type;
}

const JsonValue* unwrap_model(const JsonValue& value) {
  if (const JsonValue* model = value.field("model")) {
    if (model->type == JsonValue::Type::Object) {
      return model;
    }
  }
  return &value;
}

bool object_model_value(const JsonValue& value, const JsonValue*& model_value) {
  model_value = &value;
  if (value.type != JsonValue::Type::Object) {
    return false;
  }
  if (const JsonValue* nested = value.field("native_model")) {
    if (nested->type != JsonValue::Type::Object) {
      return false;
    }
    model_value = nested;
  } else if (const JsonValue* nested = value.field("model")) {
    if (nested->type == JsonValue::Type::Object) {
      model_value = nested;
    }
  }
  return true;
}

std::string json_encode(const JsonValue& value);

bool parse_model_reference(
    const JsonValue& value,
    const std::filesystem::path& base_directory,
    std::unique_ptr<NativeScoreModel>& output,
    std::string& error) {
  std::string path_text;
  if (string_value(&value, path_text)) {
    const std::filesystem::path model_path = std::filesystem::u8path(path_text).is_absolute()
        ? std::filesystem::u8path(path_text)
        : (base_directory / std::filesystem::u8path(path_text)).lexically_normal();
    output = NativeScoreModel::load_file(model_path.u8string(), error);
    return output != nullptr;
  }
  const JsonValue* model_value = nullptr;
  if (!object_model_value(value, model_value)) {
    error = "native model reference must be a path or object";
    return false;
  }
  if (!model_value || model_value->type != JsonValue::Type::Object) {
    error = "native model object is invalid";
    return false;
  }
  // The node is already parsed, so hand it over directly. Re-serialising it
  // with json_encode and parsing the result again meant every number in a
  // nested model made a round trip through 17-digit text; with 15 base models
  // per stack that was the bulk of model load time.
  output = NativeScoreModel::load_parsed(*model_value, "inline native model", error);
  return output != nullptr;
}

std::string json_encode_string(const std::string& value) {
  std::string output = "\"";
  for (unsigned char character : value) {
    switch (character) {
      case '\\': output += "\\\\"; break;
      case '"': output += "\\\""; break;
      case '\b': output += "\\b"; break;
      case '\f': output += "\\f"; break;
      case '\n': output += "\\n"; break;
      case '\r': output += "\\r"; break;
      case '\t': output += "\\t"; break;
      default:
        if (character < 0x20) {
          char encoded[7];
          std::snprintf(encoded, sizeof(encoded), "\\u%04x", character);
          output += encoded;
        } else {
          output.push_back(static_cast<char>(character));
        }
    }
  }
  output += '"';
  return output;
}

std::string json_encode(const JsonValue& value) {
  std::ostringstream output;
  switch (value.type) {
    case JsonValue::Type::Null:
      return "null";
    case JsonValue::Type::Boolean:
      return value.boolean_value ? "true" : "false";
    case JsonValue::Type::Number:
      output.precision(17);
      output << value.number_value;
      return output.str();
    case JsonValue::Type::String:
      return json_encode_string(value.string_value);
    case JsonValue::Type::Array:
      output << '[';
      if (value.numeric_array) {
        // Same formatting the Number case uses, so an unboxed array re-encodes
        // byte-for-byte like a boxed one.
        output.precision(17);
        for (std::size_t index = 0; index < value.number_array_value.size(); ++index) {
          if (index != 0) output << ',';
          output << value.number_array_value[index];
        }
      } else {
        for (std::size_t index = 0; index < value.array_value.size(); ++index) {
          if (index != 0) output << ',';
          output << json_encode(value.array_value[index]);
        }
      }
      output << ']';
      return output.str();
    case JsonValue::Type::Object:
      output << '{';
      std::size_t index = 0;
      for (const auto& item : value.object_value) {
        if (index++ != 0) output << ',';
        output << json_encode_string(item.first) << ':' << json_encode(item.second);
      }
      output << '}';
      return output.str();
  }
  return "null";
}

struct TreeNode {
  double value = 0.0;
  std::vector<double> class_values;
  int feature_index = -1;
  double threshold = 0.0;
  bool missing_go_left = false;
  int left = -1;
  int right = -1;
  bool leaf = false;
};

struct TreeModel {
  std::vector<TreeNode> nodes;
  bool children_relative = true;
  bool probability_leaf = false;

  double predict(const std::vector<float>& features) const {
    if (nodes.empty()) {
      return 0.0;
    }
    int node_index = 0;
    for (std::size_t guard = 0; guard < nodes.size() + 1; ++guard) {
      if (node_index < 0 || static_cast<std::size_t>(node_index) >= nodes.size()) {
        return 0.0;
      }
      const TreeNode& node = nodes[static_cast<std::size_t>(node_index)];
      if (node.leaf) {
        if (!node.class_values.empty()) {
          const double denominator = std::accumulate(node.class_values.begin(), node.class_values.end(), 0.0);
          if (denominator > 0.0) {
            return node.class_values.back() / denominator;
          }
        }
        return probability_leaf ? std::clamp(node.value, 0.0, 1.0) : node.value;
      }
      if (node.feature_index < 0 || static_cast<std::size_t>(node.feature_index) >= features.size()) {
        return 0.0;
      }
      const float feature = features[static_cast<std::size_t>(node.feature_index)];
      const bool go_left = std::isnan(feature) ? node.missing_go_left : static_cast<double>(feature) <= node.threshold;
      const int child = go_left ? node.left : node.right;
      node_index = children_relative ? child + 0 : child;
    }
    return 0.0;
  }
};

bool parse_tree_node(const JsonValue& value, TreeNode& output, std::string& error) {
  if (value.type != JsonValue::Type::Object) {
    error = "tree node must be a JSON object";
    return false;
  }
  if (const JsonValue* leaf_value = first_field(value, {"is_leaf", "leaf"})) {
    if (!boolean_value(leaf_value, output.leaf)) {
      int leaf_flag = 0;
      if (!integer_value(leaf_value, leaf_flag)) {
        error = "tree node is_leaf must be boolean or integer";
        return false;
      }
      output.leaf = leaf_flag != 0;
    }
  }
  const JsonValue* raw_value = first_field(value, {"value", "leaf_value"});
  if (raw_value && raw_value->type == JsonValue::Type::Array) {
    if (!number_array(raw_value, output.class_values)) {
      error = "tree node value array must be numeric";
      return false;
    }
  } else {
    number_value(raw_value, output.value);
  }
  if (const JsonValue* class_values = first_field(value, {"class_values", "class_counts", "counts", "probabilities"})) {
    if (!number_array(class_values, output.class_values)) {
      error = "tree node class_values must be numeric";
      return false;
    }
  }
  integer_value(first_field(value, {"feature_idx", "feature_index", "feature"}), output.feature_index);
  number_value(first_field(value, {"num_threshold", "threshold"}), output.threshold);
  if (const JsonValue* missing = first_field(value, {"missing_go_to_left", "missing_left"})) {
    if (!boolean_value(missing, output.missing_go_left)) {
      int missing_flag = 0;
      if (!integer_value(missing, missing_flag)) {
        error = "tree node missing_go_to_left must be boolean or integer";
        return false;
      }
      output.missing_go_left = missing_flag != 0;
    }
  }
  if (!integer_value(first_field(value, {"left", "left_child", "children_left"}), output.left) ||
      !integer_value(first_field(value, {"right", "right_child", "children_right"}), output.right)) {
    if (!output.leaf) {
      error = "non-leaf tree node is missing child indices";
      return false;
    }
  }
  return true;
}

bool parse_tree_array(const JsonValue* value, std::vector<TreeModel>& output, std::string& error) {
  if (!value || value->type != JsonValue::Type::Array) {
    return false;
  }
  output.clear();
  output.reserve(value->array_value.size());
  for (const JsonValue& tree_value : value->array_value) {
    const JsonValue* nodes_value = tree_value.field("nodes");
    if (!nodes_value) {
      nodes_value = &tree_value;
    }
    if (nodes_value->type != JsonValue::Type::Array) {
      error = "tree nodes must be an array";
      return false;
    }
    TreeModel tree;
    bool relative = true;
    boolean_value(tree_value.field("children_relative"), relative);
    tree.children_relative = relative;
    bool probability_leaf = false;
    boolean_value(tree_value.field("probability_leaf"), probability_leaf);
    tree.probability_leaf = probability_leaf;
    for (const JsonValue& node_value : nodes_value->array_value) {
      TreeNode node;
      if (!parse_tree_node(node_value, node, error)) {
        return false;
      }
      tree.nodes.push_back(std::move(node));
    }
    if (tree.nodes.empty()) {
      error = "tree has no nodes";
      return false;
    }
    output.push_back(std::move(tree));
  }
  return !output.empty();
}

bool parse_flat_trees(const JsonValue& root, std::vector<TreeModel>& output, std::string& error) {
  std::vector<int> offsets;
  std::vector<double> values;
  std::vector<int> feature_indices;
  std::vector<double> thresholds;
  std::vector<int> missing_left;
  std::vector<int> left;
  std::vector<int> right;
  std::vector<int> leaves;
  if (!integer_array(first_field(root, {"tree_node_offsets", "tree_offsets"}), offsets) ||
      !number_array(root.field("node_values"), values) ||
      !integer_array(root.field("node_feature_idx"), feature_indices) ||
      !number_array(root.field("node_num_thresholds"), thresholds) ||
      !integer_array(root.field("node_missing_go_to_left"), missing_left) ||
      !integer_array(root.field("node_left"), left) ||
      !integer_array(root.field("node_right"), right) ||
      !integer_array(root.field("node_is_leaf"), leaves)) {
    return false;
  }
  if (offsets.size() < 2 || values.empty() || feature_indices.size() != values.size() ||
      thresholds.size() != values.size() || missing_left.size() != values.size() ||
      left.size() != values.size() || right.size() != values.size() || leaves.size() != values.size() ||
      offsets.front() != 0 || offsets.back() != static_cast<int>(values.size())) {
    error = "flat tree arrays have inconsistent dimensions";
    return false;
  }
  output.clear();
  output.reserve(offsets.size() - 1);
  for (std::size_t tree_index = 0; tree_index + 1 < offsets.size(); ++tree_index) {
    if (offsets[tree_index] < 0 || offsets[tree_index + 1] <= offsets[tree_index] ||
        offsets[tree_index + 1] > static_cast<int>(values.size())) {
      error = "flat tree offsets are invalid";
      return false;
    }
    TreeModel tree;
    for (int node_index = offsets[tree_index]; node_index < offsets[tree_index + 1]; ++node_index) {
      const std::size_t index = static_cast<std::size_t>(node_index);
      TreeNode node;
      node.value = values[index];
      node.feature_index = feature_indices[index];
      node.threshold = thresholds[index];
      node.missing_go_left = missing_left[index] != 0;
      node.left = left[index];
      node.right = right[index];
      node.leaf = leaves[index] != 0;
      tree.nodes.push_back(std::move(node));
    }
    output.push_back(std::move(tree));
  }
  return !output.empty();
}

std::vector<double> flatten_coefficients(const JsonValue* value) {
  std::vector<double> output;
  if (!value) {
    return output;
  }
  if (value->type == JsonValue::Type::Array) {
    if (value->numeric_array) {
      return value->number_array_value;
    }
    for (const JsonValue& item : value->array_value) {
      if (item.type == JsonValue::Type::Array) {
        if (item.numeric_array) {
          output.insert(output.end(), item.number_array_value.begin(), item.number_array_value.end());
          continue;
        }
        for (const JsonValue& nested : item.array_value) {
          if (nested.type == JsonValue::Type::Number) output.push_back(nested.number_value);
        }
      } else if (item.type == JsonValue::Type::Number) {
        output.push_back(item.number_value);
      }
    }
  }
  return output;
}

}  // namespace

struct NativeScoreModel::Impl {
  std::string type;
  std::size_t feature_count = 0;
  double intercept = 0.0;
  std::vector<double> coefficients;
  std::vector<double> scaler_mean;
  std::vector<double> scaler_scale;
  std::vector<TreeModel> trees;
  double baseline = 0.0;
  bool average_tree_probabilities = false;
  bool probability_leaf = false;
};

struct NativeStackModel::Impl {
  std::vector<std::unique_ptr<NativeScoreModel>> base_models;
  std::unique_ptr<NativeScoreModel> meta_model;
  std::size_t feature_count = 0;
  std::size_t base_probability_feature_count = 6;
  bool drop_base_probability_features = false;
  float decision_threshold = 0.5f;
};

NativeScoreModel::NativeScoreModel() : impl_(std::make_unique<Impl>()) {}
NativeScoreModel::NativeScoreModel(NativeScoreModel&&) noexcept = default;
NativeScoreModel& NativeScoreModel::operator=(NativeScoreModel&&) noexcept = default;
NativeScoreModel::~NativeScoreModel() = default;

std::unique_ptr<NativeScoreModel> NativeScoreModel::load_file(
    const std::string& path,
    std::string& error) {
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    error = "native model file cannot be opened: " + path;
    return nullptr;
  }
  std::ostringstream buffer;
  buffer << input.rdbuf();
  return load_document(buffer.str(), path, error);
}

std::unique_ptr<NativeScoreModel> NativeScoreModel::load_document(
    const std::string& document,
    const std::string& source_name,
    std::string& error) {
  detail::JsonValue root;
  JsonParser parser(document);
  if (!parser.parse(root, error)) {
    error = source_name + ": " + error;
    return nullptr;
  }
  return load_parsed(root, source_name, error);
}

std::unique_ptr<NativeScoreModel> NativeScoreModel::load_parsed(
    const detail::JsonValue& root,
    const std::string& source_name,
    std::string& error) {
  const JsonValue* model_root = unwrap_model(root);
  const std::string type = json_model_type(*model_root);
  if (type.empty()) {
    error = source_name + ": native model type is missing";
    return nullptr;
  }
  auto result = std::make_unique<NativeScoreModel>();
  result->impl_->type = type;
  int feature_count = 0;
  integer_value(first_field(*model_root, {"n_features", "feature_dim", "input_dim"}), feature_count);
  if (feature_count > 0) result->impl_->feature_count = static_cast<std::size_t>(feature_count);

  if (type == "pipeline") {
    const JsonValue* steps = model_root->field("steps");
    if (!steps || steps->type != JsonValue::Type::Array || steps->array_value.empty()) {
      error = source_name + ": pipeline model steps are missing";
      return nullptr;
    }
    std::vector<double> pipeline_mean;
    std::vector<double> pipeline_scale;
    std::unique_ptr<NativeScoreModel> final_model;
    for (const JsonValue& step_value : steps->array_value) {
      const JsonValue* step = &step_value;
      if (step_value.type == JsonValue::Type::Array && step_value.array_value.size() == 2 &&
          step_value.array_value[1].type == JsonValue::Type::Object) {
        step = &step_value.array_value[1];
      } else if (step_value.type != JsonValue::Type::Object) {
        error = source_name + ": pipeline step is invalid";
        return nullptr;
      }
      std::string step_type = json_model_type(*step);
      if (step_type == "standard_scaler" || step_type == "scaler" || step_type == "standardscaler") {
        number_array(first_field(*step, {"mean", "mean_", "scaler_mean"}), pipeline_mean);
        number_array(first_field(*step, {"scale", "scale_", "scaler_scale"}), pipeline_scale);
        if (pipeline_mean.empty() || pipeline_scale.empty() || pipeline_mean.size() != pipeline_scale.size()) {
          error = source_name + ": pipeline scaler arrays are invalid";
          return nullptr;
        }
        continue;
      }
      if (step_type == "logreg" || step_type == "logistic_regression" || step_type == "logistic" || step_type == "linear") {
        final_model = NativeScoreModel::load_parsed(*step, source_name + ":pipeline_step", error);
        if (!final_model) return nullptr;
        continue;
      }
      error = source_name + ": unsupported pipeline step type: " + step_type;
      return nullptr;
    }
    if (!final_model || final_model->impl_->coefficients.empty()) {
      error = source_name + ": pipeline has no logistic final estimator";
      return nullptr;
    }
    result->impl_->type = "logreg";
    result->impl_->feature_count = final_model->impl_->feature_count;
    result->impl_->intercept = final_model->impl_->intercept;
    result->impl_->coefficients = std::move(final_model->impl_->coefficients);
    result->impl_->scaler_mean = std::move(pipeline_mean);
    result->impl_->scaler_scale = std::move(pipeline_scale);
    if (result->impl_->feature_count == 0) result->impl_->feature_count = result->impl_->coefficients.size();
    if ((!result->impl_->scaler_mean.empty() || !result->impl_->scaler_scale.empty()) &&
        (result->impl_->scaler_mean.size() != result->impl_->feature_count || result->impl_->scaler_scale.size() != result->impl_->feature_count)) {
      error = source_name + ": pipeline scaler dimension does not match logistic model";
      return nullptr;
    }
  } else if (type == "logreg" || type == "logistic_regression" || type == "logistic" || type == "linear") {
    result->impl_->coefficients = flatten_coefficients(first_field(*model_root, {"coef", "coefficients", "weights"}));
    if (result->impl_->coefficients.empty()) {
      error = source_name + ": logistic model coefficients are missing";
      return nullptr;
    }
    number_value(first_field(*model_root, {"intercept", "bias"}), result->impl_->intercept);
    if (const JsonValue* intercept = first_field(*model_root, {"intercept", "bias"}); intercept && intercept->type == JsonValue::Type::Array && intercept->array_size() != 0) {
      if (intercept->numeric_array) {
        result->impl_->intercept = intercept->number_array_value.front();
      } else {
        number_value(&intercept->array_value.front(), result->impl_->intercept);
      }
    }
    const JsonValue* scaler = model_root->field("scaler");
    if (scaler && scaler->type == JsonValue::Type::Object) {
      number_array(first_field(*scaler, {"mean", "mean_"}), result->impl_->scaler_mean);
      number_array(first_field(*scaler, {"scale", "scale_"}), result->impl_->scaler_scale);
    } else {
      number_array(first_field(*model_root, {"mean", "mean_", "scaler_mean"}), result->impl_->scaler_mean);
      number_array(first_field(*model_root, {"scale", "scale_", "scaler_scale"}), result->impl_->scaler_scale);
    }
    if (result->impl_->feature_count == 0) result->impl_->feature_count = result->impl_->coefficients.size();
    if (result->impl_->feature_count != result->impl_->coefficients.size()) {
      error = source_name + ": logistic coefficient dimension does not match feature count";
      return nullptr;
    }
    if ((!result->impl_->scaler_mean.empty() || !result->impl_->scaler_scale.empty()) &&
        (result->impl_->scaler_mean.size() != result->impl_->feature_count || result->impl_->scaler_scale.size() != result->impl_->feature_count)) {
      error = source_name + ": scaler dimension does not match feature count";
      return nullptr;
    }
  } else if (type == "hgb" || type == "hist_gradient_boosting" || type == "histgradientboostingclassifier" || type == "extra_trees" || type == "extratrees" || type == "random_forest" || type == "randomforest") {
    number_value(first_field(*model_root, {"baseline_prediction", "baseline", "intercept"}), result->impl_->baseline);
    bool probability_leaf = type == "extra_trees" || type == "extratrees" || type == "random_forest" || type == "randomforest";
    boolean_value(model_root->field("probability_leaf"), probability_leaf);
    result->impl_->probability_leaf = probability_leaf;
    result->impl_->average_tree_probabilities = probability_leaf;
    if (!parse_tree_array(model_root->field("trees"), result->impl_->trees, error) && !parse_flat_trees(*model_root, result->impl_->trees, error)) {
      error = source_name + ": tree model has no supported tree payload";
      return nullptr;
    }
    if (result->impl_->feature_count == 0) {
      std::size_t maximum_feature = 0;
      bool found_feature = false;
      for (const TreeModel& tree : result->impl_->trees) {
        for (const TreeNode& node : tree.nodes) {
          if (node.feature_index >= 0) {
            maximum_feature = std::max(maximum_feature, static_cast<std::size_t>(node.feature_index));
            found_feature = true;
          }
        }
      }
      result->impl_->feature_count = found_feature ? maximum_feature + 1 : 0;
    }
  } else if (type == "constant") {
    number_value(first_field(*model_root, {"probability", "value", "score"}), result->impl_->baseline);
    result->impl_->average_tree_probabilities = true;
  } else {
    error = source_name + ": unsupported native model type: " + type;
    return nullptr;
  }
  return result;
}

float NativeScoreModel::predict_probability(
    const std::vector<float>& features,
    std::string* error) const {
  if (error) error->clear();
  if (!impl_) {
    if (error) *error = "native model is not initialized";
    return 0.0f;
  }
  if (impl_->feature_count != 0 && features.size() != impl_->feature_count) {
    if (error) *error = "native model feature dimension mismatch";
    return 0.0f;
  }
  double probability = 0.0;
  if (impl_->type == "logreg" || impl_->type == "logistic_regression" || impl_->type == "logistic" || impl_->type == "linear") {
    double score = impl_->intercept;
    for (std::size_t index = 0; index < impl_->coefficients.size(); ++index) {
      double value = static_cast<double>(features[index]);
      if (!impl_->scaler_mean.empty()) {
        double scale = impl_->scaler_scale[index];
        if (std::fabs(scale) < 1.0e-12) scale = 1.0;
        value = (value - impl_->scaler_mean[index]) / scale;
      }
      score += impl_->coefficients[index] * value;
    }
    score = std::clamp(score, -50.0, 50.0);
    probability = 1.0 / (1.0 + std::exp(-score));
  } else if (impl_->type == "hgb" || impl_->type == "hist_gradient_boosting" || impl_->type == "histgradientboostingclassifier") {
    double score = impl_->baseline;
    for (const TreeModel& tree : impl_->trees) score += tree.predict(features);
    score = std::clamp(score, -50.0, 50.0);
    probability = 1.0 / (1.0 + std::exp(-score));
  } else if (impl_->type == "extra_trees" || impl_->type == "extratrees" || impl_->type == "random_forest" || impl_->type == "randomforest") {
    double sum = 0.0;
    for (const TreeModel& tree : impl_->trees) sum += tree.predict(features);
    probability = impl_->trees.empty() ? 0.0 : sum / static_cast<double>(impl_->trees.size());
  } else if (impl_->type == "constant") {
    probability = impl_->baseline;
  }
  if (!std::isfinite(probability)) {
    if (error) *error = "native model produced a non-finite probability";
    return 0.0f;
  }
  return static_cast<float>(std::clamp(probability, 0.0, 1.0));
}

std::size_t NativeScoreModel::feature_count() const { return impl_ ? impl_->feature_count : 0; }
const std::string& NativeScoreModel::model_type() const {
  static const std::string empty;
  return impl_ ? impl_->type : empty;
}

NativeStackModel::NativeStackModel() : impl_(std::make_unique<Impl>()) {}
NativeStackModel::NativeStackModel(NativeStackModel&&) noexcept = default;
NativeStackModel& NativeStackModel::operator=(NativeStackModel&&) noexcept = default;
NativeStackModel::~NativeStackModel() = default;

std::unique_ptr<NativeStackModel> NativeStackModel::load_file(const std::string& path, std::string& error) {
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    error = "native stack model file cannot be opened: " + path;
    return nullptr;
  }
  std::ostringstream buffer;
  buffer << input.rdbuf();
  return load_document(buffer.str(), path, error);
}

std::unique_ptr<NativeStackModel> NativeStackModel::load_document(const std::string& document, const std::string& source_name, std::string& error) {
  JsonValue root;
  JsonParser parser(document);
  if (!parser.parse(root, error)) {
    error = source_name + ": " + error;
    return nullptr;
  }
  const JsonValue* models_value = root.field("base_models");
  const JsonValue* meta_value = root.field("meta_model");
  if (!models_value || models_value->type != JsonValue::Type::Array || models_value->array_value.empty() || !meta_value) {
    error = source_name + ": stack model requires base_models and meta_model";
    return nullptr;
  }
  auto result = std::make_unique<NativeStackModel>();
  int feature_count = 0;
  integer_value(first_field(root, {"n_features", "feature_dim", "input_dim"}), feature_count);
  if (feature_count > 0) result->impl_->feature_count = static_cast<std::size_t>(feature_count);
  int probability_count = 6;
  integer_value(root.field("base_probability_feature_count"), probability_count);
  if (probability_count >= 0) result->impl_->base_probability_feature_count = static_cast<std::size_t>(probability_count);
  boolean_value(root.field("drop_base_prob_features"), result->impl_->drop_base_probability_features);
  double threshold = 0.5;
  number_value(first_field(root, {"threshold", "decision_threshold"}), threshold);
  result->impl_->decision_threshold = static_cast<float>(std::clamp(threshold, 0.0, 1.0));

  const std::filesystem::path base_directory = std::filesystem::u8path(source_name).parent_path();
  for (const JsonValue& base_value : models_value->array_value) {
    std::unique_ptr<NativeScoreModel> model;
    if (!parse_model_reference(base_value, base_directory, model, error)) {
      return nullptr;
    }
    result->impl_->base_models.push_back(std::move(model));
  }

  if (!parse_model_reference(*meta_value, base_directory, result->impl_->meta_model, error)) {
    return nullptr;
  }
  if (!result->impl_->meta_model) {
    error = source_name + ": stack meta model is empty";
    return nullptr;
  }
  if (result->impl_->base_models.empty()) {
    error = source_name + ": stack model has no base models";
    return nullptr;
  }
  const std::size_t base_feature_count = result->impl_->base_models.front()->feature_count();
  if (base_feature_count == 0) {
    error = source_name + ": stack base model has no feature dimension";
    return nullptr;
  }
  for (const auto& model : result->impl_->base_models) {
    if (!model || model->feature_count() != base_feature_count) {
      error = source_name + ": stack base model feature dimensions differ";
      return nullptr;
    }
  }
  const std::size_t expected_input_count = base_feature_count +
      (result->impl_->drop_base_probability_features
           ? result->impl_->base_probability_feature_count
           : 0);
  if (result->impl_->feature_count != 0 && result->impl_->feature_count != expected_input_count) {
    error = source_name + ": stack input feature dimension metadata is inconsistent";
    return nullptr;
  }
  result->impl_->feature_count = expected_input_count;
  return result;
}

float NativeStackModel::predict_probability(const std::vector<float>& features, std::string* error) const {
  if (error) error->clear();
  if (!impl_ || !impl_->meta_model || impl_->base_models.empty()) {
    if (error) *error = "native stack model is not initialized";
    return 0.0f;
  }
  if (impl_->feature_count != 0 && features.size() != impl_->feature_count) {
    if (error) *error = "native stack feature dimension mismatch";
    return 0.0f;
  }
  std::vector<float> scoring_features = features;
  if (impl_->drop_base_probability_features && impl_->base_probability_feature_count > scoring_features.size()) {
    if (error) *error = "native stack probability feature count exceeds input dimension";
    return 0.0f;
  }
  if (impl_->drop_base_probability_features && impl_->base_probability_feature_count > 0) {
    scoring_features.erase(scoring_features.begin(), scoring_features.begin() + static_cast<std::ptrdiff_t>(impl_->base_probability_feature_count));
  }
  std::vector<float> scores;
  scores.reserve(impl_->base_models.size());
  for (const auto& model : impl_->base_models) {
    scores.push_back(model->predict_probability(scoring_features, error));
    if (error && !error->empty()) return 0.0f;
  }
  std::vector<float> clipped = scores;
  for (float& score : clipped) score = std::clamp(score, 1.0e-6f, 1.0f - 1.0e-6f);
  std::vector<float> stack_features = clipped;
  if (!clipped.empty()) {
    double mean = 0.0;
    for (float score : clipped) mean += score;
    mean /= static_cast<double>(clipped.size());
    double variance = 0.0;
    for (float score : clipped) variance += (static_cast<double>(score) - mean) * (static_cast<double>(score) - mean);
    variance /= static_cast<double>(clipped.size());
    const auto minimum = *std::min_element(clipped.begin(), clipped.end());
    const auto maximum = *std::max_element(clipped.begin(), clipped.end());
    std::vector<float> sorted = clipped;
    std::sort(sorted.begin(), sorted.end());
    const double median = sorted.size() % 2 == 0 ? (sorted[sorted.size() / 2 - 1] + sorted[sorted.size() / 2]) * 0.5 : sorted[sorted.size() / 2];
    double logit_mean = 0.0;
    for (float score : clipped) logit_mean += std::log(static_cast<double>(score) / (1.0 - static_cast<double>(score)));
    logit_mean /= static_cast<double>(clipped.size());
    stack_features.push_back(static_cast<float>(mean));
    stack_features.push_back(static_cast<float>(std::sqrt(variance)));
    stack_features.push_back(minimum);
    stack_features.push_back(maximum);
    stack_features.push_back(maximum - minimum);
    stack_features.push_back(static_cast<float>(median));
    stack_features.push_back(static_cast<float>(logit_mean));
  }
  return impl_->meta_model->predict_probability(stack_features, error);
}

std::size_t NativeStackModel::feature_count() const { return impl_ ? impl_->feature_count : 0; }
std::size_t NativeStackModel::base_model_count() const { return impl_ ? impl_->base_models.size() : 0; }
float NativeStackModel::threshold() const { return impl_ ? impl_->decision_threshold : 0.5f; }

}  // namespace axon_loop151_native
