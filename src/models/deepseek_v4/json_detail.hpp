#pragma once

// Private loader dependency. No JSON/vendor types cross public model APIs.
#include <rapidjson/document.h>
#include <rapidjson/error/en.h>
#include <cmath>
#include <fstream>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

namespace llaisys::models::deepseek_v4::json_detail {
using Value = rapidjson::Value;
inline void require(bool ok, const std::string &message) { if (!ok) throw std::invalid_argument(message); }
inline std::string string(const Value &v) {
    require(v.IsString(), "JSON string required"); return {v.GetString(), v.GetStringLength()};
}
inline void unique(const Value &v, unsigned depth = 0) {
    require(depth < 32, "JSON nesting exceeds model metadata limit");
    if (v.IsObject()) {
        std::set<std::string> keys;
        for (auto i = v.MemberBegin(); i != v.MemberEnd(); ++i) {
            auto key = string(i->name); require(key.find('\0') == std::string::npos && keys.insert(key).second, "duplicate/NUL JSON key");
            unique(i->value, depth + 1);
        }
    } else if (v.IsArray()) for (const auto &item : v.GetArray()) unique(item, depth + 1);
}
inline rapidjson::Document parse(const std::string &raw) {
    rapidjson::Document d;
    d.Parse<rapidjson::kParseValidateEncodingFlag | rapidjson::kParseIterativeFlag>(raw.data(), raw.size());
    require(!d.HasParseError(), d.HasParseError() ? std::string("invalid JSON: ") + rapidjson::GetParseError_En(d.GetParseError()) : "");
    require(d.IsObject(), "JSON root must be an object"); unique(d); return d;
}
inline rapidjson::Document file(const std::string &path) {
    std::ifstream in(path, std::ios::binary | std::ios::ate);
    require(in.good() && in.tellg() > 0 && in.tellg() <= 2 * 1024 * 1024, "missing/oversized model JSON: " + path);
    std::string raw(static_cast<size_t>(in.tellg()), '\0'); in.seekg(0); in.read(raw.data(), raw.size());
    require(in.good(), "truncated model JSON"); return parse(raw);
}
inline const Value &get(const Value &v, const char *key) {
    require(v.IsObject() && v.HasMember(key), std::string("missing model field: ") + key); return v[key];
}
inline int64_t integer(const Value &v) { require(v.IsInt64(), "integer (not float/bool) required"); return v.GetInt64(); }
inline int64_t integer(const Value &v, const char *key) { return integer(get(v, key)); }
inline double number(const Value &v, const char *key) {
    const auto &n = get(v, key); require(n.IsNumber() && std::isfinite(n.GetDouble()), std::string("finite number required: ") + key); return n.GetDouble();
}
inline std::string string(const Value &v, const char *key) { return string(get(v, key)); }
inline std::vector<int64_t> integers(const Value &v) {
    require(v.IsArray(), "integer array required"); std::vector<int64_t> out;
    for (const auto &item : v.GetArray()) out.push_back(integer(item));
    return out;
}
inline std::vector<int64_t> integers(const Value &v, const char *key) { return integers(get(v, key)); }
inline void boolean(const Value &v, const char *key, bool expected) {
    const auto &b = get(v, key); require(b.IsBool() && b.GetBool() == expected, std::string("unsupported boolean model option: ") + key);
}
} // namespace llaisys::models::deepseek_v4::json_detail
