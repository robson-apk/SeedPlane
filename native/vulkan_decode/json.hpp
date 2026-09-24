// Minimal JSON reader/writer for SeedPlane native tools (bundle manifests, tokenizer.json, the --serve protocol).
#pragma once
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

struct Json {
    enum Type { Null, Num, Str, Arr, Obj, Bool } type = Null;
    double num = 0; std::string str; std::vector<Json> arr; std::vector<std::pair<std::string, Json>> obj;
    const Json *get(const std::string &k) const {
        for (auto &kv : obj) if (kv.first == k) return &kv.second;
        return nullptr;
    }
    double num_or(const std::string &k, double d) const { const Json *j = get(k); return j && (j->type == Num || j->type == Bool) ? j->num : d; }
    std::string str_or(const std::string &k, const std::string &d) const { const Json *j = get(k); return j && j->type == Str ? j->str : d; }
};

inline void utf8_append(std::string &s, uint32_t cp) {
    if (cp < 0x80) s += (char)cp;
    else if (cp < 0x800) { s += (char)(0xC0 | (cp >> 6)); s += (char)(0x80 | (cp & 0x3F)); }
    else if (cp < 0x10000) { s += (char)(0xE0 | (cp >> 12)); s += (char)(0x80 | ((cp >> 6) & 0x3F)); s += (char)(0x80 | (cp & 0x3F)); }
    else { s += (char)(0xF0 | (cp >> 18)); s += (char)(0x80 | ((cp >> 12) & 0x3F)); s += (char)(0x80 | ((cp >> 6) & 0x3F)); s += (char)(0x80 | (cp & 0x3F)); }
}

struct JsonParser {
    const char *p, *end;
    void ws() { while (p < end && (*p == ' ' || *p == '\n' || *p == '\r' || *p == '\t')) ++p; }
    uint32_t hex4() { uint32_t v = 0; for (int i = 0; i < 4 && p < end; ++i, ++p) { char c = *p; v = v * 16 + (c <= '9' ? c - '0' : (c | 32) - 'a' + 10); } return v; }
    std::string string() {
        std::string s; ++p;
        while (p < end && *p != '"') {
            if (*p != '\\') { s += *p++; continue; }
            ++p; char e = *p++;
            switch (e) {
                case 'n': s += '\n'; break; case 't': s += '\t'; break; case 'r': s += '\r'; break;
                case 'b': s += '\b'; break; case 'f': s += '\f'; break;
                case 'u': {
                    uint32_t cp = hex4();
                    if (cp >= 0xD800 && cp < 0xDC00 && p + 1 < end && p[0] == '\\' && p[1] == 'u') {
                        p += 2; uint32_t lo = hex4(); cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
                    }
                    utf8_append(s, cp); break;
                }
                default: s += e;
            }
        }
        ++p; return s;
    }
    Json value() {
        ws(); Json j;
        if (p >= end) return j;
        if (*p == '{') {
            j.type = Json::Obj; ++p; ws();
            if (*p == '}') { ++p; return j; }
            for (;;) { ws(); std::string k = string(); ws(); ++p; Json v = value(); j.obj.emplace_back(std::move(k), std::move(v)); ws();
                       if (*p == ',') { ++p; continue; } ++p; break; }
        } else if (*p == '[') {
            j.type = Json::Arr; ++p; ws();
            if (*p == ']') { ++p; return j; }
            for (;;) { j.arr.push_back(value()); ws(); if (*p == ',') { ++p; continue; } ++p; break; }
        } else if (*p == '"') { j.type = Json::Str; j.str = string(); }
        else if (!strncmp(p, "true", 4)) { j.type = Json::Bool; j.num = 1; p += 4; }
        else if (!strncmp(p, "false", 5)) { j.type = Json::Bool; p += 5; }
        else if (!strncmp(p, "null", 4)) { p += 4; }
        else { j.type = Json::Num; char *e; j.num = strtod(p, &e); p = e; }
        return j;
    }
};

inline Json parse_json(const std::string &s) { JsonParser jp{s.data(), s.data() + s.size()}; return jp.value(); }

// JSON string literal (UTF-8 passes through; control characters, quotes and backslashes are escaped).
inline std::string json_quote(const std::string &s) {
    std::string o = "\"";
    for (unsigned char c : s) {
        if (c == '"') o += "\\\""; else if (c == '\\') o += "\\\\"; else if (c == '\n') o += "\\n"; else if (c == '\r') o += "\\r";
        else if (c == '\t') o += "\\t"; else if (c < 0x20) { char b[8]; snprintf(b, sizeof b, "\\u%04x", c); o += b; }
        else o += (char)c;
    }
    return o + "\"";
}
