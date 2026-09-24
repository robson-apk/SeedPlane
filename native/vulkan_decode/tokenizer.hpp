// Native Qwen2 tokenizer: byte-level BPE read from the bundle's tokenizer.json (HF `tokenizers` format).
//
// Pipeline, mirroring tokenizers' configuration for Qwen2:
//   1. special (added) tokens are matched literally in the raw text (leftmost, longest);
//   2. every other segment is NFC-normalised;
//   3. pre-tokenised with the Qwen2 split regex (hand-written matcher below, one branch per alternative):
//        (?i:'s|'t|'re|'ve|'m|'ll|'d) | [^\r\n\p{L}\p{N}]?\p{L}+ | \p{N} | ?[^\s\p{L}\p{N}]+[\r\n]* |
//        \s*[\r\n]+ | \s+(?!\S) | \s+
//   4. each piece's UTF-8 bytes are mapped to GPT-2 byte symbols and merged by BPE rank.
// Decoding maps symbols back to bytes; StreamDecoder only emits complete UTF-8 sequences.
#pragma once
#include "json.hpp"
#include <algorithm>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace uc {
#include "unicode_tables.inc"

template <size_t N> inline bool in_ranges(const uint32_t (&r)[N][2], uint32_t cp) {
    size_t lo = 0, hi = N;
    while (lo < hi) { size_t m = (lo + hi) / 2; if (r[m][1] < cp) lo = m + 1; else hi = m; }
    return lo < N && r[lo][0] <= cp && cp <= r[lo][1];
}
inline bool is_letter(uint32_t cp) { return cp < 0x80 ? ((cp | 32) >= 'a' && (cp | 32) <= 'z') : in_ranges(UC_LETTER, cp); }
inline bool is_number(uint32_t cp) { return cp < 0x80 ? (cp >= '0' && cp <= '9') : in_ranges(UC_NUMBER, cp); }
inline bool is_space(uint32_t cp) { for (uint32_t w : UC_WHITE) if (w == cp) return true; return false; }
inline uint32_t ccc(uint32_t cp) {
    if (cp < 0x300) return 0;
    size_t lo = 0, hi = sizeof(UC_CCC) / sizeof(UC_CCC[0]);
    while (lo < hi) { size_t m = (lo + hi) / 2; if (UC_CCC[m][0] < cp) lo = m + 1; else hi = m; }
    return lo < sizeof(UC_CCC) / sizeof(UC_CCC[0]) && UC_CCC[lo][0] == cp ? UC_CCC[lo][1] : 0;
}
inline const uint32_t *decomp(uint32_t cp) {
    size_t n = sizeof(UC_DECOMP) / sizeof(UC_DECOMP[0]), lo = 0, hi = n;
    while (lo < hi) { size_t m = (lo + hi) / 2; if (UC_DECOMP[m][0] < cp) lo = m + 1; else hi = m; }
    return lo < n && UC_DECOMP[lo][0] == cp ? UC_DECOMP[lo] + 1 : nullptr;
}
constexpr uint32_t SB = 0xAC00, LB = 0x1100, VB = 0x1161, TB = 0x11A7, LC = 19, VC = 21, TC = 28, NC = VC * TC, SC = LC * NC;
inline uint32_t compose(uint32_t a, uint32_t b) {
    if (a >= LB && a < LB + LC && b >= VB && b < VB + VC) return SB + ((a - LB) * VC + (b - VB)) * TC;
    if (a >= SB && a < SB + SC && (a - SB) % TC == 0 && b > TB && b < TB + TC) return a + (b - TB);
    static const std::unordered_map<uint64_t, uint32_t> table = [] {
        std::unordered_map<uint64_t, uint32_t> t;
        for (auto &e : UC_COMP) t[(uint64_t)e[0] << 32 | e[1]] = e[2];
        return t; }();
    auto it = table.find((uint64_t)a << 32 | b); return it == table.end() ? 0 : it->second;
}
// Canonical composition (NFC) of a code point sequence.
inline std::vector<uint32_t> nfc(const std::vector<uint32_t> &in) {
    bool simple = true; for (uint32_t cp : in) if (cp >= 0x300) { simple = false; break; }
    if (simple) return in;                                   // nothing below U+0300 changes under NFC
    std::vector<uint32_t> d; d.reserve(in.size());
    for (uint32_t cp : in) {
        if (cp >= SB && cp < SB + SC) {
            uint32_t s = cp - SB; d.push_back(LB + s / NC); d.push_back(VB + (s % NC) / TC);
            if (s % TC) d.push_back(TB + s % TC);
        } else if (const uint32_t *x = decomp(cp)) { for (int i = 0; i < 4 && x[i]; ++i) d.push_back(x[i]); }
        else d.push_back(cp);
    }
    for (size_t i = 1; i < d.size(); ++i) {                   // canonical ordering (stable by combining class)
        uint32_t c = ccc(d[i]); if (!c) continue;
        for (size_t j = i; j > 0 && ccc(d[j - 1]) > c; --j) std::swap(d[j], d[j - 1]);
    }
    std::vector<uint32_t> out; out.reserve(d.size()); long starter = -1; int last = -1;
    for (uint32_t cp : d) {
        int c = (int)ccc(cp);
        if (starter >= 0 && (last == -1 || (last != 0 && last < c))) {
            if (uint32_t k = compose(out[(size_t)starter], cp)) { out[(size_t)starter] = k; continue; }
        }
        if (c == 0) { starter = (long)out.size(); last = -1; } else last = c;
        out.push_back(cp);
    }
    return out;
}
}  // namespace uc

inline std::vector<uint32_t> utf8_decode(const std::string &s) {
    std::vector<uint32_t> out; out.reserve(s.size());
    for (size_t i = 0; i < s.size();) {
        unsigned char c = (unsigned char)s[i]; uint32_t cp; int n;
        if (c < 0x80) { cp = c; n = 1; } else if ((c >> 5) == 6) { cp = c & 0x1F; n = 2; }
        else if ((c >> 4) == 14) { cp = c & 0x0F; n = 3; } else if ((c >> 3) == 30) { cp = c & 0x07; n = 4; } else { out.push_back(0xFFFD); ++i; continue; }
        if (i + n > s.size()) { out.push_back(0xFFFD); ++i; continue; }
        bool ok = true; for (int k = 1; k < n; ++k) { unsigned char d = (unsigned char)s[i + k]; if ((d >> 6) != 2) { ok = false; break; } cp = (cp << 6) | (d & 0x3F); }
        if (!ok) { out.push_back(0xFFFD); ++i; continue; }
        out.push_back(cp); i += n;
    }
    return out;
}

class Tokenizer {
public:
    void load(const std::string &json_text) {
        Json t = parse_json(json_text);
        const Json *model = t.get("model");
        if (!model || model->str_or("type", "") != "BPE") throw std::runtime_error("tokenizer.json: only BPE models are supported");
        for (auto &kv : model->get("vocab")->obj) set_token((int)kv.second.num, kv.first);
        for (auto &a : t.get("added_tokens")->arr) {
            int id = (int)a.get("id")->num; std::string c = a.get("content")->str;
            set_token(id, c); added_.push_back({c, id}); special_[id] = a.num_or("special", 0) != 0;
        }
        for (auto &kv : model->get("vocab")->obj) vocab_[kv.first] = (int)kv.second.num;
        int rank = 0;
        for (auto &m : model->get("merges")->arr) {
            std::string a, b;
            if (m.type == Json::Arr) { a = m.arr[0].str; b = m.arr[1].str; }
            else { size_t sp = m.str.find(' '); a = m.str.substr(0, sp); b = m.str.substr(sp + 1); }
            auto ia = vocab_.find(a), ib = vocab_.find(b), iab = vocab_.find(a + b);
            if (ia != vocab_.end() && ib != vocab_.end() && iab != vocab_.end())
                merges_[(uint64_t)(uint32_t)ia->second << 32 | (uint32_t)ib->second] = {rank, iab->second};
            ++rank;
        }
        // GPT-2 byte <-> printable symbol mapping.
        std::vector<int> bs; for (int b = '!'; b <= '~'; ++b) bs.push_back(b);
        for (int b = 0xA1; b <= 0xAC; ++b) bs.push_back(b); for (int b = 0xAE; b <= 0xFF; ++b) bs.push_back(b);
        std::vector<int> cs = bs; int n = 0;
        for (int b = 0; b < 256; ++b) if (std::find(bs.begin(), bs.end(), b) == bs.end()) { bs.push_back(b); cs.push_back(256 + n++); }
        for (size_t i = 0; i < bs.size(); ++i) {
            std::string sym; utf8_append(sym, (uint32_t)cs[i]); byte_sym_[bs[i]] = sym; sym_byte_[(uint32_t)cs[i]] = (unsigned char)bs[i];
            auto it = vocab_.find(sym); if (it == vocab_.end()) throw std::runtime_error("tokenizer.json: byte symbol missing from vocab");
            byte_id_[bs[i]] = it->second;
        }
        std::sort(added_.begin(), added_.end(), [](auto &x, auto &y) { return x.first.size() > y.first.size(); });  // longest first
    }
    size_t vocab_size() const { return id2tok_.size(); }
    bool is_special(int id) const { auto it = special_.find(id); return it != special_.end() && it->second; }
    int token_id(const std::string &s) const {
        for (auto &a : added_) if (a.first == s) return a.second;
        auto it = vocab_.find(s); return it == vocab_.end() ? -1 : it->second;
    }

    std::vector<int> encode(const std::string &text, bool allow_special = true) const {
        std::vector<int> ids; size_t seg = 0, i = 0;
        while (i < text.size()) {
            int hit = -1; size_t len = 0;
            if (allow_special && text[i] == '<')
                for (auto &a : added_) if (text.compare(i, a.first.size(), a.first) == 0) { hit = a.second; len = a.first.size(); break; }
            if (hit < 0) { ++i; continue; }
            encode_ordinary(text.substr(seg, i - seg), ids); ids.push_back(hit); i += len; seg = i;
        }
        encode_ordinary(text.substr(seg), ids);
        return ids;
    }
    // Raw bytes of one token (special tokens as their literal text unless skip_special).
    std::string token_bytes(int id, bool skip_special = false) const {
        if (id < 0 || (size_t)id >= id2tok_.size()) return "";
        if (special_.count(id)) return skip_special && is_special(id) ? "" : id2tok_[(size_t)id];
        std::string out;
        for (uint32_t cp : utf8_decode(id2tok_[(size_t)id])) { auto it = sym_byte_.find(cp); if (it != sym_byte_.end()) out += (char)it->second; }
        return out;
    }
    std::string decode(const std::vector<int> &ids, bool skip_special = false) const {
        std::string s; for (int id : ids) s += token_bytes(id, skip_special); return s;
    }
    std::string nfc(const std::string &s) const { std::string o; for (uint32_t cp : uc::nfc(utf8_decode(s))) utf8_append(o, cp); return o; }
    // Pre-tokenizer pieces as [begin, end) code point ranges (exposed for tests).
    static std::vector<std::pair<size_t, size_t>> split(const std::vector<uint32_t> &s) {
        using namespace uc;
        std::vector<std::pair<size_t, size_t>> out; const size_t n = s.size(); size_t p = 0;
        auto nl = [](uint32_t c) { return c == '\r' || c == '\n'; };
        auto lower = [](uint32_t c) { return c < 0x80 ? (c | 32) : c; };
        while (p < n) {
            size_t e = 0; uint32_t c = s[p];
            // 1. contractions, case-insensitive ('S also matches U+017F LATIN SMALL LETTER LONG S)
            if (c == '\'' && p + 1 < n) {
                uint32_t a = lower(s[p + 1]), b = p + 2 < n ? lower(s[p + 2]) : 0;
                if (a == 's' || a == 0x17F || a == 't' || a == 'm' || a == 'd') e = p + 2;
                else if ((a == 'r' && b == 'e') || (a == 'v' && b == 'e') || (a == 'l' && b == 'l')) e = p + 3;
            }
            // 2. [^\r\n\p{L}\p{N}]?\p{L}+
            if (!e) {
                size_t q = p;
                if (!nl(c) && !is_letter(c) && !is_number(c) && p + 1 < n && is_letter(s[p + 1])) q = p + 1;
                if (is_letter(s[q])) { while (q < n && is_letter(s[q])) ++q; e = q; }
            }
            // 3. \p{N}
            if (!e && is_number(c)) e = p + 1;
            // 4.  ?[^\s\p{L}\p{N}]+[\r\n]*
            if (!e) {
                auto other = [&](uint32_t x) { return !is_space(x) && !is_letter(x) && !is_number(x); };
                size_t q = p;
                if (c == ' ' && p + 1 < n && other(s[p + 1])) q = p + 1;
                if (q < n && other(s[q])) { while (q < n && other(s[q])) ++q; while (q < n && nl(s[q])) ++q; e = q; }
            }
            if (!e && is_space(c)) {
                size_t q = p; while (q < n && is_space(s[q])) ++q;          // maximal whitespace run [p, q)
                size_t last_nl = n;
                for (size_t k = p; k < q; ++k) if (nl(s[k])) last_nl = k;
                if (last_nl != n) e = last_nl + 1;                           // 5. \s*[\r\n]+
                else if (q == n) e = q;                                      // 6. \s+(?!\S) at end of text
                else if (q - p >= 2) e = q - 1;                              // 6. leave one space for the next word
                else e = q;                                                  // 7. \s+
            }
            if (!e) e = p + 1;                                               // unreachable for valid input
            out.push_back({p, e}); p = e;
        }
        return out;
    }

private:
    void set_token(int id, const std::string &s) { if ((size_t)id >= id2tok_.size()) id2tok_.resize((size_t)id + 1); id2tok_[(size_t)id] = s; }
    void encode_ordinary(const std::string &text, std::vector<int> &ids) const {
        if (text.empty()) return;
        std::vector<uint32_t> cps = uc::nfc(utf8_decode(text));
        for (auto &pc : split(cps)) {
            std::string bytes; for (size_t k = pc.first; k < pc.second; ++k) utf8_append(bytes, cps[k]);
            bpe(bytes, ids);
        }
    }
    void bpe(const std::string &bytes, std::vector<int> &ids) const {
        if (bytes.size() <= 64) { auto it = cache_.find(bytes); if (it != cache_.end()) { ids.insert(ids.end(), it->second.begin(), it->second.end()); return; } }
        std::vector<int> w; w.reserve(bytes.size());
        for (unsigned char b : bytes) w.push_back(byte_id_[b]);
        while (w.size() > 1) {                                       // lowest rank first, leftmost on ties (== HF merge queue)
            int best = -1, best_rank = 1 << 30, best_id = 0;
            for (size_t k = 0; k + 1 < w.size(); ++k) {
                auto it = merges_.find((uint64_t)(uint32_t)w[k] << 32 | (uint32_t)w[k + 1]);
                if (it != merges_.end() && it->second.first < best_rank) { best_rank = it->second.first; best = (int)k; best_id = it->second.second; }
            }
            if (best < 0) break;
            w[(size_t)best] = best_id; w.erase(w.begin() + best + 1);
        }
        if (bytes.size() <= 64) cache_[bytes] = w;
        ids.insert(ids.end(), w.begin(), w.end());
    }
    std::vector<std::string> id2tok_;
    std::unordered_map<std::string, int> vocab_;
    std::unordered_map<uint64_t, std::pair<int, int>> merges_;
    std::vector<std::pair<std::string, int>> added_;
    std::unordered_map<int, bool> special_;
    std::string byte_sym_[256]; int byte_id_[256]{};
    std::unordered_map<uint32_t, unsigned char> sym_byte_;
    mutable std::unordered_map<std::string, std::vector<int>> cache_;
};

// Streaming detokenizer: emits only complete UTF-8 sequences so multi-byte characters split across tokens print once.
class StreamDecoder {
public:
    explicit StreamDecoder(const Tokenizer &t, bool skip_special = true) : t_(t), skip_(skip_special) {}
    std::string push(int id) {
        buf_ += t_.token_bytes(id, skip_);
        size_t cut = buf_.size(), k = buf_.size();
        // find the start of a trailing incomplete sequence, if any
        int back = 0; while (k > 0 && back < 4 && ((unsigned char)buf_[k - 1] >> 6) == 2) { --k; ++back; }
        if (k > 0) {
            unsigned char lead = (unsigned char)buf_[k - 1]; int need = lead >= 0xF0 ? 4 : lead >= 0xE0 ? 3 : lead >= 0xC0 ? 2 : 1;
            if (need > 1 && back + 1 < need) cut = k - 1;
        }
        std::string out = buf_.substr(0, cut); buf_.erase(0, cut); return out;
    }
    std::string flush() { std::string o; o.swap(buf_); return o; }
private:
    const Tokenizer &t_; bool skip_; std::string buf_;
};
