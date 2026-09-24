// Tokenizer-only CLI (no Vulkan), for tests on machines without a GPU runtime:
//   tok_cli <tokenizer.json> encode <in.txt> <out.i32>
//   tok_cli <tokenizer.json> decode <in.i32> <out.txt>
#include "../tokenizer.hpp"
#include <fstream>
#include <iterator>

static std::string slurp(const char *f) { std::ifstream i(f, std::ios::binary); return std::string(std::istreambuf_iterator<char>(i), {}); }

int main(int argc, char **argv) {
    if (argc != 5) { fprintf(stderr, "usage: tok_cli <tokenizer.json> encode|decode <in> <out>\n"); return 1; }
    Tokenizer t; t.load(slurp(argv[1])); std::string mode = argv[2]; std::ofstream o(argv[4], std::ios::binary);
    if (mode == "encode") { auto ids = t.encode(slurp(argv[3])); o.write((const char *)ids.data(), (std::streamsize)(ids.size() * 4)); }
    else { std::string raw = slurp(argv[3]); std::vector<int> ids(raw.size() / 4); memcpy(ids.data(), raw.data(), ids.size() * 4); std::string s = t.decode(ids); o.write(s.data(), (std::streamsize)s.size()); }
    return 0;
}
