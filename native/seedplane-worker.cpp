// seedplane-worker — a SeedPlane window worker on top of llama.cpp's optimized kernels (CPU, Vulkan, SYCL, CUDA, Metal).
//
// It loads a GGUF model on one device and serves "windows" over TCP. A window is a list of token ids together with their
// ORIGINAL positions in the full text; it is decoded from an empty KV cache, so windows are fully independent and can be
// spread across GPUs, CPU cores and other machines.
//
//   seedplane-worker -m model.gguf --port 54000 --ngl 99          # GPU (Vulkan/SYCL/CUDA/Metal build)
//   seedplane-worker -m model.gguf --port 54001 --ngl 0 -t 4      # CPU, 4 threads
//
// Protocol (little-endian, no pickle). Auth: client sends u32 len + key bytes (env SEEDPLANE_AUTHKEY), server replies u32 1.
// Request : u32 magic 'SPW1', u32 want (0 prefill, 1 nll, 2 close), u32 n_tok, u32 core_off, u32 score_from, i32 next_tok,
//           i32 tokens[n_tok], i32 pos[n_tok]
// Response: u32 magic, f64 nll_sum, u32 n_scored, i32 argmax_last, f32 compute_ms
#include "llama.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#ifdef _WIN32
#include <winsock2.h>
#include <ws2tcpip.h>
#pragma comment(lib, "ws2_32.lib")
typedef SOCKET sock_t;
#define CLOSESOCK closesocket
#else
#include <arpa/inet.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <unistd.h>
typedef int sock_t;
#define CLOSESOCK close
#define INVALID_SOCKET (-1)
#endif

static const uint32_t MAGIC = 0x31575053;  // "SPW1"

#pragma pack(push, 1)
struct Req { uint32_t magic, want, n_tok, core_off, score_from; int32_t next_tok; };
struct Resp { uint32_t magic; double nll; uint32_t n_scored; int32_t argmax_last; float compute_ms; };
#pragma pack(pop)

static bool read_all(sock_t s, void* buf, size_t n) {
    char* p = (char*)buf;
    while (n) { int r = recv(s, p, (int)std::min<size_t>(n, 1 << 20), 0); if (r <= 0) return false; p += r; n -= r; }
    return true;
}
static bool write_all(sock_t s, const void* buf, size_t n) {
    const char* p = (const char*)buf;
    while (n) { int r = send(s, p, (int)std::min<size_t>(n, 1 << 20), 0); if (r <= 0) return false; p += r; n -= r; }
    return true;
}

int main(int argc, char** argv) {
    std::string model_path; int port = 54000, ngl = 0, threads = 1, n_ctx = 4096, n_batch = 2048, n_ubatch = 512;
    std::string host = "0.0.0.0";
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto next = [&]() { return std::string(i + 1 < argc ? argv[++i] : ""); };
        if (a == "-m") model_path = next(); else if (a == "--port") port = std::stoi(next());
        else if (a == "--ngl") ngl = std::stoi(next()); else if (a == "-t") threads = std::stoi(next());
        else if (a == "-c") n_ctx = std::stoi(next()); else if (a == "-b") n_batch = std::stoi(next());
        else if (a == "-ub") n_ubatch = std::stoi(next()); else if (a == "--host") host = next();
    }
    const char* key_env = getenv("SEEDPLANE_AUTHKEY"); std::string key = key_env ? key_env : "seedplane-local-default";
    if (model_path.empty()) { fprintf(stderr, "usage: %s -m model.gguf [--port P] [--ngl N] [-t T] [-c CTX]\n", argv[0]); return 1; }

    llama_backend_init(); ggml_backend_load_all();
    llama_model_params mp = llama_model_default_params(); mp.n_gpu_layers = ngl;
    llama_model* model = llama_model_load_from_file(model_path.c_str(), mp);
    if (!model) { fprintf(stderr, "failed to load %s\n", model_path.c_str()); return 1; }
    const llama_vocab* vocab = llama_model_get_vocab(model); const int n_vocab = llama_vocab_n_tokens(vocab);
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = n_ctx; cp.n_batch = std::min(n_batch, n_ctx); cp.n_ubatch = std::min(n_ubatch, (int)cp.n_batch);
    cp.n_threads = threads; cp.n_threads_batch = threads; cp.no_perf = true;
    llama_context* ctx = llama_init_from_model(model, cp);
    if (!ctx) { fprintf(stderr, "failed to create context\n"); return 1; }
    llama_batch batch = llama_batch_init(n_ctx, 0, 1);

#ifdef _WIN32
    WSADATA w; WSAStartup(MAKEWORD(2, 2), &w);
#endif
    sock_t ls = socket(AF_INET, SOCK_STREAM, 0); int yes = 1;
    setsockopt(ls, SOL_SOCKET, SO_REUSEADDR, (const char*)&yes, sizeof(yes));
    sockaddr_in addr{}; addr.sin_family = AF_INET; addr.sin_port = htons((uint16_t)port); inet_pton(AF_INET, host.c_str(), &addr.sin_addr);
    if (bind(ls, (sockaddr*)&addr, sizeof(addr)) != 0 || listen(ls, 4) != 0) { fprintf(stderr, "cannot listen on %d\n", port); return 1; }
    printf("seedplane-worker ready on %s:%d (ngl=%d, threads=%d, ctx=%d)\n", host.c_str(), port, ngl, threads, n_ctx); fflush(stdout);

    std::vector<int32_t> tok, pos;
    for (;;) {
        sock_t cs = accept(ls, nullptr, nullptr); if (cs == INVALID_SOCKET) continue;
        setsockopt(cs, IPPROTO_TCP, TCP_NODELAY, (const char*)&yes, sizeof(yes));
        uint32_t klen = 0; std::string got;
        if (read_all(cs, &klen, 4) && klen < 4096) { got.resize(klen); read_all(cs, &got[0], klen); }
        uint32_t ok = (got == key) ? 1u : 0u; write_all(cs, &ok, 4);
        if (!ok) { CLOSESOCK(cs); continue; }
        for (;;) {
            Req q; if (!read_all(cs, &q, sizeof(q)) || q.magic != MAGIC || q.want == 2) break;
            if ((int)q.n_tok > n_ctx || q.n_tok == 0) { fprintf(stderr, "window of %u tokens > ctx %d\n", q.n_tok, n_ctx); break; }
            tok.resize(q.n_tok); pos.resize(q.n_tok);
            if (!read_all(cs, tok.data(), 4 * q.n_tok) || !read_all(cs, pos.data(), 4 * q.n_tok)) break;
            auto t0 = std::chrono::high_resolution_clock::now();
            llama_memory_clear(llama_get_memory(ctx), true);           // every window starts from an empty KV cache
            const uint32_t first_scored = std::max(q.core_off, q.score_from);
            Resp r{MAGIC, 0.0, 0, -1, 0.f}; bool fail = false;
            for (uint32_t b0 = 0; b0 < q.n_tok && !fail; b0 += cp.n_batch) {
                const uint32_t b1 = std::min<uint32_t>(q.n_tok, b0 + cp.n_batch); batch.n_tokens = 0;
                for (uint32_t i = b0; i < b1; i++) {
                    const int k = batch.n_tokens++;
                    batch.token[k] = tok[i]; batch.pos[k] = pos[i]; batch.n_seq_id[k] = 1; batch.seq_id[k][0] = 0;
                    batch.logits[k] = (q.want == 1) ? (i >= first_scored) : (i == q.n_tok - 1);
                }
                if (llama_decode(ctx, batch) != 0) { fprintf(stderr, "llama_decode failed\n"); fail = true; break; }
                for (uint32_t i = b0; i < b1; i++) {
                    if (!batch.logits[i - b0]) continue;
                    const float* lg = llama_get_logits_ith(ctx, (int32_t)(i - b0));
                    if (i == q.n_tok - 1) r.argmax_last = (int32_t)(std::max_element(lg, lg + n_vocab) - lg);
                    if (q.want == 1) {
                        const int32_t target = (i + 1 < q.n_tok) ? tok[i + 1] : q.next_tok; if (target < 0) continue;
                        double m = lg[0]; for (int v = 1; v < n_vocab; v++) m = std::max(m, (double)lg[v]);
                        double s = 0; for (int v = 0; v < n_vocab; v++) s += std::exp((double)lg[v] - m);
                        r.nll += (m + std::log(s)) - lg[target]; r.n_scored++;
                    }
                }
            }
            if (fail) break;
            r.compute_ms = std::chrono::duration<float, std::milli>(std::chrono::high_resolution_clock::now() - t0).count();
            if (!write_all(cs, &r, sizeof(r))) break;
        }
        CLOSESOCK(cs);
    }
}
