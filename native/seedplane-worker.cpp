// seedplane-worker — a SeedPlane window worker on top of llama.cpp's optimized kernels (CPU, Vulkan, SYCL, CUDA, Metal).
//
// It loads a GGUF model ONCE on one device, keeps it warm, and serves "windows" over TCP. A window is a list of token
// ids together with their ORIGINAL positions in the full text; it is decoded from an empty KV cache, so windows are
// fully independent and can be spread across GPUs, CPU cores and other machines.
//
//   seedplane-worker --list-devices                                   # JSON: every backend/device this build sees
//   seedplane-worker -m model.gguf --dev Vulkan0 --bench 512           # JSON: prompt tok/s of this device (autotune)
//   seedplane-worker -m model.gguf --dev Vulkan0 --port 54000          # GPU
//   seedplane-worker -m model.gguf --dev CPU --slots 4 -t 1 --port 54001
//       4 independent CPU slots (1 thread each) sharing ONE copy of the weights: 4 connections run in parallel.
//
// Protocol (little-endian, no pickle). Auth: client sends u32 len + key bytes (env SEEDPLANE_AUTHKEY), server replies u32 1.
// Request : u32 magic 'SPW1', u32 want (0 prefill, 1 nll, 2 close), u32 n_tok, u32 core_off, u32 score_from, i32 next_tok,
//           i32 tokens[n_tok], i32 pos[n_tok]
// Response: u32 magic, f64 nll_sum, u32 n_scored, i32 argmax_last, f32 compute_ms
// want 3 (span prefill) / 4 (span nll): the window is a long SPAN (halo prefix of core_off tokens + many cores). It is
// decoded in chunks of --span-chunk tokens; before each chunk the KV cache drops every position older than
// --span-keep tokens before the chunk, so the halo is REUSED from the previous chunk instead of recomputed.
#include "llama.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <random>
#include <string>
#include <thread>
#include <vector>
#ifdef _WIN32
#define NOMINMAX
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

struct Slot { llama_context* ctx; llama_batch batch; int n_batch; };
static int g_span_chunk = 512, g_span_keep = 256;

// Decode one window from an empty cache; fills r (nll over scored positions and/or argmax of the last token).
static bool run_window(Slot& sl, int n_vocab, const Req& q, const std::vector<int32_t>& tok, const std::vector<int32_t>& pos, Resp& r) {
    llama_memory_clear(llama_get_memory(sl.ctx), true);
    const uint32_t first_scored = std::max(q.core_off, q.score_from);
    const bool span = q.want >= 3; const uint32_t want_nll = (q.want == 1 || q.want == 4);
    for (uint32_t b0 = 0, b1 = 0; b0 < q.n_tok; b0 = b1) {
        b1 = std::min<uint32_t>(q.n_tok, b0 + (span ? (b0 == 0 ? q.core_off + g_span_chunk : g_span_chunk) : sl.n_batch));
        if (span && b1 - b0 > (uint32_t)sl.n_batch) { fprintf(stderr, "span chunk > n_batch\n"); return false; }
        if (span && b0 > 0) llama_memory_seq_rm(llama_get_memory(sl.ctx), 0, -1, pos[b0] - g_span_keep);  // slide the KV window
        llama_batch& batch = sl.batch; batch.n_tokens = 0;
        for (uint32_t i = b0; i < b1; i++) {
            const int k = batch.n_tokens++;
            batch.token[k] = tok[i]; batch.pos[k] = pos[i]; batch.n_seq_id[k] = 1; batch.seq_id[k][0] = 0;
            batch.logits[k] = want_nll ? (i >= first_scored) : (i == q.n_tok - 1);
        }
        if (llama_decode(sl.ctx, batch) != 0) { fprintf(stderr, "llama_decode failed\n"); return false; }
        for (uint32_t i = b0; i < b1; i++) {
            if (!batch.logits[i - b0]) continue;
            const float* lg = llama_get_logits_ith(sl.ctx, (int32_t)(i - b0));
            if (i == q.n_tok - 1) r.argmax_last = (int32_t)(std::max_element(lg, lg + n_vocab) - lg);
            if (want_nll) {
                const int32_t target = (i + 1 < q.n_tok) ? tok[i + 1] : q.next_tok; if (target < 0) continue;
                double m = lg[0]; for (int v = 1; v < n_vocab; v++) m = std::max(m, (double)lg[v]);
                double s = 0; for (int v = 0; v < n_vocab; v++) s += std::exp((double)lg[v] - m);
                r.nll += (m + std::log(s)) - lg[target]; r.n_scored++;
            }
        }
    }
    return true;
}

static void list_devices() {
    printf("[\n");
    for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
        ggml_backend_dev_t d = ggml_backend_dev_get(i); size_t fr = 0, tot = 0; ggml_backend_dev_memory(d, &fr, &tot);
        const char* ty[] = {"cpu", "gpu", "igpu", "accel"}; int t = (int)ggml_backend_dev_type(d);
        printf("  {\"name\": \"%s\", \"backend\": \"%s\", \"type\": \"%s\", \"description\": \"%s\", \"free_mb\": %zu, \"total_mb\": %zu}%s\n",
               ggml_backend_dev_name(d), ggml_backend_reg_name(ggml_backend_dev_backend_reg(d)), (t >= 0 && t < 4) ? ty[t] : "?",
               ggml_backend_dev_description(d), fr >> 20, tot >> 20, i + 1 < ggml_backend_dev_count() ? "," : "");
    }
    printf("]\n");
}

int main(int argc, char** argv) {
    std::string model_path, dev_name, host = "0.0.0.0";
    int port = 54000, ngl = -1, threads = 1, n_ctx = 4096, n_batch = 2048, n_ubatch = 512, slots = 1, bench = 0;
    bool list = false;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto next = [&]() { return std::string(i + 1 < argc ? argv[++i] : ""); };
        if (a == "-m") model_path = next(); else if (a == "--port") port = std::stoi(next());
        else if (a == "--ngl") ngl = std::stoi(next()); else if (a == "-t") threads = std::stoi(next());
        else if (a == "-c") n_ctx = std::stoi(next()); else if (a == "-b") n_batch = std::stoi(next());
        else if (a == "-ub") n_ubatch = std::stoi(next()); else if (a == "--host") host = next();
        else if (a == "--dev") dev_name = next(); else if (a == "--slots") slots = std::max(1, std::stoi(next()));
        else if (a == "--bench") bench = std::stoi(next()); else if (a == "--list-devices") list = true;
        else if (a == "--span-chunk") g_span_chunk = std::stoi(next()); else if (a == "--span-keep") g_span_keep = std::stoi(next());
    }
    llama_backend_init(); ggml_backend_load_all();
    if (list) { list_devices(); return 0; }
    const char* key_env = getenv("SEEDPLANE_AUTHKEY"); std::string key = key_env ? key_env : "seedplane-local-default";
    if (model_path.empty()) {
        fprintf(stderr, "usage: %s -m model.gguf [--dev NAME] [--slots N] [-t T] [--port P] [--bench N] | --list-devices\n", argv[0]);
        return 1;
    }

    // Device choice: --dev NAME pins the whole model on that device (CPU = no offload); default = llama.cpp's choice.
    llama_model_params mp = llama_model_default_params(); std::vector<ggml_backend_dev_t> devs;
    bool cpu_only = false;
    if (!dev_name.empty()) {
        for (size_t i = 0; i < ggml_backend_dev_count(); i++)
            if (dev_name == ggml_backend_dev_name(ggml_backend_dev_get(i))) devs.push_back(ggml_backend_dev_get(i));
        if (devs.empty()) { fprintf(stderr, "device %s not found (see --list-devices)\n", dev_name.c_str()); return 1; }
        cpu_only = ggml_backend_dev_type(devs[0]) == GGML_BACKEND_DEVICE_TYPE_CPU;
        if (cpu_only) devs.clear();
        devs.push_back(nullptr); mp.devices = devs.data(); mp.split_mode = LLAMA_SPLIT_MODE_NONE;
    }
    mp.n_gpu_layers = ngl >= 0 ? ngl : (cpu_only ? 0 : 999);
    llama_model* model = llama_model_load_from_file(model_path.c_str(), mp);
    if (!model) { fprintf(stderr, "failed to load %s\n", model_path.c_str()); return 1; }
    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));

    // N slots = N contexts on ONE shared model (weights loaded once, kept warm).
    std::vector<Slot> pool;
    for (int s = 0; s < slots; s++) {
        llama_context_params cp = llama_context_default_params();
        cp.n_ctx = n_ctx; cp.n_batch = std::min(n_batch, n_ctx); cp.n_ubatch = std::min(n_ubatch, (int)cp.n_batch);
        cp.n_threads = threads; cp.n_threads_batch = threads; cp.no_perf = true; cp.n_seq_max = 1;
        llama_context* ctx = llama_init_from_model(model, cp);
        if (!ctx) { fprintf(stderr, "failed to create context %d\n", s); return 1; }
        pool.push_back({ctx, llama_batch_init(cp.n_batch, 0, 1), (int)cp.n_batch});
    }

    if (bench > 0) {   // self-benchmark for the autotuner: every slot prefills `bench` tokens AT THE SAME TIME
        std::mt19937 rng(1); std::vector<int32_t> tok(bench), pos(bench);   // (aggregate tok/s, contention included)
        for (int i = 0; i < bench; i++) { tok[i] = (int32_t)(rng() % (uint32_t)std::min(n_vocab, 32000)); pos[i] = i; }
        Req q{MAGIC, 0, (uint32_t)bench, 0, 0, -1}; std::vector<double> ts; bool ok = true;
        for (int rep = 0; rep < 4; rep++) {
            auto t0 = std::chrono::high_resolution_clock::now(); std::vector<std::thread> th;
            for (int s = 0; s < slots; s++) th.emplace_back([&, s] { Resp r{MAGIC, 0.0, 0, -1, 0.f}; if (!run_window(pool[s], n_vocab, q, tok, pos, r)) ok = false; });
            for (auto& t : th) t.join();
            if (!ok) return 1;
            if (rep) ts.push_back(std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - t0).count());
        }
        std::sort(ts.begin(), ts.end());
        printf("{\"dev\": \"%s\", \"slots\": %d, \"threads\": %d, \"tokens\": %d, \"tok_s\": %.1f}\n",
               dev_name.c_str(), slots, threads, bench, (double)bench * slots / ts[1]);
        return 0;
    }

#ifdef _WIN32
    WSADATA w; WSAStartup(MAKEWORD(2, 2), &w);
#endif
    sock_t ls = socket(AF_INET, SOCK_STREAM, 0); int yes = 1;
    setsockopt(ls, SOL_SOCKET, SO_REUSEADDR, (const char*)&yes, sizeof(yes));
    sockaddr_in addr{}; addr.sin_family = AF_INET; addr.sin_port = htons((uint16_t)port); inet_pton(AF_INET, host.c_str(), &addr.sin_addr);
    if (bind(ls, (sockaddr*)&addr, sizeof(addr)) != 0 || listen(ls, 16) != 0) { fprintf(stderr, "cannot listen on %d\n", port); return 1; }
    printf("seedplane-worker ready on %s:%d (dev=%s, slots=%d, threads/slot=%d, ctx=%d)\n",
           host.c_str(), port, dev_name.empty() ? "auto" : dev_name.c_str(), slots, threads, n_ctx); fflush(stdout);

    std::mutex mu; std::condition_variable cv; std::vector<int> free_slots;
    for (int s = slots - 1; s >= 0; s--) free_slots.push_back(s);
    for (;;) {
        sock_t cs = accept(ls, nullptr, nullptr); if (cs == INVALID_SOCKET) continue;
        int s; { std::unique_lock<std::mutex> lk(mu); cv.wait(lk, [&] { return !free_slots.empty(); }); s = free_slots.back(); free_slots.pop_back(); }
        std::thread([&, cs, s]() {
            setsockopt(cs, IPPROTO_TCP, TCP_NODELAY, (const char*)&yes, sizeof(yes));
            uint32_t klen = 0; std::string got;
            if (read_all(cs, &klen, 4) && klen < 4096) { got.resize(klen); read_all(cs, &got[0], klen); }
            uint32_t ok = (got == key) ? 1u : 0u; write_all(cs, &ok, 4);
            std::vector<int32_t> tok, pos;
            while (ok) {
                Req q; if (!read_all(cs, &q, sizeof(q)) || q.magic != MAGIC || q.want == 2) break;
                if ((q.want < 3 && (int)q.n_tok > n_ctx) || q.n_tok == 0 || q.n_tok > (1u << 24)) { fprintf(stderr, "window of %u tokens > ctx %d\n", q.n_tok, n_ctx); break; }
                tok.resize(q.n_tok); pos.resize(q.n_tok);
                if (!read_all(cs, tok.data(), 4 * q.n_tok) || !read_all(cs, pos.data(), 4 * q.n_tok)) break;
                auto t0 = std::chrono::high_resolution_clock::now(); Resp r{MAGIC, 0.0, 0, -1, 0.f};
                if (!run_window(pool[s], n_vocab, q, tok, pos, r)) break;
                r.compute_ms = std::chrono::duration<float, std::milli>(std::chrono::high_resolution_clock::now() - t0).count();
                if (!write_all(cs, &r, sizeof(r))) break;
            }
            CLOSESOCK(cs);
            { std::lock_guard<std::mutex> lk(mu); free_slots.push_back(s); } cv.notify_one();
        }).detach();
    }
}
