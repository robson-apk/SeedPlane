// SeedPlane native runtime for seedplane-bundle/2 (Qwen2 family) on Vulkan compute.
//
// Decodes the SeedPlane model: the token at position t of shard k = t / S attends to
// [sinks] + [halo of shard k-1] + [shard k up to t], with original position ids (ShardPlan.windows).
// KV caches are window-local (sinks + halo + shard slots). The default boundary mode (shadow-batch, V20) builds the
// next shard's window while the halo is decoded, in K/V-only batches of 8, so no boundary stalls the decode.
//
// The one-token graph (embedding, all layers, final norm, LM head, greedy argmax) is recorded once into a command
// buffer; each token is one vkQueueSubmit. Weights FP16, activations and KV cache FP32. Text goes through the native
// byte-level BPE tokenizer (tokenizer.hpp); sampling follows Qwen2Engine.sample (sampler.hpp).
//
// usage:
//   qwen_vk <bundle.sp> --chat [--system text] [sampling]              interactive chat in the terminal
//   qwen_vk <bundle.sp> --serve [sampling]                             JSON lines on stdin/stdout (seedplane chat --native)
//   qwen_vk <bundle.sp> [--prompt ids | --prompt-file f.i32] [-n N] [--runs R] [sampling]      benchmark
//   qwen_vk <bundle.sp> --score-file f.i32 [--nll-out f.f32]                                    teacher-forced NLL
//   qwen_vk <bundle.sp> --tokenize-file in.txt out.i32 | --nfc-file in.txt out.txt | --detok-file in.i32 out.txt
//   qwen_vk --sample-test logits.f32 T top_k top_p draws seed out.u32                           sampler self-test
//   qwen_vk <bundle.sp> --sample-test-gpu logits.f32 T top_k top_p draws seed out.u32        GPU-assisted sampler self-test
// options: [--mode shadow-batch|shadow|rebuild-batch|rebuild|reuse] [--attn split|old] [--shard S --halo H --sinks K |
//          --full] [--ctx N] [--dump-at p1,p2 --dump-dir d] [--spin] [--shaders dir]
// sampling: [--temperature T] [--top-k K] [--top-p P] [--seed S] [-n max_new_tokens]
//           sampling runs on the GPU (Gumbel-max / candidate kernels); --host-sampling uses the V21 CPU path.
//           analysis: [--copy-logits] (pay the logits copy in greedy mode)
#include <vulkan/vulkan.h>
#include "json.hpp"
#include "sampler.hpp"
#include "tokenizer.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <functional>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#ifdef _WIN32
#define NOMINMAX
#include <fcntl.h>
#include <io.h>
#include <windows.h>
#endif

namespace fs = std::filesystem;

#define VK(x) do { VkResult r_ = (x); if (r_ != VK_SUCCESS) { \
    fprintf(stderr, "%s failed: %d (line %d)\n", #x, r_, __LINE__); std::exit(2); } } while (0)

static std::string read_text(const fs::path &f) {
    std::ifstream in(f, std::ios::binary); if (!in) throw std::runtime_error("cannot open " + f.string());
    return std::string(std::istreambuf_iterator<char>(in), {});
}
static void write_bytes(const fs::path &f, const void *p, size_t n) { std::ofstream o(f, std::ios::binary); o.write((const char *)p, (std::streamsize)n); }

// ---------------------------------------------------------------- seedplane-bundle/2
// Written by `python -m seedplane.native_bundle`: seedplane.json (arch, plan, tensor table) + weights.spw.
struct Bundle {
    Json meta; std::vector<uint8_t> blob; fs::path dir;
    void load_meta(const fs::path &d) {
        dir = d; meta = parse_json(read_text(dir / "seedplane.json"));
        const Json *fmt = meta.get("format");
        if (!fmt || fmt->str != "seedplane-bundle/2")
            throw std::runtime_error("not a seedplane-bundle/2 (convert it with: seedplane convert <src> <out> --native)");
    }
    void load_weights() {
        std::ifstream in(dir / meta.get("weights")->get("file")->str, std::ios::binary | std::ios::ate);
        if (!in) throw std::runtime_error("cannot open weights file");
        blob.resize((size_t)in.tellg()); in.seekg(0); in.read((char *)blob.data(), (std::streamsize)blob.size());
    }
    const Json &arch() const { return *meta.get("arch"); }
    uint32_t a(const char *k) const { return (uint32_t)arch().get(k)->num; }
    bool has(const std::string &n) const { return meta.get("tensors")->get(n) != nullptr; }
    std::pair<const uint8_t *, size_t> tensor(const std::string &n) const {
        const Json *t = meta.get("tensors")->get(n); if (!t) throw std::runtime_error("missing tensor " + n);
        size_t off = (size_t)t->get("offset")->num, bytes = (size_t)t->get("bytes")->num;
        if (off + bytes > blob.size()) throw std::runtime_error("tensor out of range " + n);
        return {blob.data() + off, bytes};
    }
};

// ---------------------------------------------------------------- Vulkan
struct Buf { VkBuffer b = VK_NULL_HANDLE; VkDeviceMemory m = VK_NULL_HANDLE; VkDeviceSize size = 0; void *map = nullptr; bool coherent = true; };

constexpr int NB = 8;   // storage-buffer bindings per descriptor set

struct Ctx {
    VkInstance inst; VkPhysicalDevice pd; VkDevice dev; VkQueue q; uint32_t qf = 0;
    VkPhysicalDeviceMemoryProperties mp; VkCommandPool pool; VkFence fence;
    VkDescriptorSetLayout dsl; VkPipelineLayout pl; VkDescriptorPool dpool;
    Buf staging; std::string name; uint32_t subgroup = 0, shared_bytes = 0;
    bool spin = false;  // --spin polls the fence instead of sleeping (tested: did not remove the ~28 ms spikes)

    uint32_t mem_type(uint32_t bits, VkMemoryPropertyFlags want) {
        for (uint32_t i = 0; i < mp.memoryTypeCount; ++i)
            if ((bits & (1u << i)) && (mp.memoryTypes[i].propertyFlags & want) == want) return i;
        throw std::runtime_error("no suitable memory type");
    }
    // host: mapped memory. readback: prefer HOST_CACHED memory, because CPU reads from uncached (write-combined) mappings
    // are slow enough to dominate sampling (V21 G4 first run: 10.6 tok/s).
    Buf buffer(VkDeviceSize size, bool host, bool readback = false) {
        Buf r; r.size = size;
        VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        bi.size = size; bi.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        VK(vkCreateBuffer(dev, &bi, nullptr, &r.b));
        VkMemoryRequirements req; vkGetBufferMemoryRequirements(dev, r.b, &req);
        VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        ai.allocationSize = req.size;
        const VkMemoryPropertyFlags HV = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT, HC = VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                                    CA = VK_MEMORY_PROPERTY_HOST_CACHED_BIT;
        if (!host) ai.memoryTypeIndex = mem_type(req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        else if (!readback) ai.memoryTypeIndex = mem_type(req.memoryTypeBits, HV | HC);
        else {
            try { ai.memoryTypeIndex = mem_type(req.memoryTypeBits, HV | CA | HC); }
            catch (const std::runtime_error &) {
                try { ai.memoryTypeIndex = mem_type(req.memoryTypeBits, HV | CA); r.coherent = false; }
                catch (const std::runtime_error &) { ai.memoryTypeIndex = mem_type(req.memoryTypeBits, HV | HC); }
            }
        }
        VK(vkAllocateMemory(dev, &ai, nullptr, &r.m)); VK(vkBindBufferMemory(dev, r.b, r.m, 0));
        if (host) VK(vkMapMemory(dev, r.m, 0, size, 0, &r.map));
        return r;
    }
    void invalidate(const Buf &b) {
        if (b.coherent) return;
        VkMappedMemoryRange mr{VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE}; mr.memory = b.m; mr.offset = 0; mr.size = VK_WHOLE_SIZE;
        VK(vkInvalidateMappedMemoryRanges(dev, 1, &mr));
    }
    VkCommandBuffer begin_once() {
        VkCommandBufferAllocateInfo ai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        ai.commandPool = pool; ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY; ai.commandBufferCount = 1;
        VkCommandBuffer cb; VK(vkAllocateCommandBuffers(dev, &ai, &cb));
        VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO}; bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        VK(vkBeginCommandBuffer(cb, &bi)); return cb;
    }
    void submit_wait(VkCommandBuffer cb) {
        VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO}; si.commandBufferCount = 1; si.pCommandBuffers = &cb;
        VK(vkQueueSubmit(q, 1, &si, fence));
        if (spin) { VkResult r; while ((r = vkGetFenceStatus(dev, fence)) == VK_NOT_READY) {} VK(r); }
        else VK(vkWaitForFences(dev, 1, &fence, VK_TRUE, UINT64_MAX));
        VK(vkResetFences(dev, 1, &fence));
    }
    void end_submit_free(VkCommandBuffer cb) { VK(vkEndCommandBuffer(cb)); submit_wait(cb); vkFreeCommandBuffers(dev, pool, 1, &cb); }
    Buf upload(const void *data, size_t bytes) {
        Buf r = buffer(bytes, false);
        if (bytes > staging.size) throw std::runtime_error("staging too small");
        memcpy(staging.map, data, bytes);
        VkCommandBuffer cb = begin_once(); VkBufferCopy c{0, 0, bytes}; vkCmdCopyBuffer(cb, staging.b, r.b, 1, &c); end_submit_free(cb);
        return r;
    }
    template <class T> Buf upload(const std::vector<T> &v) { return upload(v.data(), v.size() * sizeof(T)); }

    void init(size_t staging_bytes) {
        VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO}; app.pApplicationName = "seedplane-qwen-vk"; app.apiVersion = VK_API_VERSION_1_2;
        VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO}; ici.pApplicationInfo = &app;
        const char *layer = "VK_LAYER_KHRONOS_validation";
        if (getenv("SP_VK_VALIDATE")) { ici.enabledLayerCount = 1; ici.ppEnabledLayerNames = &layer; }
        // Portability drivers (MoltenVK on macOS) are only enumerated when the instance asks for them.
        uint32_t ne = 0; vkEnumerateInstanceExtensionProperties(nullptr, &ne, nullptr);
        std::vector<VkExtensionProperties> iexts(ne); vkEnumerateInstanceExtensionProperties(nullptr, &ne, iexts.data());
        const char *portability_enum = "VK_KHR_portability_enumeration";
        for (auto &e : iexts) if (!strcmp(e.extensionName, portability_enum)) {
            ici.enabledExtensionCount = 1; ici.ppEnabledExtensionNames = &portability_enum; ici.flags |= 0x00000001;  // ENUMERATE_PORTABILITY_BIT_KHR
        }
        VK(vkCreateInstance(&ici, nullptr, &inst));
        uint32_t n = 0; vkEnumeratePhysicalDevices(inst, &n, nullptr); std::vector<VkPhysicalDevice> pds(n); vkEnumeratePhysicalDevices(inst, &n, pds.data());
        pd = VK_NULL_HANDLE;
        for (auto d : pds) { VkPhysicalDeviceProperties p; vkGetPhysicalDeviceProperties(d, &p);
                             if (p.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) { pd = d; break; } }
        if (!pd) { if (!n) throw std::runtime_error("no Vulkan device"); pd = pds[0]; }
        VkPhysicalDeviceSubgroupProperties sg{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES};
        VkPhysicalDeviceProperties2 p2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2}; p2.pNext = &sg;
        vkGetPhysicalDeviceProperties2(pd, &p2); name = p2.properties.deviceName; subgroup = sg.subgroupSize;
        shared_bytes = p2.properties.limits.maxComputeSharedMemorySize;
        if (!(sg.supportedOperations & VK_SUBGROUP_FEATURE_ARITHMETIC_BIT) || !(sg.supportedStages & VK_SHADER_STAGE_COMPUTE_BIT))
            throw std::runtime_error("device lacks compute subgroup arithmetic");
        vkGetPhysicalDeviceMemoryProperties(pd, &mp);
        uint32_t nq = 0; vkGetPhysicalDeviceQueueFamilyProperties(pd, &nq, nullptr); std::vector<VkQueueFamilyProperties> qfp(nq);
        vkGetPhysicalDeviceQueueFamilyProperties(pd, &nq, qfp.data());
        for (uint32_t i = 0; i < nq; ++i) if (qfp[i].queueFlags & VK_QUEUE_COMPUTE_BIT) { qf = i; break; }
        float prio = 1.f; VkDeviceQueueCreateInfo qci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO}; qci.queueFamilyIndex = qf; qci.queueCount = 1; qci.pQueuePriorities = &prio;
        VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO}; dci.queueCreateInfoCount = 1; dci.pQueueCreateInfos = &qci;
        uint32_t nd = 0; vkEnumerateDeviceExtensionProperties(pd, nullptr, &nd, nullptr);
        std::vector<VkExtensionProperties> dexts(nd); vkEnumerateDeviceExtensionProperties(pd, nullptr, &nd, dexts.data());
        const char *portability_subset = "VK_KHR_portability_subset";   // must be enabled when a device exposes it
        for (auto &e : dexts) if (!strcmp(e.extensionName, portability_subset)) { dci.enabledExtensionCount = 1; dci.ppEnabledExtensionNames = &portability_subset; }
        VK(vkCreateDevice(pd, &dci, nullptr, &dev)); vkGetDeviceQueue(dev, qf, 0, &q);
        VkCommandPoolCreateInfo cpi{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO}; cpi.queueFamilyIndex = qf; cpi.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        VK(vkCreateCommandPool(dev, &cpi, nullptr, &pool));
        VkFenceCreateInfo fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO}; VK(vkCreateFence(dev, &fci, nullptr, &fence));
        VkDescriptorSetLayoutBinding b[NB]{};
        for (int i = 0; i < NB; ++i) { b[i].binding = i; b[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; b[i].descriptorCount = 1; b[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT; }
        VkDescriptorSetLayoutCreateInfo dli{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO}; dli.bindingCount = NB; dli.pBindings = b;
        VK(vkCreateDescriptorSetLayout(dev, &dli, nullptr, &dsl));
        VkPushConstantRange pcr{VK_SHADER_STAGE_COMPUTE_BIT, 0, 32};
        VkPipelineLayoutCreateInfo pli{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO}; pli.setLayoutCount = 1; pli.pSetLayouts = &dsl; pli.pushConstantRangeCount = 1; pli.pPushConstantRanges = &pcr;
        VK(vkCreatePipelineLayout(dev, &pli, nullptr, &pl));
        VkDescriptorPoolSize ps{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, NB * 4096};
        VkDescriptorPoolCreateInfo dpi{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO}; dpi.maxSets = 4096; dpi.poolSizeCount = 1; dpi.pPoolSizes = &ps;
        VK(vkCreateDescriptorPool(dev, &dpi, nullptr, &dpool));
        staging = buffer(staging_bytes, true);
    }
    VkPipeline pipeline(const fs::path &spv) {
        std::string code = read_text(spv);
        VkShaderModuleCreateInfo smi{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO}; smi.codeSize = code.size(); smi.pCode = (const uint32_t *)code.data();
        VkShaderModule sm; VK(vkCreateShaderModule(dev, &smi, nullptr, &sm));
        VkComputePipelineCreateInfo ci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
        ci.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO}; ci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT; ci.stage.module = sm; ci.stage.pName = "main";
        ci.layout = pl; VkPipeline p; VK(vkCreateComputePipelines(dev, VK_NULL_HANDLE, 1, &ci, nullptr, &p));
        return p;
    }
    VkDescriptorSet set(std::initializer_list<const Buf *> bufs, const Buf &dummy) {
        VkDescriptorSetAllocateInfo ai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO}; ai.descriptorPool = dpool; ai.descriptorSetCount = 1; ai.pSetLayouts = &dsl;
        VkDescriptorSet s; VK(vkAllocateDescriptorSets(dev, &ai, &s));
        VkDescriptorBufferInfo info[NB]; VkWriteDescriptorSet w[NB]{}; int i = 0;
        for (auto *bp : bufs) { if (i == NB) break; const Buf &bb = bp ? *bp : dummy; info[i] = {bb.b, 0, VK_WHOLE_SIZE}; ++i; }
        for (; i < NB; ++i) info[i] = {dummy.b, 0, VK_WHOLE_SIZE};
        for (int k = 0; k < NB; ++k) { w[k].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET; w[k].dstSet = s; w[k].dstBinding = k; w[k].descriptorCount = 1;
                                      w[k].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; w[k].pBufferInfo = &info[k]; }
        vkUpdateDescriptorSets(dev, NB, w, 0, nullptr); return s;
    }
};

// ---------------------------------------------------------------- engine
struct KV { Buf k, v; };
struct Layer { Buf qkv_w, qkv_b, o_w, gate_w, up_w, down_w, in_norm, post_norm; KV cache[2], sink; };
struct Push { uint32_t u[8]{}; };
static uint32_t F(float f) { uint32_t r; memcpy(&r, &f, 4); return r; }

enum class Mode { Rebuild, RebuildBatch, Shadow, ShadowBatch, Reuse };
constexpr uint32_t BATCH = 8;   // columns of a K/V-only batch submission
struct Item { uint32_t tok, pos, slot; };

// Records dispatches into one command buffer, with a compute->compute barrier between consecutive dispatches.
struct Rec {
    Ctx &c; const Buf &dummy; VkCommandBuffer cb; int n = 0;
    void operator()(VkPipeline p, std::initializer_list<const Buf *> bufs, Push pc, uint32_t groups) {
        if (n) {
            VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER}; mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
            mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
            vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb, 0, nullptr, 0, nullptr);
        }
        VkDescriptorSet s = c.set(bufs, dummy);
        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, p);
        vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, c.pl, 0, 1, &s, 0, nullptr);
        vkCmdPushConstants(cb, c.pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(Push), &pc);
        vkCmdDispatch(cb, groups, 1, 1); ++n;
    }
};

struct Options {
    fs::path shaders; Mode mode = Mode::ShadowBatch; bool full = false, old_attn = false, spin = false, scoring = false, copy_logits = false;
    bool gpu_sampling = false, copy_candidates = false, dynamic_sampling = false;
    uint32_t fixed_sampling_mode = 0;   // 0 greedy, 1 temperature/Gumbel, 2 top-k/top-p
    long shard = -1, halo = -1, sinks = -1; uint32_t positions = 0;   // positions: RoPE table size and full-attention window
};

struct Engine {
    Ctx c; Bundle bd; Options o;
    uint32_t L, H, NH, NKV, HD, I, V, QN, S, HALO, K, W, nchunk; float eps, scale; double theta; VkDeviceSize slot_bytes;
    bool two_caches = false;
    std::vector<Layer> layers;
    Buf dummy, state, embed, head, final_norm, rope_b, x, qkv, att, act, part, logits, logits_host, nll, nll_host, xb, xn, qkvb, attb, actb, partb;
    // V22b sampling scratch: sh = 2-level histograms + selection scalars (sample_layout.glsl); cand = top-k set for top-k+top-p.
    static constexpr uint32_t CAND_CAP = 256, CAND_WORDS = 4 + 2 * CAND_CAP, STAT_WG = 64, SH_WORDS = 4104;
    Buf cand, cand_host, spart, shist;
    VkPipeline p_stats = VK_NULL_HANDLE, p_hist = VK_NULL_HANDLE, p_hist2 = VK_NULL_HANDLE, p_final = VK_NULL_HANDLE, p_gumbel = VK_NULL_HANDLE, p_candidates = VK_NULL_HANDLE;
    uint32_t sampling_counter = 0; int fallbacks = 0;
    VkCommandBuffer step[2][2]{}, batch_cb[2]{};
    int dispatches_single = 0, batch_submits = 0; double load_s = 0;
    volatile uint32_t *st = nullptr;
    const char *mode_name = "";

    uint32_t sinks_of(uint32_t c0) const { return std::min(K, c0 >= HALO ? c0 - HALO : 0u); }   // ShardPlan.windows sinks
    bool shadowing() const { return !o.full && (o.mode == Mode::Shadow || o.mode == Mode::ShadowBatch); }
    bool batch_ok() const { return !o.scoring && (o.full || o.mode == Mode::RebuildBatch || o.mode == Mode::ShadowBatch || o.mode == Mode::Reuse); }

    void init(const fs::path &bundle_dir, const Options &opt) {
        o = opt; auto t_load = std::chrono::steady_clock::now();
        bd.load_meta(bundle_dir); bd.load_weights();
        L = bd.a("layers"); H = bd.a("hidden"); NH = bd.a("heads"); NKV = bd.a("kv_heads"); HD = bd.a("head_dim");
        I = bd.a("intermediate"); V = bd.a("vocab"); QN = (NH + 2 * NKV) * HD;
        theta = bd.arch().get("rope_theta")->num; eps = (float)bd.arch().get("rms_norm_eps")->num; scale = 1.0f / std::sqrt((float)HD);
        const Json &pj = *bd.meta.get("plan");
        S = (uint32_t)(o.shard >= 0 ? o.shard : (long)pj.get("shard")->num); HALO = (uint32_t)(o.halo >= 0 ? o.halo : (long)pj.get("halo")->num);
        K = (uint32_t)(o.sinks >= 0 ? o.sinks : (long)pj.get("sinks")->num);
        const uint32_t P = o.positions;
        if (o.full) { S = P; HALO = 0; K = 0; }             // one window covering everything = original full attention
        W = std::min(S + HALO + K, P); nchunk = (W + 127) / 128;
        auto fail = [](const char *m) { throw std::runtime_error(m); };
        if (HD != 64 || P > bd.a("max_positions") || S == 0) fail("unsupported head_dim, length or shard");
        if (!o.full && o.mode != Mode::Rebuild && o.mode != Mode::RebuildBatch && (HALO > S || K > S)) fail("shadow/reuse need halo <= shard and sinks <= shard");
        if (o.old_attn && (W > 4096 || (!o.full && o.mode == Mode::Shadow))) fail("--attn old supports one column and <= 4096 slots");
        two_caches = !o.full && (o.mode == Mode::Shadow || o.mode == Mode::ShadowBatch || o.mode == Mode::Reuse);
        mode_name = o.full ? "full" : o.mode == Mode::Rebuild ? "rebuild" : o.mode == Mode::RebuildBatch ? "rebuild-batch"
                  : o.mode == Mode::Reuse ? "reuse" : o.mode == Mode::ShadowBatch ? "shadow-batch" : "shadow";

        size_t biggest = 0;
        for (auto &kv : bd.meta.get("tensors")->obj) biggest = std::max(biggest, (size_t)kv.second.get("bytes")->num);
        c.spin = o.spin; c.init(biggest);
        const bool dual_kernels = !o.full && o.mode == Mode::Shadow;          // only the dual-column shadow mode needs 2-column GEMV
        if (c.shared_bytes < (dual_kernels ? 2u : 1u) * 4864 * 4 + 256) fail("device shared memory too small for the GEMV kernels");
        auto up = [&](const std::string &n) { auto t = bd.tensor(n); return c.upload(t.first, t.second); };
        dummy = c.buffer(256, false); state = c.buffer(256, true); memset(state.map, 0, 256); st = (volatile uint32_t *)state.map;
        embed = up("embed"); head = bd.has("lm_head") ? up("lm_head") : embed; final_norm = up("final_norm");
        slot_bytes = (VkDeviceSize)NKV * HD * 4;
        layers.resize(L);
        for (uint32_t l = 0; l < L; ++l) {
            std::string p = std::to_string(l) + "."; Layer &ly = layers[l];
            ly.qkv_w = up(p + "qkv_w"); ly.qkv_b = up(p + "qkv_b"); ly.o_w = up(p + "o_w");
            ly.gate_w = up(p + "gate_w"); ly.up_w = up(p + "up_w"); ly.down_w = up(p + "down_w");
            ly.in_norm = up(p + "in_norm"); ly.post_norm = up(p + "post_norm");
            for (int k = 0; k < (two_caches ? 2 : 1); ++k) { ly.cache[k].k = c.buffer(W * slot_bytes, false); ly.cache[k].v = c.buffer(W * slot_bytes, false); }
            if (!two_caches) ly.cache[1] = ly.cache[0];
            ly.sink.k = c.buffer(std::max(K, 1u) * slot_bytes, false); ly.sink.v = c.buffer(std::max(K, 1u) * slot_bytes, false);
        }
        bd.blob.clear(); bd.blob.shrink_to_fit();
        // RoPE table for original positions, computed like the PyTorch reference (FP32 inv_freq and phase).
        std::vector<float> rope((size_t)P * (HD / 2) * 2);
        for (uint32_t pos = 0; pos < P; ++pos)
            for (uint32_t i = 0; i < HD / 2; ++i) {
                float inv = 1.0f / (float)std::pow(theta, (double)((float)(2 * i) / (float)HD)), ph = (float)pos * inv;
                rope[((size_t)pos * (HD / 2) + i) * 2] = (float)std::cos((double)ph);
                rope[((size_t)pos * (HD / 2) + i) * 2 + 1] = (float)std::sin((double)ph);
            }
        rope_b = c.upload(rope);
        x = c.buffer(2 * H * 4, false); qkv = c.buffer(2 * QN * 4, false); att = c.buffer(2 * H * 4, false); act = c.buffer(2 * I * 4, false);
        part = c.buffer((VkDeviceSize)2 * NH * nchunk * 66 * 4, false); logits = c.buffer((VkDeviceSize)V * 4, false);
        logits_host = c.buffer((VkDeviceSize)V * 4, true, true);
        nll = c.buffer((VkDeviceSize)std::max(P, 1u) * 4, false); nll_host = c.buffer((VkDeviceSize)std::max(P, 1u) * 4, true, true);
        xb = c.buffer(BATCH * H * 4, false); xn = c.buffer(BATCH * H * 4, false); qkvb = c.buffer(BATCH * QN * 4, false);
        attb = c.buffer(BATCH * H * 4, false); actb = c.buffer(BATCH * I * 4, false); partb = c.buffer((VkDeviceSize)BATCH * NH * nchunk * 66 * 4, false);
        cand = c.buffer(CAND_WORDS * 4, false); cand_host = c.buffer(CAND_WORDS * 4, true, true);
        spart = c.buffer(STAT_WG * 8, false); shist = c.buffer(SH_WORDS * 4, false);
        load_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_load).count();
        record_all();
    }

    VkPipeline pipe(const char *name) { return c.pipeline(o.shaders / name); }
    static void mem_barrier(VkCommandBuffer cb, VkPipelineStageFlags src, VkAccessFlags sa, VkPipelineStageFlags dst, VkAccessFlags da) {
        VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER}; mb.srcAccessMask = sa; mb.dstAccessMask = da;
        vkCmdPipelineBarrier(cb, src, dst, 0, 1, &mb, 0, nullptr, 0, nullptr); }

    void record_all() {
        const bool dual = !o.full && o.mode == Mode::Shadow;              // 2-column pipelines only when they are used
        VkPipeline p_embed = pipe("embed.spv"), p_gemv_n[2] = {pipe("gemv1.spv"), dual ? pipe("gemv2.spv") : VK_NULL_HANDLE},
                   p_swiglu_n[2] = {pipe("swiglu1.spv"), dual ? pipe("swiglu2.spv") : VK_NULL_HANDLE}, p_rope = pipe("rope_kv.spv"), p_attn = pipe("attn.spv"),
                   p_part = pipe("attn_part.spv"), p_comb = pipe("attn_combine.spv"), p_argmax = pipe("argmax.spv"), p_nll = pipe("nll.spv"),
                   p_embed_b = pipe("embed_b.spv"), p_norm_b = pipe("rmsnorm_b.spv"), p_gemm_b = pipe("gemm_b.spv"),
                   p_swiglu_b = pipe("swiglu_b.spv"), p_rope_b = pipe("rope_b.spv"), p_part_b = pipe("attn_part_b.spv");
        if (o.gpu_sampling) { p_stats = pipe("sample_stats.spv"); p_hist = pipe("sample_hist.spv"); p_hist2 = pipe("sample_hist2.spv");
                              p_final = pipe("sample_final.spv");
                              p_gumbel = pipe("sample_gumbel.spv"); p_candidates = pipe("sample_candidates.spv"); }
        VkCommandBufferAllocateInfo cai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO}; cai.commandPool = c.pool; cai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY; cai.commandBufferCount = 1;
        VkCommandBufferBeginInfo cbi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        const VkPipelineStageFlags CS = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, TR = VK_PIPELINE_STAGE_TRANSFER_BIT;
        const VkAccessFlags SW = VK_ACCESS_SHADER_WRITE_BIT, SRW = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        const uint32_t rows_per_wg = std::max(1u, 128u / std::max(1u, c.subgroup));
        auto gemv_groups = [&](uint32_t N) { return std::min((N + rows_per_wg - 1) / rows_per_wg, 4096u); };
        enum { BIAS = 1, RES = 2, NORM = 4 };
        // top-k / top-p passes (all exit at once unless state.smode == 2); `final_groups` > 1 only in the self-test.
        record_candidates = [this](Rec &d, bool with_final, uint32_t test, uint32_t final_groups, const Buf *draws) {
            d(p_stats, {&state, &logits, &spart, &shist, &cand}, {{V}}, STAT_WG);
            d(p_hist, {&state, &logits, &spart, &shist}, {{V, STAT_WG}}, STAT_WG);
            d(p_hist2, {&state, &logits, &shist}, {{V}}, STAT_WG);
            if (with_final) d(p_final, {&state, &logits, &shist, &cand, draws ? draws : &dummy}, {{V, test}}, final_groups);
        };
        // One-token graph for `ncols` batch columns; column 0 uses cache[cur], column 1 (shadow window) cache[1 - cur].
        auto record = [&](uint32_t ncols, int cur) {
            VkPipeline p_gemv = p_gemv_n[ncols - 1], p_swiglu = p_swiglu_n[ncols - 1];
            VkCommandBuffer cb; VK(vkAllocateCommandBuffers(c.dev, &cai, &cb)); VK(vkBeginCommandBuffer(cb, &cbi));
            mem_barrier(cb, CS | TR, SW | VK_ACCESS_TRANSFER_WRITE_BIT, CS, SRW);   // order after earlier submissions / copies
            Rec d{c, dummy, cb};
            d(p_embed, {&state, &embed, &x}, {{H, ncols}}, (H / 2 + 255) / 256);
            for (uint32_t l = 0; l < L; ++l) {
                Layer &ly = layers[l]; KV &a = ly.cache[cur], &b = ly.cache[1 - cur];
                d(p_gemv, {nullptr, &x, &ly.qkv_w, &ly.qkv_b, &qkv, &ly.in_norm}, {{QN, H, BIAS | NORM, F(eps), ncols}}, gemv_groups(QN));
                d(p_rope, {&state, &qkv, &a.k, &a.v, &rope_b, &b.k, &b.v}, {{NH, NKV, QN}}, (NH + NKV) * ncols);
                if (o.old_attn) d(p_attn, {&state, &qkv, &a.k, &a.v, &att}, {{NH, NKV, F(scale)}}, NH);
                else {
                    d(p_part, {&state, &qkv, &a.k, &a.v, &part, &b.k, &b.v}, {{NH, NKV, F(scale), QN, nchunk}}, ncols * NH * nchunk);
                    d(p_comb, {nullptr, &part, &att}, {{NH, nchunk, H}}, ncols * NH);
                }
                d(p_gemv, {nullptr, &att, &ly.o_w, nullptr, &x, nullptr}, {{H, H, RES, F(eps), ncols}}, gemv_groups(H));
                d(p_swiglu, {nullptr, &x, &ly.gate_w, &ly.up_w, &act, &ly.post_norm}, {{I, H, 0, F(eps), ncols}}, gemv_groups(I));
                d(p_gemv, {nullptr, &act, &ly.down_w, nullptr, &x, nullptr}, {{H, I, RES, F(eps), ncols}}, gemv_groups(H));
            }
            d(p_gemv_n[0], {nullptr, &x, &head, nullptr, &logits, &final_norm}, {{V, H, NORM, F(eps), 1}}, gemv_groups(V));
            if (o.dynamic_sampling || o.fixed_sampling_mode == 0u) d(p_argmax, {&state, &logits}, {{V}}, 1);
            if (o.scoring) d(p_nll, {&state, &logits, &nll}, {{V}}, 1);
            if (o.gpu_sampling) {
                if (o.dynamic_sampling || o.fixed_sampling_mode == 2u) record_candidates(d, true, 0, 1, nullptr);
                if (o.dynamic_sampling || o.fixed_sampling_mode == 1u) d(p_gumbel, {&state, &logits, &dummy}, {{V, 0}}, 1);
                if (o.copy_candidates) {
                    mem_barrier(cb, CS, SW, TR, VK_ACCESS_TRANSFER_READ_BIT);   // 2 KB status exposes overflow to host fallback
                    VkBufferCopy r{0, 0, CAND_WORDS * 4}; vkCmdCopyBuffer(cb, cand.b, cand_host.b, 1, &r);
                    mem_barrier(cb, TR, VK_ACCESS_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_HOST_BIT, VK_ACCESS_HOST_READ_BIT);
                }
            }
            if (o.copy_logits) {                                  // logits to host memory for sampling, same submission
                mem_barrier(cb, CS, SW, TR, VK_ACCESS_TRANSFER_READ_BIT);
                VkBufferCopy r{0, 0, (VkDeviceSize)V * 4}; vkCmdCopyBuffer(cb, logits.b, logits_host.b, 1, &r);
                mem_barrier(cb, TR, VK_ACCESS_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_HOST_BIT, VK_ACCESS_HOST_READ_BIT);
            }
            mem_barrier(cb, CS, SW, VK_PIPELINE_STAGE_HOST_BIT | TR, VK_ACCESS_HOST_READ_BIT | VK_ACCESS_TRANSFER_READ_BIT);
            VK(vkEndCommandBuffer(cb)); if (ncols == 1) dispatches_single = d.n; return cb;
        };
        step[0][0] = record(1, 0); step[0][1] = two_caches ? record(1, 1) : step[0][0];
        if (!o.full && o.mode == Mode::Shadow) { step[1][0] = record(2, 0); step[1][1] = record(2, 1); }
        // K/V-only batch of up to BATCH tokens (state.bn, btok/bpos/bslot) into cache[cur], causal inside the batch.
        // Only the K/V of every layer are needed, so the last layer stops after writing its K/V and there is no LM head.
        auto record_batch = [&](int cur) {
            VkCommandBuffer cb; VK(vkAllocateCommandBuffers(c.dev, &cai, &cb)); VK(vkBeginCommandBuffer(cb, &cbi));
            mem_barrier(cb, CS | TR, SW | VK_ACCESS_TRANSFER_WRITE_BIT, CS, SRW);
            Rec d{c, dummy, cb};
            d(p_embed_b, {&state, &embed, &xb}, {{H}}, (H / 2 + 255) / 256);
            for (uint32_t l = 0; l < L; ++l) {
                Layer &ly = layers[l]; KV &a = ly.cache[cur];
                d(p_norm_b, {&state, &xb, &ly.in_norm, &xn}, {{H, F(eps)}}, BATCH);
                d(p_gemm_b, {&state, &xn, &ly.qkv_w, &ly.qkv_b, &qkvb}, {{QN, H, BIAS}}, gemv_groups(QN));
                d(p_rope_b, {&state, &qkvb, &a.k, &a.v, &rope_b}, {{NH, NKV, QN}}, (NH + NKV) * BATCH);
                if (l + 1 == L) break;
                d(p_part_b, {&state, &qkvb, &a.k, &a.v, &partb}, {{NH, NKV, F(scale), QN, nchunk}}, BATCH * NH * nchunk);
                d(p_comb, {nullptr, &partb, &attb}, {{NH, nchunk, H}}, BATCH * NH);
                d(p_gemm_b, {&state, &attb, &ly.o_w, nullptr, &xb}, {{H, H, RES}}, gemv_groups(H));
                d(p_norm_b, {&state, &xb, &ly.post_norm, &xn}, {{H, F(eps)}}, BATCH);
                d(p_swiglu_b, {&state, &xn, &ly.gate_w, &ly.up_w, &actb}, {{I, H}}, gemv_groups(I));
                d(p_gemm_b, {&state, &actb, &ly.down_w, nullptr, &xb}, {{H, I, RES}}, gemv_groups(H));
            }
            VK(vkEndCommandBuffer(cb)); return cb;
        };
        batch_cb[0] = record_batch(0); batch_cb[1] = two_caches ? record_batch(1) : batch_cb[0];
    }

    // Copy `n` KV slots of every layer between caches (src/dst: 0/1 = cache index, 2 = sink buffer).
    void copy_slots(int src, uint32_t s0, int dst, uint32_t d0, uint32_t n) {
        if (!n) return;
        VkCommandBuffer cb = c.begin_once();
        mem_barrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                    VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT);
        VkBufferCopy r{s0 * slot_bytes, d0 * slot_bytes, n * slot_bytes};
        for (auto &ly : layers) {
            KV &a = src == 2 ? ly.sink : ly.cache[src], &b = dst == 2 ? ly.sink : ly.cache[dst];
            vkCmdCopyBuffer(cb, a.k.b, b.k.b, 1, &r); vkCmdCopyBuffer(cb, a.v.b, b.v.b, 1, &r);
        }
        mem_barrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                    VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);
        c.end_submit_free(cb);
    }
    void write_buffer(const Buf &dst, const void *data, size_t bytes) {
        memcpy(c.staging.map, data, bytes);
        VkCommandBuffer cb = c.begin_once(); VkBufferCopy r{0, 0, bytes}; vkCmdCopyBuffer(cb, c.staging.b, dst.b, 1, &r);
        mem_barrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT);
        c.end_submit_free(cb);
    }
    void grab(const Buf &src, const Buf &dst, VkDeviceSize bytes) {
        VkCommandBuffer cb = c.begin_once(); VkBufferCopy r{0, 0, bytes}; vkCmdCopyBuffer(cb, src.b, dst.b, 1, &r);
        mem_barrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_HOST_BIT, VK_ACCESS_HOST_READ_BIT);
        c.end_submit_free(cb); c.invalidate(dst); }
    void kv_batch(int cache, std::vector<Item> &items) {
        for (size_t b = 0; b < items.size(); b += BATCH) {
            uint32_t n = (uint32_t)std::min<size_t>(BATCH, items.size() - b); st[6] = n;
            for (uint32_t k = 0; k < n; ++k) { st[8 + k] = items[b + k].tok; st[16 + k] = items[b + k].pos; st[24 + k] = items[b + k].slot; }
            c.submit_wait(batch_cb[cache]); ++batch_submits;
        }
        items.clear();
    }
    // Sampling parameters for the next steps (state.smode: 0 greedy, 1 Gumbel-max on the GPU, 2 top-256 candidates).
    void set_sampling(const SamplerParams &sp) {
        st[32] = sp.temperature <= 0.0f ? 0u : (sp.top_k == 0 && sp.top_p >= 1.0f) ? 1u : 2u;
        float T = sp.temperature > 0.0f ? sp.temperature : 1.0f; memcpy((void *)&st[33], &T, 4);
        st[34] = (uint32_t)(sp.seed ^ (sp.seed >> 32)); st[37] = (uint32_t)std::max(sp.top_k, 0);
        float P = sp.top_p; memcpy((void *)&st[38], &P, 4); if (!o.gpu_sampling) st[32] = 0;
    }
    // Next token after a full step: GPU draw, candidate draw on the host, or (fallback / no GPU sampling) full logits.
    int next_token(Sampler &sm, uint32_t greedy_token) {
        if (sm.greedy()) return (int)greedy_token;
        const SamplerParams &sp = sm.params();
        if (o.gpu_sampling) {
            if (!(sp.top_k > 0 && sp.top_p < 1.0f)) return (int)st[36];      // Gumbel-max, full or restricted to l >= tau
            c.invalidate(cand_host); const uint32_t *cd = (const uint32_t *)cand_host.map;
            if (cd[2] <= CAND_CAP) return (int)st[36];                  // GPU sorted top-k, applied top-p, and drew exactly
            ++fallbacks; grab(logits, logits_host, (VkDeviceSize)V * 4);
        }
        return sm.sample(host_logits(), V);
    }
    std::vector<int> cand_ids_; std::vector<double> cand_probs_;
    std::function<void(Rec &, bool, uint32_t, uint32_t, const Buf *)> record_candidates;
    // Logits of the last full step (host-cached mapping, so the sampler can read it directly).
    const float *host_logits() { c.invalidate(logits_host); return (const float *)logits_host.map; }
};

// ---------------------------------------------------------------- session
// One growing token sequence decoded under the shard plan. seq may hold tokens not yet processed (e.g. the last sampled
// token, or a new chat turn); `done` counts processed positions. KV state persists between calls.
struct Session {
    Engine &e;
    std::vector<uint32_t> seq; uint32_t done = 0;
    int cur = 0, boundaries = 0, extra = 0, dual = 0, shadow_items = 0; uint32_t slot = 0, slot2 = 0; bool active = false, sinks_saved = false;
    std::vector<Item> pend_cur, pend_sh;
    explicit Session(Engine &en) : e(en) {}
    void reset() { *this = Session(e); }
    Session &operator=(const Session &s) {
        seq = s.seq; done = s.done; cur = s.cur; boundaries = s.boundaries; extra = s.extra; dual = s.dual; shadow_items = s.shadow_items;
        slot = s.slot; slot2 = s.slot2; active = s.active; sinks_saved = s.sinks_saved; pend_cur = s.pend_cur; pend_sh = s.pend_sh; return *this; }

    void start_shadow(uint32_t nb) {             // window of the next shard starts as [its sinks] (+ halo as it is decoded)
        e.kv_batch(cur, pend_cur);               // sinks may still be pending in the current window
        uint32_t ns = e.sinks_of(nb);
        e.copy_slots(sinks_saved ? 2 : cur, 0, 1 - cur, 0, ns); slot2 = ns; active = true;
    }
    // Process position i = done. need_logits: run the full one-token graph (logits + argmax); otherwise K/V only (batched).
    // target: next token for teacher-forced NLL (scoring engines only). Returns the greedy token when logits were computed.
    uint32_t step(bool need_logits, uint32_t target = 0) {
        const uint32_t i = done, S = e.S, HALO = e.HALO, K = e.K; volatile uint32_t *st = e.st;
        if (i >= e.o.positions) throw std::runtime_error("context full (" + std::to_string(e.o.positions) + " positions)");
        if (!e.o.full && i > 0 && i % S == 0) {
            ++boundaries; e.kv_batch(cur, pend_cur);
            if (!sinks_saved && e.o.mode != Mode::Rebuild && e.o.mode != Mode::RebuildBatch) { e.copy_slots(cur, 0, 2, 0, std::min(K, S)); sinks_saved = true; }
            uint32_t h0 = i >= HALO ? i - HALO : 0, ns = e.sinks_of(i);
            if (e.o.mode == Mode::Rebuild || e.o.mode == Mode::RebuildBatch) {   // exact: re-run sinks + halo inside the new window
                std::vector<Item> items; slot = 0;
                for (uint32_t j = 0; j < ns; ++j) items.push_back({seq[j], j, slot++});
                for (uint32_t j = h0; j < i; ++j) items.push_back({seq[j], j, slot++});
                extra += (int)items.size();
                if (e.o.mode == Mode::RebuildBatch) e.kv_batch(cur, items);
                else for (auto &it : items) { st[0] = it.tok; st[1] = it.pos; st[2] = it.slot; st[5] = e.o.positions - 1; e.c.submit_wait(e.step[0][cur]); }
            } else if (e.shadowing()) {                  // exact: the shadow window already holds sinks + halo
                if (!active) start_shadow(i);
                e.kv_batch(1 - cur, pend_sh);
                cur = 1 - cur; slot = slot2; active = false;
            } else {                                     // approximate: reuse halo K/V computed in the previous window
                uint32_t c0p = i - S, h0p = c0p >= HALO ? c0p - HALO : 0, pre = c0p ? e.sinks_of(c0p) + (c0p - h0p) : 0;
                e.copy_slots(2, 0, 1 - cur, 0, ns);
                e.copy_slots(cur, pre + (h0 - c0p), 1 - cur, ns, i - h0);
                cur = 1 - cur; slot = ns + (i - h0);
            }
        }
        if (e.shadowing() && !active && HALO > 0) {
            uint32_t nb = (i / S + 1) * S;
            if (nb < e.o.positions && i >= nb - HALO) start_shadow(nb);
        }
        uint32_t next = 0;
        if (!need_logits) {
            pend_cur.push_back({seq[i], i, slot++});
            if (pend_cur.size() == BATCH) e.kv_batch(cur, pend_cur);
        } else {
            e.kv_batch(cur, pend_cur);
            const bool use_dual = active && e.o.mode == Mode::Shadow;
            st[0] = seq[i]; st[1] = i; st[2] = slot; st[3] = slot2; st[4] = target; st[5] = i; st[35] = e.sampling_counter++;
            e.c.submit_wait(e.step[use_dual ? 1 : 0][cur]);
            ++slot; if (use_dual) { ++slot2; ++dual; }
            next = st[0];
        }
        if (active && e.o.mode == Mode::ShadowBatch) {   // the same token, appended to the next shard's window
            pend_sh.push_back({seq[i], i, slot2++}); ++shadow_items;
            if (pend_sh.size() == BATCH) e.kv_batch(1 - cur, pend_sh);
        }
        ++done; return next;
    }
    // Process every pending token; the last one runs the full graph so its logits are ready. Returns its greedy token.
    uint32_t catch_up() {
        uint32_t next = 0;
        while (done < seq.size()) { bool last = done + 1 == seq.size(); next = step(last || !e.batch_ok()); }
        return next;
    }
};

// ---------------------------------------------------------------- text generation
static std::vector<uint32_t> parse_list(const std::string &s) {
    std::vector<uint32_t> v; size_t i = 0;
    while (i < s.size()) { size_t e = s.find(',', i); if (e == std::string::npos) e = s.size(); v.push_back((uint32_t)std::stoul(s.substr(i, e - i))); i = e + 1; }
    return v;
}
static std::vector<uint32_t> read_ids(const std::string &f) {       // little-endian int32 token file
    std::string raw = read_text(f); std::vector<uint32_t> v(raw.size() / 4); memcpy(v.data(), raw.data(), v.size() * 4); return v;
}

struct GenResult { std::string reason = "length"; int generated = 0, prompt_tokens = 0; double prefill_s = 0, decode_s = 0; };

// Append `ids` to the session, then sample up to max_new tokens. on_token(id, text_piece) streams the output; the final
// sampled token stays in the session unprocessed (a chat turn's <|im_end|> becomes part of the history).
template <class CB>
static GenResult generate(Engine &e, Session &s, const Tokenizer &tok, const std::vector<int> &ids, const SamplerParams &sp,
                          int max_new, const std::vector<int> &stop, CB on_token) {
    using clk = std::chrono::steady_clock; GenResult g; g.prompt_tokens = (int)ids.size();
    for (int id : ids) s.seq.push_back((uint32_t)id);
    auto t0 = clk::now(); Sampler sm(sp); StreamDecoder dec(tok, true); e.set_sampling(sp);
    try {
        uint32_t greedy = s.catch_up();
        g.prefill_s = std::chrono::duration<double>(clk::now() - t0).count(); t0 = clk::now();
        for (;;) {
            int t = e.next_token(sm, greedy);
            s.seq.push_back((uint32_t)t); ++g.generated;
            bool is_stop = std::find(stop.begin(), stop.end(), t) != stop.end();
            on_token(t, is_stop ? std::string() : dec.push(t));
            if (is_stop) { g.reason = "stop"; break; }
            if (g.generated >= max_new) break;
            greedy = s.step(true);
        }
    } catch (const std::runtime_error &ex) {
        if (std::string(ex.what()).rfind("context full", 0) != 0) throw;
        g.reason = "context";
    }
    std::string rest = dec.flush(); if (!rest.empty()) on_token(-1, rest);
    g.decode_s = std::chrono::duration<double>(clk::now() - t0).count();
    return g;
}

static std::string chat_turn(const Session &s, const Tokenizer &tok, const std::string &system, const std::string &user) {
    const std::string tail = "<|im_start|>user\n" + user + "<|im_end|>\n<|im_start|>assistant\n";
    if (s.seq.empty()) return "<|im_start|>system\n" + system + "<|im_end|>\n" + tail;
    bool closed = !s.seq.empty() && (int)s.seq.back() == tok.token_id("<|im_end|>");
    return (closed ? "" : "<|im_end|>") + std::string("\n") + tail;
}

static bool read_line_utf8(std::string &line) {
#ifdef _WIN32
    HANDLE h = GetStdHandle(STD_INPUT_HANDLE); DWORD mode;
    if (GetConsoleMode(h, &mode)) {                      // console: read UTF-16 and convert, so accents/emoji survive
        std::wstring w; wchar_t buf[512]; DWORD n = 0;
        for (;;) {
            if (!ReadConsoleW(h, buf, 512, &n, nullptr) || n == 0) return false;
            w.append(buf, n); if (!w.empty() && w.back() == L'\n') break;
        }
        while (!w.empty() && (w.back() == L'\n' || w.back() == L'\r')) w.pop_back();
        int len = WideCharToMultiByte(CP_UTF8, 0, w.data(), (int)w.size(), nullptr, 0, nullptr, nullptr);
        line.assign((size_t)len, '\0'); WideCharToMultiByte(CP_UTF8, 0, w.data(), (int)w.size(), line.data(), len, nullptr, nullptr);
        return true;
    }
#endif
    if (!std::getline(std::cin, line)) return false;
    if (!line.empty() && line.back() == '\r') line.pop_back();
    return true;
}

static void out(const std::string &s) { fwrite(s.data(), 1, s.size(), stdout); fflush(stdout); }

int main(int argc, char **argv) {
#ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);
#endif
    try {
        if (argc >= 2 && std::string(argv[1]) == "--sample-test") {      // sampler self-test: counts of `draws` samples
            if (argc < 9) { fprintf(stderr, "--sample-test logits.f32 T top_k top_p draws seed out.u32\n"); return 1; }
            std::string raw = read_text(argv[2]); std::vector<float> lg(raw.size() / 4); memcpy(lg.data(), raw.data(), lg.size() * 4);
            SamplerParams sp{(float)atof(argv[3]), atoi(argv[4]), (float)atof(argv[5]), (uint64_t)atoll(argv[7])};
            Sampler sm(sp); std::vector<uint32_t> counts(lg.size()); long draws = atol(argv[6]);
            std::vector<int> ids; std::vector<double> probs; sm.distribution(lg.data(), lg.size(), ids, probs);   // same code path as sample()
            for (long d = 0; d < draws; ++d) ++counts[(size_t)sm.draw(ids, probs)];
            write_bytes(argv[8], counts.data(), counts.size() * 4); return 0;
        }
        if (argc < 2) {
            fprintf(stderr, "usage: qwen_vk <bundle.sp> --chat | --serve | [--prompt ids | --prompt-file f.i32] [-n N] | --score-file f.i32 ...\n"
                            "(see the header of qwen_vk.cpp for every option)\n");
            return 1;
        }
        fs::path bundle_dir = argv[1];
        std::string prompt_s = "9707,11,1879,0", prompt_file, score_file, nll_out, dump_dir = ".", system = "You are a helpful assistant.";
        std::string tok_in, tok_out, nfc_in, nfc_out, detok_in, detok_out;
        int n_new = -1, runs = 3; std::vector<uint32_t> dump_at; bool chat = false, serve = false, copy_only = false, host_sampling = false; long ctx = -1; std::vector<std::string> gpu_test;
        Options opt; opt.shaders = fs::path(argv[0]).parent_path() / "shaders"; SamplerParams sp;
        for (int i = 2; i < argc; ++i) {
            std::string a = argv[i];
            auto next = [&]() { if (i + 1 >= argc) throw std::runtime_error("missing value for " + a); return std::string(argv[++i]); };
            if (a == "--prompt") prompt_s = next(); else if (a == "--prompt-file") prompt_file = next(); else if (a == "-n") n_new = std::stoi(next());
            else if (a == "--runs") runs = std::stoi(next()); else if (a == "--full") opt.full = true;
            else if (a == "--shard") opt.shard = std::stol(next()); else if (a == "--halo") opt.halo = std::stol(next()); else if (a == "--sinks") opt.sinks = std::stol(next());
            else if (a == "--dump-at") dump_at = parse_list(next()); else if (a == "--dump-dir") dump_dir = next(); else if (a == "--shaders") opt.shaders = next();
            else if (a == "--spin") opt.spin = true; else if (a == "--ctx") ctx = std::stol(next());
            else if (a == "--copy-logits") copy_only = true;           // analysis: pay the logits copy without sampling
            else if (a == "--host-sampling") host_sampling = true;     // V21 path: full logits copied, sampled on the CPU
            else if (a == "--sample-test-gpu") { for (int k = 0; k < 7; ++k) gpu_test.push_back(next()); }
            else if (a == "--score-file") score_file = next(); else if (a == "--nll-out") nll_out = next();
            else if (a == "--chat") chat = true; else if (a == "--serve") serve = true; else if (a == "--system") system = next();
            else if (a == "--temperature") sp.temperature = std::stof(next()); else if (a == "--top-k") sp.top_k = std::stoi(next());
            else if (a == "--top-p") sp.top_p = std::stof(next()); else if (a == "--seed") sp.seed = std::stoull(next());
            else if (a == "--tokenize-file") { tok_in = next(); tok_out = next(); } else if (a == "--nfc-file") { nfc_in = next(); nfc_out = next(); }
            else if (a == "--detok-file") { detok_in = next(); detok_out = next(); }
            else if (a == "--attn") { std::string v = next(); if (v == "old") opt.old_attn = true; else if (v != "split") throw std::runtime_error("bad --attn"); }
            else if (a == "--mode") {
                std::string v = next();
                if (v == "rebuild") opt.mode = Mode::Rebuild; else if (v == "rebuild-batch") opt.mode = Mode::RebuildBatch; else if (v == "shadow") opt.mode = Mode::Shadow;
                else if (v == "shadow-batch") opt.mode = Mode::ShadowBatch; else if (v == "reuse") opt.mode = Mode::Reuse; else throw std::runtime_error("bad --mode");
            } else throw std::runtime_error("unknown argument " + a);
        }
        // ------------------------------------------------ tokenizer-only modes (no GPU)
        Tokenizer tok; bool have_tok = fs::exists(bundle_dir / "tokenizer.json");
        if (have_tok) tok.load(read_text(bundle_dir / "tokenizer.json"));
        if (!tok_in.empty() || !nfc_in.empty() || !detok_in.empty()) {
            if (!have_tok) throw std::runtime_error("bundle has no tokenizer.json");
            if (!tok_in.empty()) { auto ids = tok.encode(read_text(tok_in)); write_bytes(tok_out, ids.data(), ids.size() * 4); }
            if (!nfc_in.empty()) { std::string s = tok.nfc(read_text(nfc_in)); write_bytes(nfc_out, s.data(), s.size()); }
            if (!detok_in.empty()) { auto v = read_ids(detok_in); std::string s = tok.decode(std::vector<int>(v.begin(), v.end())); write_bytes(detok_out, s.data(), s.size()); }
            return 0;
        }
        Bundle meta; meta.load_meta(bundle_dir); const uint32_t max_pos = meta.a("max_positions");
        if (!gpu_test.empty()) {                               // GPU-assisted sampler self-test on fixed logits
            if (gpu_test.size() != 7) throw std::runtime_error("--sample-test-gpu logits.f32 T top_k top_p draws seed out.u32");
            opt.positions = 16; opt.gpu_sampling = true; Engine e; e.init(bundle_dir, opt);
            std::string raw = read_text(gpu_test[0]); if (raw.size() != (size_t)e.V * 4) throw std::runtime_error("logits size != vocab");
            e.write_buffer(e.logits, raw.data(), raw.size());
            SamplerParams tp{std::stof(gpu_test[1]), std::stoi(gpu_test[2]), std::stof(gpu_test[3]), std::stoull(gpu_test[5])};
            Sampler sm(tp); e.set_sampling(tp); const long draws = std::stol(gpu_test[4]); std::vector<uint32_t> counts(e.V);
            const char *path = "candidates"; uint32_t cand_count = 0;
            VkCommandBufferAllocateInfo cai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO}; cai.commandPool = e.c.pool; cai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY; cai.commandBufferCount = 1;
            VkCommandBufferBeginInfo cbi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
            VkCommandBuffer cb; VK(vkAllocateCommandBuffers(e.c.dev, &cai, &cb)); VK(vkBeginCommandBuffer(cb, &cbi));
            const uint32_t D = 4096; Buf out = e.c.buffer(D * 4, false), out_host = e.c.buffer(D * 4, true, true);
            if (tp.top_k == 0 && tp.top_p >= 1.0f) {
                path = "gumbel";
                VkDescriptorSet ds = e.c.set({&e.state, &e.logits, &out}, e.dummy); Push pc{{e.V, 1}};
                vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, e.p_gumbel);
                vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, e.c.pl, 0, 1, &ds, 0, nullptr);
                vkCmdPushConstants(cb, e.c.pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(Push), &pc); vkCmdDispatch(cb, D, 1, 1);
                Engine::mem_barrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_READ_BIT);
                VkBufferCopy r{0, 0, D * 4}; vkCmdCopyBuffer(cb, out.b, out_host.b, 1, &r);
                Engine::mem_barrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_HOST_BIT, VK_ACCESS_HOST_READ_BIT);
                VK(vkEndCommandBuffer(cb));
                for (long done = 0; done < draws; done += D) {
                    e.st[35] = (uint32_t)done; e.c.submit_wait(cb); e.c.invalidate(out_host);
                    const uint32_t *o = (const uint32_t *)out_host.map;
                    for (long k = 0; k < std::min<long>(D, draws - done); ++k) ++counts[o[k]];
                }
            } else if (!(tp.top_k > 0 && tp.top_p < 1.0f)) {       // top-k only / top-p only: GPU threshold + restricted Gumbel
                path = "restricted-gumbel";
                Rec d{e.c, e.dummy, cb}; e.record_candidates(d, true, 1, D, &out);   // same passes as every step, D draws
                Engine::mem_barrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_READ_BIT);
                VkBufferCopy r{0, 0, D * 4}; vkCmdCopyBuffer(cb, out.b, out_host.b, 1, &r);
                Engine::mem_barrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_HOST_BIT, VK_ACCESS_HOST_READ_BIT);
                VK(vkEndCommandBuffer(cb));
                for (long done = 0; done < draws; done += D) {
                    e.st[35] = (uint32_t)done; e.c.submit_wait(cb); e.c.invalidate(out_host);
                    const uint32_t *o = (const uint32_t *)out_host.map;
                    for (long k = 0; k < std::min<long>(D, draws - done); ++k) ++counts[o[k]];
                }
            } else {                                                  // top-k + top-p: GPU top-k set and GPU Gumbel draws
                Rec d{e.c, e.dummy, cb}; e.record_candidates(d, true, 0, 1, nullptr);
                Engine::mem_barrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_READ_BIT);
                VkBufferCopy r{0, 0, Engine::CAND_WORDS * 4}; vkCmdCopyBuffer(cb, e.cand.b, e.cand_host.b, 1, &r);
                Engine::mem_barrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_HOST_BIT, VK_ACCESS_HOST_READ_BIT);
                VK(vkEndCommandBuffer(cb)); e.c.submit_wait(cb); e.c.invalidate(e.cand_host);
                const uint32_t *cd = (const uint32_t *)e.cand_host.map; float M, Z; memcpy(&M, cd, 4); memcpy(&Z, cd + 1, 4);
                std::vector<int> ids; std::vector<double> probs; cand_count = cd[2];
                if (cd[2] > Engine::CAND_CAP) {
                    path = "fallback"; std::vector<float> lg(e.V); memcpy(lg.data(), raw.data(), raw.size()); sm.distribution(lg.data(), e.V, ids, probs);
                    for (long k = 0; k < draws; ++k) ++counts[(size_t)sm.draw(ids, probs)];
                } else {
                    path = "gpu-top-k+p";
                    VkCommandBuffer draw_cb; VK(vkAllocateCommandBuffers(e.c.dev, &cai, &draw_cb)); VK(vkBeginCommandBuffer(draw_cb, &cbi));
                    Engine::mem_barrier(draw_cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT,
                                        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT);
                    VkDescriptorSet ds = e.c.set({&e.state, &e.cand, &out}, e.dummy); uint32_t test = 1;
                    vkCmdBindPipeline(draw_cb, VK_PIPELINE_BIND_POINT_COMPUTE, e.p_candidates);
                    vkCmdBindDescriptorSets(draw_cb, VK_PIPELINE_BIND_POINT_COMPUTE, e.c.pl, 0, 1, &ds, 0, nullptr);
                    vkCmdPushConstants(draw_cb, e.c.pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(test), &test); vkCmdDispatch(draw_cb, D, 1, 1);
                    Engine::mem_barrier(draw_cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_READ_BIT);
                    VkBufferCopy ro{0, 0, D * 4}; vkCmdCopyBuffer(draw_cb, out.b, out_host.b, 1, &ro);
                    Engine::mem_barrier(draw_cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_HOST_BIT, VK_ACCESS_HOST_READ_BIT);
                    VK(vkEndCommandBuffer(draw_cb));
                    for (long done = 0; done < draws; done += D) {
                        e.st[35] = (uint32_t)done; e.c.submit_wait(draw_cb); e.c.invalidate(out_host);
                        const uint32_t *o = (const uint32_t *)out_host.map;
                        for (long k = 0; k < std::min<long>(D, draws - done); ++k) {
                            if (o[k] >= e.V) throw std::runtime_error("GPU top-k/top-p sampler returned invalid token");
                            ++counts[o[k]];
                        }
                    }
                }
            }
            write_bytes(gpu_test[6], counts.data(), counts.size() * 4);
            printf("{\"path\": \"%s\", \"draws\": %ld, \"candidates\": %u}\n", path, draws, cand_count); return 0;
        }
        // ------------------------------------------------ chat / serve
        if (chat || serve) {
            if (!have_tok) throw std::runtime_error("bundle has no tokenizer.json");
#ifdef _WIN32
            if (serve) { _setmode(_fileno(stdin), _O_BINARY); _setmode(_fileno(stdout), _O_BINARY); }
#endif
            opt.positions = (uint32_t)std::min<long>(ctx > 0 ? ctx : (opt.full ? 8192 : max_pos), max_pos); opt.gpu_sampling = true;
            opt.copy_candidates = serve || (sp.top_k > 0 && sp.top_p < 1.0f);
            opt.dynamic_sampling = serve;
            opt.fixed_sampling_mode = sp.temperature <= 0.0f ? 0u : (sp.top_k == 0 && sp.top_p >= 1.0f) ? 1u : 2u;
            if (n_new < 0) n_new = 512;
            if (chat && sp.temperature == 0.0f && sp.top_k == 0) { sp.temperature = 0.7f; sp.top_k = 40; sp.top_p = 0.9f; }
            Engine e; e.init(bundle_dir, opt); Session s(e);
            const std::vector<int> stops = {tok.token_id("<|im_end|>"), tok.token_id("<|endoftext|>")};
            if (chat) {
                out("SeedPlane native chat | " + e.c.name + " | mode " + e.mode_name + " | plan S=" + std::to_string(e.S) + " H=" + std::to_string(e.HALO) +
                    " K=" + std::to_string(e.K) + " | /reset, /exit\n");
                std::string line;
                for (;;) {
                    out("\nVoce> "); if (!read_line_utf8(line)) break;
                    if (line == "/exit") break;
                    if (line == "/reset") { s.reset(); out("(new conversation)\n"); continue; }
                    if (line.empty()) continue;
                    out("Qwen> ");
                    GenResult g = generate(e, s, tok, tok.encode(chat_turn(s, tok, system, line)), sp, n_new, stops, [&](int, const std::string &p) { out(p); });
                    char b[200]; snprintf(b, sizeof b, "\n[%d tokens, %.1f tok/s, prompt %d tokens in %.2fs, position %zu%s]\n", g.generated,
                                          g.generated / std::max(g.decode_s, 1e-9), g.prompt_tokens, g.prefill_s, s.seq.size(), g.reason == "context" ? ", context full" : "");
                    out(b);
                }
                return 0;
            }
            std::string line;                                   // --serve: one JSON request per line, JSON lines back
            out("{\"ready\":true,\"device\":" + json_quote(e.c.name) + ",\"mode\":" + json_quote(e.mode_name) + ",\"plan\":{\"shard\":" +
                std::to_string(e.S) + ",\"halo\":" + std::to_string(e.HALO) + ",\"sinks\":" + std::to_string(e.K) + "},\"positions\":" +
                std::to_string(e.o.positions) + ",\"window_slots\":" + std::to_string(e.W) + "}\n");
            while (std::getline(std::cin, line)) {
                if (!line.empty() && line.back() == '\r') line.pop_back();
                if (line.empty()) continue;
                try {
                    Json q = parse_json(line); std::string op = q.str_or("op", "");
                    if (op == "reset") { s.reset(); out("{\"ok\":true}\n"); continue; }
                    if (op == "tokenize") {
                        auto ids = tok.encode(q.str_or("text", ""), q.num_or("special", 1) != 0); std::string r = "{\"ids\":[";
                        for (size_t k = 0; k < ids.size(); ++k) r += (k ? "," : "") + std::to_string(ids[k]);
                        out(r + "]}\n"); continue;
                    }
                    if (op == "detokenize") {
                        std::vector<int> ids; if (const Json *a = q.get("ids")) for (auto &v : a->arr) ids.push_back((int)v.num);
                        out("{\"text\":" + json_quote(tok.decode(ids, q.num_or("skip_special", 0) != 0)) + "}\n"); continue;
                    }
                    if (op == "state") {
                        std::string r = "{\"position\":" + std::to_string(s.seq.size()) + ",\"processed\":" + std::to_string(s.done) + ",\"tokens\":[";
                        for (size_t k = 0; k < s.seq.size(); ++k) r += (k ? "," : "") + std::to_string(s.seq[k]);
                        out(r + "]}\n"); continue;
                    }
                    if (op != "chat" && op != "generate") throw std::runtime_error("unknown op " + op);
                    if (q.num_or("reset", 0)) s.reset();
                    SamplerParams p = sp;
                    p.temperature = (float)q.num_or("temperature", p.temperature); p.top_k = (int)q.num_or("top_k", p.top_k);
                    p.top_p = (float)q.num_or("top_p", p.top_p); p.seed = (uint64_t)q.num_or("seed", (double)p.seed);
                    std::vector<int> ids;
                    if (op == "chat") ids = tok.encode(chat_turn(s, tok, q.str_or("system", system), q.str_or("content", "")));
                    else if (const Json *a = q.get("ids")) for (auto &v : a->arr) ids.push_back((int)v.num);
                    else ids = tok.encode(q.str_or("text", ""));
                    std::vector<int> stop = stops;
                    if (const Json *a = q.get("stop")) { stop.clear(); for (auto &v : a->arr) stop.push_back((int)v.num); }
                    int max_new = (int)q.num_or("max_new_tokens", n_new);
                    GenResult g = generate(e, s, tok, ids, p, max_new, stop, [&](int t, const std::string &piece) {
                        out("{\"token\":" + std::to_string(t) + ",\"text\":" + json_quote(piece) + "}\n"); });
                    char b[400]; snprintf(b, sizeof b, "{\"done\":true,\"reason\":\"%s\",\"generated\":%d,\"prompt_tokens\":%d,\"prefill_s\":%.6f,"
                        "\"decode_s\":%.6f,\"decode_tok_s\":%.3f,\"position\":%zu}\n", g.reason.c_str(), g.generated, g.prompt_tokens, g.prefill_s,
                        g.decode_s, g.generated / std::max(g.decode_s, 1e-9), s.seq.size());
                    out(b);
                } catch (const std::exception &ex) { out("{\"error\":" + json_quote(ex.what()) + "}\n"); }
            }
            return 0;
        }
        // ------------------------------------------------ benchmark / scoring (V18-V20 format)
        const bool scoring = !score_file.empty();
        const std::vector<uint32_t> prompt = scoring ? read_ids(score_file) : !prompt_file.empty() ? read_ids(prompt_file) : parse_list(prompt_s);
        if (n_new < 0) n_new = 128;
        if (scoring) n_new = 0;
        const uint32_t P = (uint32_t)prompt.size(), total = P + (uint32_t)n_new;
        const bool sampling = sp.temperature > 0.0f;
        opt.scoring = scoring; opt.positions = total; opt.gpu_sampling = sampling && !host_sampling;
        opt.copy_candidates = opt.gpu_sampling && sp.top_k > 0 && sp.top_p < 1.0f;
        opt.fixed_sampling_mode = sp.temperature <= 0.0f ? 0u : (sp.top_k == 0 && sp.top_p >= 1.0f) ? 1u : 2u;
        opt.copy_logits = (sampling && host_sampling) || copy_only;
        Engine e; e.init(bundle_dir, opt);
        using clk = std::chrono::steady_clock; auto secs = [](clk::time_point a) { return std::chrono::duration<double>(clk::now() - a).count(); };
        printf("{\n  \"device\": \"%s\", \"subgroup_size\": %u, \"shared_bytes\": %u, \"dispatches_per_token\": %d, \"load_seconds\": %.3f,\n"
               "  \"mode\": \"%s\", \"attn\": \"%s\", \"fence_wait\": \"%s\", \"plan\": {\"shard\": %u, \"halo\": %u, \"sinks\": %u}, \"window_slots\": %u, \"chunks\": %u,\n"
               "  \"kv_bytes_per_cache\": %llu, \"prompt_tokens\": %u, \"generated_tokens\": %d, \"scoring\": %s,\n"
               "  \"sampling\": {\"temperature\": %g, \"top_k\": %d, \"top_p\": %g, \"seed\": %llu},\n  \"runs\": [\n",
               e.c.name.c_str(), e.c.subgroup, e.c.shared_bytes, e.dispatches_single, e.load_s, e.mode_name, opt.old_attn ? "old" : "split",
               e.c.spin ? "spin" : "sleep", e.S, e.HALO, e.K, e.W, e.nchunk, (unsigned long long)2 * e.L * e.W * e.slot_bytes, P, n_new,
               scoring ? "true" : "false", sp.temperature, sp.top_k, sp.top_p, (unsigned long long)sp.seed);
        for (int r = 0; r < runs; ++r) {
            Session s(e); s.seq = prompt; e.batch_submits = 0; e.fallbacks = 0; e.sampling_counter = 0; Sampler sm(sp); e.set_sampling(sp);
            std::vector<double> lat; lat.reserve((size_t)n_new); double prefill_s = 0, decode_s = 0, lat_max = 0; uint32_t lat_max_pos = 0;
            double step_s = 0, sample_s = 0;   // decode-phase breakdown: GPU step (incl. sampling kernels / copies) vs host sampling
            for (uint32_t i = 0; i < total; ++i) {
                auto t0 = clk::now();
                const bool need = scoring || i + 1 >= P || !e.batch_ok();
                uint32_t next = s.step(need, scoring && i + 1 < total ? s.seq[i + 1] : 0);
                double t_step = secs(t0);
                if (need && sampling && i + 1 >= P) {
                    auto ts = clk::now(); next = (uint32_t)e.next_token(sm, next); sample_s += secs(ts);
                }
                double dt = secs(t0);
                if (i >= P) step_s += t_step;
                if (i < P) prefill_s += dt; else { decode_s += dt; lat.push_back(dt); if (dt > lat_max) { lat_max = dt; lat_max_pos = i; } }
                if (r == 0 && std::find(dump_at.begin(), dump_at.end(), i) != dump_at.end()) {
                    e.grab(e.logits, e.logits_host, (VkDeviceSize)e.V * 4);
                    write_bytes(fs::path(dump_dir) / ("logits_" + std::to_string(i) + ".f32"), e.logits_host.map, (size_t)e.V * 4);
                }
                if (i + 1 >= P && s.seq.size() < total) s.seq.push_back(next);
            }
            double nll_sum = 0;
            if (scoring && total > 1) {
                e.grab(e.nll, e.nll_host, (VkDeviceSize)total * 4); const float *z = (const float *)e.nll_host.map;
                for (uint32_t i = 0; i + 1 < total; ++i) nll_sum += z[i];
                if (r == 0 && !nll_out.empty()) write_bytes(nll_out, z, (size_t)(total - 1) * 4);
            }
            std::vector<double> sl = lat; std::sort(sl.begin(), sl.end());
            auto q = [&](double f) { return sl.empty() ? 0.0 : 1000 * sl[std::min(sl.size() - 1, (size_t)(f * (sl.size() - 1) + 0.5))]; };
            printf("    {\"prefill_seconds\": %.6f, \"decode_seconds\": %.6f, \"decode_tok_s\": %.3f, \"lat_median_ms\": %.4f, \"lat_p99_ms\": %.4f,"
                   " \"lat_max_ms\": %.4f, \"lat_max_pos\": %u, \"boundaries\": %d, \"rebuild_tokens\": %d, \"dual_submits\": %d, \"shadow_batch_tokens\": %d,"
                   " \"batch_submits\": %d, \"nll_mean\": %.8f, \"step_ms_mean\": %.4f, \"sample_ms_mean\": %.4f, \"fallbacks\": %d, \"tokens\": [",
                   prefill_s, decode_s, n_new ? n_new / decode_s : 0.0, q(0.5), q(0.99), 1000 * lat_max, lat_max_pos,
                   s.boundaries, s.extra, s.dual, s.shadow_items, e.batch_submits, total > 1 ? nll_sum / (total - 1) : 0.0,
                   n_new ? 1000 * step_s / n_new : 0.0, n_new ? 1000 * sample_s / n_new : 0.0, e.fallbacks);
            for (size_t i = P; i < s.seq.size(); ++i) printf("%s%u", i > P ? "," : "", s.seq[i]);
            printf("]}%s\n", r + 1 < runs ? "," : "");
        }
        printf("  ]\n}\n");
        return 0;
    } catch (const std::exception &ex) { fprintf(stderr, "error: %s\n", ex.what()); return 1; }
}
