// Token sampling with the same definition as seedplane.qwen_engine.Qwen2Engine.sample:
//   temperature <= 0 -> argmax; otherwise z = logits / T; top-k keeps z >= k-th largest (ties kept);
//   top-p keeps the smallest prefix of the sorted distribution whose exclusive cumulative mass is <= p; then sample.
#pragma once
#include <algorithm>
#include <functional>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <random>
#include <vector>

struct SamplerParams { float temperature = 0.0f; int top_k = 0; float top_p = 1.0f; uint64_t seed = 1; };

class Sampler {
public:
    explicit Sampler(const SamplerParams &p) : p_(p), rng_(p.seed) {}
    bool greedy() const { return p_.temperature <= 0.0f; }
    // The filtered distribution (token ids and probabilities; sorted by descending probability when top_p < 1).
    // top-k: one pass with a k-element min-heap, then every token >= the k-th value (ties kept, as in the reference).
    // top-p without top-k: candidates are tokens within 40 nats of the maximum; the excluded tail holds at most
    // V * e^-40 (~6e-13 for Qwen's vocabulary) of the mass, below double-precision relevance for the cumulative cut.
    void distribution(const float *logits, size_t n, std::vector<int> &ids, std::vector<double> &probs) {
        const double T = p_.temperature; ids.clear(); probs.clear(); cand_.clear();
        if (p_.top_k > 0 && (size_t)p_.top_k < n) {
            const size_t k = (size_t)p_.top_k; heap_.assign(logits, logits + k);
            std::make_heap(heap_.begin(), heap_.end(), std::greater<float>());
            for (size_t i = k; i < n; ++i)
                if (logits[i] > heap_.front()) { std::pop_heap(heap_.begin(), heap_.end(), std::greater<float>()); heap_.back() = logits[i];
                                                 std::push_heap(heap_.begin(), heap_.end(), std::greater<float>()); }
            const float cutoff = heap_.front();
            for (size_t i = 0; i < n; ++i) if (logits[i] >= cutoff) cand_.push_back((int)i);
        } else if (p_.top_p < 1.0f) {
            float mx = *std::max_element(logits, logits + n); const float floor_ = mx - (float)(40.0 * T);
            for (size_t i = 0; i < n; ++i) if (logits[i] >= floor_) cand_.push_back((int)i);
        } else { cand_.resize(n); std::iota(cand_.begin(), cand_.end(), 0); }
        double mx = logits[cand_[0]]; for (int i : cand_) mx = std::max(mx, (double)logits[i]);
        double z = 0; for (int i : cand_) z += std::exp((logits[i] - mx) / T);
        if (p_.top_p < 1.0f) {
            auto desc = [&](int a, int b) { return logits[a] > logits[b] || (logits[a] == logits[b] && a < b); };
            size_t sorted = std::min<size_t>(cand_.size(), 1024);
            for (;;) {                                   // sort only the head until it covers the top-p mass
                std::partial_sort(cand_.begin(), cand_.begin() + sorted, cand_.end(), desc);
                double mass = 0; for (size_t i = 0; i < sorted; ++i) mass += std::exp((logits[cand_[i]] - mx) / T) / z;
                if (mass > p_.top_p + 1e-6 || sorted == cand_.size()) break;
                sorted = std::min(cand_.size(), sorted * 4);
            }
            double cum = 0;
            for (size_t i = 0; i < sorted; ++i) {
                if (cum > p_.top_p) break;               // exclusive cumulative mass > p -> dropped
                double q = std::exp((logits[cand_[i]] - mx) / T) / z; ids.push_back(cand_[i]); probs.push_back(q); cum += q;
            }
        } else for (int i : cand_) { ids.push_back(i); probs.push_back(std::exp((logits[i] - mx) / T) / z); }
        double s = 0; for (double q : probs) s += q; for (double &q : probs) q /= s;
    }
    // Same distribution from GPU candidates: every token with logit >= M - 30 T (vals/idx, any order), the
    // full-vocabulary max M and Z = sum exp((l - M) / T). Returns false when the result could depend on tokens outside
    // the candidates (top-p mass not reached); the caller then falls back to distribution() over the full logits.
    // With top-k > candidates, the missing tokens (each < e^-30 of the top token) are dropped (documented approximation).
    bool distribution_from_candidates(const float *vals, const uint32_t *idx, size_t n, float M, float Z,
                                      std::vector<int> &ids, std::vector<double> &probs) {
        const double T = p_.temperature; ids.clear(); probs.clear();
        if (n == 0) return false;
        order_.resize(n); std::iota(order_.begin(), order_.end(), 0);
        auto desc = [&](int a, int b) { return vals[a] > vals[b] || (vals[a] == vals[b] && idx[a] < idx[b]); };
        if (p_.top_k > 0) {
            const size_t kk = std::min(n, (size_t)p_.top_k);
            std::nth_element(order_.begin(), order_.begin() + (kk - 1), order_.end(), desc);
            const float cutoff = vals[order_[kk - 1]];
            size_t kept = (size_t)(std::partition(order_.begin(), order_.end(), [&](int a) { return vals[a] >= cutoff; }) - order_.begin());
            std::sort(order_.begin(), order_.begin() + kept, desc);
            const double mx = vals[order_[0]]; double z = 0;
            for (size_t i = 0; i < kept; ++i) z += std::exp((vals[order_[i]] - mx) / T);
            double cum = 0;
            for (size_t i = 0; i < kept; ++i) {
                if (p_.top_p < 1.0f && cum > p_.top_p) break;
                double q = std::exp((vals[order_[i]] - mx) / T) / z; ids.push_back((int)idx[order_[i]]); probs.push_back(q); cum += q;
            }
        } else {                                                // top-p over the full-vocabulary softmax
            std::sort(order_.begin(), order_.end(), desc);
            double cum = 0; size_t i = 0;
            for (; i < n; ++i) {
                if (cum > p_.top_p) break;
                double q = std::exp(((double)vals[order_[i]] - M) / T) / Z; ids.push_back((int)idx[order_[i]]); probs.push_back(q); cum += q;
            }
            if (i == n && cum <= p_.top_p) return false;        // mass not reached inside the candidates
        }
        double s = 0; for (double q : probs) s += q; for (double &q : probs) q /= s;
        return true;
    }
    const SamplerParams &params() const { return p_; }
    // One draw from a filtered distribution (inverse CDF with a 64-bit uniform).
    int draw(const std::vector<int> &ids, const std::vector<double> &probs) {
        double u = std::uniform_real_distribution<double>(0.0, 1.0)(rng_), acc = 0;
        for (size_t i = 0; i < ids.size(); ++i) { acc += probs[i]; if (u < acc) return ids[i]; }
        return ids.back();
    }
    int sample(const float *logits, size_t n) {
        if (greedy()) { size_t b = 0; for (size_t i = 1; i < n; ++i) if (logits[i] > logits[b]) b = i; return (int)b; }
        distribution(logits, n, ids_, probs_);
        return draw(ids_, probs_);
    }
private:
    SamplerParams p_; std::mt19937_64 rng_; std::vector<int> cand_, ids_, order_; std::vector<double> probs_; std::vector<float> heap_;
};
