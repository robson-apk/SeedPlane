// Shared by the top-k / top-p sampling passes: reduce the per-workgroup (max, sum exp) partials of sample_stats into
// the full-vocabulary max M (in units of l/T) and Z = sum exp(l/T - M). Every invocation of the workgroup must call it.
shared float red_m[256]; shared float red_s[256];
vec2 reduce_partials(uint t, uint nparts) {
    vec2 v = t < nparts ? part[t] : vec2(-3.4e38, 0.0); red_m[t] = v.x; red_s[t] = v.y; barrier();
    for (uint k = 128; k > 0; k >>= 1) {
        if (t < k) { float a = red_m[t], b = red_m[t + k], mm = max(a, b); red_s[t] = red_s[t] * exp(a - mm) + red_s[t + k] * exp(b - mm); red_m[t] = mm; }
        barrier();
    }
    vec2 r = vec2(red_m[0], red_s[0]); barrier(); return r;
}
