#extension GL_KHR_shader_subgroup_basic : require
#extension GL_KHR_shader_subgroup_arithmetic : require
shared float wg_red[64];
// Sum over the whole workgroup; every invocation must call it.
float wg_sum(float v) {
    v = subgroupAdd(v);
    if (subgroupElect()) wg_red[gl_SubgroupID] = v;
    barrier();
    float t = 0.0;
    for (uint i = 0; i < gl_NumSubgroups; ++i) t += wg_red[i];
    barrier();
    return t;
}
