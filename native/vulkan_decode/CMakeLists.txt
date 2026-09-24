# Portable build of the SeedPlane native runtime (Windows, Linux; macOS through MoltenVK is untested).
#   cmake -S native/vulkan_decode -B native/vulkan_decode/build -DCMAKE_BUILD_TYPE=Release
#   cmake --build native/vulkan_decode/build --config Release
# Output: build/qwen_vk[.exe] and build/shaders/*.spv (the engine looks for shaders next to itself).
cmake_minimum_required(VERSION 3.16)
project(seedplane_native CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
find_package(Vulkan REQUIRED)
find_program(GLSLC glslc HINTS "$ENV{VULKAN_SDK}/bin" "$ENV{VULKAN_SDK}/Bin" REQUIRED)

set(SHADER_OUT ${CMAKE_BINARY_DIR}/shaders)
file(MAKE_DIRECTORY ${SHADER_OUT})
set(SPV)
function(shader name src)
  add_custom_command(OUTPUT ${SHADER_OUT}/${name}.spv
    COMMAND ${GLSLC} --target-env=vulkan1.2 -O ${ARGN} ${CMAKE_CURRENT_SOURCE_DIR}/shaders/${src} -o ${SHADER_OUT}/${name}.spv
    DEPENDS shaders/${src} shaders/common.glsl shaders/sample_common.glsl shaders/sample_layout.glsl VERBATIM)
  set(SPV ${SPV} ${SHADER_OUT}/${name}.spv PARENT_SCOPE)
endfunction()
foreach(n 1 2)
  shader(gemv${n} gemv.comp -DNC=${n})
  shader(swiglu${n} swiglu.comp -DNC=${n})
endforeach()
foreach(s embed rope_kv attn attn_part attn_combine argmax nll embed_b rmsnorm_b gemm_b swiglu_b rope_b attn_part_b sample_stats sample_hist sample_hist2 sample_final sample_gumbel sample_candidates)
  shader(${s} ${s}.comp)
endforeach()
add_custom_target(shaders ALL DEPENDS ${SPV})

add_executable(qwen_vk qwen_vk.cpp)
target_link_libraries(qwen_vk PRIVATE Vulkan::Vulkan)
add_dependencies(qwen_vk shaders)
set_target_properties(qwen_vk PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
                      RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_BINARY_DIR} RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CMAKE_BINARY_DIR})
if(MSVC)
  target_compile_options(qwen_vk PRIVATE /utf-8 /EHsc)
else()
  target_compile_options(qwen_vk PRIVATE -O2)
endif()
