@echo off
rem Build the SeedPlane native Vulkan decoder (MSVC + Vulkan SDK).
setlocal
if "%VULKAN_SDK%"=="" set "VULKAN_SDK=E:\Vulkan SDK 1.4.357.0"
if "%VCVARS%"=="" set "VCVARS=C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
call "%VCVARS%" >nul || exit /b 1
cd /d "%~dp0"
if not exist build\shaders mkdir build\shaders
for %%n in (1 2) do for %%s in (gemv swiglu) do (
  "%VULKAN_SDK%\Bin\glslc.exe" --target-env=vulkan1.2 -O -DNC=%%n shaders\%%s.comp -o build\shaders\%%s%%n.spv || exit /b 1
)
for %%s in (embed rope_kv attn attn_part attn_combine argmax nll embed_b rmsnorm_b gemm_b swiglu_b rope_b attn_part_b sample_stats sample_hist sample_hist2 sample_final sample_gumbel sample_candidates) do (
  "%VULKAN_SDK%\Bin\glslc.exe" --target-env=vulkan1.2 -O shaders\%%s.comp -o build\shaders\%%s.spv || exit /b 1
)
cl /nologo /O2 /EHsc /std:c++17 /utf-8 /arch:AVX2 /I"%VULKAN_SDK%\Include" qwen_vk.cpp /Fo:build\ /Fe:build\qwen_vk.exe /link /LIBPATH:"%VULKAN_SDK%\Lib" vulkan-1.lib || exit /b 1
echo BUILD OK
