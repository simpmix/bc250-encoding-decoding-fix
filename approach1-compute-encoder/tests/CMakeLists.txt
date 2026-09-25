# bc250-encoding-decoding-fix v0.4.3 - https://github.com/simpmix/bc250-encoding-decoding-fix
cmake_minimum_required(VERSION 3.16)

# All four test binaries below validate exclusively via assert() - there is
# no other pass/fail signal (a failed assert() aborts with a nonzero exit
# code; ctest and manual invocation both rely on that). The top-level
# CMakeLists.txt's documented build (`-DCMAKE_BUILD_TYPE=Release`) makes
# CMake append `-DNDEBUG` to CMAKE_C_FLAGS, which compiles every assert()
# in these files into a no-op per the C standard - so a Release build of
# this test suite "passes" unconditionally, even with the exact bugs these
# tests exist to catch reintroduced (confirmed directly: temporarily
# reverting the run_before ordering fix in src/cavlc.c and rebuilding
# test_cavlc via this project's own documented Release command still
# printed "ALL CAVLC CONFORMANCE UNIT TESTS PASSED!"). -UNDEBUG below
# unconditionally re-enables assert() for these test targets only (the
# main bc250_drv_video library keeps NDEBUG/-O3 as configured) regardless
# of CMAKE_BUILD_TYPE.
add_compile_options(-UNDEBUG)

add_executable(test_bitstream test_bitstream.c ../src/bitstream.c)
target_include_directories(test_bitstream PRIVATE ../src)

add_executable(test_va_api test_va_api.c)
target_include_directories(test_va_api PRIVATE ../src ${Vulkan_INCLUDE_DIRS} ${LIBVA_INCLUDE_DIRS} ${DRM_INCLUDE_DIRS})
find_package(Threads REQUIRED)
target_link_libraries(test_va_api PRIVATE bc250_drv_video ${Vulkan_LIBRARIES} ${LIBVA_LIBRARIES} ${DRM_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT} m)

add_executable(test_cavlc test_cavlc.c ../src/cavlc.c ../src/bitstream.c)
target_include_directories(test_cavlc PRIVATE ../src)

add_executable(test_encode test_encode.c)
target_include_directories(test_encode PRIVATE ../src ${Vulkan_INCLUDE_DIRS} ${LIBVA_INCLUDE_DIRS} ${DRM_INCLUDE_DIRS})
target_link_libraries(test_encode PRIVATE bc250_drv_video ${Vulkan_LIBRARIES} ${LIBVA_LIBRARIES} ${DRM_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT} m)
if(TARGET OpenMP::OpenMP_C)
    target_link_libraries(test_encode PRIVATE OpenMP::OpenMP_C)
endif()

# HEVC bitstream structure test - GPU-free (hevc_encoder_encode_raw(), see
# encoder_h265.h), same "no Vulkan context needed" pattern as test_encode's
# H.264 path. Still links the full driver library since encoder_h265.c
# references gpu_compute_*() symbols (used only by the GPU-surface entry
# point this test doesn't call, but the linker still needs them resolved).
add_executable(test_hevc_encode test_hevc_encode.c)
target_include_directories(test_hevc_encode PRIVATE ../src ${Vulkan_INCLUDE_DIRS} ${LIBVA_INCLUDE_DIRS} ${DRM_INCLUDE_DIRS})
target_link_libraries(test_hevc_encode PRIVATE bc250_drv_video ${Vulkan_LIBRARIES} ${LIBVA_LIBRARIES} ${DRM_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT} m)

# Unit tests for Dynamic Governor and CPU SIMD Motion Estimation
add_executable(test_cpu_simd_me test_cpu_simd_me.c ../src/cpu_simd_me.c)
target_include_directories(test_cpu_simd_me PRIVATE ../src ${Vulkan_INCLUDE_DIRS})
target_link_libraries(test_cpu_simd_me PRIVATE m ${CMAKE_THREAD_LIBS_INIT})
if(TARGET OpenMP::OpenMP_C)
    target_link_libraries(test_cpu_simd_me PRIVATE OpenMP::OpenMP_C)
endif()

add_executable(test_dynamic_governor test_dynamic_governor.c ../src/dynamic_governor.c)
target_include_directories(test_dynamic_governor PRIVATE ../src)

# The H.264 decoder's own unit tests. GPU-free: the decoder reconstructs
# into plain system memory and only touches the GPU to upload a finished
# picture, so these need nothing but libc.
add_executable(test_cabac_roundtrip test_cabac_roundtrip.c
               ../src/cabac.c ../src/h264_dec_tables.c)
target_include_directories(test_cabac_roundtrip PRIVATE ../src)

add_executable(test_mc test_mc.c ../src/h264_mc.c)
target_include_directories(test_mc PRIVATE ../src)
target_link_libraries(test_mc PRIVATE m)

add_executable(test_pred test_pred.c ../src/h264_pred.c)
target_include_directories(test_pred PRIVATE ../src)
target_link_libraries(test_pred PRIVATE m)

# The H.265 inverse transforms against the matrices of clause 8.6.4.2.
add_executable(test_hevc_transform test_hevc_transform.c
               ../src/hevc_transform.c ../src/hevc_dec_tables.c)
target_include_directories(test_hevc_transform PRIVATE ../src)
target_link_libraries(test_hevc_transform PRIVATE m)

enable_testing()
add_test(NAME BitstreamTest COMMAND test_bitstream)
add_test(NAME CavlcUnitTest COMMAND test_cavlc)
add_test(NAME VaApiDriverTest COMMAND test_va_api)
add_test(NAME EncodeBitstreamTest COMMAND test_encode WORKING_DIRECTORY ${CMAKE_BINARY_DIR})
add_test(NAME HevcEncodeBitstreamTest COMMAND test_hevc_encode WORKING_DIRECTORY ${CMAKE_BINARY_DIR})
add_test(NAME CpuSimdMeTest COMMAND test_cpu_simd_me)
add_test(NAME DynamicGovernorTest COMMAND test_dynamic_governor)
add_test(NAME CabacRoundtripTest COMMAND test_cabac_roundtrip)
add_test(NAME McUnitTest COMMAND test_mc)
add_test(NAME IntraPredUnitTest COMMAND test_pred)
add_test(NAME HevcTransformUnitTest COMMAND test_hevc_transform)

# The libx264 backend, when the driver was built with it.
if(X264_FOUND)
    add_executable(test_x264 test_x264.c)
    target_include_directories(test_x264 PRIVATE ../src ${X264_INCLUDE_DIRS})
    target_link_libraries(test_x264 PRIVATE bc250_drv_video ${X264_LINK_LIBRARIES} ${Vulkan_LIBRARIES} ${LIBVA_LIBRARIES} ${DRM_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT} m)
    add_test(NAME X264BackendTest COMMAND test_x264)
endif()
