# bc250-encoding-decoding-fix v0.5.1 - https://github.com/simpmix/bc250-encoding-decoding-fix
cmake_minimum_required(VERSION 3.16)
project(bc250-compute-encoder VERSION 0.5.1 LANGUAGES C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Wextra -O2 -fPIC")

option(ENABLE_DEBUG "Enable debug build with sanitizers" OFF)
if(ENABLE_DEBUG)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -g -fsanitize=address,undefined")
endif()

# 32-bit (i386) build, for 32-bit VA-API clients. Steam Link's runtime is
# 32-bit and dlopen()s a 32-bit VA-API driver; a 64-bit bc250_drv_video.so is
# simply invisible to it, so it falls back to software encoding with no useful
# error (see issue #8). This produces the same driver, same filename, from the
# same sources - only the architecture and the install directory differ, so a
# machine can carry both side by side.
#
# Needs 32-bit libva/libdrm/vulkan-loader development packages present, AND a
# SHARED 32-bit libgomp (see the -z text note below):
#   Arch/CachyOS : lib32-libva lib32-libdrm lib32-vulkan-icd-loader lib32-gcc-libs
#   Fedora       : libva-devel.i686 libdrm-devel.i686 vulkan-loader-devel.i686 glibc-devel.i686 libgomp.i686
#   Debian/Ubuntu: gcc-multilib libva-dev:i386 libdrm-dev:i386 libvulkan-dev:i386 libgomp1:i386
option(BUILD_32BIT "Build a 32-bit (i386) driver for 32-bit VA-API clients such as Steam Link" OFF)
if(BUILD_32BIT)
    # Must be set BEFORE any check_c_compiler_flag() below, so those probes
    # test the combination actually being built rather than the 64-bit one.
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -m32")
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -m32")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -m32")

    # Refuse to produce a driver with text relocations. This is not pedantry:
    # the whole point of the 32-bit build is Steam Link, which reaches this
    # driver through dlopen(), and SELinux's deny_execmod denies dlopen() of a
    # library carrying DT_TEXTREL. Such a build succeeds, installs, and then
    # fails to load on exactly the target it was built for - a silent failure
    # of the worst kind.
    #
    # Observed for real while adding this: on a multilib box with no SHARED
    # 32-bit libgomp, -fopenmp quietly links the static libgomp.a, whose
    # non-PIC objects drag in text relocations, and the only complaint is an
    # easily-missed ld warning. -z text promotes that to a hard link error.
    # If it fires, install the shared libgomp for your distro (listed above)
    # rather than removing this flag.
    #
    # Both directions verified on real 32-bit builds, not assumed: with shared
    # libgomp present the driver links clean under this flag and carries no
    # DT_TEXTREL (NEEDED libgomp.so.1, i.e. dynamic); and the flag genuinely
    # rejects a text-relocating object rather than passing it through
    # ("read-only segment has dynamic relocations").
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,-z,text")

    # pkg-config must be pointed at the 32-bit .pc files. Left alone it
    # cheerfully returns the 64-bit -L paths and the link then fails with
    # "skipping incompatible ..." noise that doesn't name the real problem.
    if(IS_DIRECTORY "/usr/lib32/pkgconfig")
        set(ENV{PKG_CONFIG_LIBDIR} "/usr/lib32/pkgconfig:/usr/share/pkgconfig")
    elseif(IS_DIRECTORY "/usr/lib/i386-linux-gnu/pkgconfig")
        set(ENV{PKG_CONFIG_LIBDIR} "/usr/lib/i386-linux-gnu/pkgconfig:/usr/share/pkgconfig")
    elseif(IS_DIRECTORY "/usr/lib/pkgconfig" AND IS_DIRECTORY "/usr/lib64")
        # Fedora/RHEL multilib: on a lib64-style distro /usr/lib IS the 32-bit
        # tree (64-bit lives in /usr/lib64), so i686 .pc files land in
        # /usr/lib/pkgconfig. Testing for /usr/lib64 rather than /usr/lib32 is
        # the point - Fedora has no /usr/lib32 at all, and requiring one here
        # rejected a working multilib container outright.
        # Debian-multiarch is already handled above, so reaching this branch
        # means we're not on one.
        set(ENV{PKG_CONFIG_LIBDIR} "/usr/lib/pkgconfig:/usr/share/pkgconfig")
    else()
        message(FATAL_ERROR
            "BUILD_32BIT=ON but no 32-bit pkgconfig directory was found "
            "(looked for /usr/lib32/pkgconfig, /usr/lib/i386-linux-gnu/pkgconfig, "
            "/usr/lib/pkgconfig). Install the 32-bit libva/libdrm/vulkan development "
            "packages for your distro - see the comment above this check in CMakeLists.txt.")
    endif()
    message(STATUS "BUILD_32BIT: building i386 driver, PKG_CONFIG_LIBDIR=$ENV{PKG_CONFIG_LIBDIR}")
endif()

# Target-CPU codegen: this driver only ever runs on one exact piece of
# silicon, the AMD BC-250 ("Cyan Skillfish", Zen 2) - see README. The
# documented `-DCMAKE_BUILD_TYPE=Release` build leaves CMake's default
# CMAKE_C_FLAGS_RELEASE ("-O3 -DNDEBUG", appended after the -O2 above, so
# -O3 is what actually wins) at the generic x86-64 baseline: GCC/Clang do
# NOT emit BMI2 (pdep/pext/bzhi/shlx/shrx) or AVX2 there even though the
# real hardware supports them - those need an explicit -march.
#
# Measured effect (board, tools/perf_test.sh, 1280x720, 90 frames, 2 runs
# each, SKIP_BUILD builds diffing only this flag): baseline 5.95-5.98 fps /
# 163-165 ms CAVLC+bitstream per frame -> -march=znver2 6.33-6.37 fps /
# 152-153 ms, a reproducible ~6% wall-clock win. objdump confirms real
# codegen change (shlx/shrx/bzhi appear in bitstream.c.o and cavlc.c.o with
# the flag, absent without it) - this is not just a passed-but-inert flag.
# -flto was also measured (alone, and stacked with -march) and made no
# further difference either way, so it is deliberately NOT added here.
# This does not close the real-time gap by itself - CAVLC's O(n) bit-loop
# behavior still dominates total frame time - but it is a free, additive
# win that stacks with that fix, with no correctness or portability cost
# on the one machine this ever runs on.
#
# Auto-detected, not hardcoded: unconditionally forcing -march=znver2 would
# break CI (.github/workflows/build.yml builds AND RUNS ctest/test_encode
# on a generic ubuntu-22.04 runner, which is not znver2 hardware - a
# binary using Zen-2-only instructions there can SIGILL). Only opt in when
# actually configuring directly on real BC-250 hardware (which is also how
# this project is always actually built/installed - see tools/setup_bazzite.sh
# and tools/setup_steamos.sh, both of which run cmake straight on-device).
include(CheckCCompilerFlag)
execute_process(
    COMMAND grep -q "AMD BC-250" /proc/cpuinfo
    RESULT_VARIABLE BC250_IS_TARGET_HW
    OUTPUT_QUIET ERROR_QUIET
)
if(BC250_IS_TARGET_HW EQUAL 0)
    check_c_compiler_flag("-march=znver2" COMPILER_SUPPORTS_ZNVER2)
    if(COMPILER_SUPPORTS_ZNVER2)
        message(STATUS "Detected AMD BC-250 hardware: enabling -march=znver2 -mtune=znver2 (measured ~6% CAVLC/bitstream speedup, see CMakeLists.txt comment)")
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -march=znver2 -mtune=znver2")
    else()
        message(WARNING "Running on AMD BC-250 hardware but this compiler doesn't support -march=znver2 - building generic x86-64 codegen instead")
    endif()
endif()

# Find dependencies
find_package(PkgConfig REQUIRED)
pkg_check_modules(LIBVA REQUIRED libva)
pkg_check_modules(DRM REQUIRED libdrm)

# Vulkan via pkg-config on a 32-bit build: CMake's FindVulkan searches the
# host's 64-bit library paths (and $VULKAN_SDK) and will happily hand back
# /usr/lib/libvulkan.so for an -m32 link. pkg-config is already pointed at the
# 32-bit .pc files above, so it resolves the right one.
if(BUILD_32BIT)
    pkg_check_modules(VULKAN REQUIRED vulkan)
    set(Vulkan_INCLUDE_DIRS ${VULKAN_INCLUDE_DIRS})
    set(Vulkan_LIBRARIES ${VULKAN_LIBRARIES})
    message(STATUS "BUILD_32BIT: Vulkan via pkg-config: ${VULKAN_LIBRARIES}")
else()
    find_package(Vulkan REQUIRED)
endif()

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${Vulkan_INCLUDE_DIRS}
    ${LIBVA_INCLUDE_DIRS}
    ${DRM_INCLUDE_DIRS}
)

# Look for shader compilers
find_program(GLSLANG_VALIDATOR glslangValidator)
find_program(GLSLC glslc)

set(SHADER_DIR ${CMAKE_CURRENT_SOURCE_DIR}/shaders)
file(GLOB SHADERS "${SHADER_DIR}/*.comp")
set(SPV_OUTPUTS "")

# Compile shaders
foreach(SHADER ${SHADERS})
    get_filename_component(SHADER_NAME ${SHADER} NAME)
    set(SPV_FILE "${CMAKE_CURRENT_BINARY_DIR}/${SHADER_NAME}.spv")
    
    if(GLSLANG_VALIDATOR)
        add_custom_command(
            OUTPUT ${SPV_FILE}
            COMMAND ${GLSLANG_VALIDATOR} -V --target-env vulkan1.1 ${SHADER} -o ${SPV_FILE}
            DEPENDS ${SHADER}
            COMMENT "Compiling compute shader ${SHADER_NAME} with glslangValidator"
        )
        list(APPEND SPV_OUTPUTS ${SPV_FILE})
    elseif(GLSLC)
        add_custom_command(
            OUTPUT ${SPV_FILE}
            COMMAND ${GLSLC} --target-env=vulkan1.1 -c ${SHADER} -o ${SPV_FILE}
            DEPENDS ${SHADER}
            COMMENT "Compiling compute shader ${SHADER_NAME} with glslc"
        )
        list(APPEND SPV_OUTPUTS ${SPV_FILE})
    else()
        message(WARNING "Neither glslangValidator nor glslc found. Shaders must be compiled separately.")
    endif()
endforeach()

if(SPV_OUTPUTS)
    add_custom_target(compile_shaders ALL DEPENDS ${SPV_OUTPUTS})
endif()

# Build the VA-API driver
set(SOURCES
    src/va_backend.c
    src/gpu_compute.c
    src/bitstream.c
    src/cavlc.c
    src/cabac.c
    src/rate_control.c
    src/encoder_h264.c
    src/encoder_h265.c
    src/hevc_cabac.c
    src/hevc_intra.c
    src/decoder_h264.c
    src/va_decode.c
    src/decoder_h265.c
    src/va_decode_hevc.c
    src/hevc_ps.c
    src/hevc_cu.c
    src/hevc_residual.c
    src/hevc_transform.c
    src/hevc_pred.c
    src/hevc_filter.c
    src/hevc_mv.c
    src/hevc_mc.c
    src/hevc_tiles.c
    src/hevc_wpp.c
    src/hevc_dec_tables.c
    src/h264_mb_cabac.c
    src/h264_mb_cavlc.c
    src/h264_cavlc_dec.c
    src/h264_mb_motion.c
    src/h264_recon_mb.c
    src/h264_recon.c
    src/h264_pred.c
    src/h264_mc.c
    src/h264_deblock.c
    src/h264_dec_tables.c
    src/h264_threads.c
    src/cpu_simd_me.c
    src/dynamic_governor.c
)

add_library(bc250_drv_video SHARED ${SOURCES})
set_target_properties(bc250_drv_video PROPERTIES
    PREFIX ""
    OUTPUT_NAME "bc250_drv_video"
)
if(TARGET compile_shaders)
    add_dependencies(bc250_drv_video compile_shaders)
endif()

find_package(Threads REQUIRED)
find_package(OpenMP)

set(BC250_EXTRA_LIBS "")
if(TARGET OpenMP::OpenMP_C)
    list(APPEND BC250_EXTRA_LIBS OpenMP::OpenMP_C)
    message(STATUS "OpenMP C found: multi-threaded slice entropy coding enabled")
else()
    message(STATUS "OpenMP C not found: multi-threaded slice entropy coding disabled")
endif()

target_link_libraries(bc250_drv_video
    ${Vulkan_LIBRARIES}
    ${LIBVA_LIBRARIES}
    ${DRM_LIBRARIES}
    ${CMAKE_THREAD_LIBS_INIT}
    m
    # dladdr(), used to find the shaders installed beside this .so. No-op on
    # glibc >= 2.34 where it lives in libc, still needed on older ones.
    ${CMAKE_DL_LIBS}
    ${BC250_EXTRA_LIBS}
)

# H.264 through libx264 - see src/encoder_x264.h for why. Optional: without
# it H.264 is encoded by the compute encoder, as before.
option(BC250_WITH_X264 "Encode H.264 through libx264" ON)
if(BC250_WITH_X264)
    pkg_check_modules(X264 x264)
    if(X264_FOUND)
        target_sources(bc250_drv_video PRIVATE src/encoder_x264.c)
        target_include_directories(bc250_drv_video PRIVATE ${X264_INCLUDE_DIRS})
        target_link_libraries(bc250_drv_video ${X264_LINK_LIBRARIES})
        target_compile_definitions(bc250_drv_video PRIVATE BC250_HAVE_X264)
        message(STATUS "libx264 ${X264_VERSION} found: H.264 encodes through x264")
    else()
        message(WARNING "libx264 not found: H.264 falls back to the compute encoder")
    endif()
endif()

# Installation paths
if(NOT LIBVA_DRIVERS_PATH)
    if(BUILD_32BIT)
        # A 32-bit driver MUST NOT land in the 64-bit DRI directory: libva
        # picks by filename, so a wrong-arch bc250_drv_video.so there would
        # shadow the working 64-bit one and break every 64-bit client.
        if(EXISTS "/usr/lib/i386-linux-gnu/dri")
            set(LIBVA_DRIVERS_PATH "/usr/lib/i386-linux-gnu/dri/")
        else()
            set(LIBVA_DRIVERS_PATH "/usr/lib32/dri/")
        endif()
    elseif(EXISTS "/usr/lib/x86_64-linux-gnu/dri")
        set(LIBVA_DRIVERS_PATH "/usr/lib/x86_64-linux-gnu/dri/")
    elseif(EXISTS "/usr/lib64/dri")
        set(LIBVA_DRIVERS_PATH "/usr/lib64/dri/")
    else()
        set(LIBVA_DRIVERS_PATH "/usr/lib/dri/")
    endif()
endif()

# Architecture-independent: SPIR-V is the same bytes for both builds, so a
# 32-bit and a 64-bit install share this one directory rather than each
# shipping a copy that could drift out of sync with the other's .so.
set(SHADER_INSTALL_PATH "${CMAKE_INSTALL_PREFIX}/share/bc250/shaders")

install(TARGETS bc250_drv_video DESTINATION ${LIBVA_DRIVERS_PATH})

# Refuse to leave a wrong-architecture driver in a DRI directory. libva selects
# by filename, so an ELF64 bc250_drv_video.so sitting in a 32-bit DRI path (or
# the reverse) produces no error anyone can see - it silently shadows the
# correct driver and the client falls back to software encoding. Issue #14 was
# reported as exactly that shape: "the install script seems to copy the 64bit
# version to /usr/lib32/dri".
#
# Checked at install time rather than build time on purpose. The case that
# actually goes wrong is a build tree whose configured architecture no longer
# matches the binary in it, and a POST_BUILD check inside the BUILD_32BIT
# branch cannot see that - if the cache says 64-bit, the check never runs.
#
# Reads the ELF class byte directly (offset 4: 01 = ELF32, 02 = ELF64) instead
# of shelling out to readelf or file. Neither is guaranteed present - this
# project's own Fedora build container ships without `file`.
if(BUILD_32BIT)
    set(BC250_EXPECT_ELF_CLASS "01")
    set(BC250_EXPECT_ARCH_NAME "32-bit (ELF32)")
else()
    set(BC250_EXPECT_ELF_CLASS "02")
    set(BC250_EXPECT_ARCH_NAME "64-bit (ELF64)")
endif()

install(CODE "
    set(_bc250_so \"\$ENV{DESTDIR}${LIBVA_DRIVERS_PATH}/bc250_drv_video.so\")
    if(EXISTS \"\${_bc250_so}\")
        file(READ \"\${_bc250_so}\" _bc250_hdr HEX LIMIT 5)
        string(SUBSTRING \"\${_bc250_hdr}\" 8 2 _bc250_class)
        if(NOT _bc250_class STREQUAL \"${BC250_EXPECT_ELF_CLASS}\")
            message(FATAL_ERROR
                \"Refusing this install: \${_bc250_so} is not ${BC250_EXPECT_ARCH_NAME} \"
                \"(ELF class byte 0x\${_bc250_class}). libva selects drivers by filename, so a \"
                \"wrong-architecture driver in this directory silently shadows the correct one \"
                \"and clients fall back to software encoding with no visible error. This almost \"
                \"always means the build directory was configured for the other architecture - \"
                \"delete it and reconfigure instead of reusing its CMake cache.\")
        endif()
        message(STATUS \"bc250: verified installed driver is ${BC250_EXPECT_ARCH_NAME}\")
    endif()
")
if(SPV_OUTPUTS)
    install(FILES ${SPV_OUTPUTS} DESTINATION ${SHADER_INSTALL_PATH})
endif()

# ----------------------------------------------------------------------------
# rc_bench - off-board rate-control response harness (tools/rc_bench.c)
# ----------------------------------------------------------------------------
option(BUILD_RC_BENCH "Build the off-board rate-control harness (tools/rc_bench.c)" OFF)
if(BUILD_RC_BENCH)
    set(RC_BENCH_SRC ${CMAKE_CURRENT_SOURCE_DIR}/../tools/rc_bench.c)
    if(EXISTS ${RC_BENCH_SRC})
        # Linked against the driver itself rather than a list of its
        # sources: the HEVC encoder predicts through the decoder's motion
        # compensation, so that list would have to be most of the decoder.
        add_executable(rc_bench ${RC_BENCH_SRC})
        target_include_directories(rc_bench PRIVATE src ${Vulkan_INCLUDE_DIRS})
        target_link_libraries(rc_bench PRIVATE bc250_drv_video ${Vulkan_LIBRARIES} ${BC250_EXTRA_LIBS} m)
    endif()
endif()

# Tests
#
# enable_testing() MUST be called here, in the top-level CMakeLists.txt,
# before add_subdirectory(tests) - not only inside tests/CMakeLists.txt.
# CTest only writes a CTestTestfile.cmake (the file `ctest --test-dir <dir>`
# reads to discover tests, including which subdirectories to recurse into)
# for a directory whose CMakeLists.txt itself called enable_testing() (or an
# ancestor did before add_subdirectory'ing it). tests/CMakeLists.txt already
# calls enable_testing(), which correctly generates
# build/tests/CTestTestfile.cmake with all 4 add_test() entries - but with
# no enable_testing() call up here, CMake never generates
# build/CTestTestfile.cmake at all. `ctest --test-dir approach1-compute-encoder/build`
# (the exact command in .github/workflows/build.yml's "Run Test Suite" step)
# only reads the top-level file, so it always printed "No tests were
# found!!!" and exited 0 - meaning test_bitstream, test_cavlc and
# test_va_api were NEVER executed by CI, regardless of the -UNDEBUG fix in
# tests/CMakeLists.txt (confirmed by reproducing this exact CI step
# locally: it reported 0 tests both before and after that fix). Restoring
# enable_testing() here was verified to make all 4 tests discovered and run
# via the unmodified CI command (all pass on a clean tree; CavlcUnitTest
# fails loudly, as expected, when the run_before regression is
# reintroduced into src/cavlc.c).
option(BUILD_TESTS "Build test suite" ON)
if(BUILD_TESTS AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/CMakeLists.txt")
    enable_testing()
    add_subdirectory(tests)
endif()

message(STATUS "VA-API driver target: bc250_drv_video.so")
message(STATUS "Driver install path: ${LIBVA_DRIVERS_PATH}")
message(STATUS "Shader install path: ${SHADER_INSTALL_PATH}")
message(STATUS "Usage: export LIBVA_DRIVER_NAME=bc250")
