/* bc250-vulkan-encode-stopgap - https://github.com/Shalasere/bc250-vulkan-encode-stopgap */
/*
 * Copyright (c) 2026 BC-250 Project
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * gpu_contention.c - a tunable, self-calibrating Vulkan compute load
 * generator standing in for "a game using the GPU", for `tools/lab`'s
 * --load= condition (docs/backlog.md, the assessment that led here).
 *
 * WHY THIS EXISTS
 * ---------------
 * The only GPU load generator this harness had before this file was
 * ffmpeg's `nlmeans_vulkan` filter, and the project's own docs already
 * flagged it as "a pathologically heavy compute filter, almost certainly
 * harsher than a game... a synthetic worst case, not a 'what a game does'
 * number" (README's Known Limitations, backlog.md). It was never designed
 * to model GPU contention and was never calibrated against anything real.
 *
 * The real-time-systems literature has an established answer for this
 * class of problem: a *contention generator* - a synthetic, tunable
 * workload whose intensity is calibrated against a measured target
 * (typically GPU busy%) rather than picked arbitrarily. This is that
 * generator, tuned by wall-clock duty cycle instead of the OS-reported
 * busy% this board's amdgpu happens not to expose (see below).
 *
 * WHY DUTY CYCLE, NOT gpu_busy_percent
 * -------------------------------------
 * The obvious calibration target would be amdgpu's own
 * `gpu_busy_percent` sysfs node. On this board it returns `Operation not
 * supported` (checked directly, 2026-09-22) - a real hardware/driver gap,
 * not a permissions issue (confirmed present as a file, not merely
 * unreadable). `pp_dpm_sclk` only reports which of 3 coarse DPM rungs is
 * active, nowhere near fine enough to target a percentage. So this tool
 * measures its OWN duty cycle instead: the fraction of wall-clock time its
 * own dispatches are actually executing on the GPU, timed with Vulkan
 * timestamp queries, not read from any OS counter. That is the same
 * underlying quantity "GPU busy%" means at the hardware level - just
 * self-instrumented because the vendor counter for it doesn't work here.
 * Report this honestly as "self-measured duty cycle of our own submitted
 * work", never as "board-reported GPU utilization" - they are not the
 * same provenance.
 *
 * HOW THE TARGET DUTY CYCLE IS HELD
 * ----------------------------------
 * Each cycle: dispatch a large, ALU-bound, provably-live compute shader
 * (see shaders/gpu_contention.comp's own header for why it can't be
 * optimized away), time it with a timestamp query pair, then sleep long
 * enough that busy_ns / (busy_ns + sleep_ns) == the target duty cycle.
 * Two corrections keep this honest under real conditions rather than
 * trusting one-shot arithmetic:
 *
 *  - PROPORTIONAL FEEDBACK against the *cumulative* realized duty cycle
 *    (running_busy_ns / elapsed-since-start), not just this cycle's own
 *    math - nanosleep() reliably oversleeps by some scheduler-dependent
 *    slop, which would otherwise bias the realized duty cycle below
 *    target indefinitely.
 *  - PERIODIC RECALIBRATION of the iteration count when a dispatch drifts
 *    far outside a workable timing window (too short to time accurately,
 *    too long to hold a stable duty cycle) - this also absorbs the DPM
 *    clock changing mid-run, which this generator's own load is what
 *    would trigger in the first place.
 *
 * USAGE
 * -----
 *   gpu_contention [--duty=70] [--duration=30] [--groups=4096]
 *                  [--report=2] [--shader=PATH]
 *
 * Exit 0 after --duration seconds, having printed periodic status lines
 * and a final summary with the realized (measured, not assumed) duty
 * cycle for the whole run.
 */
#include <vulkan/vulkan.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <time.h>
#include <unistd.h>
#include <libgen.h>
#include <errno.h>

#define AMD_VENDOR_ID   0x1002
#define BC250_DEVICE_ID 0x13FE

#define VK_CHECK(x) do { \
    VkResult _r = (x); \
    if (_r != VK_SUCCESS) { \
        fprintf(stderr, "gpu_contention: %s failed, VkResult=%d (%s:%d)\n", #x, (int)_r, __FILE__, __LINE__); \
        exit(1); \
    } \
} while (0)

static uint64_t now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
}

static void sleep_ns(uint64_t ns) {
    if (ns == 0) return;
    struct timespec ts = { .tv_sec = (time_t)(ns / 1000000000ull), .tv_nsec = (long)(ns % 1000000000ull) };
    while (nanosleep(&ts, &ts) == -1) { /* EINTR: ts already holds the remainder */ }
}

/* Resolve "gpu_contention.comp.spv" next to this executable, the same
 * "ship it beside the binary" convention CLAUDE.md requires for the
 * driver's own shaders, via /proc/self/exe rather than dladdr() (this is
 * an executable, not a shared library the dynamic linker maps). */
static void default_shader_path(char *out, size_t out_len) {
    char exe[4096];
    ssize_t n = readlink("/proc/self/exe", exe, sizeof(exe) - 1);
    if (n <= 0) { snprintf(out, out_len, "gpu_contention.comp.spv"); return; }
    exe[n] = '\0';
    char *dir = dirname(exe); /* dirname() may modify its argument; exe is a local copy already */
    snprintf(out, out_len, "%s/gpu_contention.comp.spv", dir);
}

static uint32_t *read_spirv(const char *path, size_t *out_words) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "gpu_contention: cannot open shader '%s': %s\n", path, strerror(errno)); exit(1); }
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    uint32_t *buf = malloc((size_t)sz);
    if (fread(buf, 1, (size_t)sz, f) != (size_t)sz) { fprintf(stderr, "gpu_contention: short read on '%s'\n", path); exit(1); }
    fclose(f);
    *out_words = (size_t)sz / sizeof(uint32_t);
    return buf;
}

int main(int argc, char **argv) {
    double target_duty = 0.70;
    double duration_s = 30.0;
    double report_s = 2.0;
    uint32_t groups = 4096;
    char shader_path[4096];
    default_shader_path(shader_path, sizeof(shader_path));

    for (int i = 1; i < argc; i++) {
        if      (!strncmp(argv[i], "--duty=", 7))     target_duty = atof(argv[i] + 7) / 100.0;
        else if (!strncmp(argv[i], "--duration=", 11)) duration_s = atof(argv[i] + 11);
        else if (!strncmp(argv[i], "--groups=", 9))    groups = (uint32_t)atoi(argv[i] + 9);
        else if (!strncmp(argv[i], "--report=", 9))    report_s = atof(argv[i] + 9);
        else if (!strncmp(argv[i], "--shader=", 9))    snprintf(shader_path, sizeof(shader_path), "%s", argv[i] + 9);
        else { fprintf(stderr, "usage: %s [--duty=70] [--duration=30] [--groups=4096] [--report=2] [--shader=PATH]\n", argv[0]); return 2; }
    }
    if (target_duty <= 0.0 || target_duty >= 1.0) { fprintf(stderr, "gpu_contention: --duty must be in (0,100)\n"); return 2; }

    /* ---- Vulkan init: instance, device pick (same priority as
     * bc250_gpu_init(): BC-250 by device ID, else any AMD, else whatever's
     * first - but none of that function's dma-buf/semaphore/priority
     * machinery, which this tool has no use for). ---- */
    VkApplicationInfo app_info = { .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO, .pApplicationName = "gpu_contention", .apiVersion = VK_API_VERSION_1_2 };
    VkInstanceCreateInfo inst_info = { .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, .pApplicationInfo = &app_info };
    VkInstance instance;
    VK_CHECK(vkCreateInstance(&inst_info, NULL, &instance));

    uint32_t dev_count = 0;
    vkEnumeratePhysicalDevices(instance, &dev_count, NULL);
    if (dev_count == 0) { fprintf(stderr, "gpu_contention: no Vulkan physical devices\n"); return 1; }
    VkPhysicalDevice *devices = malloc(dev_count * sizeof(VkPhysicalDevice));
    vkEnumeratePhysicalDevices(instance, &dev_count, devices);

    VkPhysicalDevice phys = VK_NULL_HANDLE;
    VkPhysicalDeviceProperties props;
    for (uint32_t i = 0; i < dev_count; i++) {
        vkGetPhysicalDeviceProperties(devices[i], &props);
        if (props.deviceID == BC250_DEVICE_ID) { phys = devices[i]; break; }
    }
    if (!phys) for (uint32_t i = 0; i < dev_count; i++) {
        vkGetPhysicalDeviceProperties(devices[i], &props);
        if (props.vendorID == AMD_VENDOR_ID) { phys = devices[i]; break; }
    }
    if (!phys) { phys = devices[0]; vkGetPhysicalDeviceProperties(devices[0], &props); }
    free(devices);
    fprintf(stderr, "gpu_contention: using %s\n", props.deviceName);
    double timestamp_period_ns = (double)props.limits.timestampPeriod;

    uint32_t qf_count = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(phys, &qf_count, NULL);
    VkQueueFamilyProperties *qf = malloc(qf_count * sizeof(VkQueueFamilyProperties));
    vkGetPhysicalDeviceQueueFamilyProperties(phys, &qf_count, qf);
    uint32_t family = UINT32_MAX;
    /* Prefer a queue family that reports usable timestamp bits - without
     * that, the whole calibration scheme below has nothing to measure. */
    for (uint32_t i = 0; i < qf_count; i++)
        if ((qf[i].queueFlags & VK_QUEUE_COMPUTE_BIT) && qf[i].timestampValidBits > 0) { family = i; break; }
    if (family == UINT32_MAX)
        for (uint32_t i = 0; i < qf_count; i++)
            if (qf[i].queueFlags & VK_QUEUE_COMPUTE_BIT) { family = i; break; }
    free(qf);
    if (family == UINT32_MAX) { fprintf(stderr, "gpu_contention: no compute queue family\n"); return 1; }

    float prio = 1.0f;
    VkDeviceQueueCreateInfo q_info = { .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO, .queueFamilyIndex = family, .queueCount = 1, .pQueuePriorities = &prio };
    VkDeviceCreateInfo dev_info = { .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, .queueCreateInfoCount = 1, .pQueueCreateInfos = &q_info };
    VkDevice dev;
    VK_CHECK(vkCreateDevice(phys, &dev_info, NULL, &dev));
    VkQueue queue;
    vkGetDeviceQueue(dev, family, 0, &queue);

    /* ---- Sink buffer: one uint, DEVICE_LOCAL only - the host never
     * reads it mid-run, it exists purely so every shader invocation has a
     * real, visible side effect to write (see the shader's own comment). */
    VkBufferCreateInfo buf_info = { .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = 4, .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, .sharingMode = VK_SHARING_MODE_EXCLUSIVE };
    VkBuffer sink_buf;
    VK_CHECK(vkCreateBuffer(dev, &buf_info, NULL, &sink_buf));
    VkMemoryRequirements mreq;
    vkGetBufferMemoryRequirements(dev, sink_buf, &mreq);
    VkPhysicalDeviceMemoryProperties mprops;
    vkGetPhysicalDeviceMemoryProperties(phys, &mprops);
    uint32_t mem_type = UINT32_MAX;
    for (uint32_t i = 0; i < mprops.memoryTypeCount; i++) {
        if ((mreq.memoryTypeBits & (1u << i)) && (mprops.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)) { mem_type = i; break; }
    }
    if (mem_type == UINT32_MAX) mem_type = __builtin_ctz(mreq.memoryTypeBits); /* any type the buffer accepts */
    VkMemoryAllocateInfo mem_alloc = { .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, .allocationSize = mreq.size, .memoryTypeIndex = mem_type };
    VkDeviceMemory sink_mem;
    VK_CHECK(vkAllocateMemory(dev, &mem_alloc, NULL, &sink_mem));
    VK_CHECK(vkBindBufferMemory(dev, sink_buf, sink_mem, 0));

    /* ---- Pipeline: one storage-buffer binding, one uint push constant ---- */
    VkDescriptorSetLayoutBinding dsl_binding = { .binding = 0, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT };
    VkDescriptorSetLayoutCreateInfo dsl_info = { .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, .bindingCount = 1, .pBindings = &dsl_binding };
    VkDescriptorSetLayout dsl;
    VK_CHECK(vkCreateDescriptorSetLayout(dev, &dsl_info, NULL, &dsl));

    VkPushConstantRange pc_range = { .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT, .offset = 0, .size = sizeof(uint32_t) };
    VkPipelineLayoutCreateInfo pl_info = { .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, .setLayoutCount = 1, .pSetLayouts = &dsl, .pushConstantRangeCount = 1, .pPushConstantRanges = &pc_range };
    VkPipelineLayout pipeline_layout;
    VK_CHECK(vkCreatePipelineLayout(dev, &pl_info, NULL, &pipeline_layout));

    size_t spv_words;
    uint32_t *spv = read_spirv(shader_path, &spv_words);
    VkShaderModuleCreateInfo sm_info = { .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, .codeSize = spv_words * sizeof(uint32_t), .pCode = spv };
    VkShaderModule shader;
    VK_CHECK(vkCreateShaderModule(dev, &sm_info, NULL, &shader));
    free(spv);

    VkComputePipelineCreateInfo pipe_info = {
        .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        .stage = { .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, .stage = VK_SHADER_STAGE_COMPUTE_BIT, .module = shader, .pName = "main" },
        .layout = pipeline_layout
    };
    VkPipeline pipeline;
    VK_CHECK(vkCreateComputePipelines(dev, VK_NULL_HANDLE, 1, &pipe_info, NULL, &pipeline));

    VkDescriptorPoolSize dp_size = { .type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 1 };
    VkDescriptorPoolCreateInfo dp_info = { .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO, .maxSets = 1, .poolSizeCount = 1, .pPoolSizes = &dp_size };
    VkDescriptorPool dpool;
    VK_CHECK(vkCreateDescriptorPool(dev, &dp_info, NULL, &dpool));
    VkDescriptorSetAllocateInfo ds_alloc = { .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO, .descriptorPool = dpool, .descriptorSetCount = 1, .pSetLayouts = &dsl };
    VkDescriptorSet dset;
    VK_CHECK(vkAllocateDescriptorSets(dev, &ds_alloc, &dset));
    VkDescriptorBufferInfo db_info = { .buffer = sink_buf, .offset = 0, .range = 4 };
    VkWriteDescriptorSet write = { .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = dset, .dstBinding = 0, .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .pBufferInfo = &db_info };
    vkUpdateDescriptorSets(dev, 1, &write, 0, NULL);

    VkCommandPoolCreateInfo cp_info = { .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO, .queueFamilyIndex = family, .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT };
    VkCommandPool cpool;
    VK_CHECK(vkCreateCommandPool(dev, &cp_info, NULL, &cpool));
    VkCommandBufferAllocateInfo cb_alloc = { .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, .commandPool = cpool, .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY, .commandBufferCount = 1 };
    VkCommandBuffer cmd;
    VK_CHECK(vkAllocateCommandBuffers(dev, &cb_alloc, &cmd));

    VkQueryPoolCreateInfo qp_info = { .sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO, .queryType = VK_QUERY_TYPE_TIMESTAMP, .queryCount = 2 };
    VkQueryPool qpool;
    VK_CHECK(vkCreateQueryPool(dev, &qp_info, NULL, &qpool));

    VkFenceCreateInfo fence_info = { .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO };
    VkFence fence;
    VK_CHECK(vkCreateFence(dev, &fence_info, NULL, &fence));

    /* ---- Calibration: pick an iteration count that lands a single
     * dispatch in an 8-16ms window - long enough that submission/query
     * overhead is negligible against it, short enough that the duty-cycle
     * sleep below has fine-grained control. Starts from a deliberately
     * tiny guess and scales by direct proportion (dispatch cost is linear
     * in iterations by construction - a dependent ALU chain, no shortcuts
     * available to the compiler or the hardware). ---- */
    uint32_t iters = 20000;
    const double WINDOW_LO = 0.008, WINDOW_HI = 0.016, WINDOW_MID = 0.012;

    uint64_t run_start = now_ns();
    uint64_t running_busy_ns = 0;
    uint64_t last_report = run_start;
    uint64_t last_calib_check = run_start;
    int cycles = 0;

    fprintf(stderr, "gpu_contention: target duty=%.0f%%, groups=%u, shader=%s\n", target_duty * 100.0, groups, shader_path);

    while ((double)(now_ns() - run_start) / 1e9 < duration_s) {
        VK_CHECK(vkResetCommandBuffer(cmd, 0));
        VkCommandBufferBeginInfo begin = { .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT };
        VK_CHECK(vkBeginCommandBuffer(cmd, &begin));
        vkCmdResetQueryPool(cmd, qpool, 0, 2);
        vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, qpool, 0);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_layout, 0, 1, &dset, 0, NULL);
        vkCmdPushConstants(cmd, pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(uint32_t), &iters);
        vkCmdDispatch(cmd, groups, 1, 1);
        vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, qpool, 1);
        VK_CHECK(vkEndCommandBuffer(cmd));

        VK_CHECK(vkResetFences(dev, 1, &fence));
        VkSubmitInfo submit = { .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .commandBufferCount = 1, .pCommandBuffers = &cmd };
        VK_CHECK(vkQueueSubmit(queue, 1, &submit, fence));
        VK_CHECK(vkWaitForFences(dev, 1, &fence, VK_TRUE, UINT64_MAX));

        uint64_t ts[2];
        VK_CHECK(vkGetQueryPoolResults(dev, qpool, 0, 2, sizeof(ts), ts, sizeof(uint64_t), VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT));
        uint64_t busy_ns = (uint64_t)((double)(ts[1] - ts[0]) * timestamp_period_ns);
        running_busy_ns += busy_ns;
        cycles++;

        /* Base sleep from this cycle's own measurement, then a small
         * proportional correction against the CUMULATIVE realized duty
         * cycle so nanosleep()'s habitual oversleep doesn't bias the
         * long-run average below target - see the file header comment. */
        double base_sleep_ns = (double)busy_ns * (1.0 - target_duty) / target_duty;
        double elapsed_s = (double)(now_ns() - run_start) / 1e9;
        double realized_duty = elapsed_s > 0 ? (double)running_busy_ns / 1e9 / elapsed_s : target_duty;
        double error = target_duty - realized_duty; /* >0 : we're running too little, sleep less */
        double correction_ns = -error * base_sleep_ns * 2.0; /* gain=2: converges within a few cycles without oscillating at this duty range */
        double sleep_target_ns = base_sleep_ns + correction_ns;
        if (sleep_target_ns < 0) sleep_target_ns = 0;
        sleep_ns((uint64_t)sleep_target_ns);

        /* Recalibrate iters if the dispatch has drifted well outside the
         * timing window - most likely because our own load just pushed
         * the GPU onto a different DPM clock rung. */
        double busy_s = (double)busy_ns / 1e9;
        if ((now_ns() - last_calib_check) > 1000000000ull && (busy_s < WINDOW_LO || busy_s > WINDOW_HI) && busy_s > 0) {
            uint32_t new_iters = (uint32_t)((double)iters * WINDOW_MID / busy_s);
            if (new_iters < 1000) new_iters = 1000;
            iters = new_iters;
            last_calib_check = now_ns();
        }

        if ((now_ns() - last_report) > (uint64_t)(report_s * 1e9)) {
            fprintf(stderr, "gpu_contention: t=%.1fs cycles=%d iters=%u dispatch=%.2fms realized_duty=%.1f%%\n",
                    elapsed_s, cycles, iters, busy_s * 1000.0, realized_duty * 100.0);
            last_report = now_ns();
        }
    }

    double total_s = (double)(now_ns() - run_start) / 1e9;
    double final_duty = (double)running_busy_ns / 1e9 / total_s;
    fprintf(stderr, "gpu_contention: DONE. %d cycles over %.1fs, realized duty cycle = %.2f%% (target %.0f%%)\n",
            cycles, total_s, final_duty * 100.0, target_duty * 100.0);

    vkDestroyFence(dev, fence, NULL);
    vkDestroyQueryPool(dev, qpool, NULL);
    vkDestroyCommandPool(dev, cpool, NULL);
    vkDestroyDescriptorPool(dev, dpool, NULL);
    vkDestroyPipeline(dev, pipeline, NULL);
    vkDestroyPipelineLayout(dev, pipeline_layout, NULL);
    vkDestroyDescriptorSetLayout(dev, dsl, NULL);
    vkDestroyShaderModule(dev, shader, NULL);
    vkFreeMemory(dev, sink_mem, NULL);
    vkDestroyBuffer(dev, sink_buf, NULL);
    vkDestroyDevice(dev, NULL);
    vkDestroyInstance(instance, NULL);
    return 0;
}
