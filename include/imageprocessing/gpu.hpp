#pragma once

#include "imageprocessing/convolution.hpp"

namespace imageprocessing {

enum class Kernel { Naive, Constant, Tiled };
enum class HostMemory { Pageable, Pinned };

struct GpuConfig {
    Kernel kernel = Kernel::Naive;
    HostMemory host_memory = HostMemory::Pageable;
    int iterations = 100;
    int warmup = 10;
};

struct GpuResult {
    Image output;
    double allocation_ms = 0;
    double h2d_ms = 0;
    double filter_upload_ms = 0;
    double kernel_ms = 0;
    double d2h_ms = 0;
    double pipeline_ms = 0;
};

struct DeviceInfo {
    std::string name;
    std::size_t memory_bytes = 0;
    int major = 0;
    int minor = 0;
    int runtime_version = 0;
    int driver_version = 0;
};

bool gpu_available();
DeviceInfo device_info();
const char* kernel_name(Kernel kernel);
const char* memory_name(HostMemory memory);
GpuResult convolve_gpu(const Image& image, const Filter& filter, const GpuConfig& config);

}  // namespace imageprocessing
