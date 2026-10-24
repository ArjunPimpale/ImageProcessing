#include "imageprocessing/gpu.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <memory>
#include <stdexcept>

namespace imageprocessing {
namespace {

using Clock = std::chrono::steady_clock;
double milliseconds(Clock::time_point start) {
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

void check(cudaError_t error, const char* operation) {
    if (error != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(error));
    }
}

class DeviceBuffer {
public:
    explicit DeviceBuffer(std::size_t count) {
        check(cudaMalloc(reinterpret_cast<void**>(&data_), count * sizeof(float)), "cudaMalloc");
    }
    ~DeviceBuffer() { cudaFree(data_); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    float* get() const { return data_; }
private:
    float* data_ = nullptr;
};

class HostBuffer {
public:
    HostBuffer(std::size_t count, HostMemory kind) : pinned_(kind == HostMemory::Pinned) {
        if (pinned_) {
            check(cudaMallocHost(reinterpret_cast<void**>(&data_), count * sizeof(float)), "cudaMallocHost");
        } else {
            pageable_ = std::make_unique<float[]>(count);
            data_ = pageable_.get();
        }
    }
    ~HostBuffer() { if (pinned_) cudaFreeHost(data_); }
    HostBuffer(const HostBuffer&) = delete;
    HostBuffer& operator=(const HostBuffer&) = delete;
    float* get() const { return data_; }
private:
    bool pinned_;
    float* data_ = nullptr;
    std::unique_ptr<float[]> pageable_;
};

class Event {
public:
    Event() { check(cudaEventCreate(&event_), "cudaEventCreate"); }
    ~Event() { cudaEventDestroy(event_); }
    Event(const Event&) = delete;
    Event& operator=(const Event&) = delete;
    cudaEvent_t get() const { return event_; }
private:
    cudaEvent_t event_{};
};

__constant__ float constant_filter[9];
constexpr int tile = 16;
constexpr int input_tile = tile + 2;

template <bool use_constant>
__global__ void direct_kernel(const float* input, const float* filter, float* output,
                              int width, int height) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    float sum = 0;
    #pragma unroll
    for (int fy = 0; fy < 3; ++fy) {
        #pragma unroll
        for (int fx = 0; fx < 3; ++fx) {
            const int ix = x + fx - 1;
            const int iy = y + fy - 1;
            if (ix >= 0 && ix < width && iy >= 0 && iy < height) {
                const float weight = use_constant ? constant_filter[fy * 3 + fx] : filter[fy * 3 + fx];
                sum += weight * input[static_cast<std::size_t>(iy) * width + ix];
            }
        }
    }
    output[static_cast<std::size_t>(y) * width + x] = sum;
}

__global__ void tiled_kernel(const float* input, float* output, int width, int height) {
    __shared__ float pixels[input_tile][input_tile];
    // All threads participate in loading the tile and its one-pixel halo.
    const int thread = threadIdx.y * tile + threadIdx.x;
    for (int index = thread; index < input_tile * input_tile; index += tile * tile) {
        const int sy = index / input_tile;
        const int sx = index % input_tile;
        const int ix = static_cast<int>(blockIdx.x) * tile + sx - 1;
        const int iy = static_cast<int>(blockIdx.y) * tile + sy - 1;
        pixels[sy][sx] = (ix >= 0 && ix < width && iy >= 0 && iy < height)
            ? input[static_cast<std::size_t>(iy) * width + ix] : 0.0f;
    }
    __syncthreads();
    const int x = blockIdx.x * tile + threadIdx.x;
    const int y = blockIdx.y * tile + threadIdx.y;
    // The boundary return must follow the barrier, including in partial blocks.
    if (x >= width || y >= height) return;
    float sum = 0;
    #pragma unroll
    for (int fy = 0; fy < 3; ++fy) {
        #pragma unroll
        for (int fx = 0; fx < 3; ++fx) {
            sum += constant_filter[fy * 3 + fx] * pixels[threadIdx.y + fy][threadIdx.x + fx];
        }
    }
    output[static_cast<std::size_t>(y) * width + x] = sum;
}

void launch(Kernel kernel, const float* input, const float* filter, float* output, int width, int height) {
    const dim3 block(tile, tile);
    const dim3 grid((width + tile - 1) / tile, (height + tile - 1) / tile);
    switch (kernel) {
        case Kernel::Naive:
            direct_kernel<false><<<grid, block>>>(input, filter, output, width, height);
            break;
        case Kernel::Constant:
            direct_kernel<true><<<grid, block>>>(input, filter, output, width, height);
            break;
        case Kernel::Tiled:
            tiled_kernel<<<grid, block>>>(input, output, width, height);
            break;
    }
    check(cudaGetLastError(), "Convolution kernel launch");
}

}  // namespace

const char* kernel_name(Kernel kernel) {
    switch (kernel) {
        case Kernel::Naive: return "naive";
        case Kernel::Constant: return "constant";
        case Kernel::Tiled: return "tiled";
    }
    return "unknown";
}

const char* memory_name(HostMemory memory) {
    return memory == HostMemory::Pinned ? "pinned" : "pageable";
}

bool gpu_available() {
    int count = 0;
    return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
}

DeviceInfo device_info() {
    int device = 0;
    check(cudaGetDevice(&device), "cudaGetDevice");
    cudaDeviceProp properties{};
    check(cudaGetDeviceProperties(&properties, device), "cudaGetDeviceProperties");
    DeviceInfo info;
    info.name = properties.name;
    info.memory_bytes = properties.totalGlobalMem;
    info.major = properties.major;
    info.minor = properties.minor;
    check(cudaRuntimeGetVersion(&info.runtime_version), "cudaRuntimeGetVersion");
    check(cudaDriverGetVersion(&info.driver_version), "cudaDriverGetVersion");
    return info;
}

GpuResult convolve_gpu(const Image& image, const Filter& filter, const GpuConfig& config) {
    image.validate();
    if (config.iterations <= 0 || config.warmup < 0) {
        throw std::invalid_argument("Iterations must be positive and warmup must be nonnegative");
    }
    check(cudaFree(nullptr), "Initialize CUDA context");
    GpuResult result;
    result.output = Image{image.width, image.height, std::vector<float>(image.pixels.size())};
    const std::size_t count = image.pixels.size();
    const std::size_t bytes = count * sizeof(float);
    auto start = Clock::now();
    HostBuffer host_input(count, config.host_memory);
    HostBuffer host_output(count, config.host_memory);
    HostBuffer host_filter(filter.size(), config.host_memory);
    DeviceBuffer device_input(count);
    DeviceBuffer device_output(count);
    DeviceBuffer device_filter(filter.size());
    result.allocation_ms = milliseconds(start);
    std::copy(filter.begin(), filter.end(), host_filter.get());

    const auto upload_filter = [&] {
        if (config.kernel == Kernel::Naive) {
            check(cudaMemcpy(device_filter.get(), host_filter.get(), sizeof(Filter), cudaMemcpyHostToDevice), "Upload global filter");
        } else {
            check(cudaMemcpyToSymbol(constant_filter, host_filter.get(), sizeof(Filter)), "Upload constant filter");
        }
    };
    const auto execute = [&] {
        launch(config.kernel, device_input.get(), device_filter.get(), device_output.get(), image.width, image.height);
    };

    std::copy(image.pixels.begin(), image.pixels.end(), host_input.get());
    check(cudaMemcpy(device_input.get(), host_input.get(), bytes, cudaMemcpyHostToDevice), "Upload input");
    upload_filter();
    check(cudaDeviceSynchronize(), "Synchronize initial transfers");
    for (int i = 0; i < config.warmup; ++i) execute();
    check(cudaDeviceSynchronize(), "Synchronize warmup");

    Event begin, end;
    check(cudaEventRecord(begin.get()), "Record kernel start");
    for (int i = 0; i < config.iterations; ++i) execute();
    check(cudaEventRecord(end.get()), "Record kernel end");
    check(cudaEventSynchronize(end.get()), "Wait for kernel benchmark");
    float kernel_total_ms = 0;
    check(cudaEventElapsedTime(&kernel_total_ms, begin.get(), end.get()), "Measure kernels");
    result.kernel_ms = kernel_total_ms / config.iterations;

    // Measure complete synchronous pipelines independently of the kernel-only batch.
    // Host staging copies are included in pipeline_ms, not in transfer-only columns.
    for (int i = 0; i < config.iterations; ++i) {
        const auto pipeline_start = Clock::now();
        std::copy(image.pixels.begin(), image.pixels.end(), host_input.get());
        start = Clock::now();
        check(cudaMemcpy(device_input.get(), host_input.get(), bytes, cudaMemcpyHostToDevice), "Upload input");
        check(cudaDeviceSynchronize(), "Synchronize H2D");
        result.h2d_ms += milliseconds(start);
        start = Clock::now();
        upload_filter();
        check(cudaDeviceSynchronize(), "Synchronize filter upload");
        result.filter_upload_ms += milliseconds(start);
        execute();
        check(cudaDeviceSynchronize(), "Synchronize convolution");
        start = Clock::now();
        check(cudaMemcpy(host_output.get(), device_output.get(), bytes, cudaMemcpyDeviceToHost), "Download output");
        result.d2h_ms += milliseconds(start);
        std::copy(host_output.get(), host_output.get() + count, result.output.pixels.begin());
        result.pipeline_ms += milliseconds(pipeline_start);
    }
    result.h2d_ms /= config.iterations;
    result.filter_upload_ms /= config.iterations;
    result.d2h_ms /= config.iterations;
    result.pipeline_ms /= config.iterations;
    return result;
}

}  // namespace imageprocessing
