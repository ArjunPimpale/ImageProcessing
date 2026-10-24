#include "imageprocessing/convolution.hpp"
#ifdef IMAGEPROCESSING_HAS_CUDA
#include "imageprocessing/gpu.hpp"
#endif

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>

namespace ip = imageprocessing;
using Clock = std::chrono::steady_clock;

struct Options {
    std::string input;
    std::string output = "output";
    std::string filter = "high-pass";
    std::string backend = "all";
    std::string csv;
    int iterations = 100;
    int warmup = 10;
    int width = 2048;
    int height = 1328;
};

void usage() {
    std::cout <<
        "Usage: image_filters [options]\n"
        "  --input PATH         Load JPEG/PNG/PNM/etc.; otherwise use seeded synthetic input\n"
        "  --filter NAME        sharpen|high-pass|low-pass|gaussian|d-gaussian\n"
        "  --backend NAME       cpu|gpu|all (default: all; GPU requests require CUDA)\n"
        "  --output DIR         PNG output directory (default: output)\n"
        "  --csv PATH           Write benchmark CSV with device and correctness metadata\n"
        "  --iterations N       Measured iterations (default: 100)\n"
        "  --warmup N           Warmup iterations (default: 10)\n"
        "  --width N --height N Synthetic image dimensions (default: 2048 x 1328)\n"
        "  --help               Show this help\n";
}

int integer(const std::string& value, bool allow_zero = false) {
    std::size_t consumed = 0;
    const long long parsed = std::stoll(value, &consumed);
    if (consumed != value.size() || parsed < (allow_zero ? 0 : 1) ||
        parsed > std::numeric_limits<int>::max()) {
        throw std::invalid_argument("Invalid integer: " + value);
    }
    return static_cast<int>(parsed);
}

Options parse(int argc, char** argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string key = argv[i];
        if (key == "--help") { usage(); std::exit(0); }
        if (i + 1 >= argc) throw std::invalid_argument("Missing value for " + key);
        const std::string value = argv[++i];
        if (key == "--input") options.input = value;
        else if (key == "--output") options.output = value;
        else if (key == "--filter") options.filter = value;
        else if (key == "--backend") options.backend = value;
        else if (key == "--csv") options.csv = value;
        else if (key == "--iterations") options.iterations = integer(value);
        else if (key == "--warmup") options.warmup = integer(value, true);
        else if (key == "--width") options.width = integer(value);
        else if (key == "--height") options.height = integer(value);
        else throw std::invalid_argument("Unknown option: " + key);
    }
    if (options.backend != "cpu" && options.backend != "gpu" && options.backend != "all") {
        throw std::invalid_argument("Backend must be cpu, gpu, or all");
    }
    return options;
}

std::string csv_quote(const std::string& value) {
    std::string result = "\"";
    for (char character : value) {
        if (character == '"') result += '"';
        result += character;
    }
    return result + '"';
}

struct Row {
    std::string backend, kernel, memory, device;
    int major = 0, minor = 0, runtime = 0, driver = 0;
    double allocation = 0, h2d = 0, filter_upload = 0, kernel_ms = 0, d2h = 0, pipeline = 0;
    ip::Comparison comparison;
};

double gflops(std::size_t pixels, double kernel_ms) {
    return kernel_ms > 0 ? static_cast<double>(pixels) * 18.0 / (kernel_ms * 1e6) : 0;
}

void print_row(const Row& row, std::size_t pixels) {
    std::cout << std::left << std::setw(9) << row.backend << std::setw(11) << row.kernel
              << std::setw(11) << row.memory << std::right << std::setw(12) << row.kernel_ms
              << std::setw(14) << row.pipeline << std::setw(12) << gflops(pixels, row.kernel_ms)
              << "  " << (row.comparison.passed ? "PASS" : "FAIL")
              << " (max error " << std::scientific << row.comparison.max_absolute_error
              << std::fixed << ")\n";
}

void write_csv(const std::string& path, const Options& options, const ip::Image& image,
               const std::vector<Row>& rows) {
    const std::filesystem::path destination(path);
    if (destination.has_parent_path()) std::filesystem::create_directories(destination.parent_path());
    std::ofstream output(path);
    if (!output) throw std::runtime_error("Cannot create CSV: " + path);
    output << "build_mode,cuda_compiler,backend,kernel,host_memory,device,compute_major,compute_minor,cuda_runtime,cuda_driver,"
              "width,height,filter,iterations,warmup,allocation_ms,h2d_ms,filter_upload_ms,kernel_ms,"
              "d2h_ms,pipeline_ms,kernel_gflops,correct,max_absolute_error,mismatches\n";
    output << std::setprecision(12);
    for (const auto& row : rows) {
        output << IMAGEPROCESSING_BUILD_MODE << ',' << IMAGEPROCESSING_CUDA_COMPILER_VERSION
               << ',' << row.backend << ',' << row.kernel << ',' << row.memory << ',' << csv_quote(row.device)
               << ',' << row.major << ',' << row.minor << ',' << row.runtime << ',' << row.driver
               << ',' << image.width << ',' << image.height << ',' << options.filter
               << ',' << options.iterations << ',' << options.warmup << ',' << row.allocation
               << ',' << row.h2d << ',' << row.filter_upload << ',' << row.kernel_ms
               << ',' << row.d2h << ',' << row.pipeline << ',' << gflops(image.pixels.size(), row.kernel_ms)
               << ',' << (row.comparison.passed ? "true" : "false")
               << ',' << row.comparison.max_absolute_error << ',' << row.comparison.mismatches << '\n';
    }
    if (!output) throw std::runtime_error("Failed while writing CSV: " + path);
}

int main(int argc, char** argv) {
    try {
        const Options options = parse(argc, argv);
        std::cout << "Build: " << IMAGEPROCESSING_BUILD_MODE
                  << " | CUDA compiler: " << IMAGEPROCESSING_CUDA_COMPILER_VERSION << '\n';
        const bool use_gpu = options.backend != "cpu";
        // Explicit GPU requests must never silently become CPU benchmark results.
#ifdef IMAGEPROCESSING_HAS_CUDA
        ip::DeviceInfo device;
        if (use_gpu) {
            device = ip::device_info();
            std::cout << "GPU: " << device.name << " | compute " << device.major << '.' << device.minor
                      << " | VRAM " << device.memory_bytes / (1024.0 * 1024 * 1024) << " GiB"
                      << " | CUDA runtime " << device.runtime_version
                      << " | driver API " << device.driver_version << '\n';
        }
#else
        if (use_gpu) throw std::runtime_error("CPU-only build: use --backend cpu, or rebuild with CUDA enabled");
#endif
        const auto filter = ip::make_filter(options.filter);
        const auto image = options.input.empty() ? ip::synthetic_image(options.width, options.height)
                                                  : ip::load_image(options.input);
        std::filesystem::create_directories(options.output);
        std::cout << "Image: " << image.width << " x " << image.height << " | filter: " << options.filter
                  << " | iterations: " << options.iterations << " | warmup: " << options.warmup << '\n';

        // A CPU reference is always computed, including for --backend gpu.
        ip::Image reference = ip::convolve_cpu(image, filter);
        std::vector<Row> rows;
        if (options.backend != "gpu") {
            for (int i = 0; i < options.warmup; ++i) reference = ip::convolve_cpu(image, filter);
            const auto start = Clock::now();
            for (int i = 0; i < options.iterations; ++i) reference = ip::convolve_cpu(image, filter);
            const double cpu_ms = std::chrono::duration<double, std::milli>(Clock::now() - start).count() / options.iterations;
            Row cpu;
            cpu.backend = "cpu";
            cpu.kernel = "reference";
            cpu.memory = "pageable";
            cpu.device = "host CPU";
            cpu.kernel_ms = cpu_ms;
            cpu.pipeline = cpu_ms;
            rows.push_back(cpu);
            ip::save_png((std::filesystem::path(options.output) / "cpu.png").string(), reference);
        }
        std::cout << std::fixed << std::setprecision(6)
                  << "backend  kernel     memory        kernel_ms   pipeline_ms      GFLOPS  validation\n";
        if (!rows.empty()) print_row(rows.back(), image.pixels.size());
        bool passed = true;
#ifdef IMAGEPROCESSING_HAS_CUDA
        if (use_gpu) {
            for (auto memory : {ip::HostMemory::Pageable, ip::HostMemory::Pinned}) {
                for (auto kernel : {ip::Kernel::Naive, ip::Kernel::Constant, ip::Kernel::Tiled}) {
                    const ip::GpuConfig config{kernel, memory, options.iterations, options.warmup};
                    const auto result = ip::convolve_gpu(image, filter, config);
                    Row row;
                    row.backend = "gpu";
                    row.kernel = ip::kernel_name(kernel);
                    row.memory = ip::memory_name(memory);
                    row.device = device.name;
                    row.major = device.major; row.minor = device.minor;
                    row.runtime = device.runtime_version; row.driver = device.driver_version;
                    row.allocation = result.allocation_ms; row.h2d = result.h2d_ms;
                    row.filter_upload = result.filter_upload_ms; row.kernel_ms = result.kernel_ms;
                    row.d2h = result.d2h_ms; row.pipeline = result.pipeline_ms;
                    row.comparison = ip::compare(reference, result.output);
                    passed = passed && row.comparison.passed;
                    rows.push_back(row);
                    print_row(row, image.pixels.size());
                    if (row.comparison.passed) {
                        ip::save_png((std::filesystem::path(options.output) /
                            (row.kernel + "_" + row.memory + ".png")).string(), result.output);
                    }
                }
            }
        }
#endif
        if (!options.csv.empty()) write_csv(options.csv, options, image, rows);
        if (!passed) {
            std::cerr << "GPU validation failed; these timings must not be used as performance claims.\n";
            return 2;
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    }
}
