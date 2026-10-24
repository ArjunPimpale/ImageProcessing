#include "imageprocessing/gpu.hpp"

#include <iostream>
#include <utility>

namespace ip = imageprocessing;

int main() {
    if (!ip::gpu_available()) {
        std::cout << "SKIP: no accessible CUDA GPU\n";
        return 77;
    }
    try {
        const auto device = ip::device_info();
        std::cout << "Testing on " << device.name << '\n';
        int cases = 0;
        // Thin images, partial blocks, halo boundaries, and both rectangle orientations.
        for (auto dimensions : {std::pair{1, 1}, std::pair{1, 19}, std::pair{19, 1},
                                std::pair{2, 3}, std::pair{16, 16}, std::pair{31, 17},
                                std::pair{19, 35}, std::pair{64, 48}}) {
            const auto image = ip::synthetic_image(dimensions.first, dimensions.second);
            for (const auto& name : ip::filter_names()) {
                const auto filter = ip::make_filter(name);
                const auto expected = ip::convolve_cpu(image, filter);
                for (auto memory : {ip::HostMemory::Pageable, ip::HostMemory::Pinned}) {
                    for (auto kernel : {ip::Kernel::Naive, ip::Kernel::Constant, ip::Kernel::Tiled}) {
                        const auto actual = ip::convolve_gpu(image, filter, {kernel, memory, 2, 1});
                        const auto comparison = ip::compare(expected, actual.output);
                        if (!comparison.passed) {
                            std::cerr << "FAIL " << image.width << 'x' << image.height << ' ' << name
                                      << ' ' << ip::kernel_name(kernel) << ' ' << ip::memory_name(memory)
                                      << " mismatches=" << comparison.mismatches
                                      << " max_error=" << comparison.max_absolute_error << '\n';
                            return 1;
                        }
                        ++cases;
                    }
                }
            }
        }
        std::cout << "Passed " << cases << " GPU/CPU comparisons.\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
