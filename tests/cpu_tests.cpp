#include "imageprocessing/convolution.hpp"

#include <cmath>
#include <filesystem>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <utility>

namespace ip = imageprocessing;

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

int main() {
    try {
        const ip::Filter identity{0, 0, 0, 0, 1, 0, 0, 0, 0};
        for (auto dimensions : {std::pair{1, 1}, std::pair{2, 3}, std::pair{31, 17}, std::pair{19, 35}}) {
            auto image = ip::synthetic_image(dimensions.first, dimensions.second);
            require(ip::compare(image, ip::convolve_cpu(image, identity), 0, 0).passed, "Rectangular identity failed");
        }
        const ip::Image input{3, 2, {1, 2, 3, 4, 5, 6}};
        const ip::Filter ones{1, 1, 1, 1, 1, 1, 1, 1, 1};
        const ip::Image sums{3, 2, {12, 21, 16, 12, 21, 16}};
        require(ip::compare(sums, ip::convolve_cpu(input, ones), 0, 0).passed, "Zero-padded rectangular convolution failed");
        // Asymmetric weights distinguish cross-correlation from flipped convolution.
        const ip::Filter right_neighbor{0, 0, 0, 0, 0, 1, 0, 0, 0};
        const ip::Image shifted{3, 2, {2, 3, 0, 5, 6, 0}};
        require(ip::compare(shifted, ip::convolve_cpu(input, right_neighbor), 0, 0).passed, "Filter orientation failed");
        for (const auto& name : ip::filter_names()) {
            const auto filter = ip::make_filter(name);
            const auto output = ip::convolve_cpu(ip::Image{1, 1, {0.5f}}, filter);
            require(std::abs(output.pixels[0] - 0.5f * filter[4]) < 1e-6f, "Single-pixel boundary failed");
        }
        bool rejected = false;
        try { ip::convolve_cpu(ip::Image{2, 2, {1}}, identity); }
        catch (const std::invalid_argument&) { rejected = true; }
        require(rejected, "Invalid image buffer was accepted");
        rejected = false;
        try { ip::make_filter("invalid"); } catch (const std::invalid_argument&) { rejected = true; }
        require(rejected, "Invalid filter was accepted");
        auto invalid = input;
        invalid.pixels[0] = std::numeric_limits<float>::quiet_NaN();
        require(!ip::compare(input, invalid).passed, "NaN comparison incorrectly passed");
        const auto temporary = std::filesystem::temp_directory_path() /
            ("imageprocessing-test-" + std::to_string(std::random_device{}()) + ".png");
        const ip::Image grayscale{3, 2, {0, 0.2f, 0.4f, 0.6f, 0.8f, 1}};
        ip::save_png(temporary.string(), grayscale);
        const auto reloaded = ip::load_image(temporary.string());
        std::filesystem::remove(temporary);
        require(ip::compare(grayscale, reloaded, 1.0f/255 + 1e-6f, 0).passed, "PNG round trip failed");
        std::cout << "CPU tests passed: rectangles, borders, orientation, five filters, errors, PNG I/O.\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
