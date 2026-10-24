#include "imageprocessing/convolution.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <random>
#include <stdexcept>

namespace imageprocessing {

void Image::validate() const {
    if (width <= 0 || height <= 0 ||
        pixels.size() != static_cast<std::size_t>(width) * static_cast<std::size_t>(height)) {
        throw std::invalid_argument("Image dimensions must be positive and match its pixel buffer");
    }
}

const std::vector<std::string>& filter_names() {
    static const std::vector<std::string> names = {
        "sharpen", "high-pass", "low-pass", "gaussian", "d-gaussian"};
    return names;
}

Filter make_filter(const std::string& name) {
    if (name == "sharpen") {
        // Preserve the upstream sharpening formula at its default alpha = 0.8.
        constexpr float alpha = 0.8f;
        constexpr float neighbor = -alpha / (9 - 9 * alpha);
        constexpr float center = (9 - alpha) / (9 - 9 * alpha);
        return {neighbor, neighbor, neighbor, neighbor, center, neighbor,
                neighbor, neighbor, neighbor};
    }
    if (name == "high-pass") return {-1, -1, -1, -1, 8, -1, -1, -1, -1};
    if (name == "low-pass") {
        Filter filter{};
        filter.fill(1.0f / 9.0f);
        return filter;
    }
    if (name == "gaussian") {
        return {1.0f/16, 2.0f/16, 1.0f/16, 2.0f/16, 4.0f/16,
                2.0f/16, 1.0f/16, 2.0f/16, 1.0f/16};
    }
    // Retain the upstream d_Gaussian coefficients and naming for traceability.
    if (name == "d-gaussian") return {-2, 1, -2, 1, 4, 1, -2, 1, -2};
    throw std::invalid_argument("Unknown filter: " + name);
}

Image convolve_cpu(const Image& image, const Filter& filter) {
    image.validate();
    Image output{image.width, image.height, std::vector<float>(image.pixels.size())};
    for (int y = 0; y < image.height; ++y) {
        for (int x = 0; x < image.width; ++x) {
            float sum = 0;
            for (int fy = 0; fy < 3; ++fy) {
                for (int fx = 0; fx < 3; ++fx) {
                    const int iy = y + fy - 1;
                    const int ix = x + fx - 1;
                    if (iy >= 0 && iy < image.height && ix >= 0 && ix < image.width) {
                        sum += filter[fy * 3 + fx] *
                            image.pixels[static_cast<std::size_t>(iy) * image.width + ix];
                    }
                }
            }
            output.pixels[static_cast<std::size_t>(y) * image.width + x] = sum;
        }
    }
    return output;
}

Image synthetic_image(int width, int height, unsigned seed) {
    if (width <= 0 || height <= 0) throw std::invalid_argument("Invalid synthetic image size");
    Image image{width, height, std::vector<float>(static_cast<std::size_t>(width) * height)};
    std::mt19937 generator(seed);
    std::uniform_real_distribution<float> distribution(0.0f, 1.0f);
    for (auto& pixel : image.pixels) pixel = distribution(generator);
    return image;
}

Comparison compare(const Image& expected, const Image& actual, float atol, float rtol) {
    expected.validate();
    actual.validate();
    if (expected.width != actual.width || expected.height != actual.height) {
        throw std::invalid_argument("Cannot compare images with different dimensions");
    }
    Comparison result;
    for (std::size_t i = 0; i < expected.pixels.size(); ++i) {
        const float a = expected.pixels[i];
        const float b = actual.pixels[i];
        if (!std::isfinite(a) || !std::isfinite(b)) {
            ++result.mismatches;
            result.max_absolute_error = std::numeric_limits<float>::infinity();
            continue;
        }
        const float difference = std::abs(a - b);
        result.max_absolute_error = std::max(result.max_absolute_error, difference);
        if (difference > atol + rtol * std::abs(a)) ++result.mismatches;
    }
    result.passed = result.mismatches == 0;
    return result;
}

}  // namespace imageprocessing
