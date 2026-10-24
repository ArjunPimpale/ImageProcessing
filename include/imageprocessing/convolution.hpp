#pragma once

#include <array>
#include <cstddef>
#include <string>
#include <vector>

namespace imageprocessing {

struct Image {
    int width = 0;
    int height = 0;
    std::vector<float> pixels;
    void validate() const;
};

using Filter = std::array<float, 9>;
const std::vector<std::string>& filter_names();
Filter make_filter(const std::string& name);
Image convolve_cpu(const Image& image, const Filter& filter);
Image load_image(const std::string& path);
void save_png(const std::string& path, const Image& image);
Image synthetic_image(int width, int height, unsigned seed = 42);

struct Comparison {
    bool passed = true;
    float max_absolute_error = 0.0f;
    std::size_t mismatches = 0;
};
Comparison compare(const Image& expected, const Image& actual,
                   float absolute_tolerance = 1e-5f, float relative_tolerance = 1e-5f);

}  // namespace imageprocessing
