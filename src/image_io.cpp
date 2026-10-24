#include "imageprocessing/convolution.hpp"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include <algorithm>
#include <memory>
#include <stdexcept>

namespace imageprocessing {

Image load_image(const std::string& path) {
    int width = 0, height = 0, channels = 0;
    using StbBuffer = std::unique_ptr<unsigned char, decltype(&stbi_image_free)>;
    StbBuffer rgb(stbi_load(path.c_str(), &width, &height, &channels, 3), stbi_image_free);
    if (!rgb) {
        const char* reason = stbi_failure_reason();
        throw std::runtime_error("Cannot load image '" + path + "': " + (reason ? reason : "unknown error"));
    }
    Image image{width, height, std::vector<float>(static_cast<std::size_t>(width) * height)};
    for (std::size_t i = 0; i < image.pixels.size(); ++i) {
        image.pixels[i] = (0.299f * rgb.get()[3*i] + 0.587f * rgb.get()[3*i+1] +
                           0.114f * rgb.get()[3*i+2]) / 255.0f;
    }
    return image;
}

void save_png(const std::string& path, const Image& image) {
    image.validate();
    std::vector<unsigned char> bytes(image.pixels.size());
    for (std::size_t i = 0; i < bytes.size(); ++i) {
        bytes[i] = static_cast<unsigned char>(std::clamp(image.pixels[i], 0.0f, 1.0f) * 255.0f);
    }
    if (!stbi_write_png(path.c_str(), image.width, image.height, 1, bytes.data(), image.width)) {
        throw std::runtime_error("Cannot write PNG: " + path);
    }
}

}  // namespace imageprocessing
