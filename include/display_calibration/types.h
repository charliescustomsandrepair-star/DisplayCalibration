#pragma once

#include <cstdint>
#include <array>
#include <vector>

namespace display_calibration {

// Represents a logical pixel with RGBA channels (0..1 float)
struct PixelRGBA {
    float r, g, b, a;
};

// Per‑subpixel coverage values (0..1)
struct SubpixelCoverage {
    float r_cover;
    float g_cover;
    float b_cover;
};

// Weighting factors from patent [0012]
constexpr float PERCEPTUAL_WEIGHT_R = 1.1f;
constexpr float PERCEPTUAL_WEIGHT_G = 0.95f;
constexpr float PERCEPTUAL_WEIGHT_B = 1.05f;

// Calibration results [0026]
struct SubpixelGeometry {
    enum class Layout {
        RGB_STRIPE,
        BGR_STRIPE,
        PENTILE_RG,      // green on separate plane
        DIAMOND_PENTILE,  // e.g., Samsung diamond
        VERTICAL_STRIPE,  // subpixels stacked vertically
        UNKNOWN
    };
    Layout layout = Layout::UNKNOWN;
    bool horizontal_orientation = true; // true = subpixels side‑by‑side horizontally
    float pitch_x = 1.0f / 3.0f;        // logical pixel width per subpixel (fraction of logical pixel)
    float pitch_y = 1.0f;                // logical pixel height per subpixel
    std::array<float, 3> offset_x = {0.0f, 1.0f/3.0f, 2.0f/3.0f}; // R,G,B offsets in logical pixels
    std::array<float, 3> offset_y = {0.0f, 0.0f, 0.0f};
    // For Pentile, some subpixels may be missing or shared
};

// Edge detection result [0015]
struct EdgeInfo {
    float gradient_magnitude;   // luminance gradient
    bool is_edge;
};

// Temporal state [0022]
struct TemporalAccumulation {
    std::vector<float> prev_r, prev_g, prev_b; // previous frame subpixel values
    std::vector<float> motion_vectors;          // simplified (per tile)
};

} // namespace display_calibration
