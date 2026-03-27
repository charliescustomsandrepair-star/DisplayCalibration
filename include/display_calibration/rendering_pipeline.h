#pragma once

#include "types.h"
#include <vector>
#include <memory>

namespace display_calibration {

// Main rendering pipeline class
class RenderingPipeline {
public:
    RenderingPipeline();
    ~RenderingPipeline();

    // Set calibration data (after auto‑calibration)
    void set_subpixel_geometry(const SubpixelGeometry& geom);

    // Process one frame of image data (vector of logical pixels)
    // Input: logical RGBA pixels in linear space (0..1)
    // Output: enhanced RGBA pixels for display
    std::vector<PixelRGBA> process_frame(const std::vector<PixelRGBA>& input,
                                          int width, int height,
                                          bool temporal_enable = true);

    // Edge detection on luminance (internal, but exposed for testing)
    EdgeInfo detect_edge(const PixelRGBA* neighborhood, int stride) const;

private:
    class Impl;
    std::unique_ptr<Impl> pimpl;
};

} // namespace display_calibration
