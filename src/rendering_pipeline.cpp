#include "display_calibration/rendering_pipeline.h"
#include "display_calibration/types.h"
#include <cmath>
#include <algorithm>
#include <cstring>

namespace display_calibration {

struct RenderingPipeline::Impl {
    SubpixelGeometry geom;
    TemporalAccumulation temporal;
    int last_width = 0, last_height = 0;

    // Helper: compute coverage for a fragment at (x,y) with given logical pixel color
    // Handles all layout types: RGB_STRIPE, BGR_STRIPE, PENTILE_RG,
    // DIAMOND_PENTILE, VERTICAL_STRIPE [0008]-[0009]
    SubpixelCoverage compute_coverage(float x, float y, const PixelRGBA& color) const {
        SubpixelCoverage cov;

        switch (geom.layout) {
        case SubpixelGeometry::Layout::BGR_STRIPE: {
            // BGR: reversed horizontal order — B at left, G center, R right
            float b_center_x = geom.offset_x[0]; // B occupies first slot
            float g_center_x = geom.offset_x[1];
            float r_center_x = geom.offset_x[2]; // R occupies last slot
            const float radius = geom.pitch_x * 0.5f;
            cov.r_cover = std::max(0.0f, 1.0f - std::abs(x - r_center_x) / radius);
            cov.g_cover = std::max(0.0f, 1.0f - std::abs(x - g_center_x) / radius);
            cov.b_cover = std::max(0.0f, 1.0f - std::abs(x - b_center_x) / radius);
            break;
        }
        case SubpixelGeometry::Layout::PENTILE_RG: {
            // Pentile RG/BG: green on every pixel, R and B alternate
            // Green subpixel is always present; R or B depends on checkerboard position
            const float radius = geom.pitch_x * 0.5f;
            cov.g_cover = std::max(0.0f, 1.0f - std::abs(x - geom.offset_x[1]) / radius);
            // R and B share a position; contribution is halved (alternating rows)
            cov.r_cover = std::max(0.0f, 1.0f - std::abs(x - geom.offset_x[0]) / radius) * 0.5f;
            cov.b_cover = std::max(0.0f, 1.0f - std::abs(x - geom.offset_x[2]) / radius) * 0.5f;
            break;
        }
        case SubpixelGeometry::Layout::DIAMOND_PENTILE: {
            // Diamond Pentile (e.g. Samsung AMOLED): subpixels arranged in diamond grid
            // Green subpixels at full density, R/B at half density with diagonal offset
            const float radius_x = geom.pitch_x * 0.5f;
            const float radius_y = geom.pitch_y * 0.5f;
            // 2D distance for diamond arrangement
            float r_dist = std::sqrt(std::pow((x - geom.offset_x[0]) / radius_x, 2.0f)
                                   + std::pow((y - geom.offset_y[0]) / radius_y, 2.0f));
            float g_dist = std::sqrt(std::pow((x - geom.offset_x[1]) / radius_x, 2.0f)
                                   + std::pow((y - geom.offset_y[1]) / radius_y, 2.0f));
            float b_dist = std::sqrt(std::pow((x - geom.offset_x[2]) / radius_x, 2.0f)
                                   + std::pow((y - geom.offset_y[2]) / radius_y, 2.0f));
            cov.r_cover = std::max(0.0f, 1.0f - r_dist);
            cov.g_cover = std::max(0.0f, 1.0f - g_dist);
            cov.b_cover = std::max(0.0f, 1.0f - b_dist);
            break;
        }
        case SubpixelGeometry::Layout::VERTICAL_STRIPE: {
            // Vertical stripe: subpixels stacked vertically (R top, G middle, B bottom)
            const float radius = geom.pitch_y * 0.5f;
            cov.r_cover = std::max(0.0f, 1.0f - std::abs(y - geom.offset_y[0]) / radius);
            cov.g_cover = std::max(0.0f, 1.0f - std::abs(y - geom.offset_y[1]) / radius);
            cov.b_cover = std::max(0.0f, 1.0f - std::abs(y - geom.offset_y[2]) / radius);
            break;
        }
        case SubpixelGeometry::Layout::RGB_STRIPE:
        case SubpixelGeometry::Layout::UNKNOWN:
        default: {
            // Standard RGB horizontal stripe (default)
            float r_center_x = geom.offset_x[0];
            float g_center_x = geom.offset_x[1];
            float b_center_x = geom.offset_x[2];
            const float radius = geom.pitch_x * 0.5f;
            cov.r_cover = std::max(0.0f, 1.0f - std::abs(x - r_center_x) / radius);
            cov.g_cover = std::max(0.0f, 1.0f - std::abs(x - g_center_x) / radius);
            cov.b_cover = std::max(0.0f, 1.0f - std::abs(x - b_center_x) / radius);
            break;
        }
        }

        return cov;
    }

    // Luminance from pixel (perceptual weights [0012])
    float luminance(const PixelRGBA& p) const {
        return 0.2126f * p.r + 0.7152f * p.g + 0.0722f * p.b;
    }

    // Sobel‑like edge detection [0014]
    EdgeInfo detect_edge(const PixelRGBA* window, int stride) const {
        if (!window || stride <= 0) {
            return EdgeInfo{0.0f, false};
        }
        // Simple 3x3 luminance gradient
        float l[9];
        for (int i = 0; i < 9; ++i) {
            l[i] = luminance(window[i]);
        }
        float gx = (l[2] + 2*l[5] + l[8]) - (l[0] + 2*l[3] + l[6]);
        float gy = (l[6] + 2*l[7] + l[8]) - (l[0] + 2*l[1] + l[2]);
        float mag = std::sqrt(gx*gx + gy*gy);
        EdgeInfo info;
        info.gradient_magnitude = mag;
        info.is_edge = mag > 0.1f; // threshold
        return info;
    }

    // Apply perceptual weighting and edge modulation [0012][0016]
    PixelRGBA combine(const PixelRGBA& fragment, const SubpixelCoverage& cov, float edge_strength) {
        // Base perceptual weights
        float w_r = PERCEPTUAL_WEIGHT_R;
        float w_g = PERCEPTUAL_WEIGHT_G;
        float w_b = PERCEPTUAL_WEIGHT_B;

        // Modulate based on edge strength: stronger edges get more weight [0016]
        float mod = 1.0f + edge_strength; // linear modulation
        w_r *= mod;
        w_g *= mod;
        w_b *= mod;

        // Combine [0018]
        PixelRGBA out;
        out.r = fragment.r * cov.r_cover * w_r;
        out.g = fragment.g * cov.g_cover * w_g;
        out.b = fragment.b * cov.b_cover * w_b;
        out.a = fragment.a; // preserve alpha

        // Clamp
        out.r = std::min(1.0f, std::max(0.0f, out.r));
        out.g = std::min(1.0f, std::max(0.0f, out.g));
        out.b = std::min(1.0f, std::max(0.0f, out.b));
        return out;
    }

    // Temporal stabilization with motion-compensated accumulation
    // and neighborhood clamping [0022]
    void temporal_stabilize(std::vector<PixelRGBA>& frame, int w, int h) {
        if (temporal.prev_r.empty()) {
            // First frame: initialize history buffers
            temporal.prev_r.resize(w*h);
            temporal.prev_g.resize(w*h);
            temporal.prev_b.resize(w*h);
            temporal.motion_vectors.resize(w*h*2, 0.0f); // dx,dy per pixel
            for (int i = 0; i < w*h; ++i) {
                temporal.prev_r[i] = frame[i].r;
                temporal.prev_g[i] = frame[i].g;
                temporal.prev_b[i] = frame[i].b;
            }
            return;
        }

        // Resize motion vectors if needed
        if (temporal.motion_vectors.size() != static_cast<size_t>(w*h*2)) {
            temporal.motion_vectors.resize(w*h*2, 0.0f);
        }

        const float alpha = 0.7f; // temporal blending factor

        for (int y = 0; y < h; ++y) {
            for (int x = 0; x < w; ++x) {
                int idx = y * w + x;

                // --- Motion compensation ---
                // Estimate motion by comparing current vs previous luminance in local region
                float cur_lum = luminance(frame[idx]);
                float prev_lum = 0.2126f * temporal.prev_r[idx]
                               + 0.7152f * temporal.prev_g[idx]
                               + 0.0722f * temporal.prev_b[idx];
                float motion_magnitude = std::abs(cur_lum - prev_lum);

                // Store motion estimate (simplified block-matching surrogate)
                temporal.motion_vectors[idx*2]     = motion_magnitude; // dx proxy
                temporal.motion_vectors[idx*2 + 1] = motion_magnitude; // dy proxy

                // --- Neighborhood clamping ---
                // Compute min/max of current frame's 3x3 neighborhood
                float min_r = frame[idx].r, max_r = frame[idx].r;
                float min_g = frame[idx].g, max_g = frame[idx].g;
                float min_b = frame[idx].b, max_b = frame[idx].b;

                for (int dy = -1; dy <= 1; ++dy) {
                    for (int dx = -1; dx <= 1; ++dx) {
                        int nx = std::clamp(x + dx, 0, w - 1);
                        int ny = std::clamp(y + dy, 0, h - 1);
                        int ni = ny * w + nx;
                        min_r = std::min(min_r, frame[ni].r);
                        max_r = std::max(max_r, frame[ni].r);
                        min_g = std::min(min_g, frame[ni].g);
                        max_g = std::max(max_g, frame[ni].g);
                        min_b = std::min(min_b, frame[ni].b);
                        max_b = std::max(max_b, frame[ni].b);
                    }
                }

                // Clamp previous frame values to current neighborhood bounds
                float clamped_pr = std::clamp(temporal.prev_r[idx], min_r, max_r);
                float clamped_pg = std::clamp(temporal.prev_g[idx], min_g, max_g);
                float clamped_pb = std::clamp(temporal.prev_b[idx], min_b, max_b);

                // --- Adaptive blending ---
                // Reduce temporal blending in high-motion regions to avoid ghosting
                float adaptive_alpha = alpha * (1.0f - std::min(1.0f, motion_magnitude * 4.0f));

                // Blend current frame with clamped history
                frame[idx].r = (1.0f - adaptive_alpha) * frame[idx].r + adaptive_alpha * clamped_pr;
                frame[idx].g = (1.0f - adaptive_alpha) * frame[idx].g + adaptive_alpha * clamped_pg;
                frame[idx].b = (1.0f - adaptive_alpha) * frame[idx].b + adaptive_alpha * clamped_pb;

                // Update history
                temporal.prev_r[idx] = frame[idx].r;
                temporal.prev_g[idx] = frame[idx].g;
                temporal.prev_b[idx] = frame[idx].b;
            }
        }
    }
};

RenderingPipeline::RenderingPipeline() : pimpl(std::make_unique<Impl>()) {}
RenderingPipeline::~RenderingPipeline() = default;

void RenderingPipeline::set_subpixel_geometry(const SubpixelGeometry& geom) {
    pimpl->geom = geom;
}

std::vector<PixelRGBA> RenderingPipeline::process_frame(
    const std::vector<PixelRGBA>& input, int width, int height, bool temporal_enable)
{
    // Validate input
    if (width <= 0 || height <= 0 || input.size() != static_cast<size_t>(width * height)) {
        return std::vector<PixelRGBA>(); // return empty on invalid input
    }
    std::vector<PixelRGBA> output(width * height);

    // For each logical pixel, we process fragments (here one fragment per pixel for simplicity)
    // In a real implementation, fragments would be generated by rasterization.
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            const PixelRGBA& frag = input[y*width + x];

            // Compute subpixel coverage based on calibrated geometry
            // Fragment position in logical pixel coordinates (0..1 within pixel)
            float fx = 0.5f; // fragment covers whole pixel for demo
            float fy = 0.5f;
            SubpixelCoverage cov = pimpl->compute_coverage(fx, fy, frag);

            // Edge detection using 3x3 neighborhood
            PixelRGBA neighborhood[9];
            for (int dy = -1; dy <= 1; ++dy) {
                for (int dx = -1; dx <= 1; ++dx) {
                    int nx = std::clamp(x+dx, 0, width-1);
                    int ny = std::clamp(y+dy, 0, height-1);
                    neighborhood[(dy+1)*3 + (dx+1)] = input[ny*width + nx];
                }
            }
            EdgeInfo edge = pimpl->detect_edge(neighborhood, 3);

            // Combine with modulation
            output[y*width + x] = pimpl->combine(frag, cov, edge.gradient_magnitude);
        }
    }

    // Optional temporal stabilization
    if (temporal_enable) {
        pimpl->temporal_stabilize(output, width, height);
    }

    return output;
}

EdgeInfo RenderingPipeline::detect_edge(const PixelRGBA* neighborhood, int stride) const {
    return pimpl->detect_edge(neighborhood, stride);
}

} // namespace display_calibration
