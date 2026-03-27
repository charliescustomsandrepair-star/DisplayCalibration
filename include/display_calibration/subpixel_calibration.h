#pragma once

#include "types.h"
#include <vector>
#include <memory>
#include <opencv2/core.hpp>

namespace display_calibration {

// Auto‑calibration subsystem using camera captures
class SubpixelCalibration {
public:
    SubpixelCalibration();
    ~SubpixelCalibration();

    // Perform full calibration:
    // - project fringe patterns (via display)
    // - capture images (from camera)
    // - return detected geometry
    SubpixelGeometry calibrate_from_camera(
        const std::vector<cv::Mat>& captured_fringes,  // sequence of phase‑shifted images
        float expected_pattern_freq = 0.1f,            // cycles per pixel
        bool patterns_horizontal = true                 // fringe orientation
    );

    // Alternative: calibrate from screenshots (if display can capture its own output)
    SubpixelGeometry calibrate_from_screenshots(
        const std::vector<cv::Mat>& screenshots
    );

    // Gamma estimation and linearization [0024]
    static cv::Mat gamma_linearize(const cv::Mat& img, double gamma = 2.2);

    // Quality‑guided phase unwrapping [0025]
    static cv::Mat unwrap_phase(const cv::Mat& wrapped_phase,
                                 const cv::Mat& modulation_quality);

    // Weighted quadratic phase fitting [0025]
    static cv::Mat fit_quadratic_surface(const cv::Mat& unwrapped_phase,
                                          const cv::Mat& weight_mask);

    // FFT layout detection [0026]
    static SubpixelGeometry::Layout detect_layout(const cv::Mat& fringe_image);

private:
    class Impl;
    std::unique_ptr<Impl> pimpl;
};

} // namespace display_calibration
