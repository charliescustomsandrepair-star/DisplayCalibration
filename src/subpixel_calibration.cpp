#include "display_calibration/subpixel_calibration.h"
#include <opencv2/opencv.hpp>
#include <complex>
#include <queue>
#include <cmath>

namespace display_calibration {

struct SubpixelCalibration::Impl {
    // Helper: return wrapped phase and modulation as pair
    static std::pair<cv::Mat, cv::Mat> compute_modulation(const std::vector<cv::Mat>& phase_shifted_imgs) {
        // Standard phase‑shifting algorithm for N steps
        int N = phase_shifted_imgs.size();
        CV_Assert(N >= 3);
        cv::Mat sum_sin, sum_cos;
        for (int i = 0; i < N; ++i) {
            double phase = 2.0 * CV_PI * i / N;
            cv::Mat img;
            phase_shifted_imgs[i].convertTo(img, CV_32F);
            if (i == 0) {
                sum_sin = cv::Mat::zeros(img.size(), CV_32F);
                sum_cos = cv::Mat::zeros(img.size(), CV_32F);
            }
            sum_sin += img * std::sin(phase);
            sum_cos += img * std::cos(phase);
        }
        cv::Mat wrapped_phase, modulation;
        cv::phase(sum_cos, sum_sin, wrapped_phase);
        cv::magnitude(sum_cos, sum_sin, modulation);
        modulation /= (static_cast<float>(N) / 2.0f);
        return {wrapped_phase, modulation};
    }

    // Gamma linearization: apply inverse gamma to linearize [0024]
    static cv::Mat gamma_linearize(const cv::Mat& img, double gamma) {
        CV_Assert(!img.empty() && gamma > 0.0);
        cv::Mat linear;
        img.convertTo(linear, CV_32F, 1.0/255.0);
        cv::pow(linear, 1.0/gamma, linear);  // linear = image^(1/gamma) to linearize
        return linear;
    }

    // Quality‑guided unwrapping using priority queue [0025]
    static cv::Mat quality_guided_unwrap(const cv::Mat& wrapped, const cv::Mat& quality) {
        CV_Assert(wrapped.type() == CV_32F && quality.type() == CV_32F);
        cv::Mat unwrapped = cv::Mat::zeros(wrapped.size(), CV_32F);
        cv::Mat processed = cv::Mat::zeros(wrapped.size(), CV_8U);

        // Find pixel with highest quality
        double maxQ;
        cv::Point maxLoc;
        cv::minMaxLoc(quality, nullptr, &maxQ, nullptr, &maxLoc);

        using Pixel = std::tuple<float, int, int>; // (quality, y, x)
        auto cmp = [](const Pixel& a, const Pixel& b) {
            return std::get<0>(a) < std::get<0>(b);
        };
        std::priority_queue<Pixel, std::vector<Pixel>, decltype(cmp)> pq(cmp);

        auto add_neighbors = [&](int y, int x) {
            const int dy[] = {-1,1,0,0};
            const int dx[] = {0,0,-1,1};
            for (int k = 0; k < 4; ++k) {
                int ny = y + dy[k];
                int nx = x + dx[k];
                if (ny >= 0 && ny < wrapped.rows && nx >= 0 && nx < wrapped.cols &&
                    !processed.at<uchar>(ny, nx)) {
                    pq.emplace(quality.at<float>(ny, nx), ny, nx);
                }
            }
        };

        // Initialize
        unwrapped.at<float>(maxLoc) = wrapped.at<float>(maxLoc);
        processed.at<uchar>(maxLoc) = 1;
        add_neighbors(maxLoc.y, maxLoc.x);

        while (!pq.empty()) {
            auto [q, y, x] = pq.top();
            pq.pop();
            if (processed.at<uchar>(y, x)) continue;

            // Unwrap using the best already unwrapped neighbor
            float ref = 0.0f;
            int count = 0;
            const int dy[] = {-1,1,0,0};
            const int dx[] = {0,0,-1,1};
            for (int k = 0; k < 4; ++k) {
                int ny = y + dy[k];
                int nx = x + dx[k];
                if (ny >= 0 && ny < wrapped.rows && nx >= 0 && nx < wrapped.cols &&
                    processed.at<uchar>(ny, nx)) {
                    ref += unwrapped.at<float>(ny, nx);
                    count++;
                }
            }
            if (count == 0) continue; // should not happen

            ref /= count;
            float w = wrapped.at<float>(y, x);
            float diff = w - ref;
            float nwrap = std::round(diff / (2*CV_PI));
            unwrapped.at<float>(y, x) = w - nwrap * 2*CV_PI;

            processed.at<uchar>(y, x) = 1;
            add_neighbors(y, x);
        }

        return unwrapped;
    }

    // Fit quadratic surface: phase = a*x^2 + b*y^2 + c*x*y + d*x + e*y + f
    // Returns coefficients as 1x6 Mat
    static cv::Mat fit_quadratic(const cv::Mat& unwrapped, const cv::Mat& weight) {
        CV_Assert(!unwrapped.empty() && !weight.empty() && unwrapped.size() == weight.size());
        int N = unwrapped.rows * unwrapped.cols;
        cv::Mat A(N, 6, CV_32F);
        cv::Mat b(N, 1, CV_32F);
        cv::Mat W = cv::Mat::diag(weight.reshape(1, N));

        int idx = 0;
        for (int y = 0; y < unwrapped.rows; ++y) {
            for (int x = 0; x < unwrapped.cols; ++x) {
                float X = x;
                float Y = y;
                A.at<float>(idx, 0) = X*X;
                A.at<float>(idx, 1) = Y*Y;
                A.at<float>(idx, 2) = X*Y;
                A.at<float>(idx, 3) = X;
                A.at<float>(idx, 4) = Y;
                A.at<float>(idx, 5) = 1.0f;
                b.at<float>(idx, 0) = unwrapped.at<float>(y, x);
                idx++;
            }
        }

        // Weighted least squares: (A^T W A) \ A^T W b
        cv::Mat AtWA = A.t() * W * A;
        cv::Mat AtWb = A.t() * W * b;
        cv::Mat coeffs;
        cv::solve(AtWA, AtWb, coeffs, cv::DECOMP_SVD);
        return coeffs;
    }

    // FFT layout detection [0026]
    static SubpixelGeometry::Layout detect_layout_fft(const cv::Mat& fringe) {
        // Convert to grayscale float
        cv::Mat gray;
        if (fringe.channels() == 3)
            cv::cvtColor(fringe, gray, cv::COLOR_BGR2GRAY);
        else
            gray = fringe.clone();
        gray.convertTo(gray, CV_32F, 1.0/255.0);

        // Apply window (Hann) to reduce edge effects
        cv::Mat win;
        cv::createHanningWindow(win, gray.size(), CV_32F);
        gray = gray.mul(win);

        // Compute FFT
        cv::Mat planes[] = {gray, cv::Mat::zeros(gray.size(), CV_32F)};
        cv::Mat complex;
        cv::merge(planes, 2, complex);
        cv::dft(complex, complex);

        // Split into magnitude
        cv::Mat mag_planes[2];
        cv::split(complex, mag_planes);
        cv::Mat mag;
        cv::magnitude(mag_planes[0], mag_planes[1], mag);

        // Shift to center
        int cx = mag.cols/2;
        int cy = mag.rows/2;
        cv::Mat q0(mag, cv::Rect(0,0,cx,cy));
        cv::Mat q1(mag, cv::Rect(cx,0,cx,cy));
        cv::Mat q2(mag, cv::Rect(0,cy,cx,cy));
        cv::Mat q3(mag, cv::Rect(cx,cy,cx,cy));
        cv::Mat tmp;
        q0.copyTo(tmp);
        q3.copyTo(q0);
        tmp.copyTo(q3);
        q1.copyTo(tmp);
        q2.copyTo(q1);
        tmp.copyTo(q2);

        // Look for peaks at 1/3 and 1/2 of Nyquist
        // Simplified: just check energy at expected frequencies
        float energy_stripe = 0.0f, energy_pentile = 0.0f;
        int radius = 3;
        for (int dy = -radius; dy <= radius; ++dy) {
            for (int dx = -radius; dx <= radius; ++dx) {
                int x_stripe = mag.cols/3 + dx;
                int y_stripe = mag.rows/2 + dy;
                if (x_stripe >=0 && x_stripe<mag.cols && y_stripe>=0 && y_stripe<mag.rows)
                    energy_stripe += mag.at<float>(y_stripe, x_stripe);

                int x_pentile = mag.cols/2 + dx;
                int y_pentile = mag.rows/2 + dy;
                if (x_pentile>=0 && x_pentile<mag.cols && y_pentile>=0 && y_pentile<mag.rows)
                    energy_pentile += mag.at<float>(y_pentile, x_pentile);
            }
        }

        if (energy_pentile > energy_stripe * 1.5)
            return SubpixelGeometry::Layout::PENTILE_RG;
        else
            return SubpixelGeometry::Layout::RGB_STRIPE;
    }
};

SubpixelCalibration::SubpixelCalibration() : pimpl(std::make_unique<Impl>()) {}
SubpixelCalibration::~SubpixelCalibration() = default;

cv::Mat SubpixelCalibration::gamma_linearize(const cv::Mat& img, double gamma) {
    return Impl::gamma_linearize(img, gamma);
}

cv::Mat SubpixelCalibration::unwrap_phase(const cv::Mat& wrapped, const cv::Mat& quality) {
    return Impl::quality_guided_unwrap(wrapped, quality);
}

cv::Mat SubpixelCalibration::fit_quadratic_surface(const cv::Mat& unwrapped, const cv::Mat& weight) {
    return Impl::fit_quadratic(unwrapped, weight);
}

SubpixelGeometry::Layout SubpixelCalibration::detect_layout(const cv::Mat& fringe_image) {
    return Impl::detect_layout_fft(fringe_image);
}

SubpixelGeometry SubpixelCalibration::calibrate_from_camera(
    const std::vector<cv::Mat>& captured_fringes,
    float expected_pattern_freq,
    bool patterns_horizontal)
{
    // Validate input
    if (captured_fringes.empty()) {
        return SubpixelGeometry(); // return default geometry
    }

    // Phase shifting must use a fixed number of steps per carrier frequency.
    // Full mobile capture uses 15 frames (3 frequencies × 5 steps). Use the
    // middle carrier (indices 5–9) for a valid 5-bucket PSI solve; otherwise
    // take the first contiguous group of five frames.
    std::vector<cv::Mat> group;
    if (captured_fringes.size() >= 15) {
        group.assign(captured_fringes.begin() + 5, captured_fringes.begin() + 10);
    } else if (captured_fringes.size() >= 5) {
        group.assign(captured_fringes.begin(), captured_fringes.begin() + 5);
    } else {
        return SubpixelGeometry();
    }

    // Step 1: Gamma linearize each captured image [0024]
    std::vector<cv::Mat> linear_fringes;
    for (const auto& img : group) {
        linear_fringes.push_back(gamma_linearize(img, 2.2)); // assume gamma 2.2
    }

    // Step 2: Compute modulation quality and wrapped phase [0025]
    auto [wrapped, quality] = Impl::compute_modulation(linear_fringes);

    // Step 3: Quality‑guided unwrapping
    cv::Mat unwrapped = unwrap_phase(wrapped, quality);

    // Step 4: Fit quadratic surface to correct perspective [0025]
    cv::Mat weight = quality.clone(); // use modulation as weight
    cv::Mat coeffs = fit_quadratic_surface(unwrapped, weight);

    // Step 5: Derive subpixel pitch and offsets from gradients
    SubpixelGeometry geom;
    // From coefficients: phase = a x^2 + b y^2 + c x y + d x + e y + f
    // Gradient in x: d phase/dx = 2a x + c y + d
    // At center (x=0,y=0) gradient is d
    float grad_x = coeffs.at<float>(3); // d
    float grad_y = coeffs.at<float>(4); // e

    // Frequency = gradient / (2pi)   (phase in radians)
    float freq_x = grad_x / (2 * CV_PI);
    float freq_y = grad_y / (2 * CV_PI);

    // Subpixel pitch = 1 / (freq * logical_pixel_size) but logical pixel size unknown.
    // Instead we compute offset ratios:
    geom.pitch_x = 1.0f / (freq_x * expected_pattern_freq); // simplified
    geom.pitch_y = 1.0f / (freq_y * expected_pattern_freq);

    // Offsets: from constant term? Actually we need phase at subpixel positions.
    // For now, default to stripe with offsets.
    geom.offset_x = {0.0f, geom.pitch_x, 2*geom.pitch_x};
    geom.offset_y = {0.0f, 0.0f, 0.0f};

    // Step 6: Layout detection using FFT [0026]
    geom.layout = detect_layout(group[0]);

    // Step 7: Orientation detection (if patterns_horizontal, we can infer)
    geom.horizontal_orientation = patterns_horizontal; // true means subpixels side‑by‑side horizontally

    return geom;
}

SubpixelGeometry SubpixelCalibration::calibrate_from_screenshots(
    const std::vector<cv::Mat>& screenshots)
{
    // Similar to camera, but assume no gamma distortion (screenshots are already linear?)
    // We'll still linearize as screenshots may be in sRGB.
    return calibrate_from_camera(screenshots, 0.1f, true);
}

} // namespace display_calibration
