#include "display_calibration/calibration_token.h"
#include "display_calibration/rendering_pipeline.h"
#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

static void print_usage(const char* exe) {
    std::cerr
        << "Usage: " << exe << " [--token <calibration_result.json>] [--input <image.png>]\n"
        << "\n"
        << "  --token   JSON from capture-app (calibration_result.json or API export).\n"
        << "            Maps layout, pitch_*, offset_* into the perceptual renderer.\n"
        << "  --input   Source image (default: input.png). BGR/RGB image file.\n"
        << "\n"
        << "Without --token, defaults to RGB stripe geometry (1/3 subpixel spacing).\n";
}

int main(int argc, char** argv) {
    std::string token_path;
    std::string input_path = "input.png";

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--help" || a == "-h") {
            print_usage(argv[0]);
            return 0;
        }
        if (a == "--token" && i + 1 < argc) {
            token_path = argv[++i];
            continue;
        }
        if (a == "--input" && i + 1 < argc) {
            input_path = argv[++i];
            continue;
        }
        std::cerr << "Unknown argument: " << a << "\n";
        print_usage(argv[0]);
        return 2;
    }

    display_calibration::SubpixelGeometry geom;
    if (!token_path.empty()) {
        std::string err;
        if (!display_calibration::load_subpixel_geometry_from_token_json(token_path, geom, &err)) {
            std::cerr << "Token load failed: " << err << std::endl;
            return 1;
        }
        std::cout << "Loaded calibration token: " << token_path << std::endl;
    } else {
        geom.layout = display_calibration::SubpixelGeometry::Layout::RGB_STRIPE;
        geom.horizontal_orientation = true;
        std::cout << "Using default RGB stripe geometry (no --token)." << std::endl;
    }

    cv::Mat img = cv::imread(input_path);
    if (img.empty()) {
        std::cerr << "Could not load image: " << input_path << std::endl;
        return -1;
    }
    cv::cvtColor(img, img, cv::COLOR_BGR2RGBA);
    img.convertTo(img, CV_32F, 1.0 / 255.0);

    std::vector<display_calibration::PixelRGBA> input(static_cast<size_t>(img.total()));
    for (int i = 0; i < img.total(); ++i) {
        cv::Vec4f p = img.at<cv::Vec4f>(i);
        input[static_cast<size_t>(i)] = {p[0], p[1], p[2], p[3]};
    }

    display_calibration::RenderingPipeline pipeline;
    pipeline.set_subpixel_geometry(geom);

    auto output = pipeline.process_frame(input, img.cols, img.rows, false);

    std::cout << "Demo completed successfully." << std::endl;
    std::cout << "Input: " << input_path << " (" << img.cols << "x" << img.rows << ")" << std::endl;
    std::cout << "Output pixels: " << output.size() << std::endl;
    std::cout << "Feed output to your display path or GL shader using the same geometry.\n";

    return 0;
}
