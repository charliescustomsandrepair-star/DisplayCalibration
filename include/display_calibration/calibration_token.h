#pragma once

#include "types.h"
#include <string>

namespace display_calibration {

// Load SubpixelGeometry from JSON produced by capture-app (calibration_result.json
// or /api/calibration-token). Unknown or missing fields keep sensible defaults.
bool load_subpixel_geometry_from_token_json(const std::string& path,
                                            SubpixelGeometry& out,
                                            std::string* error_message = nullptr);

} // namespace display_calibration
