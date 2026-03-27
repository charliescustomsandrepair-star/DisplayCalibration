#include "display_calibration/calibration_token.h"

#include <fstream>
#include <regex>
#include <sstream>

namespace display_calibration {

namespace {

SubpixelGeometry::Layout parse_layout_string(const std::string& s) {
    if (s == "RGB_STRIPE") return SubpixelGeometry::Layout::RGB_STRIPE;
    if (s == "BGR_STRIPE") return SubpixelGeometry::Layout::BGR_STRIPE;
    if (s == "PENTILE_RG") return SubpixelGeometry::Layout::PENTILE_RG;
    if (s == "DIAMOND_PENTILE") return SubpixelGeometry::Layout::DIAMOND_PENTILE;
    if (s == "VERTICAL_STRIPE") return SubpixelGeometry::Layout::VERTICAL_STRIPE;
    return SubpixelGeometry::Layout::UNKNOWN;
}

bool read_file(const std::string& path, std::string& out, std::string* err) {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        if (err) *err = "Cannot open file: " + path;
        return false;
    }
    std::ostringstream ss;
    ss << f.rdbuf();
    out = ss.str();
    if (out.size() >= 3 &&
        static_cast<unsigned char>(out[0]) == 0xEF &&
        static_cast<unsigned char>(out[1]) == 0xBB &&
        static_cast<unsigned char>(out[2]) == 0xBF) {
        out.erase(0, 3);
    }
    return true;
}

bool parse_float_after_key(const std::string& j, const char* key, float& v) {
    std::string pat = std::string(R"(\")") + key + R"(\"\s*:\s*([-0-9.eE+]+))";
    std::smatch m;
    if (!std::regex_search(j, m, std::regex(pat))) return false;
    try {
        v = std::stof(m[1].str());
    } catch (...) {
        return false;
    }
    return true;
}

bool parse_bool_after_key(const std::string& j, const char* key, bool& v) {
    std::string pat = std::string(R"(\")") + key + R"(\"\s*:\s*(true|false))";
    std::smatch m;
    if (!std::regex_search(j, m, std::regex(pat))) return false;
    v = (m[1].str() == "true");
    return true;
}

bool parse_layout_value(const std::string& j, std::string& s) {
    static const std::regex re(R"re("layout"\s*:\s*"([^"]*)")re");
    std::smatch m;
    if (!std::regex_search(j, m, re)) return false;
    s = m[1].str();
    return true;
}

bool parse_float3_array(const std::string& j, const char* key, std::array<float, 3>& arr) {
    std::string pat = std::string(R"(\")") + key +
                      R"(\"\s*:\s*\[\s*([-0-9.eE+]+)\s*,\s*([-0-9.eE+]+)\s*,\s*([-0-9.eE+]+)\s*\])";
    std::smatch m;
    if (!std::regex_search(j, m, std::regex(pat))) return false;
    try {
        arr[0] = std::stof(m[1].str());
        arr[1] = std::stof(m[2].str());
        arr[2] = std::stof(m[3].str());
    } catch (...) {
        return false;
    }
    return true;
}

} // namespace

bool load_subpixel_geometry_from_token_json(const std::string& path,
                                            SubpixelGeometry& out,
                                            std::string* error_message) {
    std::string j;
    if (!read_file(path, j, error_message)) return false;

    out = SubpixelGeometry{};

    std::string layout_s;
    if (parse_layout_value(j, layout_s))
        out.layout = parse_layout_string(layout_s);

    parse_bool_after_key(j, "horizontal_orientation", out.horizontal_orientation);

    float px = out.pitch_x, py = out.pitch_y;
    if (parse_float_after_key(j, "pitch_x", px)) out.pitch_x = px;
    if (parse_float_after_key(j, "pitch_y", py)) out.pitch_y = py;

    std::array<float, 3> ox = out.offset_x, oy = out.offset_y;
    if (parse_float3_array(j, "offset_x", ox)) out.offset_x = ox;
    if (parse_float3_array(j, "offset_y", oy)) out.offset_y = oy;

    if (out.layout == SubpixelGeometry::Layout::UNKNOWN) {
        out.layout = SubpixelGeometry::Layout::RGB_STRIPE;
    }

    return true;
}

} // namespace display_calibration
