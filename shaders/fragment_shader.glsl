#version 330 core
in vec2 TexCoord;
out vec4 FragColor;

uniform sampler2D screenTexture;
uniform vec3 subpixelOffsets;    // x offsets for R,G,B (in logical pixel fraction)
uniform vec3 subpixelOffsetsY;   // y offsets for R,G,B (for vertical/diamond layouts)
uniform float pixelWidth;        // logical pixel width in screen coordinates
uniform float pixelHeight;       // logical pixel height in screen coordinates

// Layout types matching SubpixelGeometry::Layout enum
// 0 = RGB_STRIPE, 1 = BGR_STRIPE, 2 = PENTILE_RG,
// 3 = DIAMOND_PENTILE, 4 = VERTICAL_STRIPE, 5 = UNKNOWN (defaults to RGB)
uniform int layoutType;

// Perceptual weights from patent
const float WEIGHT_R = 1.1;
const float WEIGHT_G = 0.95;
const float WEIGHT_B = 1.05;

void main() {
    // Get the logical pixel color
    vec4 color = texture(screenTexture, TexCoord);

    // Determine fragment's position within the logical pixel (0..1)
    float fx = fract(gl_FragCoord.x / pixelWidth);
    float fy = fract(gl_FragCoord.y / pixelHeight);

    // Compute subpixel coverage based on layout type
    float r_cov, g_cov, b_cov;

    if (layoutType == 1) {
        // BGR_STRIPE: reversed horizontal order — B left, G center, R right
        b_cov = max(0.0, 1.0 - abs(fx - subpixelOffsets.x) * 3.0);
        g_cov = max(0.0, 1.0 - abs(fx - subpixelOffsets.y) * 3.0);
        r_cov = max(0.0, 1.0 - abs(fx - subpixelOffsets.z) * 3.0);
    } else if (layoutType == 2) {
        // PENTILE_RG: green always present, R/B alternate at half density
        g_cov = max(0.0, 1.0 - abs(fx - subpixelOffsets.y) * 3.0);
        r_cov = max(0.0, 1.0 - abs(fx - subpixelOffsets.x) * 3.0) * 0.5;
        b_cov = max(0.0, 1.0 - abs(fx - subpixelOffsets.z) * 3.0) * 0.5;
    } else if (layoutType == 3) {
        // DIAMOND_PENTILE: 2D distance for diamond grid arrangement
        float r_dist = sqrt(pow((fx - subpixelOffsets.x) * 3.0, 2.0)
                          + pow((fy - subpixelOffsetsY.x) * 3.0, 2.0));
        float g_dist = sqrt(pow((fx - subpixelOffsets.y) * 3.0, 2.0)
                          + pow((fy - subpixelOffsetsY.y) * 3.0, 2.0));
        float b_dist = sqrt(pow((fx - subpixelOffsets.z) * 3.0, 2.0)
                          + pow((fy - subpixelOffsetsY.z) * 3.0, 2.0));
        r_cov = max(0.0, 1.0 - r_dist);
        g_cov = max(0.0, 1.0 - g_dist);
        b_cov = max(0.0, 1.0 - b_dist);
    } else if (layoutType == 4) {
        // VERTICAL_STRIPE: subpixels stacked vertically (coverage based on y)
        r_cov = max(0.0, 1.0 - abs(fy - subpixelOffsetsY.x) * 3.0);
        g_cov = max(0.0, 1.0 - abs(fy - subpixelOffsetsY.y) * 3.0);
        b_cov = max(0.0, 1.0 - abs(fy - subpixelOffsetsY.z) * 3.0);
    } else {
        // RGB_STRIPE (default, also handles UNKNOWN)
        r_cov = max(0.0, 1.0 - abs(fx - subpixelOffsets.x) * 3.0);
        g_cov = max(0.0, 1.0 - abs(fx - subpixelOffsets.y) * 3.0);
        b_cov = max(0.0, 1.0 - abs(fx - subpixelOffsets.z) * 3.0);
    }

    // Edge detection using Sobel operator on luminance
    vec2 offsets[9] = vec2[](
        vec2(-1,-1), vec2(0,-1), vec2(1,-1),
        vec2(-1, 0), vec2(0, 0), vec2(1, 0),
        vec2(-1, 1), vec2(0, 1), vec2(1, 1)
    );
    float gx = 0.0, gy = 0.0;
    float kernel_x[9] = float[](
        -1, 0, 1,
        -2, 0, 2,
        -1, 0, 1
    );
    float kernel_y[9] = float[](
        -1,-2,-1,
         0, 0, 0,
         1, 2, 1
    );
    vec2 texelSize = vec2(1.0 / pixelWidth, 1.0 / pixelHeight);
    for (int i = 0; i < 9; ++i) {
        vec2 offset = offsets[i] * texelSize;
        float lum = dot(texture(screenTexture, TexCoord + offset).rgb, vec3(0.2126, 0.7152, 0.0722));
        gx += kernel_x[i] * lum;
        gy += kernel_y[i] * lum;
    }
    float edgeStrength = sqrt(gx*gx + gy*gy) / 8.0; // normalized gradient

    // Modulate weights
    float mod = 1.0 + edgeStrength;
    float wR = WEIGHT_R * mod;
    float wG = WEIGHT_G * mod;
    float wB = WEIGHT_B * mod;

    // Combine
    FragColor.r = color.r * r_cov * wR;
    FragColor.g = color.g * g_cov * wG;
    FragColor.b = color.b * b_cov * wB;
    FragColor.a = color.a;
}
