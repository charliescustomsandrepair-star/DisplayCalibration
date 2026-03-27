#version 330 core
// Simple sinusoidal fringe pattern generator for calibration
out vec4 FragColor;

uniform float time;
uniform float phaseShift; // 0..2pi
uniform float freq;       // cycles per screen
uniform vec2 screenResolution; // display resolution in pixels

void main() {
    // Normalized screen coordinates
    float x = gl_FragCoord.x / screenResolution.x;
    float y = gl_FragCoord.y / screenResolution.y;

    // Horizontal fringes
    float value = 0.5 + 0.5 * sin(2.0 * 3.14159 * (freq * x + phaseShift));
    FragColor = vec4(value, value, value, 1.0);
}
