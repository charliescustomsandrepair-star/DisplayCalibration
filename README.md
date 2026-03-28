# DisplayCalibration

**Subpixel-aware perceptual rendering + automated calibration toolkit**

This project lets you calibrate the exact subpixel layout of any display (RGB stripe, Pentile, Diamond Pentile, etc.) using only a phone camera, then applies advanced perceptual rendering to improve image quality on that display.

### Features
- Fully automated structured-light calibration (15 phase-shifted patterns)
- Phone-based capture with real-time edge detection and cropping
- High-quality token generation (`calibration_result.json`)
- C++ rendering pipeline with:
  - Subpixel coverage computation for multiple layouts
  - Perceptual luminance weighting + edge modulation
  - Optional temporal stabilization
- Cross-platform demo executable (Windows, macOS, Linux)
- Pre-built binaries available via GitHub Releases

---

## Quick Start (Easiest Way)

### 1. Download Pre-built Demo
Go to the [Releases page](https://github.com/YOURUSERNAME/DisplayCalibration/releases) and download the latest version for your platform:

- `demo-windows.zip` → `demo.exe`
- `demo-macos.zip` → `demo` (make executable with `chmod +x demo`)
- `demo-linux.zip` → `demo`

### 2. Run Calibration (Phone + PC)

1. Clone or download this repository.
2. Open a terminal in the `capture-app` folder.
3. Run the server:

   ```bash
   python -m venv .venv
   # Windows:
   .\.venv\Scripts\Activate.ps1
   # macOS/Linux:
   source .venv/bin/activate
   pip install -r requirements.txt
   python server.py


--
## Quick Start (Easiest Way)

### 1. Download Pre-built Demo
Go to the [Releases page] and download the latest version for your platform:

- `demo-windows.zip` → `demo.exe`
- `demo-macos.zip` → `demo` (make executable with `chmod +x demo`)
- `demo-linux.zip` → `demo`

### 2. Run Calibration (Phone + PC)

1. Clone or download this repository.
2. Open a terminal in the `capture-app` folder.
3. Run the server:

   ```bash
   python -m venv .venv
   # Windows:
   .\.venv\Scripts\Activate.ps1
   # macOS/Linux:
   source .venv/bin/activate
   pip install -r requirements.txt
   python server.py
-

## 3. C++ demo (optional)

This demo needs a C++ toolchain (MSVC) and dependencies (OpenCV/GLEW/GLFW).

### One-time setup (Windows)

1) Install a compiler:
- Install **Visual Studio Build Tools** (or Visual Studio) with **Desktop development with C++**.

2) Install dependencies via vcpkg:
- Install vcpkg (default expected path is `C:\src\vcpkg`).
- Optional: set `VCPKG_ROOT` to your vcpkg folder if you installed it elsewhere.
- Then CMake will auto-install dependencies from `vcpkg.json` during configure.
 - If `ninja` isn’t found during configure, install it (e.g. `winget install Ninja-build.Ninja`) or ensure it’s on `PATH`.

### Build + run

```powershell
cd C:\Users\charl\Desktop\DisplayCalibration
cmake --preset demo-vcpkg
cmake --build --preset demo-vcpkg
.\build\demo-vcpkg\demo.exe --token capture-app\calibration_result.json --input your.png
```

On your PC monitor:
Open the URL shown (e.g. http://192.168.x.x:5000/patterns)
Press F11 for fullscreen

On your phone (same Wi-Fi):
Open the /capture page (or scan the QR code)
Align the camera so the monitor fills ~95% of the view
Tap Start Capture


After ~15 captures, you'll get capture-app/calibration_result.json
3. Run the Demo
Bash# Windows
.\demo.exe --token capture-app\calibration_result.json --input your_image.png

# macOS / Linux
./demo --token capture-app/calibration_result.json --input your_image.png
Without a token it falls back to standard RGB stripe.

Building from Source (Advanced)
Prerequisites
Windows:

Visual Studio Build Tools (Desktop development with C++)
vcpkg (recommended at C:\vcpkg)

macOS:
Bashbrew install opencv glew glfw cmake ninja
Linux:
Bashsudo apt-get update
sudo apt-get install -y libopencv-dev libglew-dev libglfw3-dev cmake ninja-build pkg-config
Build Commands
Bashcmake --preset demo-vcpkg   # Windows (uses vcpkg)
# or
cmake --preset demo         # macOS / Linux (uses system packages)

cmake --build --preset demo-vcpkg --config Release   # Windows
# or
cmake --build --preset demo                          # macOS / Linux
The executable will be in build/demo-vcpkg/Release/demo.exe (Windows) or build/demo (others).

Project Structure

capture-app/ – Python Flask + SocketIO server + calibration pipeline
src/ – C++ core library (display_calibration)
shaders/ – GLSL reference shaders (CPU pipeline is the source of truth)
CMakeLists.txt + presets – Cross-platform build configuration


How It Works (Technical Overview)

Capture: PC displays sinusoidal fringes → phone captures 15 images (3 frequencies × 5 phases)
Processing: Gamma correction, phase shifting, quality-guided unwrapping, quadratic surface fitting, FFT layout detection
Token: Produces calibration_result.json containing layout, pitch, and subpixel offsets
Rendering: The C++ pipeline uses the token to compute accurate subpixel coverage, perceptual weights, and edge-aware output


Notes

Make sure PC and phone are on the same Wi-Fi network.
Dim ambient lighting gives best results.
The demo currently processes the image and prints statistics. You can easily extend main_demo.cpp to save the output image or feed it into your own renderer/shader.
Firewall: Allow Python on private networks if the phone cannot connect.


Contributing / Feedback
Feel free to open issues or pull requests. This project was built in ~5 months as a solo effort — feedback is very welcome!
---



---



-

-

-
