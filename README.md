#1 DisplayCalibration

**How to run:** use **§2** below.  



## 2. Run calibration (phone + PC)

```powershell
cd C:\Users\charl\Desktop\DisplayCalibration\capture-app
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python server.py
```

- PC monitor: URL printed for **`/patterns`** → **F11** fullscreen  
- Phone (same Wi‑Fi): **`/capture`** or QR from the PC  
- Result: `capture-app\calibration_result.json` after 15 captures  

Firewall: allow Python on private networks if the phone cannot connect.

---

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

---



---



-

-

-
