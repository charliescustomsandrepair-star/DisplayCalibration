cmake_minimum_required(VERSION 3.15)

# Auto-detect vcpkg toolchain if the user did not pass one explicitly.
# This avoids a hard failure when VCPKG_ROOT is unset in presets.
if(NOT DEFINED CMAKE_TOOLCHAIN_FILE)
    # Prefer a standalone vcpkg checkout if present (avoids VS-bundled vcpkg baseline rules).
    if(EXISTS "C:/vcpkg/scripts/buildsystems/vcpkg.cmake")
        set(CMAKE_TOOLCHAIN_FILE "C:/vcpkg/scripts/buildsystems/vcpkg.cmake" CACHE STRING "vcpkg toolchain")
    elseif(EXISTS "C:/src/vcpkg/scripts/buildsystems/vcpkg.cmake")
        set(CMAKE_TOOLCHAIN_FILE "C:/src/vcpkg/scripts/buildsystems/vcpkg.cmake" CACHE STRING "vcpkg toolchain")
    elseif(DEFINED ENV{VCPKG_ROOT} AND EXISTS "$ENV{VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake")
        set(CMAKE_TOOLCHAIN_FILE "$ENV{VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake" CACHE STRING "vcpkg toolchain")
    endif()
endif()

project(DisplayCalibration VERSION 1.0.0 LANGUAGES CXX)

# C++17 required
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Options
option(BUILD_DEMO "Build demo application" OFF)
option(USE_OPENCV "Use OpenCV for calibration image processing" OFF)

# Library - core without dependencies
add_library(display_calibration STATIC
    src/rendering_pipeline.cpp
    src/calibration_token.cpp
)

target_include_directories(display_calibration
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# Conditional: Add subpixel_calibration only if OpenCV is available
if(USE_OPENCV)
    # Workaround: in some environments CMake may have Protobuf_FIND_VERSION set
    # to an empty string, causing vcpkg's Protobuf config discovery to reject
    # the installed Protobuf as "not compatible with requested version ''".
    unset(Protobuf_FIND_VERSION CACHE)
    unset(Protobuf_FIND_VERSION_MAJOR CACHE)
    unset(Protobuf_FIND_VERSION_MINOR CACHE)
    unset(Protobuf_FIND_VERSION_PATCH CACHE)
    unset(Protobuf_FIND_VERSION_TWEAK CACHE)

    unset(Protobuf_FIND_VERSION)
    unset(Protobuf_FIND_VERSION_MAJOR)
    unset(Protobuf_FIND_VERSION_MINOR)
    unset(Protobuf_FIND_VERSION_PATCH)
    unset(Protobuf_FIND_VERSION_TWEAK)

    # Prefer a config package (vcpkg ships one). Fallback to module mode if needed.
    find_package(OpenCV REQUIRED COMPONENTS core imgproc highgui CONFIG)
    target_sources(display_calibration PRIVATE src/subpixel_calibration.cpp)
    target_include_directories(display_calibration PRIVATE ${OpenCV_INCLUDE_DIRS})
    target_link_libraries(display_calibration ${OpenCV_LIBS})
    message(STATUS "Building with OpenCV support")
else()
    message(STATUS "Building core library only (no OpenCV)")
endif()

# Demo executable (optional, requires all dependencies)
if(BUILD_DEMO)
    if(NOT USE_OPENCV)
        message(WARNING "Demo requires OpenCV. Set USE_OPENCV=ON to build demo.")
    else()
        find_package(OpenGL REQUIRED)
        find_package(GLEW REQUIRED CONFIG)
        find_package(glfw3 REQUIRED CONFIG)

        # vcpkg/exported targets can have different names across ports/toolchains.
        # Pick whichever imported target exists so linking is deterministic.
        if(TARGET OpenGL::GL)
            set(DEMO_OPENGL_TARGET OpenGL::GL)
        elseif(DEFINED OPENGL_gl_LIBRARY)
            set(DEMO_OPENGL_TARGET ${OPENGL_gl_LIBRARY})
        else()
            message(FATAL_ERROR "OpenGL target/library not found after find_package(OpenGL).")
        endif()

        set(DEMO_GLEW_TARGET "")
        if(TARGET GLEW::GLEW)
            set(DEMO_GLEW_TARGET GLEW::GLEW)
        elseif(TARGET GLEW::glew)
            set(DEMO_GLEW_TARGET GLEW::glew)
        endif()
        if(DEMO_GLEW_TARGET STREQUAL "")
            message(FATAL_ERROR "GLEW imported target not found after find_package(GLEW).")
        endif()

        set(DEMO_GLFW_TARGET "")
        if(TARGET glfw)
            set(DEMO_GLFW_TARGET glfw)
        elseif(TARGET glfw3::glfw)
            set(DEMO_GLFW_TARGET glfw3::glfw)
        elseif(TARGET glfw3)
            set(DEMO_GLFW_TARGET glfw3)
        endif()
        if(DEMO_GLFW_TARGET STREQUAL "")
            message(FATAL_ERROR "glfw imported target not found after find_package(glfw3).")
        endif()
        
        add_executable(demo src/main_demo.cpp)
        target_link_libraries(demo display_calibration
            ${OpenCV_LIBS}
            ${DEMO_OPENGL_TARGET}
            ${DEMO_GLEW_TARGET}
            ${DEMO_GLFW_TARGET}
        )
        
        # Copy shaders to build directory
        add_custom_command(TARGET demo POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_directory
                ${CMAKE_CURRENT_SOURCE_DIR}/shaders
                $<TARGET_FILE_DIR:demo>/shaders
        )
    endif()
endif()

message(STATUS "Use -DUSE_OPENCV=ON to enable full demo")
message(STATUS "Use -DBUILD_DEMO=ON to build demo application")
