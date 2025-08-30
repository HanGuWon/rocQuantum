@echo off
REM build_rocq.bat - Build script for rocQuantum-1

REM --- User Configuration ---
REM **IMPORTANT**: Please set the following variables to match your system.

REM 1. Set the path to your ROCm installation.
REM    This is the directory containing the bin, lib, include, etc. folders.
set "ROCM_PATH=C:\Program Files\AMD\ROCm\6.1"

REM 2. Set the target AMD GPU architecture.
REM    You can specify one or more targets, separated by semicolons.
REM    Common targets: gfx906, gfx908, gfx90a, gfx1030, gfx1100
set "AMDGPU_TARGETS=gfx906"

REM --- End of User Configuration ---

REM --- Script ---
echo ==================================================================
echo                rocQuantum-1 Build Script
echo ==================================================================

REM Check if ROCM_PATH exists
if not exist "%ROCM_PATH%" (
    echo Error: ROCM_PATH not found at "%ROCM_PATH%".
    echo Please edit this script and set the ROCM_PATH variable.
    goto :eof
)

echo ROCm Path: %ROCM_PATH%
echo AMD GPU Targets: %AMDGPU_TARGETS%

REM Set up the environment
set "PATH=%ROCM_PATH%\bin;%PATH%"
set "HIP_DEVICE_LIB_PATH=%ROCM_PATH%\lib"

REM Create a build directory
if not exist .\build (
    mkdir build
)
cd build

REM Configure the project with CMake
echo.
echo --- Running CMake --- 
echo.
cmake .. -G "Visual Studio 17 2022" -A x64 ^
    -DCMAKE_BUILD_TYPE=Release ^
    -DCMAKE_CXX_COMPILER="%ROCM_PATH%/bin/hipcc.bat" ^
    -DCMAKE_INSTALL_PREFIX="..\install" ^
    -DAMDGPU_TARGETS="%AMDGPU_TARGETS%"

if %errorlevel% neq 0 (
    echo CMake configuration failed.
    goto :eof
)

REM Build the project
echo.
echo --- Building Project --- 
echo.
cmake --build . --config Release

if %errorlevel% neq 0 (
    echo Build failed.
    goto :eof
)

echo.
echo ==================================================================
echo                  Build completed successfully!
echo ==================================================================
echo.