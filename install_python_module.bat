@echo off
setlocal

REM =================================================================
REM PLEASE SET THE FOLLOWING PATHS
REM =================================================================

REM Set the path to your ROCm installation directory.
REM 예: set ROCM_PATH=C:\Users\한구원\Desktop\ROCm-develop
set "ROCM_PATH=C:\Users\한구원\Desktop\ROCm-develop"

REM Set the path to the directory containing the cmake.exe executable.
REM CMake를 설치한 경로의 bin 폴더를 지정해야 합니다.
REM 예: set CMAKE_BIN_PATH=C:\Program Files\CMake\bin
set "CMAKE_BIN_PATH="

REM =================================================================
REM DO NOT EDIT BELOW THIS LINE
REM =================================================================

echo ROCM_PATH is set to: %ROCM_PATH%
echo CMAKE_BIN_PATH is set to: %CMAKE_BIN_PATH%

if not defined CMAKE_BIN_PATH (
    echo ERROR: CMAKE_BIN_PATH is not set.
    echo Please edit this script and set the path to the directory containing cmake.exe.
    exit /b 1
)

if not exist "%CMAKE_BIN_PATH%\cmake.exe" (
    echo ERROR: cmake.exe not found at the specified CMAKE_BIN_PATH.
    echo Please verify the path: %CMAKE_BIN_PATH%
    exit /b 1
)

REM Add CMake to the PATH for this script's execution
set "PATH=%CMAKE_BIN_PATH%;%PATH%"

echo Temporarily updated PATH to include CMake:
echo %PATH%
echo.

echo Starting Python module installation...
echo Using pip to install from the current directory.
echo.

pip install .

if %errorlevel% neq 0 (
    echo.
    echo ERROR: Python module installation failed.
    echo Please check the output above for errors.
    exit /b %errorlevel%
)

echo.
echo Successfully installed the rocq Python module.
echo You can now try running the example scripts.

endlocal
