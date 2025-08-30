@echo off
REM Installs the rocQuantum-1 project.

echo Changing directory to build...
cd build

if %errorlevel% neq 0 (
    echo Failed to change directory to build. Make sure the project has been configured with build_rocq.bat first.
    goto :eof
)

echo Running install command...
cmake --install . --config Release

if %errorlevel% neq 0 (
    echo CMake install failed.
    goto :eof
)

echo Installation completed.
