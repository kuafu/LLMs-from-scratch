@echo off
setlocal DisableDelayedExpansion

cd /d "%~dp0"

echo =======================================================
echo     Generating Visual Studio Solution (.sln)
echo =======================================================

rem An optional first argument selects a standalone LibTorch installation.
set "LIBTORCH_PATH=%~1"
set "BUILD_DIR=%~2"

if not defined BUILD_DIR (
    if defined LIBTORCH_PATH (
        set "BUILD_DIR=build-libtorch"
    ) else (
        set "BUILD_DIR=build-python-torch"
    )
)

echo [INFO] Build directory: %BUILD_DIR%
echo [INFO] CMake Generator: Visual Studio 17 2022 (x64)
echo [INFO] CUDA: Disabled (CPU mode)
echo.

if defined LIBTORCH_PATH (
    echo [INFO] Using the specified LibTorch directory: %LIBTORCH_PATH%
    cmake -B "%BUILD_DIR%" -S . -G "Visual Studio 17 2022" -A x64 "-DCMAKE_PREFIX_PATH=%LIBTORCH_PATH%" -DUSE_CUDA=OFF
) else (
    echo [INFO] Using Python PyTorch unless a Torch path is set in the environment or cache.
    cmake -B "%BUILD_DIR%" -S . -G "Visual Studio 17 2022" -A x64 -DUSE_CUDA=OFF
)

if errorlevel 1 goto failed

echo.
echo =======================================================
echo [SUCCESS] Visual Studio solution generated successfully!
echo Solution file located at:
echo   %BUILD_DIR%\gpt_cpp.sln
echo =======================================================
pause
exit /b 0

:failed
echo.
echo [FAILED] CMake configuration failed. Please check your environment.
pause
exit /b 1
