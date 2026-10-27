@echo off
setlocal enabledelayedexpansion

cd /d "%~dp0"

echo =======================================================
echo     Generating Visual Studio Solution (.sln)
echo =======================================================

rem Set LibTorch path from argument 1, or auto-detect H: and F: drives
set "LIBTORCH_PATH=%~1"

if "%LIBTORCH_PATH%"=="" (
    if exist "H:\LLM\libtorch" (
        set "LIBTORCH_PATH=H:\LLM\libtorch"
    ) else if exist "F:\LLM\libtorch" (
        set "LIBTORCH_PATH=F:\LLM\libtorch"
    ) else (
        echo [ERROR] LibTorch not found at H:\LLM\libtorch or F:\LLM\libtorch.
        echo Please pass the LibTorch directory as an argument:
        echo   %~nx0 D:\path\to\libtorch
        pause
        exit /b 1
    )
)

echo [INFO] Using LibTorch from: %LIBTORCH_PATH%
echo [INFO] CMake Generator: Visual Studio 17 2022 (x64)
echo.

cmake -B build -S . -G "Visual Studio 17 2022" -A x64 -DCMAKE_PREFIX_PATH="%LIBTORCH_PATH%"

if %ERRORLEVEL% equ 0 (
    echo.
    echo =======================================================
    echo [SUCCESS] Visual Studio solution generated successfully!
    echo Solution file located at:
    echo   %~dp0build\gpt_cpp.sln
    echo =======================================================
) else (
    echo.
    echo [FAILED] CMake configuration failed. Please check your environment.
)

echo.
pause
