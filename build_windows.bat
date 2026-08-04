@echo off
REM Build BungVision Label Studio into a standalone Windows folder + exe.
REM Run this from a Windows machine with Python 3.10-3.12 installed.
REM
REM Output: dist\BungVisionLabelStudio\BungVisionLabelStudio.exe
REM
REM This is a FULL build: torch/ultralytics are bundled so Model Test,
REM Auto-label, and Pre-label & Review work without any per-machine setup.
REM torch comes from the cu128 index so it carries Blackwell (sm_120) kernels
REM for RTX 50-series cards. Expect a large download and a slow build.
REM
REM For a labeling-only build without the ML stack:
REM     set BUNGVISION_LEAN=1
REM     build_windows.bat

setlocal

echo === Installing build dependencies ===
python -m pip install --upgrade pip || goto :error
python -m pip install "PySide6>=6.6" "opencv-python>=4.8" "numpy>=1.24" "PyYAML>=6.0" "pypylon>=3.0" || goto :error
python -m pip install "pyinstaller~=6.6" || goto :error

if "%BUNGVISION_LEAN%"=="1" (
    echo === Lean build: torch/ultralytics excluded ===
) else (
    echo === Full build: installing cu128 torch + ultralytics ===
    REM cu128, NOT the default index: default gives CPU-only torch, and cu124
    REM has no sm_120 kernels, so either would fail on RTX 50-series hardware.
    python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128 || goto :error
    python -m pip install "ultralytics>=8.3.0" || goto :error
)

echo === Cleaning previous build ===
if exist build rmdir /s /q build
if exist dist rmdir /s /q dist

echo === Generating app icon ===
python scripts\make_icon.py || goto :error

echo === Running PyInstaller ===
pyinstaller BungVisionLabelStudio.spec --noconfirm || goto :error

echo === Staging seed data beside the exe ===
REM Frozen builds resolve user data next to the .exe, so starter recipes and
REM the class config must live there rather than inside _internal.
mkdir "dist\BungVisionLabelStudio\data\recipes" 2>nul
copy /y "data\class_config.json" "dist\BungVisionLabelStudio\data\" >nul
copy /y "data\recipes\*.json" "dist\BungVisionLabelStudio\data\recipes\" >nul
copy /y "README.md" "dist\BungVisionLabelStudio\" >nul

REM GPU setup tooling, so deployment machines can install the cu128 torch that
REM the yolo CLI needs on RTX 50-series cards.
mkdir "dist\BungVisionLabelStudio\scripts" 2>nul
copy /y "setup_gpu.bat" "dist\BungVisionLabelStudio\" >nul
copy /y "scripts\setup_gpu.py" "dist\BungVisionLabelStudio\scripts\" >nul
copy /y "scripts\__init__.py" "dist\BungVisionLabelStudio\scripts\" >nul
copy /y "requirements.txt" "dist\BungVisionLabelStudio\" >nul

echo.
echo === BUILD COMPLETE ===
echo Run: dist\BungVisionLabelStudio\BungVisionLabelStudio.exe
echo Zip the dist\BungVisionLabelStudio folder to distribute it.
goto :eof

:error
echo.
echo BUILD FAILED with error code %errorlevel%.
exit /b %errorlevel%
