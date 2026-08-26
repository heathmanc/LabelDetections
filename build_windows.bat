@echo off
REM Build LabelVision Studio on Windows.
REM
REM   build_windows.bat full       everything bundled (default) -- Model Test,
REM                                Auto-label and Pre-label & Review work with
REM                                no per-machine setup. ~3 GB, slow build.
REM   build_windows.bat lean       labeling/capture/export only. ~400 MB, fast.
REM                                Model Test / Auto-label / Pre-label are NOT
REM                                available -- a frozen app cannot import
REM                                Ultralytics from the system Python.
REM   build_windows.bat full installer   also build the setup .exe (Inno Setup)
REM   build_windows.bat lean installer
REM
REM Output:
REM   dist\LabelVisionStudio\LabelVisionStudio.exe
REM   installer\LabelVisionStudio-<version>-<edition>-setup.exe

setlocal enabledelayedexpansion

set EDITION=%~1
if "%EDITION%"=="" set EDITION=full
if /i "%EDITION%"=="full" goto :edition_ok
if /i "%EDITION%"=="lean" goto :edition_ok
echo ERROR: first argument must be "lean" or "full" (got "%EDITION%").
exit /b 2
:edition_ok

set MAKE_INSTALLER=0
if /i "%~2"=="installer" set MAKE_INSTALLER=1

echo ==========================================================
echo   LabelVision Studio -- %EDITION% build
echo ==========================================================

echo.
echo === Installing build dependencies ===
python -m pip install --upgrade pip || goto :error
python -m pip install "PySide6>=6.6" "opencv-python>=4.8" "numpy>=1.24" "PyYAML>=6.0" "pypylon>=3.0" || goto :error
python -m pip install "pyinstaller~=6.6" || goto :error

if /i "%EDITION%"=="lean" (
    set LABELVISION_LEAN=1
    echo.
    echo === Lean build: torch/ultralytics excluded ===
) else (
    set LABELVISION_LEAN=
    echo.
    echo === Full build: installing cu128 torch + ultralytics ===
    REM cu128, NOT the default index: the default gives CPU-only torch and cu124
    REM has no sm_120 kernels, so either would fail on RTX 50-series hardware.
    python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128 || goto :error
    python -m pip install "ultralytics>=8.3.0" || goto :error
    python -c "import torch; print('bundling torch', torch.__version__, 'cuda', torch.version.cuda); print('archs', torch.cuda.get_arch_list())" || goto :error
)

echo.
echo === Cleaning previous build ===
if exist build rmdir /s /q build
if exist dist rmdir /s /q dist

echo.
echo === Generating app icon ===
python scripts\make_icon.py || goto :error

echo.
echo === Running PyInstaller ===
pyinstaller LabelVisionStudio.spec --noconfirm || goto :error

echo.
echo === Staging support files ===
REM GPU setup tooling for the yolo CLI that Training/Evaluate shell out to.
mkdir "dist\LabelVisionStudio\scripts" 2>nul
copy /y "setup_gpu.bat" "dist\LabelVisionStudio\" >nul
copy /y "scripts\setup_gpu.py" "dist\LabelVisionStudio\scripts\" >nul
copy /y "scripts\__init__.py" "dist\LabelVisionStudio\scripts\" >nul
copy /y "README.md" "dist\LabelVisionStudio\" >nul

if "%MAKE_INSTALLER%"=="1" (
    echo.
    call "%~dp0make_installer.bat" %EDITION% || goto :error
)

echo.
echo ==========================================================
echo   BUILD COMPLETE (%EDITION%)
echo ==========================================================
echo   App: dist\LabelVisionStudio\LabelVisionStudio.exe
if "%MAKE_INSTALLER%"=="1" echo   Installer: installer\
echo.
goto :eof

:error
echo.
echo BUILD FAILED with error code %errorlevel%.
exit /b %errorlevel%
