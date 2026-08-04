@echo off
REM Build BungVision Label Studio into a standalone Windows folder + exe.
REM Run this from a Windows machine with Python 3.10-3.12 installed.
REM
REM Output: dist\BungVisionLabelStudio\BungVisionLabelStudio.exe
REM
REM By default this excludes torch/ultralytics (~2.5GB) to keep the build small.
REM For a full build that bundles the ML stack, run:
REM     set BUNGVISION_BUNDLE_ML=1
REM     build_windows.bat

setlocal

echo === Installing build dependencies ===
python -m pip install --upgrade pip || goto :error
python -m pip install -r requirements.txt || goto :error
python -m pip install "pyinstaller~=6.6" || goto :error

if "%BUNGVISION_BUNDLE_ML%"=="1" (
    echo === Full build: installing ultralytics ===
    python -m pip install "ultralytics>=8.3.0" || goto :error
) else (
    echo === Lean build: torch/ultralytics excluded ===
)

echo === Cleaning previous build ===
if exist build rmdir /s /q build
if exist dist rmdir /s /q dist

echo === Running PyInstaller ===
pyinstaller BungVisionLabelStudio.spec --noconfirm || goto :error

echo === Staging seed data beside the exe ===
REM Frozen builds resolve user data next to the .exe, so starter recipes and
REM the class config must live there rather than inside _internal.
mkdir "dist\BungVisionLabelStudio\data\recipes" 2>nul
copy /y "data\class_config.json" "dist\BungVisionLabelStudio\data\" >nul
copy /y "data\recipes\*.json" "dist\BungVisionLabelStudio\data\recipes\" >nul
copy /y "README.md" "dist\BungVisionLabelStudio\" >nul

echo.
echo === BUILD COMPLETE ===
echo Run: dist\BungVisionLabelStudio\BungVisionLabelStudio.exe
echo Zip the dist\BungVisionLabelStudio folder to distribute it.
goto :eof

:error
echo.
echo BUILD FAILED with error code %errorlevel%.
exit /b %errorlevel%
