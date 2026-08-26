@echo off
REM Detect the NVIDIA GPU and install the matching PyTorch CUDA build.
REM
REM Fixes: "CUDA error: no kernel image is available for execution on the device"
REM which happens when the installed torch wheel has no kernels compiled for
REM this GPU's architecture (e.g. cu124 wheels on an RTX 50-series / sm_120).
REM
REM Run from the project folder with the venv active:
REM     .venv\Scripts\activate
REM     setup_gpu.bat
REM
REM Options:
REM     setup_gpu.bat --dry-run   show the pip command without installing
REM     setup_gpu.bat --check     only verify the current install

setlocal

python scripts\setup_gpu.py %*
if errorlevel 1 goto :error

echo.
echo === GPU SETUP COMPLETE ===
goto :eof

:error
echo.
echo GPU setup did not complete successfully. See the messages above.
exit /b %errorlevel%
