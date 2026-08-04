@echo off
REM Package the built app into a single setup .exe using Inno Setup.
REM
REM   make_installer.bat            package whatever is in dist\ (edition "full")
REM   make_installer.bat lean       label the output as the lean edition
REM
REM Requires Inno Setup 6:  winget install JRSoftware.InnoSetup
REM Run build_windows.bat first -- this only packages, it does not build.

setlocal enabledelayedexpansion

set EDITION=%~1
if "%EDITION%"=="" set EDITION=full

if not exist "dist\BungVisionLabelStudio\BungVisionLabelStudio.exe" (
    echo ERROR: dist\BungVisionLabelStudio\BungVisionLabelStudio.exe not found.
    echo Run "build_windows.bat %EDITION%" first.
    exit /b 1
)

REM Locate the Inno Setup compiler. winget may install per-machine or per-user,
REM so check both plus PATH.
set ISCC=
for %%P in (
    "%ProgramFiles(x86)%\Inno Setup 6\ISCC.exe"
    "%ProgramFiles%\Inno Setup 6\ISCC.exe"
    "%LOCALAPPDATA%\Programs\Inno Setup 6\ISCC.exe"
    "%ProgramFiles(x86)%\Inno Setup 5\ISCC.exe"
) do (
    if exist %%P set ISCC=%%P
)
if "%ISCC%"=="" (
    where iscc >nul 2>nul && set ISCC=iscc
)
if "%ISCC%"=="" (
    echo ERROR: Inno Setup 6 not found.
    echo.
    echo Install it with EITHER:
    echo     winget install JRSoftware.InnoSetup
    echo         ^(note the spelling: JRSoftware, not JRSoftwre^)
    echo     choco install innosetup
    echo.
    echo Or download the installer directly:
    echo     https://jrsoftware.org/isdl.php
    echo.
    echo If it is already installed somewhere unusual, add its folder to PATH.
    exit /b 1
)
echo Using Inno Setup: %ISCC%

REM Read the version from the single source of truth.
for /f "usebackq delims=" %%V in (`python -c "import re,pathlib;print(re.search(r'APP_VERSION\s*=\s*\"([^\"]+)\"',pathlib.Path('bung_labeler/version.py').read_text()).group(1))"`) do set APPVER=%%V
if "%APPVER%"=="" (
    echo ERROR: could not read APP_VERSION from bung_labeler\version.py
    exit /b 1
)

echo === Building installer: v%APPVER% (%EDITION%) ===
echo     Compressing a full build takes a while.

%ISCC% "installer\BungVisionLabelStudio.iss" /DAppVersion=%APPVER% /DEdition=%EDITION% || goto :error

echo.
echo === INSTALLER READY ===
dir /b "installer\BungVisionLabelStudio-%APPVER%-%EDITION%-setup.exe"
echo     installer\BungVisionLabelStudio-%APPVER%-%EDITION%-setup.exe
goto :eof

:error
echo.
echo INSTALLER BUILD FAILED with error code %errorlevel%.
exit /b %errorlevel%
