@echo off
REM Package the built app into a single setup .exe using Inno Setup.
REM
REM   make_installer.bat            package whatever is in dist\ (edition "full")
REM   make_installer.bat lean       label the output as the lean edition
REM
REM Requires Inno Setup 6:  winget install JRSoftware.InnoSetup
REM Run build_windows.bat first -- this only packages, it does not build.

setlocal

set "EDITION=%~1"
if not defined EDITION set "EDITION=full"

if not exist "dist\BungVisionLabelStudio\BungVisionLabelStudio.exe" (
    echo ERROR: dist\BungVisionLabelStudio\BungVisionLabelStudio.exe not found.
    echo Run "build_windows.bat %EDITION%" first.
    exit /b 1
)

REM Locate the Inno Setup compiler. winget may install per-machine or per-user,
REM so check both plus PATH.
REM
REM Deliberately straight-line rather than a `for %%P in (...)` loop: the literal
REM parentheses in %ProgramFiles(x86)% terminate a parenthesized block early,
REM which produced "... was unexpected at this time" on accounts whose name
REM contains a space. Quoting with set "VAR=..." also keeps the stored path free
REM of quotes so it can be re-quoted exactly once at the call site.
set "ISCC="
if exist "%ProgramFiles(x86)%\Inno Setup 6\ISCC.exe" set "ISCC=%ProgramFiles(x86)%\Inno Setup 6\ISCC.exe"
if not defined ISCC if exist "%ProgramFiles%\Inno Setup 6\ISCC.exe" set "ISCC=%ProgramFiles%\Inno Setup 6\ISCC.exe"
if not defined ISCC if exist "%LOCALAPPDATA%\Programs\Inno Setup 6\ISCC.exe" set "ISCC=%LOCALAPPDATA%\Programs\Inno Setup 6\ISCC.exe"
if not defined ISCC if exist "%ProgramFiles%\Inno Setup 5\ISCC.exe" set "ISCC=%ProgramFiles%\Inno Setup 5\ISCC.exe"
if not defined ISCC if exist "%ProgramFiles(x86)%\Inno Setup 5\ISCC.exe" set "ISCC=%ProgramFiles(x86)%\Inno Setup 5\ISCC.exe"
if not defined ISCC for /f "delims=" %%I in ('where iscc 2^>nul') do if not defined ISCC set "ISCC=%%I"

if not defined ISCC (
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
REM Via a helper script, not an inline python -c: cmd counts parentheses inside
REM a `for /f ... in (...)` block before expanding anything, so the parens and
REM escaped quotes in a one-liner terminate the block early.
set "APPVER="
for /f "delims=" %%V in ('python scripts\print_version.py') do set "APPVER=%%V"
if not defined APPVER (
    echo ERROR: could not read APP_VERSION from bung_labeler\version.py
    exit /b 1
)

echo === Building installer: v%APPVER% ^(%EDITION%^) ===
echo     Compressing a full build takes a while - 10-30 min with the CPU pegged.

REM ISCC must be quoted: it commonly lives under "Program Files (x86)" or a
REM user profile containing a space.
"%ISCC%" "installer\BungVisionLabelStudio.iss" /DAppVersion=%APPVER% /DEdition=%EDITION% || goto :error

echo.
echo === INSTALLER READY ===
dir /b "installer\BungVisionLabelStudio-%APPVER%-%EDITION%-setup.exe"
echo     installer\BungVisionLabelStudio-%APPVER%-%EDITION%-setup.exe
goto :eof

:error
echo.
echo INSTALLER BUILD FAILED with error code %errorlevel%.
exit /b %errorlevel%
