# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for BungVision Label Studio (Windows release build).

Targets PyInstaller 6.x.

Build on a Windows machine or CI runner with the deps installed:
    pyinstaller BungVisionLabelStudio.spec --noconfirm

By default this produces a LEAN build that excludes torch/ultralytics. Those
pull in ~2.5 GB of tensor/CUDA libraries, and the app imports ultralytics
lazily -- labeling, capture, review, and dataset export all work without it,
and the app already shows an actionable "pip install ultralytics" message
wherever a model is actually required. Set BUNGVISION_BUNDLE_ML=1 for a full
build that bundles the ML stack (much larger, much slower to build).
"""
import os

from PyInstaller.utils.hooks import collect_submodules

BUNDLE_ML = os.environ.get("BUNGVISION_BUNDLE_ML", "").strip().lower() in ("1", "true", "yes")

# UI assets that must live inside the bundle. The stylesheet resolves this via
# Path(__file__).parent / "assets", which lands in <_internal>/bung_labeler/ui/assets.
# NOTE: user data (captures/labels/exports) is deliberately NOT bundled -- it is
# resolved next to the .exe at runtime by storage._app_root().
datas = [
    ("bung_labeler/ui/assets", "bung_labeler/ui/assets"),
]

hiddenimports = [
    "pypylon",
    "pypylon.pylon",
]

# Trimming these keeps the build small and startup fast. Qt WebEngine alone is
# several hundred MB and nothing in this app uses it.
excludes = [
    "tkinter",
    "matplotlib",
    "gi",  # Linux-only GStreamer bindings; guarded by try/except in camera.py
    "PySide6.QtWebEngineCore",
    "PySide6.QtWebEngineWidgets",
    "PySide6.QtWebEngineQuick",
    "PySide6.QtQuick",
    "PySide6.QtQml",
    "PySide6.Qt3DCore",
    "PySide6.QtCharts",
    "PySide6.QtDataVisualization",
    "PySide6.QtMultimedia",
    "PySide6.QtBluetooth",
    "PySide6.QtNetworkAuth",
    "PySide6.QtPositioning",
    "PySide6.QtSensors",
    "PySide6.QtSerialPort",
    "PySide6.QtTest",
    "PySide6.QtDesigner",
]

if BUNDLE_ML:
    hiddenimports += collect_submodules("ultralytics")
else:
    excludes += [
        "torch",
        "torchvision",
        "ultralytics",
        "scipy",
        "sympy",
        "networkx",
    ]

a = Analysis(
    ["main.py"],
    pathex=["."],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="BungVisionLabelStudio",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,  # GUI app: no console window (training logs stream in-app)
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="BungVisionLabelStudio",
)
