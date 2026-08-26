# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for LabelVision Studio (Windows release build).

Targets PyInstaller 6.x.

Build on a Windows machine or CI runner with the deps installed:
    pyinstaller LabelVisionStudio.spec --noconfirm

By default this produces a FULL build with torch/ultralytics bundled, so Model
Test, Auto-label, and Pre-label & Review work with no per-machine setup. Those
features import Ultralytics in-process, and a frozen app cannot import from the
system Python -- a lean build loses them permanently, no matter what the user
pip-installs.

Install torch from the cu128 index before building (see build_windows.bat) so
the bundled wheel carries Blackwell/sm_120 kernels for RTX 50-series cards.

Set LABELVISION_LEAN=1 for a labeling-only build without the ML stack.
"""
import os

from PyInstaller.utils.hooks import collect_submodules

# Full build is the default: Model Test, Auto-label, and Pre-label & Review
# import Ultralytics in-process, and a frozen app cannot import from the system
# Python -- so a lean build silently loses those features no matter what the
# user pip-installs. Set LABELVISION_LEAN=1 only for a labeling-only build.
LEAN = os.environ.get("LABELVISION_LEAN", "").strip().lower() in ("1", "true", "yes")
BUNDLE_ML = not LEAN

# UI assets that must live inside the bundle. The stylesheet resolves this via
# Path(__file__).parent / "assets", which lands in <_internal>/label_detections/ui/assets.
# NOTE: user data (captures/labels/exports) is deliberately NOT bundled -- it is
# resolved next to the .exe at runtime by storage._app_root().
datas = [
    ("label_detections/ui/assets", "label_detections/ui/assets"),
    # No seed data. The detector families are written by
    # storage.load_class_config() on first run and the label library starts
    # empty on purpose -- a shipped placeholder label would only need deleting.
]

ICON = "label_detections/ui/assets/app.ico"

hiddenimports = [
    "pypylon",
    "pypylon.pylon",
]

# Trimming these keeps the build small and startup fast. Qt WebEngine alone is
# several hundred MB and nothing in this app uses it.
excludes = [
    # tkinter stays excluded even in full builds: matplotlib only needs it for
    # the TkAgg backend, and main.py forces MPLBACKEND=Agg.
    "tkinter",
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
    # Ultralytics imports matplotlib unconditionally (plotting/annotators), so
    # excluding it makes `from ultralytics import YOLO` raise
    # "No module named 'matplotlib'" -- which the app surfaces as its generic
    # "Ultralytics is not installed" message. It must be present in full builds.
    hiddenimports += ["matplotlib", "matplotlib.backends.backend_agg"]
else:
    excludes += [
        "torch",
        "torchvision",
        "ultralytics",
        "matplotlib",
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
    name="LabelVisionStudio",
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
    icon=ICON,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="LabelVisionStudio",
)
