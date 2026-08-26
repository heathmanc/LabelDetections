"""Launcher, so the app starts the same way from source and from a build."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from label_detections.app import main

if __name__ == "__main__":
    raise SystemExit(main())
