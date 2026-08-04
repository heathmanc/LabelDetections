#!/usr/bin/env python3
"""Print APP_VERSION on stdout, nothing else.

Exists so the Windows build scripts can read the version with a trivial
`for /f ... in ('python scripts\\print_version.py')`. Inlining the equivalent
`python -c "..."` puts parentheses and escaped quotes inside cmd's `for /f`
block, where cmd counts parens before expanding anything and mis-parses the
command.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bung_labeler.version import APP_VERSION

print(APP_VERSION)
