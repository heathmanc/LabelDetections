"""Single source of truth for the application name and version.

Keeping these here avoids the version-string drift that comes from editing the
window title and the review-stamp string separately.
"""
from __future__ import annotations

APP_NAME = "LabelVision Studio"
APP_VERSION = "0.1.0"
APP_TITLE = f"{APP_NAME} v{APP_VERSION}"

# Written into review markers so an imported annotation from another tool can
# never be mistaken for one an operator approved in here.
REVIEW_SOURCE = "labelvision_studio"
