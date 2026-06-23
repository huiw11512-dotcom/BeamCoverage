from __future__ import annotations

import sys
from pathlib import Path

APP_NAME = "BeamCoverage"
APP_VERSION = "3.1.6"
APP_TITLE = APP_NAME
CLI_DESCRIPTION = f"{APP_NAME} {APP_VERSION}"
APP_EXE_NAME = f"{APP_NAME}.exe"
APP_APK_NAME = f"{APP_NAME}.apk"
APP_WORKBOOK_NAME = f"{APP_NAME}.xlsx"
APP_SCAN_UNION_HTML_NAME = f"{APP_NAME}_ScanUnion3D.html"
APP_RELEASE_DIR_NAME = f"{APP_NAME}_release"
APP_RELEASE_ZIP_NAME = f"{APP_RELEASE_DIR_NAME}.zip"
APP_SYNC_ZIP_NAME = f"{APP_NAME}.zip"
APP_SPEC_NAME = f"{APP_NAME}.spec"
APP_EXCEL_TITLE = f"{APP_NAME} Excel"
APP_SCAN_UNION_HTML_TITLE = f"{APP_NAME} 3D Scan-Union Envelope"


def resource_path(*parts: str) -> Path:
    base = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))
    return base.joinpath(*parts)
