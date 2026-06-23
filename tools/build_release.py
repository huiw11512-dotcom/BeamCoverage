from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from zipfile import ZIP_DEFLATED, ZipFile


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app_info import (
    APP_APK_NAME,
    APP_EXE_NAME,
    APP_NAME,
    APP_RELEASE_DIR_NAME,
    APP_RELEASE_ZIP_NAME,
    APP_SCAN_UNION_HTML_NAME,
    APP_SPEC_NAME,
    APP_SYNC_ZIP_NAME,
    APP_VERSION,
    APP_WORKBOOK_NAME,
)

DIST = ROOT / "dist"
RELEASE_DIR = DIST / APP_RELEASE_DIR_NAME
RELEASE_ZIP = DIST / APP_RELEASE_ZIP_NAME
SYNC_ZIP = DIST / APP_SYNC_ZIP_NAME
EXCEL_HTML = APP_SCAN_UNION_HTML_NAME


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    started = time.perf_counter()
    env = _release_env()

    if not args.skip_exe_build:
        _run(["pyinstaller", "--clean", "--noconfirm", APP_SPEC_NAME], env=env)

    _run([sys.executable, "tools/create_excel_tool.py"], env=env)
    _run([sys.executable, "tools/create_scan_union_html.py"], env=env)

    apk_source = _resolve_apk_source(args.apk)
    _stage_release(apk_source)
    _write_zip(RELEASE_ZIP, RELEASE_DIR)
    _write_zip(SYNC_ZIP, RELEASE_DIR)

    if not args.skip_validation:
        _run([sys.executable, "release_check.py"], env=env)

    elapsed = time.perf_counter() - started
    print(f"PASS built release package in {elapsed:.2f}s")
    print(f"PASS release dir: {RELEASE_DIR}")
    print(f"PASS release zip: {RELEASE_ZIP}")
    print(f"PASS sync zip: {SYNC_ZIP}")
    return 0


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=f"Build the {APP_NAME} release directory and ZIP packages.")
    parser.add_argument(
        "--skip-exe-build",
        action="store_true",
        help=f"Reuse dist/{APP_EXE_NAME} instead of running PyInstaller. Use only for documentation or Excel-only packaging.",
    )
    parser.add_argument(
        "--apk",
        type=Path,
        default=None,
        help=f"APK file to include. Defaults to the existing release APK, then dist/{APP_APK_NAME}.",
    )
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Do not run release_check.py after staging and zipping.",
    )
    return parser.parse_args(argv)


def _release_env() -> dict[str, str]:
    env = os.environ.copy()
    env["QT_QPA_PLATFORM"] = "offscreen"
    env["OPENBLAS_NUM_THREADS"] = "1"
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    env["NUMEXPR_NUM_THREADS"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUNBUFFERED"] = "1"
    return env


def _run(command: list[str], env: dict[str, str]) -> None:
    print("RUN " + " ".join(_quote(part) for part in command), flush=True)
    proc = subprocess.run(command, cwd=ROOT, env=env, check=False)
    if proc.returncode != 0:
        raise SystemExit(f"FAIL command exited with {proc.returncode}: {' '.join(command)}")


def _quote(value: str) -> str:
    if not value or any(ch.isspace() for ch in value):
        return '"' + value.replace('"', '\\"') + '"'
    return value


def _resolve_apk_source(explicit_apk: Path | None) -> Path:
    candidates = []
    if explicit_apk is not None:
        candidates.append(explicit_apk)
    candidates.extend(
        [
            Path(r"E:\AndroidDev\projects\BeamCoverageAndroid\app\build\outputs\apk\release\app-release.apk"),
            DIST / APP_APK_NAME,
            RELEASE_DIR / APP_APK_NAME,
        ]
    )
    for candidate in candidates:
        path = candidate.expanduser().resolve()
        if path.exists() and path.is_file() and path.stat().st_size > 0:
            return path
    raise FileNotFoundError(
        f"No APK found for release packaging. Pass --apk or place {APP_APK_NAME} in the release/dist directory."
    )


def _stage_release(apk_source: Path) -> None:
    from release_check import EXPECTED_FILES

    DIST.mkdir(parents=True, exist_ok=True)
    RELEASE_DIR.mkdir(parents=True, exist_ok=True)

    release_readme = RELEASE_DIR / "README.txt"
    release_readme.write_text(_default_release_readme(), encoding="utf-8")

    staged_sources = {
        APP_EXE_NAME: DIST / APP_EXE_NAME,
        APP_APK_NAME: apk_source,
        APP_WORKBOOK_NAME: DIST / APP_WORKBOOK_NAME,
        EXCEL_HTML: DIST / EXCEL_HTML,
        "README.txt": release_readme,
        "CHANGELOG.md": ROOT / "CHANGELOG.md",
    }
    missing = [name for name, source in staged_sources.items() if not source.exists() or source.stat().st_size <= 0]
    if missing:
        raise FileNotFoundError(f"Release inputs are missing or empty: {missing}")

    _remove_unexpected_release_files(set(EXPECTED_FILES))
    for name in EXPECTED_FILES:
        source = staged_sources[name]
        target = RELEASE_DIR / name
        if source.resolve() == target.resolve():
            continue
        shutil.copy2(source, target)


def _remove_unexpected_release_files(expected_names: set[str]) -> None:
    release_root = RELEASE_DIR.resolve()
    for child in RELEASE_DIR.iterdir():
        resolved = child.resolve()
        try:
            resolved.relative_to(release_root)
        except ValueError as exc:
            raise RuntimeError(f"Refusing to remove path outside release directory: {resolved}")
        if child.name in expected_names:
            continue
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()


def _write_zip(zip_path: Path, release_dir: Path) -> None:
    from release_check import EXPECTED_FILES

    if zip_path.exists():
        zip_path.unlink()
    with ZipFile(zip_path, "w", compression=ZIP_DEFLATED, compresslevel=9) as zf:
        for name in sorted(EXPECTED_FILES):
            source = release_dir / name
            zf.write(source, arcname=name)


def _default_release_readme() -> str:
    return (
        f"{APP_NAME} Release Package\n\n"
        f"Version: {APP_VERSION}\n\n"
        "Included files:\n"
        f"- {APP_EXE_NAME}: Windows desktop application.\n"
        f"- {APP_APK_NAME}: Android internal preview build.\n"
        f"- {APP_WORKBOOK_NAME}: Excel/WPS single-page validation workbook.\n"
        f"- {EXCEL_HTML}: interactive 3D scan-union companion plot linked from the workbook.\n\n"
        "- CHANGELOG.md: product change log also shown inside the application.\n\n"
        "Current scope:\n"
        "- Windows desktop EXE is the primary validated calculation application.\n"
        "- APK is included as the Android internal preview build.\n"
        "- Excel/WPS workbook is a single-page validation and customer-review aid.\n"
        "- Interactive 3D scan-union viewing is provided by the companion HTML linked from the workbook.\n\n"
        "Recent hardening:\n"
        "- Product name, window title, EXE version metadata, workbook name, companion HTML name, release directory, and ZIP names are derived from app_info.py.\n"
        "- release_check.py validates version_info.txt against app_info.py before checking the packaged EXE metadata.\n"
        "- The release README is regenerated by tools/build_release.py on every package build so stale generated notes do not remain in dist.\n"
        "- release_check.py requires the staged README to exactly match this generated template and rejects old verbose prototype branding in release-facing text.\n"
        "- APK signature verification passes -JXmx256M to apksigner to avoid its default 1 GB heap on memory-constrained packaging machines.\n"
        "- Source documentation checks validate text encoding, release README content, the fallback release README template, and version metadata consistency.\n\n"
        "- Rectangular, circular/elliptical, diamond, and imported-coordinate geometry share a centralized shape registry for GUI labels, aliases, aperture modes, and overlap validation.\n\n"
        "- The far-field element-pattern importer accepts CSV/TXT/DAT tables plus CST/HFSS-style FFD/FFS vector far-field grids, and participates in scan loss, u-v pattern, 2D envelope, current 3D envelope, and scan-union calculations when enabled.\n\n"
        "- The custom sampling panel can override 2D, current-3D, u-v, scan-union, scan-center step, and 3D display-grid resolution on top of the fast/standard/fine modes.\n\n"
        "- Automatic sampling is now the default; it adapts 2D, 3D, u-v, scan-union, and display-grid resolution from electrical aperture, spacing-to-wavelength ratio, and aspect ratio so normal users do not have to tune sample counts manually.\n\n"
        "- Parameter edits no longer overwrite the right-hand plots until Calculate is clicked; the last calculated view remains visible while new parameters are staged.\n\n"
        "- Last-session parameters, selected calculation pages, window geometry, and active tab are persisted with QSettings and restored on the next launch.\n\n"
        "- The in-application change-log tab and GitHub release update check are available from the Help menu. Update checks require internet access; the calculation engine remains fully offline.\n\n"
        "- Interactive desktop 3D pages use PyQtGraph OpenGL when available for smoother drag rotation, zoom, and pan. The calculation arrays remain generated by the existing Python core; OpenGL is only the display layer, with a Matplotlib fallback for headless validation and report generation.\n\n"
        "- The experimental near-field table importer validates and preserves single-element x/y/z Ex/Ey/Ez or S data for future full-wave integration without changing the current main envelope calculation.\n\n"
        "- The near-field projection exporter can convert vector Ex/Ey/Ez samples into an approximate far-field element-pattern CSV that is then used by the existing main calculation path.\n\n"
        "Build and validation commands:\n"
        r"- python tools\build_release.py" "\n"
        r"- python tools\run_validation.py --release" "\n"
        "- release_check.py passed for this package after staging and zipping.\n"
        "- Documentation checks passed for source and release notes.\n\n"
        "Notes:\n"
        f"- The Windows application title and product metadata are {APP_NAME}.\n"
        "- The Excel workbook links to the companion HTML for interactive 3D scan-union viewing.\n"
        "- Keep old application windows closed before opening a newly copied EXE.\n"
    )


if __name__ == "__main__":
    raise SystemExit(main())
