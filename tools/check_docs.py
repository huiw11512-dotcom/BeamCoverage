from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app_info import (
    APP_APK_NAME,
    APP_EXE_NAME,
    APP_NAME,
    APP_RELEASE_DIR_NAME,
    APP_SCAN_UNION_HTML_NAME,
    APP_VERSION,
    APP_WORKBOOK_NAME,
)


ROOT_README = ROOT / "README.md"
RELEASE_README = ROOT / "dist" / APP_RELEASE_DIR_NAME / "README.txt"
SOURCE_TEXT_SUFFIXES = {".py", ".md", ".txt", ".spec"}
SOURCE_TEXT_EXCLUDED_DIRS = {
    ".git",
    ".pytest_cache",
    "__pycache__",
    "build",
    "dist",
    "export_smoke_output",
}

MOJIBAKE_FRAGMENTS = [
    "\ufffd",
    "\u8103",
    "\u8117",
    "\u93c9\u2569\u57b5",
    "\u942e\u7535\u53a7",
    "\u95ba\u50da\u6d23\u5a06",
    "\u95b8\u696c\u6d16\u9358",
    "\u95c2\u51ae\u6f67\u9358",
    "\u6fee\u30e5\u7e10\u5b95",
    "\u6d34\uff48\u6cd5\u93c1",
    "\u7f01\ue1d8\u7caf\u93c6",
    "\u6960\u708c\u723c\u6ce9",
    "\u5a62\u8235\u7257\u752f",
    "\u95c1\u63d2\u6d26\u9417",
    "\u6fa7\u20ac\u525d\u5a62",
    "\u95ba\u5d86\u6d24\u9363",
    "\u9359\u509b\u669f",
    "\u93c3\u72b3\u6665",
    "\u7487\u8702\u6168\u59dd",
    "\u942d\u2541\u8230",
    "\u9366\u55d7\u8230",
    "\u599e\u9366",
    "\u947f\u535e\u8230",
    "\u7035\u714e\u53c6",
    "\u9367\u612d\u7223",
    "\u95c3\u975b\u57aa",
    "\u95c3\u975b\u5393",
    "\u9357\u66de\u5393",
    "\u9359\uff45\u7dde",
    "\u95b2\u5d85\u5f54",
    "\u7481\uff04\u757b",
    "\u7f01\u64b4\u702f",
    "\u93c2\u7470\u609c\u9365",
    "\u9477\u9354",
    "\u93b5\u5b2a\u59e9",
    "\u8e47\u95ab",
    "\u93cd\u56e7\u566f",
    "\u7eee\u5267\u7c8f",
]


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    check_source_docs()

    if args.release:
        check_release_docs()
        print("PASS documentation/text-integrity checks including release README.")
    else:
        print("PASS source documentation/text-integrity checks.")
    return 0


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=f"Validate {APP_NAME} documentation is readable and release-aware.")
    parser.add_argument(
        "--release",
        action="store_true",
        help=f"Also require and validate dist/{APP_RELEASE_DIR_NAME}/README.txt.",
    )
    return parser.parse_args(argv)


def check_source_docs() -> None:
    root_text = _check_utf8_text(ROOT_README)
    _check_root_readme(root_text)
    _check_source_text_files()
    _check_default_release_readme_template()
    _check_version_metadata_sources()


def check_release_docs(readme_path: Path = RELEASE_README) -> None:
    release_text = _check_utf8_text(readme_path)
    _check_release_readme(release_text)


def _check_utf8_text(path: Path) -> str:
    if not path.exists():
        raise AssertionError(f"Documentation file is missing: {path}")
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise AssertionError(f"Documentation file is not valid UTF-8: {path}: {exc}") from exc
    hits = [fragment for fragment in MOJIBAKE_FRAGMENTS if fragment in text]
    if hits:
        raise AssertionError(f"Documentation file contains mojibake markers {hits}: {path}")
    return text


def _check_source_text_files() -> None:
    for path in _iter_source_text_paths():
        _check_utf8_text(path)


def _check_default_release_readme_template() -> None:
    from tools.build_release import _default_release_readme

    _check_release_readme(_default_release_readme())


def _check_version_metadata_sources() -> None:
    from release_check import check_version_resource_source

    check_version_resource_source(verbose=False)


def _iter_source_text_paths() -> list[Path]:
    paths: list[Path] = []
    for path in ROOT.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in SOURCE_TEXT_SUFFIXES:
            continue
        relative = path.relative_to(ROOT)
        if any(part in SOURCE_TEXT_EXCLUDED_DIRS for part in relative.parts[:-1]):
            continue
        paths.append(path)
    return sorted(paths)


def _check_root_readme(text: str) -> None:
    required = [
        f"# {APP_NAME}",
        r"python tools\build_release.py",
        r"python tools\run_validation.py --release",
        APP_SCAN_UNION_HTML_NAME,
        "\u5bfc\u5165\u9635\u5143\u5750\u6807 CSV",
        "\u5bfc\u5165\u5355\u5143\u8fdc\u573a\u65b9\u5411\u56fe\u6587\u4ef6",
        "\u5bfc\u51fa\u5355\u5143\u8fdc\u573a Real/Imag \u6a21\u677f",
        "\u5bfc\u51fa\u5355\u5143\u8fdc\u573a\u5e45\u76f8\u6a21\u677f",
        "\u5bfc\u5165\u5355\u5143\u8fd1\u573a\u6587\u4ef6",
        "\u5bfc\u51fa\u5355\u5143\u8fd1\u573a Ex/Ey/Ez \u6a21\u677f",
        "\u4ece\u5355\u5143\u8fd1\u573a\u5bfc\u51fa\u8fdc\u573a\u65b9\u5411\u56fe CSV",
        "near_field_sampled_far_field_extrapolated",
        "far_field_coefficient_union",
        "core/aperture_shapes.py",
        "ARRAY_LAYOUT_CHOICES",
        "element_overlap_metric",
        "Imported element near-field files",
        "Custom sampling",
        "Automatic sampling",
        "GitHub update check",
        "CHANGELOG.md",
    ]
    _require_fragments("README.md", text, required)


def _check_release_readme(text: str) -> None:
    required = [
        f"Version: {APP_VERSION}",
        APP_EXE_NAME,
        APP_APK_NAME,
        APP_WORKBOOK_NAME,
        APP_SCAN_UNION_HTML_NAME,
        "CHANGELOG.md",
        r"python tools\build_release.py",
        r"python tools\run_validation.py --release",
        "Documentation checks passed",
        "release_check.py passed",
        "centralized shape registry",
        "far-field element-pattern importer",
        "near-field table importer",
        "near-field projection exporter",
        "Automatic sampling",
        "GitHub release update check",
    ]
    _require_fragments("release README.txt", text, required)


def _require_fragments(label: str, text: str, required: list[str]) -> None:
    missing = [fragment for fragment in required if fragment not in text]
    if missing:
        raise AssertionError(f"{label} is missing required documentation fragments: {missing}")


if __name__ == "__main__":
    raise SystemExit(main())
