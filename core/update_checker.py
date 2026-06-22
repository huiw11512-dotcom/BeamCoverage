from __future__ import annotations

from dataclasses import dataclass
import json
import re
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


GITHUB_REPO = "huiw11512-dotcom/BeamCoverage"
LATEST_RELEASE_API = f"https://api.github.com/repos/{GITHUB_REPO}/releases/latest"
RELEASES_PAGE = f"https://github.com/{GITHUB_REPO}/releases"


@dataclass(frozen=True)
class UpdateInfo:
    current_version: str
    latest_version: str
    tag_name: str
    html_url: str
    name: str
    body: str
    is_newer: bool


def check_for_update(current_version: str, *, timeout_s: float = 4.0) -> UpdateInfo:
    request = Request(
        LATEST_RELEASE_API,
        headers={
            "Accept": "application/vnd.github+json",
            "User-Agent": "BeamCoverage-update-checker",
            "X-GitHub-Api-Version": "2022-11-28",
        },
    )
    try:
        with urlopen(request, timeout=float(timeout_s)) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        raise RuntimeError(f"GitHub returned HTTP {exc.code}") from exc
    except URLError as exc:
        raise RuntimeError(str(exc.reason)) from exc

    tag_name = str(payload.get("tag_name") or "")
    latest_version = _version_from_tag(tag_name)
    if not latest_version:
        latest_version = _version_from_tag(str(payload.get("name") or ""))
    if not latest_version:
        raise RuntimeError("latest release does not contain a semantic version tag")
    html_url = str(payload.get("html_url") or RELEASES_PAGE)
    return UpdateInfo(
        current_version=str(current_version),
        latest_version=latest_version,
        tag_name=tag_name,
        html_url=html_url,
        name=str(payload.get("name") or tag_name or latest_version),
        body=str(payload.get("body") or ""),
        is_newer=_compare_versions(latest_version, current_version) > 0,
    )


def format_update_message(info: UpdateInfo) -> str:
    notes = _compact_release_notes(info.body)
    message = (
        f"当前版本：{info.current_version}\n"
        f"最新版本：{info.latest_version}\n\n"
        "是否打开 GitHub Releases 页面下载新版？"
    )
    if notes:
        message += f"\n\n更新摘要：\n{notes}"
    return message


def _compact_release_notes(text: str, max_lines: int = 8) -> str:
    lines = []
    for raw in str(text).replace("\r\n", "\n").splitlines():
        line = raw.strip()
        if not line:
            continue
        if len(line) > 120:
            line = line[:117] + "..."
        lines.append(line)
        if len(lines) >= max_lines:
            break
    return "\n".join(lines)


def _version_from_tag(text: str) -> str:
    match = re.search(r"v?(\d+(?:\.\d+){1,3})", str(text), flags=re.IGNORECASE)
    return match.group(1) if match else ""


def _compare_versions(left: str, right: str) -> int:
    left_parts = _version_parts(left)
    right_parts = _version_parts(right)
    width = max(len(left_parts), len(right_parts), 3)
    left_parts.extend([0] * (width - len(left_parts)))
    right_parts.extend([0] * (width - len(right_parts)))
    if left_parts > right_parts:
        return 1
    if left_parts < right_parts:
        return -1
    return 0


def _version_parts(value: Any) -> list[int]:
    version = _version_from_tag(str(value)) or str(value)
    parts: list[int] = []
    for token in str(version).split("."):
        match = re.match(r"(\d+)", token)
        parts.append(int(match.group(1)) if match else 0)
    return parts
