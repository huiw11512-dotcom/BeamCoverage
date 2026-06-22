from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np


def _as_1d(value: Any) -> np.ndarray:
    return np.asarray(value).ravel()


def _write_columns(path: str | Path, columns: dict[str, Any]) -> None:
    arrays = {key: _as_1d(value) for key, value in columns.items()}
    lengths = [arr.size for arr in arrays.values()]
    n_rows = max(lengths, default=0)
    expanded: dict[str, np.ndarray] = {}
    for key, arr in arrays.items():
        if arr.size == n_rows:
            expanded[key] = arr
        elif arr.size == 1:
            expanded[key] = np.full(n_rows, arr.item(), dtype=object)
        else:
            raise ValueError(f"CSV column {key!r} has length {arr.size}; expected {n_rows}.")

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        headers = list(expanded.keys())
        writer.writerow(headers)
        for row_idx in range(n_rows):
            writer.writerow([_csv_value(expanded[key][row_idx]) for key in headers])


def _csv_value(value: Any) -> Any:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and np.isnan(value):
        return ""
    return value


def export_2d_cuts(cuts: list[dict[str, Any]], path: str | Path, s0_w_cm2: float) -> None:
    rows: list[dict[str, Any]] = []
    for idx, cut in enumerate(cuts, start=1):
        n = _as_1d(cut["alpha_deg"]).size
        alpha = _as_1d(cut["alpha_deg"])
        r_env = _as_1d(cut["r_env_m"])
        rho = _as_1d(cut["rho_env_m"])
        x = _as_1d(cut["x_env_m"])
        y = _as_1d(cut["y_env_m"])
        z = _as_1d(cut["z_env_m"])
        has_env = _as_1d(cut["has_envelope"])
        far = _as_1d(cut["far_extended"])
        clipped = _as_1d(cut["clipped"])
        for row_idx in range(n):
            rows.append(
                {
                    "cut_id": idx,
                    "cut_name": cut["name"],
                    "phi_cut_deg": cut["phi_cut_deg"],
                    "alpha_deg": alpha[row_idx],
                    "r_env_m": r_env[row_idx],
                    "rho_m": rho[row_idx],
                    "x_m": x[row_idx],
                    "y_m": y[row_idx],
                    "z_m": z[row_idx],
                    "has_envelope": has_env[row_idx],
                    "far_field_extrapolated": far[row_idx],
                    "range_clipped": clipped[row_idx],
                    "S0_Wcm2": s0_w_cm2,
                    "envelope_method": cut.get("envelope_method", ""),
                    "envelope_method_label": cut.get("envelope_method_label", ""),
                }
            )
    headers = [
        "cut_id",
        "cut_name",
        "phi_cut_deg",
        "alpha_deg",
        "r_env_m",
        "rho_m",
        "x_m",
        "y_m",
        "z_m",
        "has_envelope",
        "far_field_extrapolated",
        "range_clipped",
        "S0_Wcm2",
        "envelope_method",
        "envelope_method_label",
    ]
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row[key]) for key in headers})


def export_structure(points: dict[str, Any], path: str | Path) -> None:
    _write_columns(path, points)


def export_current_3d(envelope: dict[str, Any], path: str | Path, s0_w_cm2: float) -> None:
    _write_columns(
        path,
        {
            "theta_deg": envelope["THETA"],
            "phi_deg": envelope["PHI"],
            "x_m": envelope["Xsurf"],
            "y_m": envelope["Ysurf"],
            "z_m": envelope["Zsurf"],
            "r_env_m": envelope["r_env_m"],
            "has_envelope": envelope["has_envelope"],
            "far_field_extrapolated": envelope["far_extended"],
            "range_clipped": envelope["clipped"],
            "S0_Wcm2": s0_w_cm2,
            "envelope_method": envelope.get("envelopeMethod", ""),
            "envelope_method_label": envelope.get("envelopeMethodLabel", ""),
        },
    )


def export_uv_pattern(uv: dict[str, Any], path: str | Path) -> None:
    _write_columns(
        path,
        {
            "u": uv["U"],
            "v": uv["V"],
            "w": uv["W"],
            "visible": uv["visible"],
            "pattern_db": uv["pattern_db"],
        },
    )


def export_scan_union(info: dict[str, Any], path: str | Path) -> None:
    rows: list[dict[str, Any]] = []
    theta = _as_1d(info["THETA"])
    phi = _as_1d(info["PHI"])
    u = _as_1d(info["sx"])
    v = _as_1d(info["sy"])
    w = _as_1d(info["sz"])
    x = _as_1d(info["Xsurf"])
    y = _as_1d(info["Ysurf"])
    z = _as_1d(info["Zsurf"])
    r = _as_1d(info["Rsurf"])
    best_x = _as_1d(info["bestScanX_deg"])
    best_y = _as_1d(info["bestScanY_deg"])
    max_coef = _as_1d(info["maxCoef"])
    for idx in range(r.size):
        rows.append(
            {
                "dataset": "3d_union",
                "cut_name": "",
                "phi_cut_deg": "",
                "alpha_deg": "",
                "theta_deg": theta[idx],
                "phi_deg": phi[idx],
                "u": u[idx],
                "v": v[idx],
                "w": w[idx],
                "x_m": x[idx],
                "y_m": y[idx],
                "z_m": z[idx],
                "r_env_m": r[idx],
                "best_scan_x_deg": best_x[idx],
                "best_scan_y_deg": best_y[idx],
                "max_coef": max_coef[idx],
                "envelope_method": info.get("envelopeMethod", ""),
                "envelope_method_label": info.get("envelopeMethodLabel", ""),
            }
        )

    for cut in info.get("unionCuts", []):
        alpha = _as_1d(cut["alpha_deg"])
        x_cut = _as_1d(cut["x_env_m"])
        y_cut = _as_1d(cut["y_env_m"])
        z_cut = _as_1d(cut["z_env_m"])
        r_cut = _as_1d(cut["r_env_m"])
        best_x_cut = _as_1d(cut["bestScanX_deg"])
        best_y_cut = _as_1d(cut["bestScanY_deg"])
        for idx in range(r_cut.size):
            rows.append(
                {
                    "dataset": "2d_union_cut",
                    "cut_name": cut["short_name"],
                    "phi_cut_deg": cut["phi_cut_deg"],
                    "alpha_deg": alpha[idx],
                    "theta_deg": "",
                    "phi_deg": "",
                    "u": "",
                    "v": "",
                    "w": "",
                    "x_m": x_cut[idx],
                    "y_m": y_cut[idx],
                    "z_m": z_cut[idx],
                    "r_env_m": r_cut[idx],
                    "best_scan_x_deg": best_x_cut[idx],
                    "best_scan_y_deg": best_y_cut[idx],
                    "max_coef": "",
                    "envelope_method": cut.get("envelope_method", info.get("envelopeMethod", "")),
                    "envelope_method_label": cut.get("envelope_method_label", info.get("envelopeMethodLabel", "")),
                }
            )

    headers = [
        "dataset",
        "cut_name",
        "phi_cut_deg",
        "alpha_deg",
        "theta_deg",
        "phi_deg",
        "u",
        "v",
        "w",
        "x_m",
        "y_m",
        "z_m",
        "r_env_m",
        "best_scan_x_deg",
        "best_scan_y_deg",
        "max_coef",
        "envelope_method",
        "envelope_method_label",
    ]
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row[key]) for key in headers})
