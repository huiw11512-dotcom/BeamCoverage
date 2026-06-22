from __future__ import annotations

from pathlib import Path

import numpy as np

from core.geometry import DerivedParams
from core.near_field import export_element_near_field_vector_template


def export_current_array_layout_template(derived: DerivedParams, path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    x = np.asarray(derived.element_x_m, dtype=float).ravel()
    y = np.asarray(derived.element_y_m, dtype=float).ravel()
    power = np.asarray(derived.element_power_w, dtype=float).ravel()
    phase = np.asarray(derived.element_phase_offset_rad, dtype=float).ravel()
    if not (x.size == y.size == power.size == phase.size):
        raise ValueError("Current array state has inconsistent element arrays.")
    rows = ["x_m,y_m,power_w,phase_deg,enabled"]
    phase_deg = np.rad2deg(phase)
    for xi, yi, pi, ph in zip(x, y, power, phase_deg):
        rows.append(f"{xi:.12g},{yi:.12g},{pi:.12g},{ph:.12g},1")
    target.write_text("\n".join(rows) + "\n", encoding="utf-8")


def export_element_pattern_vector_template(path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    rows = ["Theta [deg],Phi [deg],Real(ETheta),Imag(ETheta),Real(EPhi),Imag(EPhi)"]
    for theta in (0.0, 15.0, 30.0, 45.0, 60.0, 75.0, 85.0):
        theta_scale = max(0.02, np.cos(np.deg2rad(theta)))
        for phi in (-180.0, -120.0, -60.0, 0.0, 60.0, 120.0, 180.0):
            phi_rad = np.deg2rad(phi)
            etheta = float(theta_scale ** 1.1)
            ephi = float(0.25 * theta_scale * np.cos(phi_rad))
            rows.append(f"{theta:.8g},{phi:.8g},{etheta:.12g},0,{ephi:.12g},0")
    target.write_text("\n".join(rows) + "\n", encoding="utf-8")


def export_element_pattern_abs_phase_template(path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    rows = ["Theta [deg],Phi [deg],Abs(Theta),Phase(Theta),Abs(Phi),Phase(Phi)"]
    for theta in (0.0, 15.0, 30.0, 45.0, 60.0, 75.0, 85.0):
        theta_rad = np.deg2rad(theta)
        theta_scale = max(0.02, np.cos(theta_rad))
        for phi in (-180.0, -120.0, -60.0, 0.0, 60.0, 120.0, 180.0):
            phi_rad = np.deg2rad(phi)
            etheta_abs = float(theta_scale ** 1.1)
            ephi_abs = float(abs(0.25 * theta_scale * np.cos(phi_rad)))
            etheta_phase_deg = float(12.0 * np.sin(phi_rad) * np.sin(theta_rad))
            ephi_phase_deg = float(-25.0 + 8.0 * np.cos(theta_rad))
            rows.append(
                f"{theta:.8g},{phi:.8g},{etheta_abs:.12g},{etheta_phase_deg:.12g},"
                f"{ephi_abs:.12g},{ephi_phase_deg:.12g}"
            )
    target.write_text("\n".join(rows) + "\n", encoding="utf-8")
