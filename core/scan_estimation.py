from __future__ import annotations

import numpy as np

from core.geometry import BeamParams, DerivedParams, X3_DB, asind, sind


def calc_scan_coverage_empirical(
    params: BeamParams,
    derived: DerivedParams,
    step_deg: float = 0.25,
    margin_deg: float = 2.0,
) -> dict[str, np.ndarray | float]:
    hpbw_x_full = 2.0 * asind(min(1.0, X3_DB * derived.wavelength_m / derived.dx_aperture_m))
    hpbw_y_full = 2.0 * asind(min(1.0, X3_DB * derived.wavelength_m / derived.dy_aperture_m))
    hpbw_x_half = float(hpbw_x_full / 2.0)
    hpbw_y_half = float(hpbw_y_full / 2.0)
    scan_limit_x = derived.scan_limit_x_deg
    scan_limit_y = derived.scan_limit_y_deg

    x_max = scan_limit_x + hpbw_x_half + margin_deg
    y_max = scan_limit_y + hpbw_y_half + margin_deg
    x_grid = np.arange(-x_max, x_max + 0.5 * step_deg, step_deg)
    y_grid = np.arange(-y_max, y_max + 0.5 * step_deg, step_deg)
    xg, yg = np.meshgrid(x_grid, y_grid)
    center_mask = (np.abs(xg) <= scan_limit_x) & (np.abs(yg) <= scan_limit_y)

    cx = np.arange(-scan_limit_x, scan_limit_x + 0.5 * step_deg, step_deg)
    cy = np.arange(-scan_limit_y, scan_limit_y + 0.5 * step_deg, step_deg)
    cxx, cyy = np.meshgrid(cx, cy)
    centers_x = cxx.ravel()
    centers_y = cyy.ravel()
    cover_count = np.zeros_like(xg, dtype=float)
    block = 256
    for start in range(0, centers_x.size, block):
        stop = min(start + block, centers_x.size)
        dx = (xg[..., None] - centers_x[start:stop]) / max(hpbw_x_half, np.finfo(float).eps)
        dy = (yg[..., None] - centers_y[start:stop]) / max(hpbw_y_half, np.finfo(float).eps)
        cover_count += np.sum(dx * dx + dy * dy <= 1.0, axis=2)
    cover_mask = cover_count > 0.0
    boundary_x, boundary_y = boundary_from_mask(cover_mask, x_grid, y_grid)

    if np.any(cover_mask):
        cover_x_min = float(np.min(xg[cover_mask]))
        cover_x_max = float(np.max(xg[cover_mask]))
        cover_y_min = float(np.min(yg[cover_mask]))
        cover_y_max = float(np.max(yg[cover_mask]))
    else:
        cover_x_min = cover_x_max = cover_y_min = cover_y_max = float("nan")

    return {
        "scanLimitX_deg": scan_limit_x,
        "scanLimitY_deg": scan_limit_y,
        "hpbwX_full_deg": float(hpbw_x_full),
        "hpbwY_full_deg": float(hpbw_y_full),
        "hpbwX_half_deg": hpbw_x_half,
        "hpbwY_half_deg": hpbw_y_half,
        "xGrid_deg": x_grid,
        "yGrid_deg": y_grid,
        "Xg": xg,
        "Yg": yg,
        "centerMask": center_mask,
        "coverCount": cover_count,
        "coverMask": cover_mask,
        "boundaryX_deg": boundary_x,
        "boundaryY_deg": boundary_y,
        "coverX_min_deg": cover_x_min,
        "coverX_max_deg": cover_x_max,
        "coverY_min_deg": cover_y_min,
        "coverY_max_deg": cover_y_max,
    }


def boundary_from_mask(mask: np.ndarray, x_grid: np.ndarray, y_grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    if mask.shape[0] < 3 or mask.shape[1] < 3 or not np.any(mask):
        return np.array([np.nan]), np.array([np.nan])
    edge = np.zeros_like(mask, dtype=bool)
    inner = mask[1:-1, 1:-1]
    neighbors = (
        mask[:-2, :-2]
        & mask[:-2, 1:-1]
        & mask[:-2, 2:]
        & mask[1:-1, :-2]
        & mask[1:-1, 2:]
        & mask[2:, :-2]
        & mask[2:, 1:-1]
        & mask[2:, 2:]
    )
    edge[1:-1, 1:-1] = inner & ~neighbors
    rr, cc = np.where(edge)
    if rr.size == 0:
        return np.array([np.nan]), np.array([np.nan])
    rc = np.mean(rr)
    cc0 = np.mean(cc)
    order = np.argsort(np.arctan2(rr - rc, cc - cc0))
    return x_grid[cc[order]], y_grid[rr[order]]


def scan_coverage_3d_points(info: dict[str, np.ndarray | float], radius_m: float = 1.0) -> dict[str, np.ndarray]:
    xang = info["Xg"]
    yang = info["Yg"]
    mask = info["coverMask"]
    u = sind(xang)
    v = sind(yang)
    visible = u * u + v * v <= 1.0
    w = np.sqrt(np.maximum(1.0 - u * u - v * v, 0.0))
    valid = mask & visible
    return {
        "scan_angle_x_deg": xang[valid],
        "scan_angle_y_deg": yang[valid],
        "u": u[valid],
        "v": v[valid],
        "w": w[valid],
        "x_m": radius_m * u[valid],
        "y_m": radius_m * v[valid],
        "z_m": radius_m * w[valid],
        "cover_count": info["coverCount"][valid],
    }

