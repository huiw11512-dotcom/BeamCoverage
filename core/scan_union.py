from __future__ import annotations

import math
from time import perf_counter

import numpy as np

from core.element_pattern import ElementPattern, element_response_components_fast
from core.envelope import envelope_cut_summary, find_outer_envelope_hybrid
from core.geometry import BeamParams, DerivedParams, ModeSettings, asind, atan2d, cosd, make_range_grid, sind


SCAN_UNION_METHOD = "far_field_coefficient_union"
SCAN_UNION_METHOD_LABEL = "远场系数扫描并集"
SCAN_UNION_CUT_METHOD = "near_field_sampled_far_field_scan_union"
SCAN_UNION_CUT_METHOD_LABEL = "近场采样 + 远场外推扫描并集"


def compute_scan_union_envelope_3d(
    params: BeamParams,
    derived: DerivedParams,
    elem: ElementPattern,
    settings: ModeSettings,
) -> dict[str, np.ndarray | float | int | list[dict[str, np.ndarray | float | str]]]:
    scan_x, scan_y, scan_u, scan_v = _make_scan_centers(params, derived, settings, include_current_scan=True)
    return _compute_scan_union_from_centers(params, derived, elem, settings, scan_x, scan_y, scan_u, scan_v)


def compute_scan_union_base_envelope_3d(
    params: BeamParams,
    derived: DerivedParams,
    elem: ElementPattern,
    settings: ModeSettings,
) -> dict[str, np.ndarray | float | int | list[dict[str, np.ndarray | float | str]]]:
    scan_x, scan_y, scan_u, scan_v = _make_scan_centers(params, derived, settings, include_current_scan=False)
    result = _compute_scan_union_from_centers(params, derived, elem, settings, scan_x, scan_y, scan_u, scan_v)
    result["baseGridOnly"] = True
    return result


def merge_scan_union_with_current_scan(
    base_union: dict[str, np.ndarray | float | int | list[dict[str, np.ndarray | float | str]]],
    params: BeamParams,
    derived: DerivedParams,
    elem: ElementPattern,
    settings: ModeSettings,
) -> dict[str, np.ndarray | float | int | bool | list[dict[str, np.ndarray | float | str]]]:
    current = _current_scan_center(params, derived)
    if current is None:
        result = dict(base_union)
        result["currentScanOverlayApplied"] = False
        return result

    cur_x, cur_y, cur_u, cur_v = current
    if _scan_pair_in_list(base_union.get("scanXList_deg"), base_union.get("scanYList_deg"), cur_x[0], cur_y[0]):
        result = dict(base_union)
        result["currentScanOverlayApplied"] = False
        return result

    t0 = perf_counter()
    current_union = _compute_scan_union_from_centers(params, derived, elem, settings, cur_x, cur_y, cur_u, cur_v)
    result = _merge_scan_union_results(base_union, current_union, derived)
    current_timings = current_union.get("timings", {}) if isinstance(current_union, dict) else {}
    timings = dict(current_timings) if isinstance(current_timings, dict) else {}
    timings["scan_union_current_overlay_s"] = perf_counter() - t0
    result["timings"] = timings
    result["currentScanOverlayApplied"] = True
    return result


def _compute_scan_union_from_centers(
    params: BeamParams,
    derived: DerivedParams,
    elem: ElementPattern,
    settings: ModeSettings,
    scan_x: np.ndarray,
    scan_y: np.ndarray,
    scan_u: np.ndarray,
    scan_v: np.ndarray,
) -> dict[str, np.ndarray | float | int | list[dict[str, np.ndarray | float | str]]]:
    total_t0 = perf_counter()
    step = settings.scan_union_step_deg
    scan_limit_x = derived.scan_limit_x_deg
    scan_limit_y = derived.scan_limit_y_deg
    scan_x = np.asarray(scan_x, dtype=float).ravel()
    scan_y = np.asarray(scan_y, dtype=float).ravel()
    scan_u = np.asarray(scan_u, dtype=float).ravel()
    scan_v = np.asarray(scan_v, dtype=float).ravel()

    dirs = _make_uv_union_directions(settings.scan_union_theta_n, settings.scan_union_phi_n)
    sx = dirs["sx"]
    sy = dirs["sy"]
    sz = dirs["sz"]
    sxv = sx.ravel()
    syv = sy.ravel()
    szv = sz.ravel()

    active = _active_elements(derived)
    t0 = perf_counter()
    max_coef, best_scan_x, best_scan_y = _max_scan_coefficients(
        params,
        derived,
        elem,
        settings,
        sxv,
        syv,
        szv,
        scan_x,
        scan_y,
        scan_u,
        scan_v,
        active,
    )
    timing_3d_s = perf_counter() - t0

    r_env = np.sqrt(np.maximum(max_coef, 0.0) / derived.s0_w_m2)
    r_env[~np.isfinite(r_env)] = np.nan
    r_grid = r_env.reshape(sx.shape)
    xsurf = r_grid * sx
    ysurf = r_grid * sy
    zsurf = r_grid * sz
    max_range = float(np.nanmax(r_grid)) if np.any(np.isfinite(r_grid)) else float("nan")
    best_scan_x_grid = best_scan_x.reshape(sx.shape)
    best_scan_y_grid = best_scan_y.reshape(sx.shape)
    t0 = perf_counter()
    union_cuts = _make_high_resolution_scan_union_cuts(
        params,
        derived,
        elem,
        settings,
        scan_x,
        scan_y,
        scan_u,
        scan_v,
        active,
    )
    cut_summary = _scan_union_cut_range_summary(union_cuts)
    timing_2d_cuts_s = perf_counter() - t0
    timing_total_s = perf_counter() - total_t0

    return {
        **dirs,
        "Rsurf": r_grid,
        "Xsurf": xsurf,
        "Ysurf": ysurf,
        "Zsurf": zsurf,
        "bestScanX_deg": best_scan_x_grid,
        "bestScanY_deg": best_scan_y_grid,
        "unionCuts": union_cuts,
        "maxCoef": max_coef.reshape(sx.shape),
        "scanLimitX_deg": scan_limit_x,
        "scanLimitY_deg": scan_limit_y,
        "hpbwX_full_deg": derived.hpbw_x_deg,
        "hpbwY_full_deg": derived.hpbw_y_deg,
        "scanStepDeg": step,
        "scanXList_deg": scan_x,
        "scanYList_deg": scan_y,
        "numScanCenters": int(scan_u.size),
        "numDirections": int(sxv.size),
        "maxRange_m": max_range,
        **cut_summary,
        "envelopeMethod": SCAN_UNION_METHOD,
        "envelopeMethodLabel": SCAN_UNION_METHOD_LABEL,
        "timings": {
            "scan_union_3d_s": timing_3d_s,
            "scan_union_2d_cuts_s": timing_2d_cuts_s,
            "scan_union_compute_s": timing_total_s,
        },
    }


def _make_scan_centers(
    params: BeamParams,
    derived: DerivedParams,
    settings: ModeSettings,
    *,
    include_current_scan: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    scan_limit_x = derived.scan_limit_x_deg
    scan_limit_y = derived.scan_limit_y_deg
    scan_x_list = _symmetric_scan_list(scan_limit_x, settings.scan_union_step_deg)
    scan_y_list = _symmetric_scan_list(scan_limit_y, settings.scan_union_step_deg)
    scan_x_grid, scan_y_grid = np.meshgrid(scan_x_list, scan_y_list)
    scan_x = scan_x_grid.ravel()
    scan_y = scan_y_grid.ravel()

    if include_current_scan:
        current = _current_scan_center(params, derived)
        if current is not None:
            cur_x, cur_y, _cur_u, _cur_v = current
            scan_x = np.append(scan_x, cur_x[0])
            scan_y = np.append(scan_y, cur_y[0])

    scan_u = sind(scan_x)
    scan_v = sind(scan_y)
    valid_scan = scan_u * scan_u + scan_v * scan_v < 1.0
    scan_x = scan_x[valid_scan]
    scan_y = scan_y[valid_scan]
    scan_u = scan_u[valid_scan]
    scan_v = scan_v[valid_scan]

    if scan_x.size:
        key = np.round(np.column_stack([scan_x, scan_y]), decimals=9)
        _, unique_idx = np.unique(key, axis=0, return_index=True)
        unique_idx.sort()
        scan_x = scan_x[unique_idx]
        scan_y = scan_y[unique_idx]
        scan_u = scan_u[unique_idx]
        scan_v = scan_v[unique_idx]

    return scan_x.astype(float), scan_y.astype(float), scan_u.astype(float), scan_v.astype(float)


def _current_scan_center(
    params: BeamParams,
    derived: DerivedParams,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    if abs(float(params.scan_x_deg)) > abs(float(derived.scan_limit_x_deg)) + 1.0e-9:
        return None
    if abs(float(params.scan_y_deg)) > abs(float(derived.scan_limit_y_deg)) + 1.0e-9:
        return None
    scan_x = np.asarray([float(params.scan_x_deg)], dtype=float)
    scan_y = np.asarray([float(params.scan_y_deg)], dtype=float)
    scan_u = sind(scan_x)
    scan_v = sind(scan_y)
    if not bool(scan_u[0] * scan_u[0] + scan_v[0] * scan_v[0] < 1.0):
        return None
    return scan_x, scan_y, scan_u.astype(float), scan_v.astype(float)


def _scan_pair_in_list(scan_x: object, scan_y: object, current_x: float, current_y: float) -> bool:
    try:
        x = np.asarray(scan_x, dtype=float).ravel()
        y = np.asarray(scan_y, dtype=float).ravel()
    except Exception:
        return False
    if x.size == 0 or y.size != x.size:
        return False
    return bool(np.any(np.isclose(x, float(current_x), rtol=0.0, atol=1.0e-8) & np.isclose(y, float(current_y), rtol=0.0, atol=1.0e-8)))


def _scan_union_cut_range_summary(cuts: object) -> dict[str, float]:
    cut_list = list(cuts) if isinstance(cuts, list) else []
    values: dict[str, float] = {
        "maxRangeNearFieldCuts_m": float("nan"),
        "maxRangeNearFieldXCut_m": float("nan"),
        "maxRangeNearFieldYCut_m": float("nan"),
    }
    maxima: list[float] = []
    for cut in cut_list:
        if not isinstance(cut, dict):
            continue
        short_name = str(cut.get("short_name", ""))
        value = float(cut.get("max_range_m", float("nan")))
        if short_name == "scan_union_xz_fixed_cut":
            values["maxRangeNearFieldXCut_m"] = value
        elif short_name == "scan_union_yz_fixed_cut":
            values["maxRangeNearFieldYCut_m"] = value
        if math.isfinite(value):
            maxima.append(value)
    if maxima:
        values["maxRangeNearFieldCuts_m"] = float(max(maxima))
    return values


def _merge_scan_union_results(
    base: dict[str, np.ndarray | float | int | list[dict[str, np.ndarray | float | str]]],
    extra: dict[str, np.ndarray | float | int | list[dict[str, np.ndarray | float | str]]],
    derived: DerivedParams,
) -> dict[str, np.ndarray | float | int | bool | list[dict[str, np.ndarray | float | str]]]:
    result = dict(base)
    base_coef = np.asarray(base["maxCoef"], dtype=float)
    extra_coef = np.asarray(extra["maxCoef"], dtype=float)
    extra_wins = np.isfinite(extra_coef) & (~np.isfinite(base_coef) | (extra_coef > base_coef))
    max_coef = np.where(extra_wins, extra_coef, base_coef)
    r_grid = np.sqrt(np.maximum(max_coef, 0.0) / derived.s0_w_m2)
    r_grid[~np.isfinite(r_grid)] = np.nan

    sx = np.asarray(base["sx"], dtype=float)
    sy = np.asarray(base["sy"], dtype=float)
    sz = np.asarray(base["sz"], dtype=float)
    result["maxCoef"] = max_coef
    result["Rsurf"] = r_grid
    result["Xsurf"] = r_grid * sx
    result["Ysurf"] = r_grid * sy
    result["Zsurf"] = r_grid * sz
    result["bestScanX_deg"] = np.where(extra_wins, np.asarray(extra["bestScanX_deg"], dtype=float), np.asarray(base["bestScanX_deg"], dtype=float))
    result["bestScanY_deg"] = np.where(extra_wins, np.asarray(extra["bestScanY_deg"], dtype=float), np.asarray(base["bestScanY_deg"], dtype=float))
    result["maxRange_m"] = float(np.nanmax(r_grid)) if np.any(np.isfinite(r_grid)) else float("nan")

    result["unionCuts"] = _merge_scan_union_cut_lists(
        base.get("unionCuts", []),
        extra.get("unionCuts", []),
    )
    result.update(_scan_union_cut_range_summary(result["unionCuts"]))
    scan_x, scan_y = _merged_scan_lists(base.get("scanXList_deg"), base.get("scanYList_deg"), extra.get("scanXList_deg"), extra.get("scanYList_deg"))
    result["scanXList_deg"] = scan_x
    result["scanYList_deg"] = scan_y
    result["numScanCenters"] = int(scan_x.size)
    result["baseGridOnly"] = False
    return result


def _merge_scan_union_cut_lists(
    base_cuts: object,
    extra_cuts: object,
) -> list[dict[str, np.ndarray | float | str]]:
    base_list = list(base_cuts) if isinstance(base_cuts, list) else []
    extra_list = list(extra_cuts) if isinstance(extra_cuts, list) else []
    merged: list[dict[str, np.ndarray | float | str]] = []
    for idx, base_cut in enumerate(base_list):
        extra_cut = extra_list[idx] if idx < len(extra_list) else None
        if isinstance(base_cut, dict) and isinstance(extra_cut, dict):
            merged.append(_merge_scan_union_cut(base_cut, extra_cut))
        elif isinstance(base_cut, dict):
            merged.append(dict(base_cut))
    return merged


def _merge_scan_union_cut(
    base_cut: dict[str, np.ndarray | float | str],
    extra_cut: dict[str, np.ndarray | float | str],
) -> dict[str, np.ndarray | float | str]:
    result = dict(base_cut)
    base_r = np.asarray(base_cut["r_env_m"], dtype=float)
    extra_r = np.asarray(extra_cut["r_env_m"], dtype=float)
    extra_wins = np.isfinite(extra_r) & (~np.isfinite(base_r) | (extra_r > base_r))
    for key in ("r_env_m", "rho_env_m", "x_env_m", "y_env_m", "z_env_m", "bestScanX_deg", "bestScanY_deg"):
        if key in base_cut and key in extra_cut:
            result[key] = np.where(extra_wins, np.asarray(extra_cut[key], dtype=float), np.asarray(base_cut[key], dtype=float))
    for key in ("far_extended", "has_envelope", "clipped"):
        if key in base_cut and key in extra_cut:
            result[key] = np.where(extra_wins, np.asarray(extra_cut[key], dtype=bool), np.asarray(base_cut[key], dtype=bool))
    result.update(envelope_cut_summary(np.asarray(result["alpha_deg"], dtype=float), np.asarray(result["r_env_m"], dtype=float)))
    return result


def _merged_scan_lists(
    base_x: object,
    base_y: object,
    extra_x: object,
    extra_y: object,
) -> tuple[np.ndarray, np.ndarray]:
    x = np.concatenate([np.asarray(base_x, dtype=float).ravel(), np.asarray(extra_x, dtype=float).ravel()])
    y = np.concatenate([np.asarray(base_y, dtype=float).ravel(), np.asarray(extra_y, dtype=float).ravel()])
    if x.size == 0:
        return x.astype(float), y.astype(float)
    key = np.round(np.column_stack([x, y]), decimals=9)
    _, unique_idx = np.unique(key, axis=0, return_index=True)
    unique_idx.sort()
    return x[unique_idx].astype(float), y[unique_idx].astype(float)


def _active_elements(derived: DerivedParams) -> dict[str, np.ndarray]:
    active = np.asarray(derived.element_power_w, dtype=float).ravel() > 0.0
    return {
        "x": np.asarray(derived.element_x_m, dtype=float).ravel()[active],
        "y": np.asarray(derived.element_y_m, dtype=float).ravel()[active],
        "power": np.asarray(derived.element_power_w, dtype=float).ravel()[active],
        "phase_offset": np.asarray(derived.element_phase_offset_rad, dtype=float).ravel()[active],
    }


def _max_scan_coefficients(
    params: BeamParams,
    derived: DerivedParams,
    elem: ElementPattern,
    settings: ModeSettings,
    sxv: np.ndarray,
    syv: np.ndarray,
    szv: np.ndarray,
    scan_x: np.ndarray,
    scan_y: np.ndarray,
    scan_u: np.ndarray,
    scan_v: np.ndarray,
    active: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xe = np.asarray(active["x"], dtype=float)
    ye = np.asarray(active["y"], dtype=float)
    pin = np.asarray(active["power"], dtype=float)
    phase_offset = np.asarray(active["phase_offset"], dtype=float)

    sxv = np.asarray(sxv, dtype=float).ravel()
    syv = np.asarray(syv, dtype=float).ravel()
    szv = np.asarray(szv, dtype=float).ravel()
    valid_dir = np.isfinite(sxv) & np.isfinite(syv) & np.isfinite(szv)
    max_coef = np.zeros(sxv.size, dtype=float)
    best_scan_x = np.full(sxv.size, np.nan)
    best_scan_y = np.full(sxv.size, np.nan)
    if xe.size == 0 or scan_u.size == 0 or not np.any(valid_dir):
        return max_coef, best_scan_x, best_scan_y

    dir_idx = np.where(valid_dir)[0]
    sxd = sxv[dir_idx]
    syd = syv[dir_idx]
    szd = szv[dir_idx]
    ux = np.broadcast_to(sxd, (xe.size, sxd.size))
    uy = np.broadcast_to(syd, (xe.size, sxd.size))
    uz = np.broadcast_to(szd, (xe.size, sxd.size))
    response_theta, response_phi = element_response_components_fast(ux, uy, uz, elem)
    aperture_phase = np.exp(1j * derived.k_rad_m * (xe[:, None] * sxd[None, :] + ye[:, None] * syd[None, :]))
    dir_matrix_theta = response_theta * aperture_phase
    dir_matrix_phi = response_phi * aperture_phase
    amp = np.sqrt(params.efficiency * pin)

    local_max = np.zeros(sxd.size, dtype=float)
    local_best_x = np.full(sxd.size, np.nan)
    local_best_y = np.full(sxd.size, np.nan)
    for start in range(0, scan_u.size, settings.scan_block_size):
        stop = min(start + settings.scan_block_size, scan_u.size)
        phase = phase_offset[None, :] - derived.k_rad_m * (
            scan_u[start:stop, None] * xe[None, :] + scan_v[start:stop, None] * ye[None, :]
        )
        weights = amp[None, :] * np.exp(1j * phase)
        field_theta = weights @ dir_matrix_theta
        field_phi = weights @ dir_matrix_phi
        coef = (np.abs(field_theta) ** 2 + np.abs(field_phi) ** 2) / (4.0 * math.pi)
        block_best_idx = np.argmax(coef, axis=0)
        block_best = coef[block_best_idx, np.arange(coef.shape[1])]
        update = block_best > local_max
        if np.any(update):
            local_max[update] = block_best[update]
            absolute_idx = start + block_best_idx[update]
            local_best_x[update] = scan_x[absolute_idx]
            local_best_y[update] = scan_y[absolute_idx]

    max_coef[dir_idx] = local_max
    best_scan_x[dir_idx] = local_best_x
    best_scan_y[dir_idx] = local_best_y
    return max_coef, best_scan_x, best_scan_y


def _symmetric_scan_list(limit_deg: float, step_deg: float) -> np.ndarray:
    limit = abs(float(limit_deg))
    step = max(abs(float(step_deg)), 1.0e-9)
    if limit <= 1.0e-12:
        return np.array([0.0], dtype=float)
    n = int(math.floor(limit / step))
    values = [0.0, -limit, limit]
    values.extend(float(k * step) for k in range(-n, n + 1))
    values = [v for v in values if abs(v) <= limit + 1.0e-9]
    values.sort()
    unique: list[float] = []
    for value in values:
        if not unique or abs(value - unique[-1]) > 1.0e-7:
            unique.append(0.0 if abs(value) < 1.0e-12 else value)
    return np.asarray(unique, dtype=float)


def _make_high_resolution_scan_union_cuts(
    params: BeamParams,
    derived: DerivedParams,
    elem: ElementPattern,
    settings: ModeSettings,
    scan_x: np.ndarray,
    scan_y: np.ndarray,
    scan_u: np.ndarray,
    scan_v: np.ndarray,
    active: dict[str, np.ndarray],
) -> list[dict[str, np.ndarray | float | str]]:
    alpha = np.linspace(-85.0, 85.0, max(3, int(settings.n_alpha_2d)))
    r_min = max(0.05, 0.25 * derived.wavelength_m)
    r_near = make_range_grid(r_min, derived.rff_m, settings.n_range_2d)
    cuts: list[dict[str, np.ndarray | float | str]] = []
    for name, short_name, phi_cut in (
        ("Scan union fixed x-z cut", "scan_union_xz_fixed_cut", 0.0),
        ("Scan union fixed y-z cut", "scan_union_yz_fixed_cut", 90.0),
    ):
        sx = sind(alpha) * cosd(phi_cut)
        sy = sind(alpha) * sind(phi_cut)
        sz = cosd(alpha)
        (
            r_env,
            best_scan_x,
            best_scan_y,
            far_extended,
            has_envelope,
            clipped,
        ) = _near_field_scan_union_cut(
            derived,
            elem,
            settings,
            r_near,
            sx,
            sy,
            sz,
            scan_x,
            scan_y,
            scan_u,
            scan_v,
            active,
        )
        x_env = r_env * sx
        y_env = r_env * sy
        z_env = r_env * sz
        cuts.append(
            _make_scan_union_cut(
                name=name,
                short_name=short_name,
                phi_cut_deg=phi_cut,
                signed_direction=sx if abs(phi_cut) < 45.0 else sy,
                r_env=r_env,
                x_env=x_env,
                y_env=y_env,
                z_env=z_env,
                best_scan_x=best_scan_x,
                best_scan_y=best_scan_y,
                far_extended=far_extended,
                has_envelope=has_envelope,
                clipped=clipped,
                envelope_method=SCAN_UNION_CUT_METHOD,
                envelope_method_label=SCAN_UNION_CUT_METHOD_LABEL,
            )
        )
    return cuts


def _near_field_scan_union_cut(
    derived: DerivedParams,
    elem: ElementPattern,
    settings: ModeSettings,
    r_near: np.ndarray,
    sx: np.ndarray,
    sy: np.ndarray,
    sz: np.ndarray,
    scan_x: np.ndarray,
    scan_y: np.ndarray,
    scan_u: np.ndarray,
    scan_v: np.ndarray,
    active: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    sx = np.asarray(sx, dtype=float).ravel()
    sy = np.asarray(sy, dtype=float).ravel()
    sz = np.asarray(sz, dtype=float).ravel()
    r_near = np.asarray(r_near, dtype=float).ravel()
    x_obs = r_near[:, None] * sx[None, :]
    y_obs = r_near[:, None] * sy[None, :]
    z_obs = r_near[:, None] * sz[None, :]

    n_alpha = sx.size
    best_r = np.full(n_alpha, np.nan, dtype=float)
    best_scan_x = np.full(n_alpha, np.nan, dtype=float)
    best_scan_y = np.full(n_alpha, np.nan, dtype=float)
    best_far_extended = np.zeros(n_alpha, dtype=bool)
    best_has_envelope = np.zeros(n_alpha, dtype=bool)
    best_clipped = np.zeros(n_alpha, dtype=bool)

    xe = np.asarray(active["x"], dtype=float)
    ye = np.asarray(active["y"], dtype=float)
    pin = np.asarray(active["power"], dtype=float)
    phase_offset = np.asarray(active["phase_offset"], dtype=float)
    if xe.size == 0 or scan_u.size == 0:
        return best_r, best_scan_x, best_scan_y, best_far_extended, best_has_envelope, best_clipped

    x_flat = x_obs.ravel()
    y_flat = y_obs.ravel()
    z_flat = z_obs.ravel()
    n_pts = x_flat.size
    n_r = r_near.size
    base_amp = np.sqrt(elem.eta_rad * pin / (4.0 * math.pi))
    chunk_size = max(1, int(settings.chunk_size))
    scan_block_size = max(1, int(settings.scan_block_size))

    for scan_start in range(0, scan_u.size, scan_block_size):
        scan_stop = min(scan_start + scan_block_size, scan_u.size)
        phase = phase_offset[None, :] - derived.k_rad_m * (
            scan_u[scan_start:scan_stop, None] * xe[None, :] + scan_v[scan_start:scan_stop, None] * ye[None, :]
        )
        weights = base_amp[None, :] * np.exp(1j * phase)
        s_block = np.zeros((scan_stop - scan_start, n_pts), dtype=float)

        for obs_start in range(0, n_pts, chunk_size):
            obs_stop = min(obs_start + chunk_size, n_pts)
            xs = x_flat[obs_start:obs_stop]
            ys = y_flat[obs_start:obs_stop]
            zs = z_flat[obs_start:obs_stop]
            power = _scan_block_power_density(
                xs,
                ys,
                zs,
                xe,
                ye,
                weights,
                derived.k_rad_m,
                elem,
            )
            s_block[:, obs_start:obs_stop] = power.T

        for local_idx in range(scan_stop - scan_start):
            s_near = s_block[local_idx].reshape(n_r, n_alpha)
            r_env, far_extended, has_envelope, clipped = find_outer_envelope_hybrid(r_near, s_near, derived.s0_w_m2)
            update = np.isfinite(r_env) & (~np.isfinite(best_r) | (r_env > best_r))
            if not np.any(update):
                continue
            idx = scan_start + local_idx
            best_r[update] = r_env[update]
            best_scan_x[update] = scan_x[idx]
            best_scan_y[update] = scan_y[idx]
            best_far_extended[update] = far_extended[update]
            best_has_envelope[update] = has_envelope[update]
            best_clipped[update] = clipped[update]

    return best_r, best_scan_x, best_scan_y, best_far_extended, best_has_envelope, best_clipped


def _scan_block_power_density(
    x_obs: np.ndarray,
    y_obs: np.ndarray,
    z_obs: np.ndarray,
    xe: np.ndarray,
    ye: np.ndarray,
    weights: np.ndarray,
    k: float,
    elem: ElementPattern,
) -> np.ndarray:
    x = np.asarray(x_obs, dtype=float).ravel()
    y = np.asarray(y_obs, dtype=float).ravel()
    z = np.asarray(z_obs, dtype=float).ravel()
    out = np.zeros((x.size, weights.shape[0]), dtype=float)
    valid = (z > 0.0) & np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    if not np.any(valid):
        return out

    xv = x[valid]
    yv = y[valid]
    zv = z[valid]
    rx = xv[:, None] - xe[None, :]
    ry = yv[:, None] - ye[None, :]
    rz = zv[:, None]
    rr = np.sqrt(rx * rx + ry * ry + rz * rz)
    rr = np.maximum(rr, 1.0e-12)
    ux = rx / rr
    uy = ry / rr
    uz = rz / rr
    response_theta, response_phi = element_response_components_fast(ux, uy, uz, elem)
    propagation = np.exp(-1j * float(k) * rr) / rr
    transfer_theta = response_theta * propagation
    transfer_phi = response_phi * propagation
    field_theta = transfer_theta @ weights.T
    field_phi = transfer_phi @ weights.T
    out[valid, :] = np.maximum(np.abs(field_theta) ** 2 + np.abs(field_phi) ** 2, 0.0)
    return out


def _make_scan_union_cuts(
    dirs: dict[str, np.ndarray],
    r_grid: np.ndarray,
    best_scan_x_grid: np.ndarray,
    best_scan_y_grid: np.ndarray,
) -> list[dict[str, np.ndarray | float | str]]:
    u = np.asarray(dirs["u"], dtype=float)
    v = np.asarray(dirs["v"], dtype=float)
    sx = np.asarray(dirs["sx"], dtype=float)
    sy = np.asarray(dirs["sy"], dtype=float)
    sz = np.asarray(dirs["sz"], dtype=float)

    v0_idx = int(np.nanargmin(np.abs(v)))
    u0_idx = int(np.nanargmin(np.abs(u)))
    return [
        _make_scan_union_cut(
            name="Scan union fixed x-z cut",
            short_name="scan_union_xz_fixed_cut",
            phi_cut_deg=0.0,
            signed_direction=sx[v0_idx, :],
            r_env=r_grid[v0_idx, :],
            x_env=r_grid[v0_idx, :] * sx[v0_idx, :],
            y_env=r_grid[v0_idx, :] * sy[v0_idx, :],
            z_env=r_grid[v0_idx, :] * sz[v0_idx, :],
            best_scan_x=best_scan_x_grid[v0_idx, :],
            best_scan_y=best_scan_y_grid[v0_idx, :],
        ),
        _make_scan_union_cut(
            name="Scan union fixed y-z cut",
            short_name="scan_union_yz_fixed_cut",
            phi_cut_deg=90.0,
            signed_direction=sy[:, u0_idx],
            r_env=r_grid[:, u0_idx],
            x_env=r_grid[:, u0_idx] * sx[:, u0_idx],
            y_env=r_grid[:, u0_idx] * sy[:, u0_idx],
            z_env=r_grid[:, u0_idx] * sz[:, u0_idx],
            best_scan_x=best_scan_x_grid[:, u0_idx],
            best_scan_y=best_scan_y_grid[:, u0_idx],
        ),
    ]


def _make_scan_union_cut(
    *,
    name: str,
    short_name: str,
    phi_cut_deg: float,
    signed_direction: np.ndarray,
    r_env: np.ndarray,
    x_env: np.ndarray,
    y_env: np.ndarray,
    z_env: np.ndarray,
    best_scan_x: np.ndarray,
    best_scan_y: np.ndarray,
    far_extended: np.ndarray | None = None,
    has_envelope: np.ndarray | None = None,
    clipped: np.ndarray | None = None,
    envelope_method: str = SCAN_UNION_METHOD,
    envelope_method_label: str = SCAN_UNION_METHOD_LABEL,
) -> dict[str, np.ndarray | float | str]:
    signed = np.asarray(signed_direction, dtype=float)
    r_env = np.asarray(r_env, dtype=float)
    rho = np.asarray(x_env if abs(phi_cut_deg) < 45.0 else y_env, dtype=float)
    alpha_deg = asind(np.clip(signed, -1.0, 1.0))
    summary = envelope_cut_summary(alpha_deg, r_env)
    if has_envelope is None:
        has_envelope = np.isfinite(r_env)
    if far_extended is None:
        far_extended = np.zeros_like(r_env, dtype=bool)
    if clipped is None:
        clipped = np.zeros_like(r_env, dtype=bool)
    return {
        "name": name,
        "short_name": short_name,
        "phi_cut_deg": float(phi_cut_deg),
        "alpha_deg": alpha_deg,
        "r_env_m": r_env,
        "rho_env_m": rho,
        "x_env_m": np.asarray(x_env, dtype=float),
        "y_env_m": np.asarray(y_env, dtype=float),
        "z_env_m": np.asarray(z_env, dtype=float),
        "bestScanX_deg": np.asarray(best_scan_x, dtype=float),
        "bestScanY_deg": np.asarray(best_scan_y, dtype=float),
        "far_extended": np.asarray(far_extended, dtype=bool),
        "has_envelope": np.asarray(has_envelope, dtype=bool),
        "clipped": np.asarray(clipped, dtype=bool),
        **summary,
        "envelope_method": envelope_method,
        "envelope_method_label": envelope_method_label,
    }


def _odd_count(value: int) -> int:
    n = max(3, int(value))
    return n if n % 2 == 1 else n + 1


def _make_uv_union_directions(v_n: int, u_n: int, uv_max: float = 0.999) -> dict[str, np.ndarray]:
    v_count = _odd_count(v_n)
    u_count = _odd_count(u_n)
    u = np.linspace(-uv_max, uv_max, u_count)
    v = np.linspace(-uv_max, uv_max, v_count)
    u_grid, v_grid = np.meshgrid(u, v)
    uv2 = u_grid * u_grid + v_grid * v_grid
    visible = uv2 <= uv_max * uv_max
    w_grid = np.full_like(u_grid, np.nan, dtype=float)
    w_grid[visible] = np.sqrt(np.maximum(1.0 - uv2[visible], 0.0))
    sx = np.where(visible, u_grid, np.nan)
    sy = np.where(visible, v_grid, np.nan)
    sz = w_grid
    theta_grid = np.full_like(u_grid, np.nan, dtype=float)
    phi_grid = np.full_like(u_grid, np.nan, dtype=float)
    theta_grid[visible] = asind(np.sqrt(uv2[visible]))
    phi_grid[visible] = atan2d(v_grid[visible], u_grid[visible])
    return {
        "u": u,
        "v": v,
        "U": u_grid,
        "V": v_grid,
        "visible": visible,
        "theta": np.linspace(0.0, 89.0, v_count),
        "phi": np.linspace(-180.0, 180.0, u_count, endpoint=False),
        "THETA": theta_grid,
        "PHI": phi_grid,
        "sx": sx,
        "sy": sy,
        "sz": sz,
    }
