from __future__ import annotations

import math

import numpy as np

from core.element_pattern import ElementPattern, element_response_components_fast, element_response_fast
from core.geometry import BeamParams, DerivedParams


def calc_far_field_pattern(
    pin: np.ndarray,
    feed_phase: np.ndarray,
    x_elem: np.ndarray,
    y_elem: np.ndarray,
    sx: np.ndarray,
    sy: np.ndarray,
    sz: np.ndarray,
    k: float,
    elem: ElementPattern,
    chunk_size: int = 30000,
) -> np.ndarray:
    sx_arr = np.asarray(sx, dtype=float)
    out_shape = sx_arr.shape
    sxv = sx_arr.ravel()
    syv = np.asarray(sy, dtype=float).ravel()
    szv = np.asarray(sz, dtype=float).ravel()

    xe_grid, ye_grid = np.meshgrid(x_elem, y_elem)
    active = pin.ravel() > 0.0
    if not np.any(active):
        return np.zeros(out_shape, dtype=complex)

    xe = xe_grid.ravel()[active]
    ye = ye_grid.ravel()[active]
    powers = pin.ravel()[active]
    phase = feed_phase.ravel()[active]
    weights = np.sqrt(elem.eta_rad * powers) * np.exp(1j * phase)

    result = np.empty(sxv.size, dtype=complex)
    for start in range(0, sxv.size, chunk_size):
        stop = min(start + chunk_size, sxv.size)
        sx_c = sxv[start:stop]
        sy_c = syv[start:stop]
        sz_c = szv[start:stop]
        ux = np.broadcast_to(sx_c, (weights.size, sx_c.size))
        uy = np.broadcast_to(sy_c, (weights.size, sx_c.size))
        uz = np.broadcast_to(sz_c, (weights.size, sx_c.size))
        response = element_response_fast(ux, uy, uz, elem)
        af = np.exp(1j * k * (xe[:, None] * sx_c[None, :] + ye[:, None] * sy_c[None, :]))
        result[start:stop] = np.sum((response * weights[:, None]) * af, axis=0)
    return result.reshape(out_shape)


def calc_far_field_pattern_elements(
    element_power_w: np.ndarray,
    element_feed_phase_rad: np.ndarray,
    element_x_m: np.ndarray,
    element_y_m: np.ndarray,
    sx: np.ndarray,
    sy: np.ndarray,
    sz: np.ndarray,
    k: float,
    elem: ElementPattern,
    chunk_size: int = 30000,
) -> np.ndarray:
    sx_arr = np.asarray(sx, dtype=float)
    out_shape = sx_arr.shape
    sxv = sx_arr.ravel()
    syv = np.asarray(sy, dtype=float).ravel()
    szv = np.asarray(sz, dtype=float).ravel()

    active = np.asarray(element_power_w, dtype=float).ravel() > 0.0
    if not np.any(active):
        return np.zeros(out_shape, dtype=complex)

    xe = np.asarray(element_x_m, dtype=float).ravel()[active]
    ye = np.asarray(element_y_m, dtype=float).ravel()[active]
    powers = np.asarray(element_power_w, dtype=float).ravel()[active]
    phase = np.asarray(element_feed_phase_rad, dtype=float).ravel()[active]
    weights = np.sqrt(elem.eta_rad * powers) * np.exp(1j * phase)

    result = np.empty(sxv.size, dtype=complex)
    for start in range(0, sxv.size, chunk_size):
        stop = min(start + chunk_size, sxv.size)
        sx_c = sxv[start:stop]
        sy_c = syv[start:stop]
        sz_c = szv[start:stop]
        ux = np.broadcast_to(sx_c, (weights.size, sx_c.size))
        uy = np.broadcast_to(sy_c, (weights.size, sx_c.size))
        uz = np.broadcast_to(sz_c, (weights.size, sx_c.size))
        response = element_response_fast(ux, uy, uz, elem)
        af = np.exp(1j * k * (xe[:, None] * sx_c[None, :] + ye[:, None] * sy_c[None, :]))
        result[start:stop] = np.sum((response * weights[:, None]) * af, axis=0)
    return result.reshape(out_shape)


def calc_far_field_power_elements(
    element_power_w: np.ndarray,
    element_feed_phase_rad: np.ndarray,
    element_x_m: np.ndarray,
    element_y_m: np.ndarray,
    sx: np.ndarray,
    sy: np.ndarray,
    sz: np.ndarray,
    k: float,
    elem: ElementPattern,
    chunk_size: int = 30000,
) -> np.ndarray:
    sx_arr = np.asarray(sx, dtype=float)
    out_shape = sx_arr.shape
    sxv = sx_arr.ravel()
    syv = np.asarray(sy, dtype=float).ravel()
    szv = np.asarray(sz, dtype=float).ravel()

    active = np.asarray(element_power_w, dtype=float).ravel() > 0.0
    if not np.any(active):
        return np.zeros(out_shape, dtype=float)

    xe = np.asarray(element_x_m, dtype=float).ravel()[active]
    ye = np.asarray(element_y_m, dtype=float).ravel()[active]
    powers = np.asarray(element_power_w, dtype=float).ravel()[active]
    phase = np.asarray(element_feed_phase_rad, dtype=float).ravel()[active]
    weights = np.sqrt(elem.eta_rad * powers) * np.exp(1j * phase)

    result = np.empty(sxv.size, dtype=float)
    for start in range(0, sxv.size, chunk_size):
        stop = min(start + chunk_size, sxv.size)
        sx_c = sxv[start:stop]
        sy_c = syv[start:stop]
        sz_c = szv[start:stop]
        ux = np.broadcast_to(sx_c, (weights.size, sx_c.size))
        uy = np.broadcast_to(sy_c, (weights.size, sx_c.size))
        uz = np.broadcast_to(sz_c, (weights.size, sx_c.size))
        response_theta, response_phi = element_response_components_fast(ux, uy, uz, elem)
        af = np.exp(1j * k * (xe[:, None] * sx_c[None, :] + ye[:, None] * sy_c[None, :]))
        field_theta = np.sum((response_theta * weights[:, None]) * af, axis=0)
        field_phi = np.sum((response_phi * weights[:, None]) * af, axis=0)
        result[start:stop] = np.maximum(np.abs(field_theta) ** 2 + np.abs(field_phi) ** 2, 0.0)
    return result.reshape(out_shape)


def calc_uv_pattern(
    params: BeamParams,
    derived: DerivedParams,
    elem: ElementPattern,
    n_uv: int,
    floor_db: float = -40.0,
) -> dict[str, np.ndarray | float]:
    u = np.linspace(-1.0, 1.0, int(n_uv))
    v = np.linspace(-1.0, 1.0, int(n_uv))
    u_grid, v_grid = np.meshgrid(u, v)
    visible = u_grid * u_grid + v_grid * v_grid <= 1.0
    w_grid = np.sqrt(np.maximum(1.0 - u_grid * u_grid - v_grid * v_grid, 0.0))
    sz = w_grid.copy()
    sz[~visible] = -1.0

    power = calc_far_field_power_elements(
        derived.element_power_w,
        derived.element_feed_phase_rad,
        derived.element_x_m,
        derived.element_y_m,
        u_grid,
        v_grid,
        sz,
        derived.k_rad_m,
        elem,
    )
    power[~visible] = np.nan
    max_power = np.nanmax(power) if np.any(np.isfinite(power)) else 1.0
    floor_linear = 10.0 ** (floor_db / 10.0)
    pdb = 10.0 * np.log10(np.maximum(power / max(max_power, np.finfo(float).tiny), floor_linear))
    pdb[~visible] = np.nan
    return {
        "u": u,
        "v": v,
        "U": u_grid,
        "V": v_grid,
        "W": w_grid,
        "visible": visible,
        "pattern_db": pdb,
        "floor_db": floor_db,
        "scan_u": derived.u0,
        "scan_v": derived.v0,
        "theta_deg": derived.theta_deg,
        "phi_deg": derived.phi_deg,
    }


def far_field_power_coefficient(field: np.ndarray) -> np.ndarray:
    return np.abs(field) ** 2 / (4.0 * math.pi)
