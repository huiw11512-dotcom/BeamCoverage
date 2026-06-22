from __future__ import annotations

import math

import numpy as np

from core.element_pattern import ElementPattern, element_response_components_fast

try:
    from numba import njit, prange

    NUMBA_AVAILABLE = True
except Exception:  # pragma: no cover - optional acceleration
    NUMBA_AVAILABLE = False
    njit = None
    prange = range


if NUMBA_AVAILABLE:

    @njit
    def _sinc_numba(x: float) -> float:
        if abs(x) <= 1.0e-12:
            return 1.0
        return math.sin(math.pi * x) / (math.pi * x)


    @njit(parallel=True)
    def _power_density_kernel(
        x_obs: np.ndarray,
        y_obs: np.ndarray,
        z_obs: np.ndarray,
        xe: np.ndarray,
        ye: np.ndarray,
        pin: np.ndarray,
        phase: np.ndarray,
        k: float,
        eta_rad: float,
        size_x: float,
        size_y: float,
        wavelength: float,
        gain_norm: float,
        use_element_pattern: bool,
        q: float,
    ) -> np.ndarray:
        npts = x_obs.size
        ne = xe.size
        out = np.zeros(npts, dtype=np.float64)
        for ip in prange(npts):
            x = x_obs[ip]
            y = y_obs[ip]
            z = z_obs[ip]
            if not (z > 0.0 and math.isfinite(x) and math.isfinite(y) and math.isfinite(z)):
                out[ip] = 0.0
                continue
            er = 0.0
            ei = 0.0
            for ie in range(ne):
                rx = x - xe[ie]
                ry = y - ye[ie]
                rz = z
                r = math.sqrt(rx * rx + ry * ry + rz * rz)
                if r < 1.0e-12:
                    r = 1.0e-12
                ux = rx / r
                uy = ry / r
                uz = rz / r
                if uz <= 0.0:
                    continue
                if use_element_pattern:
                    fx = _sinc_numba(size_x / wavelength * ux)
                    fy = _sinc_numba(size_y / wavelength * uy)
                    raw_g = fx * fx * fy * fy * (uz ** q)
                    gain = gain_norm * raw_g
                else:
                    gain = gain_norm
                amp = math.sqrt(max(gain, 0.0)) * math.sqrt(eta_rad * pin[ie] / (4.0 * math.pi)) / r
                ar = amp * math.cos(phase[ie])
                ai = amp * math.sin(phase[ie])
                c = math.cos(k * r)
                s = math.sin(k * r)
                er += ar * c + ai * s
                ei += ai * c - ar * s
            out[ip] = er * er + ei * ei
        return out


def _calc_power_density_numpy(
    x_obs: np.ndarray,
    y_obs: np.ndarray,
    z_obs: np.ndarray,
    xe: np.ndarray,
    ye: np.ndarray,
    pin: np.ndarray,
    phase: np.ndarray,
    k: float,
    elem: ElementPattern,
) -> np.ndarray:
    valid = (z_obs > 0.0) & np.isfinite(x_obs) & np.isfinite(y_obs) & np.isfinite(z_obs)
    out = np.zeros(x_obs.size, dtype=float)
    if not np.any(valid):
        return out
    x = x_obs[valid]
    y = y_obs[valid]
    z = z_obs[valid]
    rx = x[:, None] - xe[None, :]
    ry = y[:, None] - ye[None, :]
    rz = z[:, None]
    r = np.sqrt(rx * rx + ry * ry + rz * rz)
    r = np.maximum(r, 1.0e-12)
    ux = rx / r
    uy = ry / r
    uz = rz / r
    response_theta, response_phi = element_response_components_fast(ux, uy, uz, elem)
    amp0 = np.sqrt(elem.eta_rad * pin / (4.0 * math.pi)) * np.exp(1j * phase)
    propagation = np.exp(-1j * k * r) / r
    field_theta = np.sum((response_theta * amp0[None, :]) * propagation, axis=1)
    field_phi = np.sum((response_phi * amp0[None, :]) * propagation, axis=1)
    out[valid] = np.maximum(np.abs(field_theta) ** 2 + np.abs(field_phi) ** 2, 0.0)
    return out


def calc_array_power_density_chunked(
    pin_matrix: np.ndarray,
    feed_phase: np.ndarray,
    x_elem: np.ndarray,
    y_elem: np.ndarray,
    x_obs: np.ndarray,
    y_obs: np.ndarray,
    z_obs: np.ndarray,
    k: float,
    elem: ElementPattern,
    chunk_size: int = 8000,
    prefer_numba: bool = True,
) -> np.ndarray:
    xe_grid, ye_grid = np.meshgrid(x_elem, y_elem)
    active = pin_matrix.ravel() > 0.0
    if not np.any(active):
        return np.zeros_like(np.asarray(x_obs, dtype=float))
    return calc_array_power_density_elements(
        xe_grid.ravel()[active],
        ye_grid.ravel()[active],
        pin_matrix.ravel()[active],
        feed_phase.ravel()[active],
        x_obs,
        y_obs,
        z_obs,
        k,
        elem,
        chunk_size=chunk_size,
        prefer_numba=prefer_numba,
    )


def calc_array_power_density_elements(
    element_x_m: np.ndarray,
    element_y_m: np.ndarray,
    element_power_w: np.ndarray,
    element_feed_phase_rad: np.ndarray,
    x_obs: np.ndarray,
    y_obs: np.ndarray,
    z_obs: np.ndarray,
    k: float,
    elem: ElementPattern,
    chunk_size: int = 8000,
    prefer_numba: bool = True,
) -> np.ndarray:
    x_flat = np.asarray(x_obs, dtype=float).ravel()
    y_flat = np.asarray(y_obs, dtype=float).ravel()
    z_flat = np.asarray(z_obs, dtype=float).ravel()
    active = np.asarray(element_power_w, dtype=float).ravel() > 0.0
    if not np.any(active):
        return np.zeros_like(np.asarray(x_obs, dtype=float))

    xe = np.ascontiguousarray(np.asarray(element_x_m, dtype=float).ravel()[active].astype(np.float64))
    ye = np.ascontiguousarray(np.asarray(element_y_m, dtype=float).ravel()[active].astype(np.float64))
    pin = np.ascontiguousarray(np.asarray(element_power_w, dtype=float).ravel()[active].astype(np.float64))
    phase = np.ascontiguousarray(np.asarray(element_feed_phase_rad, dtype=float).ravel()[active].astype(np.float64))

    out = np.zeros(x_flat.size, dtype=float)
    for start in range(0, x_flat.size, int(chunk_size)):
        stop = min(start + int(chunk_size), x_flat.size)
        xs = np.ascontiguousarray(x_flat[start:stop])
        ys = np.ascontiguousarray(y_flat[start:stop])
        zs = np.ascontiguousarray(z_flat[start:stop])
        use_numba_kernel = prefer_numba and NUMBA_AVAILABLE and (not elem.use_element_pattern or elem.mode == "rect")
        if use_numba_kernel:
            out[start:stop] = _power_density_kernel(
                xs,
                ys,
                zs,
                xe,
                ye,
                pin,
                phase,
                float(k),
                float(elem.eta_rad),
                float(elem.size_x_m),
                float(elem.size_y_m),
                float(elem.wavelength_m),
                float(elem.gain_norm),
                bool(elem.use_element_pattern),
                float(elem.obliquity_q),
            )
        else:
            out[start:stop] = _calc_power_density_numpy(xs, ys, zs, xe, ye, pin, phase, k, elem)
    return out.reshape(np.asarray(x_obs).shape)
