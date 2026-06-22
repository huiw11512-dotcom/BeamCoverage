from __future__ import annotations

import math

import numpy as np

from core.element_pattern import ElementPattern
from core.geometry import (
    BeamParams,
    DerivedParams,
    ModeSettings,
    asind,
    atan2d,
    cosd,
    make_hemisphere_directions,
    make_range_grid,
    make_vertical_cut_directions,
    sind,
)
from core.power_density import calc_array_power_density_elements


CURRENT_ENVELOPE_METHOD = "near_field_sampled_far_field_extrapolated"
CURRENT_ENVELOPE_METHOD_LABEL = "近场采样 + 远场外推"


def envelope_cut_summary(alpha_deg: np.ndarray, r_env_m: np.ndarray) -> dict[str, float | int]:
    alpha = np.asarray(alpha_deg, dtype=float).ravel()
    r_env = np.asarray(r_env_m, dtype=float).ravel()
    finite = np.isfinite(r_env)
    if not np.any(finite):
        return {
            "max_range_m": float("nan"),
            "max_alpha_deg": float("nan"),
            "finite_direction_count": 0,
            "total_direction_count": int(r_env.size),
        }
    finite_indices = np.flatnonzero(finite)
    local_max = int(np.nanargmax(r_env[finite]))
    idx = int(finite_indices[local_max])
    alpha_value = float(alpha[idx]) if idx < alpha.size else float("nan")
    return {
        "max_range_m": float(r_env[idx]),
        "max_alpha_deg": alpha_value,
        "finite_direction_count": int(np.count_nonzero(finite)),
        "total_direction_count": int(r_env.size),
    }


def find_outer_envelope_hybrid(
    r: np.ndarray,
    power_density: np.ndarray,
    s0_w_m2: float,
    max_envelope_range_m: float = math.inf,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    r = np.asarray(r, dtype=float).ravel()
    s = np.asarray(power_density, dtype=float)
    nr = r.size
    ndir = s.shape[1]
    r_ff = r[-1]
    r_env = np.full(ndir, np.nan, dtype=float)
    far_extended = np.zeros(ndir, dtype=bool)
    has_envelope = np.zeros(ndir, dtype=bool)
    clipped = np.zeros(ndir, dtype=bool)

    for jj in range(ndir):
        sj = s[:, jj]
        if not np.any(np.isfinite(sj)):
            continue
        above = sj >= s0_w_m2
        if not np.any(above):
            continue
        has_envelope[jj] = True
        if sj[-1] >= s0_w_m2:
            r_out = r_ff * math.sqrt(max(float(sj[-1]), np.finfo(float).tiny) / s0_w_m2)
            far_extended[jj] = True
        else:
            idx_last = int(np.where(above)[0][-1])
            if idx_last >= nr - 1:
                r_out = r_ff
            else:
                s1 = max(float(sj[idx_last]), np.finfo(float).tiny)
                s2 = max(float(sj[idx_last + 1]), np.finfo(float).tiny)
                r1 = float(r[idx_last])
                r2 = float(r[idx_last + 1])
                if s1 == s2 or r1 == r2:
                    r_out = r1
                else:
                    t = (math.log(s0_w_m2) - math.log(s1)) / (math.log(s2) - math.log(s1))
                    t = min(max(t, 0.0), 1.0)
                    r_out = math.exp(math.log(r1) + t * (math.log(r2) - math.log(r1)))
        if math.isfinite(max_envelope_range_m) and r_out > max_envelope_range_m:
            r_out = max_envelope_range_m
            clipped[jj] = True
        r_env[jj] = r_out
    return r_env, far_extended, has_envelope, clipped


def append_far_field_for_plot(
    r_near: np.ndarray,
    s_near: np.ndarray,
    r_plot_max: float,
    rff_m: float,
    n_far: int,
) -> tuple[np.ndarray, np.ndarray]:
    r_near = np.asarray(r_near, dtype=float).ravel()
    if r_plot_max <= rff_m * 1.001:
        return r_near, s_near
    r_far = np.logspace(math.log10(rff_m * 1.001), math.log10(r_plot_max), int(n_far))
    sff = s_near[-1, :]
    s_far = (rff_m / r_far[:, None]) ** 2 * sff[None, :]
    return np.concatenate([r_near, r_far]), np.vstack([s_near, s_far])


def compute_2d_cuts(
    params: BeamParams,
    derived: DerivedParams,
    elem: ElementPattern,
    settings: ModeSettings,
) -> list[dict[str, np.ndarray | str | float]]:
    r_min = max(0.05, 0.25 * derived.wavelength_m)
    r_near = make_range_grid(r_min, derived.rff_m, settings.n_range_2d)

    cuts: list[dict[str, np.ndarray | str | float]] = []
    for name, short_name, phi_cut in (
        ("Global x-z vertical cut", "global_xz_vertical_cut", 0.0),
        ("Global y-z vertical cut", "global_yz_vertical_cut", 90.0),
    ):
        alpha_abs_max = _fixed_cut_alpha_abs_max(params, settings, phi_cut)
        alpha = np.linspace(-alpha_abs_max, alpha_abs_max, settings.n_alpha_2d)
        sx, sy, sz = make_vertical_cut_directions(alpha, phi_cut)
        x_obs = r_near[:, None] * sx[None, :]
        y_obs = r_near[:, None] * sy[None, :]
        z_obs = r_near[:, None] * sz[None, :]
        s_near = calc_array_power_density_elements(
            derived.element_x_m,
            derived.element_y_m,
            derived.element_power_w,
            derived.element_feed_phase_rad,
            x_obs,
            y_obs,
            z_obs,
            derived.k_rad_m,
            elem,
            settings.chunk_size,
        )
        r_env, far_extended, has_envelope, clipped = find_outer_envelope_hybrid(r_near, s_near, derived.s0_w_m2)
        summary = envelope_cut_summary(alpha, r_env)
        rho_env = r_env * sind(alpha)
        z_env = r_env * cosd(alpha)
        x_env = rho_env * cosd(phi_cut)
        y_env = rho_env * sind(phi_cut)
        cuts.append(
            {
                "name": name,
                "short_name": short_name,
                "phi_cut_deg": phi_cut,
                "alpha_abs_max_deg": alpha_abs_max,
                "alpha_window_source": "scan_x_deg" if abs(phi_cut) < 45.0 else "scan_y_deg",
                "alpha_deg": alpha,
                "sx": sx,
                "sy": sy,
                "sz": sz,
                "r_near_m": r_near,
                "s_near_w_m2": s_near,
                "s_near_w_cm2": s_near / 1.0e4,
                "r_env_m": r_env,
                "rho_env_m": rho_env,
                "z_env_m": z_env,
                "x_env_m": x_env,
                "y_env_m": y_env,
                "far_extended": far_extended,
                "has_envelope": has_envelope,
                "clipped": clipped,
                **summary,
                "envelope_method": CURRENT_ENVELOPE_METHOD,
                "envelope_method_label": CURRENT_ENVELOPE_METHOD_LABEL,
            }
        )
    return cuts


def _fixed_cut_alpha_abs_max(params: BeamParams, settings: ModeSettings, phi_cut_deg: float) -> float:
    scan_component_deg = params.scan_x_deg if abs(phi_cut_deg) < 45.0 else params.scan_y_deg
    return min(85.0, max(float(settings.cut_half_width_deg), abs(float(scan_component_deg)) + float(settings.cut_half_width_deg)))


def compute_current_3d_envelope(
    params: BeamParams,
    derived: DerivedParams,
    elem: ElementPattern,
    settings: ModeSettings,
) -> dict[str, np.ndarray | float]:
    r_min = max(0.05, 0.25 * derived.wavelength_m)
    r_near = make_range_grid(r_min, derived.rff_m, settings.n_range_3d)
    dirs = make_uv_hemisphere_directions(settings.theta_3d_n, settings.phi_3d_n)
    sx = dirs["sx"]
    sy = dirs["sy"]
    sz = dirs["sz"]
    sxv = sx.ravel()
    syv = sy.ravel()
    szv = sz.ravel()
    x_obs = r_near[:, None] * sxv[None, :]
    y_obs = r_near[:, None] * syv[None, :]
    z_obs = r_near[:, None] * szv[None, :]
    s_near = calc_array_power_density_elements(
        derived.element_x_m,
        derived.element_y_m,
        derived.element_power_w,
        derived.element_feed_phase_rad,
        x_obs,
        y_obs,
        z_obs,
        derived.k_rad_m,
        elem,
        settings.chunk_size,
    )
    r_env, far_extended, has_envelope, clipped = find_outer_envelope_hybrid(r_near, s_near, derived.s0_w_m2)
    r_env_grid = r_env.reshape(sx.shape)
    xsurf = r_env_grid * sx
    ysurf = r_env_grid * sy
    zsurf = r_env_grid * sz
    max_range = np.nanmax(r_env_grid) if np.any(np.isfinite(r_env_grid)) else np.nan
    return {
        **dirs,
        "r_env_m": r_env_grid,
        "Xsurf": xsurf,
        "Ysurf": ysurf,
        "Zsurf": zsurf,
        "far_extended": far_extended.reshape(sx.shape),
        "has_envelope": has_envelope.reshape(sx.shape),
        "clipped": clipped.reshape(sx.shape),
        "max_range_m": max_range,
        "envelopeMethod": CURRENT_ENVELOPE_METHOD,
        "envelopeMethodLabel": CURRENT_ENVELOPE_METHOD_LABEL,
    }


def make_uv_hemisphere_directions(v_n: int, u_n: int, uv_max: float = 0.999) -> dict[str, np.ndarray]:
    v_n = _odd_count(v_n)
    u_n = _odd_count(u_n)
    u = np.linspace(-uv_max, uv_max, int(u_n))
    v = np.linspace(-uv_max, uv_max, int(v_n))
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
        "theta": np.linspace(0.0, 89.0, int(v_n)),
        "phi": np.linspace(-180.0, 180.0, int(u_n), endpoint=False),
        "THETA": theta_grid,
        "PHI": phi_grid,
        "sx": sx,
        "sy": sy,
        "sz": sz,
    }


def _odd_count(value: int) -> int:
    n = max(3, int(value))
    return n if n % 2 == 1 else n + 1
