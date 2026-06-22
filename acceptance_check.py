from __future__ import annotations

import math
from pathlib import Path
import sys
import tempfile
import time

import numpy as np

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.analysis import derive_params_with_element
from core.aperture_shapes import (
    CANONICAL_ARRAY_LAYOUTS,
    CANONICAL_ELEMENT_SHAPES,
    element_overlap_metric,
    element_shape_to_mode,
    normalize_shape_name,
    shape_label,
)
from core.array_factor import calc_uv_pattern
from core.element_pattern import element_gain_fast, element_response_components_fast, element_response_fast, make_element_pattern, scan_loss_db_for_direction
from core.envelope import compute_2d_cuts, compute_current_3d_envelope
from core.geometry import BeamParams, base_cache_key, derive_params, get_effective_mode_settings, get_mode_settings, make_pin_matrix
from core.near_field import (
    export_element_near_field_vector_template,
    export_near_field_projected_far_field_pattern,
    load_imported_element_near_field,
    near_field_summary,
)
from core.scan_union import compute_scan_union_base_envelope_3d, compute_scan_union_envelope_3d, merge_scan_union_with_current_scan


def close(value: float, target: float, tol: float, label: str) -> None:
    if not math.isfinite(value) or abs(value - target) > tol:
        raise AssertionError(f"{label}: expected {target} +/- {tol}, got {value}")


def scan_definition_checks() -> None:
    cases = [
        (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        (5.0, 0.0, math.sin(math.radians(5.0)), 0.0, 5.0, 0.0),
        (0.0, 5.0, 0.0, math.sin(math.radians(5.0)), 5.0, 90.0),
        (
            5.0,
            5.0,
            math.sin(math.radians(5.0)),
            math.sin(math.radians(5.0)),
            math.degrees(math.asin(math.sqrt(2.0 * math.sin(math.radians(5.0)) ** 2))),
            45.0,
        ),
    ]
    for scan_x, scan_y, u, v, theta, phi in cases:
        _, d = derive_params(BeamParams(scan_x_deg=scan_x, scan_y_deg=scan_y))
        close(d.u0, u, 1.0e-12, f"u0 scanX={scan_x}, scanY={scan_y}")
        close(d.v0, v, 1.0e-12, f"v0 scanX={scan_x}, scanY={scan_y}")
        close(d.theta_deg, theta, 1.0e-10, f"theta scanX={scan_x}, scanY={scan_y}")
        close(d.phi_deg, phi, 1.0e-10, f"phi scanX={scan_x}, scanY={scan_y}")


def auto_result_checks() -> None:
    p = BeamParams(frequency_ghz=10.0, nx=14, ny=8, dx_m=0.1, dy_m=0.1, ax_m=0.1, ay_m=0.1)
    _, d = derive_params(p)
    close(d.dx_aperture_m, 1.4, 1.0e-12, "Dx")
    close(d.dy_aperture_m, 0.8, 1.0e-12, "Dy")
    close(d.hpbw_x_deg, 1.0877, 0.03, "HPBWx")
    close(d.hpbw_y_deg, 1.9036, 0.05, "HPBWy")
    close(d.rff_m, 130.6667, 0.05, "Rff")


def fixed_cut_checks() -> None:
    p = BeamParams(scan_x_deg=5.0, scan_y_deg=5.0, nx=4, ny=4, element_power_w=1.0e4)
    p, d = derive_params(p)
    elem = make_element_pattern(p, d.wavelength_m)
    settings0 = get_mode_settings("fast")
    settings = settings0.__class__(
        **{
            **settings0.__dict__,
            "n_alpha_2d": 41,
            "n_range_2d": 20,
            "chunk_size": 512,
        }
    )
    cuts = compute_2d_cuts(p, d, elem, settings)
    close(float(cuts[0]["phi_cut_deg"]), 0.0, 0.0, "x-z cut phi")
    close(float(cuts[1]["phi_cut_deg"]), 90.0, 0.0, "y-z cut phi")
    if not all(cut.get("envelope_method") == "near_field_sampled_far_field_extrapolated" for cut in cuts):
        raise AssertionError("2D cuts did not expose current-envelope method metadata.")
    if not np.allclose(cuts[0]["sy"], 0.0, atol=1.0e-14):
        raise AssertionError("x-z cut has nonzero sy; it is not fixed at phi=0.")
    if not np.allclose(cuts[1]["sx"], 0.0, atol=1.0e-14):
        raise AssertionError("y-z cut has nonzero sx; it is not fixed at phi=90.")

    p_asym = BeamParams(scan_x_deg=25.0, scan_y_deg=0.0, nx=4, ny=4, element_power_w=1.0e4)
    p_asym, d_asym = derive_params(p_asym)
    elem_asym = make_element_pattern(p_asym, d_asym.wavelength_m)
    cuts_asym = compute_2d_cuts(p_asym, d_asym, elem_asym, settings)
    close(float(cuts_asym[0]["alpha_abs_max_deg"]), 35.0, 1.0e-12, "x-z cut alpha window follows scanX")
    close(float(cuts_asym[1]["alpha_abs_max_deg"]), 10.0, 1.0e-12, "y-z cut alpha window ignores scanX")
    if cuts_asym[0].get("alpha_window_source") != "scan_x_deg" or cuts_asym[1].get("alpha_window_source") != "scan_y_deg":
        raise AssertionError("2D fixed-cut alpha-window metadata is missing or incorrect.")


def uv_pattern_extreme_spacing_checks() -> None:
    settings0 = get_mode_settings("fast")
    settings = settings0.__class__(**{**settings0.__dict__, "n_uv": 201})
    for scan_x in (0.0, 5.0):
        p = BeamParams(nx=8, ny=8, dx_m=0.7, dy_m=0.1, ax_m=0.1, ay_m=0.1, frequency_ghz=10.0, scan_x_deg=scan_x, scan_y_deg=0.0)
        p, d = derive_params(p)
        elem = make_element_pattern(p, d.wavelength_m)
        uv = calc_uv_pattern(p, d, elem, settings.n_uv, settings.pattern_floor_db)
        pdb = uv["pattern_db"]
        row = pdb[np.argmin(np.abs(uv["v"] - 0.0)), :]
        u_axis = uv["u"]
        peaks: list[float] = []
        for idx in range(1, u_axis.size - 1):
            if np.isfinite(row[idx]) and row[idx] >= row[idx - 1] and row[idx] >= row[idx + 1] and row[idx] > -10.0:
                peaks.append(float(u_axis[idx]))
        if len(peaks) < 5:
            raise AssertionError("Extreme dx case did not show multiple strong u-axis lobes.")

        period = d.wavelength_m / p.dx_m
        orders = np.arange(math.floor((-1.0 - d.u0) / period), math.ceil((1.0 - d.u0) / period) + 1)
        expected = d.u0 + orders * period
        expected = expected[(expected >= -1.0) & (expected <= 1.0)]
        grid_tol = 2.1 * float(u_axis[1] - u_axis[0])
        misses = [peak for peak in peaks[:8] if np.min(np.abs(expected - peak)) > grid_tol]
        if misses:
            raise AssertionError(f"u-v grating-lobe family does not match u0+m*lambda/dx for scanX={scan_x}: {misses}")


def scan_union_checks() -> None:
    p = BeamParams(nx=4, ny=4, element_power_w=1.0e4, calc_mode="fast")
    p, d = derive_params(p)
    elem = make_element_pattern(p, d.wavelength_m)
    settings0 = get_mode_settings("fast")
    settings = settings0.__class__(
        **{
            **settings0.__dict__,
            "scan_union_step_deg": 8.0,
            "scan_union_theta_n": 12,
            "scan_union_phi_n": 24,
            "scan_block_size": 16,
        }
    )
    union = compute_scan_union_envelope_3d(p, d, elem, settings)
    for key in ("Xsurf", "Ysurf", "Zsurf", "Rsurf", "bestScanX_deg", "bestScanY_deg", "unionCuts"):
        if key not in union:
            raise AssertionError(f"Scan union missing {key}")
    if union.get("envelopeMethod") != "far_field_coefficient_union":
        raise AssertionError("Scan union did not expose far-field union method metadata.")
    timing = union.get("timings", {})
    for key in ("scan_union_3d_s", "scan_union_2d_cuts_s", "scan_union_compute_s"):
        value = float(timing.get(key, float("nan"))) if isinstance(timing, dict) else float("nan")
        if not math.isfinite(value) or value < 0.0:
            raise AssertionError(f"Scan union timing {key} is missing or invalid: {value}")
    if union["Rsurf"].shape != union["Xsurf"].shape:
        raise AssertionError("Scan union range/grid shape mismatch.")
    if not np.nanmax(union["Zsurf"]) > 1.0:
        raise AssertionError("Scan union Zsurf is not a meter-scale spatial envelope.")
    if float(union["maxRange_m"]) <= 1.0:
        raise AssertionError("Scan union maxRange_m is not a spatial S=S0 range.")
    if "maxRangeNearFieldCuts_m" not in union:
        raise AssertionError("Scan union did not expose maxRangeNearFieldCuts_m.")
    if not float(union["maxRangeNearFieldCuts_m"]) > 1.0:
        raise AssertionError("Scan union near-field fixed-cut max range is not a spatial S=S0 range.")
    if not np.allclose(union["Xsurf"], union["Rsurf"] * union["sx"], equal_nan=True):
        raise AssertionError("Scan union Xsurf is not R*u in meters.")
    if not np.allclose(union["Ysurf"], union["Rsurf"] * union["sy"], equal_nan=True):
        raise AssertionError("Scan union Ysurf is not R*v in meters.")
    if not np.allclose(union["Zsurf"], union["Rsurf"] * union["sz"], equal_nan=True):
        raise AssertionError("Scan union Zsurf is not R*w in meters.")
    cuts = union["unionCuts"]
    if not isinstance(cuts, list) or len(cuts) != 2:
        raise AssertionError("Scan union did not produce fixed x-z/y-z 2D union cuts.")
    if float(cuts[0]["phi_cut_deg"]) != 0.0 or float(cuts[1]["phi_cut_deg"]) != 90.0:
        raise AssertionError("Scan union 2D cuts are not fixed at phi=0 and phi=90.")
    cut0_y = np.asarray(cuts[0]["y_env_m"], dtype=float)
    if not np.allclose(cut0_y[np.isfinite(cut0_y)], 0.0, atol=1.0e-12):
        raise AssertionError("Scan union x-z cut has nonzero y coordinates.")
    cut1_x = np.asarray(cuts[1]["x_env_m"], dtype=float)
    if not np.allclose(cut1_x[np.isfinite(cut1_x)], 0.0, atol=1.0e-12):
        raise AssertionError("Scan union y-z cut has nonzero x coordinates.")
    for cut in cuts:
        if not np.any(np.isfinite(cut["r_env_m"])):
            raise AssertionError(f"Scan union cut {cut['short_name']} has no finite envelope.")
        if np.asarray(cut["r_env_m"]).size != settings.n_alpha_2d:
            raise AssertionError("Scan union 2D cuts must use the 2D cut angular resolution, not the coarse 3D union grid.")
        if cut.get("envelope_method") != "near_field_sampled_far_field_scan_union":
            raise AssertionError("Scan union fixed cut did not expose near-field scan-union cut method metadata.")
        for key in ("max_range_m", "max_alpha_deg", "finite_direction_count", "total_direction_count"):
            if key not in cut:
                raise AssertionError(f"Scan union fixed cut did not expose {key}.")
    has_zero_scan = any(
        abs(float(x)) < 1.0e-12 and abs(float(y)) < 1.0e-12
        for x, y in zip(union["scanXList_deg"], union["scanYList_deg"])
    )
    if not has_zero_scan:
        raise AssertionError("Scan union centers do not include the 0/0 boresight scan.")

    p_current = BeamParams(
        nx=8,
        ny=8,
        scan_x_deg=3.2,
        scan_y_deg=-2.1,
        scan_limit_mode="manual",
        manual_scan_limit_x_deg=8.0,
        manual_scan_limit_y_deg=8.0,
        calc_mode="fast",
    )
    p_current, d_current = derive_params(p_current)
    elem_current = make_element_pattern(p_current, d_current.wavelength_m)
    settings_current = settings0.__class__(
        **{
            **settings0.__dict__,
            "scan_union_step_deg": 4.0,
            "scan_union_theta_n": 8,
            "scan_union_phi_n": 12,
            "n_alpha_2d": 31,
            "scan_block_size": 16,
        }
    )
    union_current = compute_scan_union_envelope_3d(p_current, d_current, elem_current, settings_current)
    current_cuts = compute_2d_cuts(p_current, d_current, elem_current, settings_current.__class__(
        **{**settings_current.__dict__, "cut_half_width_deg": 85.0}
    ))
    includes_current = any(
        abs(float(x) - p_current.scan_x_deg) < 1.0e-9 and abs(float(y) - p_current.scan_y_deg) < 1.0e-9
        for x, y in zip(union_current["scanXList_deg"], union_current["scanYList_deg"])
    )
    if not includes_current:
        raise AssertionError("Scan union did not include the current scan center even though it is inside the allowed scan limits.")
    for union_cut, current_cut in zip(union_current["unionCuts"], current_cuts):
        if union_cut.get("envelope_method") != "near_field_sampled_far_field_scan_union":
            raise AssertionError("Scan union fixed cut did not use the near-field sampled union method.")
        current_r = np.asarray(current_cut["r_env_m"], dtype=float)
        union_r = np.asarray(union_cut["r_env_m"], dtype=float)
        finite_current = np.isfinite(current_r)
        if np.any(finite_current) and not np.all(union_r[finite_current] + 1.0e-6 >= current_r[finite_current]):
            raise AssertionError("Near-field scan-union cut is smaller than the current-scan envelope even though current scan is included.")


def scan_union_cache_decomposition_checks() -> None:
    settings0 = get_mode_settings("fast")
    settings = settings0.__class__(
        **{
            **settings0.__dict__,
            "n_alpha_2d": 31,
            "n_range_2d": 24,
            "scan_union_step_deg": 3.0,
            "scan_union_theta_n": 11,
            "scan_union_phi_n": 13,
            "chunk_size": 512,
            "scan_block_size": 12,
        }
    )
    params = BeamParams(
        frequency_ghz=10.0,
        nx=4,
        ny=5,
        dx_m=0.12,
        dy_m=0.10,
        ax_m=0.08,
        ay_m=0.07,
        element_power_w=2.0e4,
        s0_w_cm2=25.0,
        scan_x_deg=2.2,
        scan_y_deg=-1.1,
        calc_mode="fast",
    )
    params, derived = derive_params(params)
    elem = make_element_pattern(params, derived.wavelength_m)
    full = compute_scan_union_envelope_3d(params, derived, elem, settings)
    base = compute_scan_union_base_envelope_3d(params, derived, elem, settings)
    merged = merge_scan_union_with_current_scan(base, params, derived, elem, settings)

    if not bool(merged.get("currentScanOverlayApplied")):
        raise AssertionError("Off-grid current scan did not trigger a current-scan overlay.")
    for key in ("Rsurf", "maxCoef", "Xsurf", "Ysurf", "Zsurf"):
        if not np.allclose(np.asarray(full[key]), np.asarray(merged[key]), rtol=0.0, atol=1.0e-9, equal_nan=True):
            diff = float(np.nanmax(np.abs(np.asarray(full[key]) - np.asarray(merged[key]))))
            raise AssertionError(f"Split scan-union cache changed {key}; max abs diff={diff:g}")
    for idx, (full_cut, merged_cut) in enumerate(zip(full["unionCuts"], merged["unionCuts"])):
        for key in ("r_env_m", "rho_env_m", "x_env_m", "y_env_m", "z_env_m"):
            if not np.allclose(np.asarray(full_cut[key]), np.asarray(merged_cut[key]), rtol=0.0, atol=1.0e-9, equal_nan=True):
                diff = float(np.nanmax(np.abs(np.asarray(full_cut[key]) - np.asarray(merged_cut[key]))))
                raise AssertionError(f"Split scan-union cache changed cut {idx} {key}; max abs diff={diff:g}")


def cache_key_checks() -> None:
    params = BeamParams(nx=4, ny=4, element_power_w=1.0e4)
    settings0 = get_mode_settings("fast")
    settings_a = settings0.__class__(**{**settings0.__dict__, "scan_union_theta_n": 11, "scan_union_phi_n": 13})
    settings_b = settings0.__class__(**{**settings0.__dict__, "scan_union_theta_n": 15, "scan_union_phi_n": 17})
    if base_cache_key(params, settings_a) == base_cache_key(params, settings_b):
        raise AssertionError("Scan-union cache key ignored actual ModeSettings sampling resolution.")

    inactive_layout_with_path = BeamParams(nx=4, ny=4, array_layout="rectangular", array_layout_file="remembered_but_inactive.csv")
    inactive_layout_without_path = BeamParams(nx=4, ny=4, array_layout="rectangular", array_layout_file="")
    if base_cache_key(inactive_layout_with_path, settings_a) != base_cache_key(inactive_layout_without_path, settings_a):
        raise AssertionError("Inactive imported-layout path leaked into the cache key.")

    inactive_pattern_with_path = BeamParams(
        nx=4,
        ny=4,
        use_element_pattern=False,
        element_pattern_file="remembered_but_disabled.csv",
    )
    inactive_pattern_without_path = BeamParams(nx=4, ny=4, use_element_pattern=False, element_pattern_file="")
    if base_cache_key(inactive_pattern_with_path, settings_a) != base_cache_key(inactive_pattern_without_path, settings_a):
        raise AssertionError("Disabled imported element-pattern path leaked into the cache key.")


def custom_sampling_checks() -> None:
    params = BeamParams(
        calc_mode="standard",
        custom_sampling_enabled=True,
        sample_2d_alpha_n=233,
        sample_2d_range_n=77,
        sample_3d_theta_n=41,
        sample_3d_phi_n=83,
        sample_3d_range_n=59,
        sample_uv_n=155,
        sample_scan_union_step_deg=0.75,
        sample_scan_union_theta_n=43,
        sample_scan_union_phi_n=87,
        display_3d_grid_n=240,
    )
    settings = get_effective_mode_settings(params)
    if settings.name != "standard+custom":
        raise AssertionError(f"Custom sampling did not mark settings name: {settings.name!r}")
    expected = {
        "n_alpha_2d": 233,
        "n_range_2d": 77,
        "theta_3d_n": 41,
        "phi_3d_n": 83,
        "n_range_3d": 59,
        "n_uv": 155,
        "scan_union_step_deg": 0.75,
        "scan_union_theta_n": 43,
        "scan_union_phi_n": 87,
    }
    for key, value in expected.items():
        actual = getattr(settings, key)
        if actual != value:
            raise AssertionError(f"Custom sampling {key}={actual!r}, expected {value!r}")
    default_settings = get_effective_mode_settings(BeamParams(calc_mode="standard", custom_sampling_enabled=False))
    if base_cache_key(params, settings) == base_cache_key(BeamParams(calc_mode="standard"), default_settings):
        raise AssertionError("Custom sampling did not change scan-union cache key.")


def shape_model_checks() -> None:
    alias_cases = {
        "矩形排布": "rectangular",
        "矩形单元": "rectangular",
        "圆形": "ellipse",
        "圆形/椭圆单元": "ellipse",
        "菱形排布": "diamond",
        "导入坐标CSV": "custom",
    }
    for alias, expected in alias_cases.items():
        actual = normalize_shape_name(alias)
        if actual != expected:
            raise AssertionError(f"shape alias {alias!r} normalized to {actual!r}, expected {expected!r}")
    if shape_label("ellipse", kind="array") != "圆形/椭圆排布":
        raise AssertionError("shape_label did not return the centralized array label.")
    if shape_label("diamond", kind="element_model") != "菱形解析口径":
        raise AssertionError("shape_label did not return the centralized element-model label.")
    if element_shape_to_mode("矩形单元") != "rect":
        raise AssertionError("Rectangular element did not map to the rectangular aperture mode.")
    if element_shape_to_mode("圆形/椭圆单元") != "ellipse":
        raise AssertionError("Ellipse element did not map to the jinc aperture mode.")
    if element_shape_to_mode("菱形单元") != "diamond":
        raise AssertionError("Diamond element did not map to the diamond aperture mode.")
    overlap_examples = {
        "rectangular": element_overlap_metric("rectangular", 0.05, 0.00, 0.10, 0.10),
        "ellipse": element_overlap_metric("ellipse", 0.05, 0.00, 0.10, 0.10),
        "diamond": element_overlap_metric("diamond", 0.05, 0.00, 0.10, 0.10),
    }
    if not (overlap_examples["rectangular"] < 1.0 and overlap_examples["ellipse"] < 1.0 and overlap_examples["diamond"] < 1.0):
        raise AssertionError(f"Centralized overlap metrics failed for close elements: {overlap_examples}")
    signed_metric = element_overlap_metric("diamond", -0.05, -0.02, 0.10, 0.10)
    if not math.isclose(signed_metric, 0.70, rel_tol=0.0, abs_tol=1.0e-12):
        raise AssertionError(f"Overlap metric should be sign-independent, got {signed_metric}")

    expected_active = {
        "rectangular": 81,
        "ellipse": 69,
        "diamond": 41,
    }
    for shape, expected_count in expected_active.items():
        p = BeamParams(nx=9, ny=9, dx_m=0.1, dy_m=0.1, ax_m=0.1, ay_m=0.1, array_layout=shape, element_shape=shape)
        p, d = derive_params(p)
        pin = make_pin_matrix(p)
        active_count = int(np.count_nonzero(pin))
        if active_count != expected_count:
            raise AssertionError(f"{shape} active element count changed: expected {expected_count}, got {active_count}")
        close(d.total_input_power_w, active_count * p.element_power_w, 1.0e-6, f"{shape} total input power")

        elem = make_element_pattern(p, d.wavelength_m)
        elem_loss = scan_loss_db_for_direction(d.u0, d.v0, d.w0, elem)
        close(d.scan_loss_db, elem_loss, 1.0e-10, f"{shape} derived scan loss matches element model")
        settings0 = get_mode_settings("fast")
        settings = settings0.__class__(
            **{
                **settings0.__dict__,
                "n_alpha_2d": 21,
                "n_range_2d": 14,
                "chunk_size": 256,
            }
        )
        cuts = compute_2d_cuts(p, d, elem, settings)
        for cut in cuts:
            if not np.any(np.isfinite(cut["r_env_m"])):
                raise AssertionError(f"{shape} produced no finite 2D envelope")

    thin = BeamParams(nx=12, ny=2, dx_m=0.45, dy_m=0.43, ax_m=0.20, ay_m=0.04, array_layout="diamond")
    thin, _ = derive_params(thin)
    if int(np.count_nonzero(make_pin_matrix(thin))) <= 0:
        raise AssertionError("Diamond layout fallback produced an empty active array.")

    isotropic = BeamParams(use_element_pattern=False, scan_x_deg=12.0, scan_y_deg=5.0)
    _, isotropic_d = derive_params(isotropic)
    close(isotropic_d.scan_loss_db, 0.0, 1.0e-12, "isotropic scan loss")

    dense = BeamParams(
        frequency_ghz=9.5,
        nx=9,
        ny=9,
        dx_m=0.30,
        dy_m=0.30,
        ax_m=0.30,
        ay_m=0.30,
        element_power_w=1.0e6,
        s0_w_cm2=100.0,
        scan_x_deg=0.0,
        scan_y_deg=0.0,
        calc_mode="fast",
    )
    dense, dense_d = derive_params(dense)
    dense_elem = make_element_pattern(dense, dense_d.wavelength_m)
    dense_settings = settings0.__class__(
        **{
            **settings0.__dict__,
            "n_alpha_2d": 61,
            "n_range_2d": 40,
            "theta_3d_n": 18,
            "phi_3d_n": 24,
            "n_range_3d": 24,
            "chunk_size": 512,
        }
    )
    dense_cuts = compute_2d_cuts(dense, dense_d, dense_elem, dense_settings)
    for cut in dense_cuts:
        finite = np.isfinite(cut["r_env_m"])
        if int(np.count_nonzero(finite)) < dense_settings.n_alpha_2d // 2:
            raise AssertionError("9x9 equal-pitch dense case produced an almost empty 2D envelope.")
        if not float(np.nanmax(cut["r_env_m"])) > dense_d.rff_m:
            raise AssertionError("9x9 equal-pitch dense case did not extend beyond the far-field boundary as expected.")


def shape_combination_matrix_checks() -> None:
    layout_shapes = tuple(shape for shape in CANONICAL_ARRAY_LAYOUTS if shape != "custom")
    element_modes = {"rectangular": "rect", "ellipse": "ellipse", "diamond": "diamond"}
    settings0 = get_mode_settings("fast")
    settings = settings0.__class__(
        **{
            **settings0.__dict__,
            "n_alpha_2d": 31,
            "n_range_2d": 18,
            "theta_3d_n": 13,
            "phi_3d_n": 17,
            "n_range_3d": 18,
            "n_uv": 41,
            "chunk_size": 384,
        }
    )
    for array_layout in layout_shapes:
        for element_shape in CANONICAL_ELEMENT_SHAPES:
            params = BeamParams(
                frequency_ghz=10.0,
                nx=7,
                ny=7,
                dx_m=0.12,
                dy_m=0.12,
                ax_m=0.08,
                ay_m=0.07,
                element_power_w=5.0e4,
                s0_w_cm2=1.0,
                scan_x_deg=6.0,
                scan_y_deg=-4.0,
                calc_mode="fast",
                array_layout=array_layout,
                element_shape=element_shape,
            )
            params, derived = derive_params(params)
            active_count = int(derived.element_x_m.size)
            if active_count <= 0:
                raise AssertionError(f"{array_layout}/{element_shape} produced no active elements.")
            close(
                derived.total_input_power_w,
                active_count * params.element_power_w,
                1.0e-6,
                f"{array_layout}/{element_shape} total input power",
            )

            elem = make_element_pattern(params, derived.wavelength_m)
            expected_mode = element_modes[element_shape]
            if elem.mode != expected_mode:
                raise AssertionError(f"{array_layout}/{element_shape} used element mode {elem.mode!r}, expected {expected_mode!r}.")
            elem_loss = scan_loss_db_for_direction(derived.u0, derived.v0, derived.w0, elem)
            close(
                derived.scan_loss_db,
                elem_loss,
                1.0e-10,
                f"{array_layout}/{element_shape} derived scan loss matches active element model",
            )

            cuts = compute_2d_cuts(params, derived, elem, settings)
            if len(cuts) != 2:
                raise AssertionError(f"{array_layout}/{element_shape} did not produce both fixed 2D cuts.")
            for cut in cuts:
                r = np.asarray(cut["r_env_m"], dtype=float)
                if not np.any(np.isfinite(r)):
                    raise AssertionError(f"{array_layout}/{element_shape} produced an empty {cut['short_name']} envelope.")

            uv = calc_uv_pattern(params, derived, elem, settings.n_uv, settings.pattern_floor_db)
            pdb = np.asarray(uv["pattern_db"], dtype=float)
            if not np.any(np.isfinite(pdb)):
                raise AssertionError(f"{array_layout}/{element_shape} produced an empty u-v pattern.")
            if not math.isclose(float(np.nanmax(pdb)), 0.0, abs_tol=1.0e-9):
                raise AssertionError(f"{array_layout}/{element_shape} u-v pattern is not normalized to 0 dB.")

            current_3d = compute_current_3d_envelope(params, derived, elem, settings)
            r3 = np.asarray(current_3d["r_env_m"], dtype=float)
            if not np.any(np.isfinite(r3)):
                raise AssertionError(f"{array_layout}/{element_shape} produced an empty current 3D envelope.")
            if not float(current_3d["max_range_m"]) > 0.0:
                raise AssertionError(f"{array_layout}/{element_shape} current 3D envelope max range is invalid.")


def imported_element_pattern_checks() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = Path(tmpdir) / "element_pattern.csv"
        u_axis = np.linspace(-0.9, 0.9, 25)
        v_axis = np.linspace(-0.9, 0.9, 25)
        rows = ["u,v,gain_db"]
        for u in u_axis:
            for v in v_axis:
                if u * u + v * v <= 1.0:
                    gain = max(0.02, (1.0 - 0.72 * u * u) * (1.0 - 0.18 * v * v))
                    rows.append(f"{u:.8f},{v:.8f},{10.0 * math.log10(gain):.8f}")
        csv_path.write_text("\n".join(rows), encoding="utf-8")

        p = BeamParams(
            nx=4,
            ny=4,
            dx_m=0.1,
            dy_m=0.1,
            ax_m=0.08,
            ay_m=0.08,
            element_pattern_file=str(csv_path),
            scan_x_deg=18.0,
            scan_y_deg=0.0,
        )
        p, d = derive_params(p)
        elem = make_element_pattern(p, d.wavelength_m)
        if elem.mode != "table" or not elem.source_path:
            raise AssertionError("Imported element pattern did not create a table-backed ElementPattern.")
        broadside = float(element_gain_fast(np.asarray(0.0), np.asarray(0.0), np.asarray(1.0), elem))
        off_axis = float(element_gain_fast(np.asarray(0.7), np.asarray(0.0), np.asarray(math.sqrt(1.0 - 0.7 * 0.7)), elem))
        if not (broadside > off_axis > 0.0):
            raise AssertionError("Imported element pattern interpolation did not preserve the expected u-axis taper.")
        imported_loss = scan_loss_db_for_direction(d.u0, d.v0, d.w0, elem)
        if not imported_loss < 0.0:
            raise AssertionError("Imported element pattern scan loss should be negative off boresight.")
        p_with_elem, d_with_elem, elem_with_elem = derive_params_with_element(p)
        if elem_with_elem.mode != "table":
            raise AssertionError("Element-aware derivation did not preserve imported table element model.")
        helper_loss = scan_loss_db_for_direction(d_with_elem.u0, d_with_elem.v0, d_with_elem.w0, elem_with_elem)
        close(d_with_elem.scan_loss_db, helper_loss, 1.0e-12, "element-aware imported scan loss")
        if p_with_elem.element_pattern_file != p.element_pattern_file:
            raise AssertionError("Element-aware derivation changed the imported element pattern path.")

        settings0 = get_mode_settings("fast")
        settings = settings0.__class__(**{**settings0.__dict__, "n_uv": 61})
        uv = calc_uv_pattern(p, d, elem, settings.n_uv, settings.pattern_floor_db)
        if not np.any(np.isfinite(uv["pattern_db"])):
            raise AssertionError("Imported element pattern produced an empty u-v pattern.")

        main_settings = settings0.__class__(
            **{
                **settings0.__dict__,
                "n_alpha_2d": 51,
                "n_range_2d": 36,
                "n_far_plot_2d": 16,
                "theta_3d_n": 12,
                "phi_3d_n": 24,
                "n_range_3d": 24,
                "n_uv": 51,
                "chunk_size": 2500,
            }
        )
        isotropic_params = BeamParams.from_dict(p_with_elem.to_dict())
        isotropic_params.use_element_pattern = False
        isotropic_params.element_pattern_file = ""
        isotropic_params, isotropic_derived, isotropic_elem = derive_params_with_element(isotropic_params)
        imported_uv = calc_uv_pattern(p_with_elem, d_with_elem, elem_with_elem, main_settings.n_uv, main_settings.pattern_floor_db)
        isotropic_uv = calc_uv_pattern(
            isotropic_params,
            isotropic_derived,
            isotropic_elem,
            main_settings.n_uv,
            main_settings.pattern_floor_db,
        )
        uv_delta = np.nanmax(np.abs(np.asarray(imported_uv["pattern_db"]) - np.asarray(isotropic_uv["pattern_db"])))
        if not float(uv_delta) > 0.05:
            raise AssertionError("Imported far-field element pattern did not change the u-v pattern calculation.")

        imported_cuts = compute_2d_cuts(p_with_elem, d_with_elem, elem_with_elem, main_settings)
        isotropic_cuts = compute_2d_cuts(isotropic_params, isotropic_derived, isotropic_elem, main_settings)
        cut_delta = np.nanmax(
            np.abs(np.asarray(imported_cuts[0]["r_env_m"], dtype=float) - np.asarray(isotropic_cuts[0]["r_env_m"], dtype=float))
        )
        reference_range = max(float(np.nanmax(np.asarray(isotropic_cuts[0]["r_env_m"], dtype=float))), 1.0)
        if not float(cut_delta) > 1.0e-3 * reference_range:
            raise AssertionError("Imported far-field element pattern did not change the 2D envelope calculation.")

        hfss_path = Path(tmpdir) / "hfss_style_pattern.csv"
        rows = ["Theta [deg],Phi [deg],dB(GainTotal)"]
        for theta in np.linspace(0.0, 80.0, 17):
            for phi in np.linspace(-180.0, 180.0, 37):
                gain = max(0.02, math.cos(math.radians(theta)) ** 2 * (1.0 - 0.08 * math.cos(math.radians(phi)) ** 2))
                rows.append(f"{theta:.8f},{phi:.8f},{10.0 * math.log10(gain):.8f}")
        hfss_path.write_text("\n".join(rows), encoding="utf-8")
        _assert_imported_pattern_loads(hfss_path, "HFSS dB(GainTotal) style")

        radian_path = Path(tmpdir) / "theta_phi_radian_pattern.csv"
        rows = ["Theta [rad],Phi [rad],dB(GainTotal)"]
        for theta in np.linspace(0.0, math.radians(80.0), 17):
            for phi in np.linspace(-math.pi, math.pi, 37):
                gain = max(0.02, math.cos(theta) ** 2 * (1.0 - 0.08 * math.cos(phi) ** 2))
                rows.append(f"{theta:.10f},{phi:.10f},{10.0 * math.log10(gain):.8f}")
        radian_path.write_text("\n".join(rows), encoding="utf-8")
        _assert_imported_pattern_loads(radian_path, "Theta/Phi radians style")
        radian_params = BeamParams(nx=3, ny=3, element_pattern_file=str(radian_path))
        _, radian_derived = derive_params(radian_params)
        radian_elem = make_element_pattern(radian_params, radian_derived.wavelength_m)
        close(float(radian_elem.table_theta_max_deg), 80.0, 0.2, "radian theta_max")

        semicolon_path = Path(tmpdir) / "semicolon_linear_pattern.csv"
        rows = ["Theta [deg];Phi [deg];Abs(GainTotal)"]
        for theta in np.linspace(0.0, 75.0, 16):
            for phi in np.linspace(-180.0, 180.0, 25):
                gain = max(0.03, math.cos(math.radians(theta)) ** 1.5 * (1.0 - 0.06 * math.sin(math.radians(phi)) ** 2))
                rows.append(f"{theta:.8f};{phi:.8f};{gain:.8f}")
        semicolon_path.write_text("\n".join(rows), encoding="utf-8")
        _assert_imported_pattern_loads(semicolon_path, "semicolon Abs(GainTotal) style")

        complex_path = Path(tmpdir) / "complex_field_pattern.csv"
        rows = ["u,v,field_real,field_imag"]
        for u in np.linspace(-0.8, 0.8, 17):
            for v in np.linspace(-0.8, 0.8, 17):
                if u * u + v * v <= 1.0:
                    amp = max(0.05, 1.0 - 0.35 * u * u - 0.12 * v * v)
                    phase = math.radians(45.0 + 20.0 * u)
                    rows.append(f"{u:.8f},{v:.8f},{amp * math.cos(phase):.8f},{amp * math.sin(phase):.8f}")
        complex_path.write_text("\n".join(rows), encoding="utf-8")
        p_complex = BeamParams(nx=3, ny=3, element_pattern_file=str(complex_path), scan_x_deg=8.0, scan_y_deg=0.0)
        p_complex, d_complex = derive_params(p_complex)
        elem_complex = make_element_pattern(p_complex, d_complex.wavelength_m)
        response = complex(element_response_fast(np.asarray(0.0), np.asarray(0.0), np.asarray(1.0), elem_complex))
        if not (abs(response.imag) > 0.2 * abs(response.real) > 0.0):
            raise AssertionError("Complex imported element pattern did not preserve field phase.")
        uv_complex = calc_uv_pattern(p_complex, d_complex, elem_complex, 61, -40.0)
        if not np.any(np.isfinite(uv_complex["pattern_db"])):
            raise AssertionError("Complex imported element pattern produced an empty u-v pattern.")

        vector_path = Path(tmpdir) / "vector_component_pattern.csv"
        rows = ["theta_deg,phi_deg,Real(ETheta),Imag(ETheta),Real(EPhi),Imag(EPhi)"]
        for theta in np.linspace(0.0, 80.0, 17):
            for phi in np.linspace(-180.0, 180.0, 37):
                amp_theta = max(0.05, math.cos(math.radians(theta)) ** 1.2)
                amp_phi = 0.45 * max(0.05, math.cos(math.radians(theta)) ** 1.1) * (1.0 + 0.12 * math.cos(math.radians(phi)))
                phase_theta = math.radians(20.0 * math.sin(math.radians(phi)))
                phase_phi = math.radians(-35.0 + 12.0 * math.cos(math.radians(theta)))
                etheta = amp_theta * complex(math.cos(phase_theta), math.sin(phase_theta))
                ephi = amp_phi * complex(math.cos(phase_phi), math.sin(phase_phi))
                rows.append(f"{theta:.8f},{phi:.8f},{etheta.real:.8f},{etheta.imag:.8f},{ephi.real:.8f},{ephi.imag:.8f}")
        vector_path.write_text("\n".join(rows), encoding="utf-8")
        p_vector = BeamParams(nx=3, ny=3, element_pattern_file=str(vector_path), scan_x_deg=6.0, scan_y_deg=2.0)
        p_vector, d_vector = derive_params(p_vector)
        elem_vector = make_element_pattern(p_vector, d_vector.wavelength_m)
        if not elem_vector.table_has_vector_components:
            raise AssertionError("Imported ETheta/EPhi component pattern was not marked as vector.")
        rt, rp = element_response_components_fast(np.asarray(0.0), np.asarray(0.0), np.asarray(1.0), elem_vector)
        gain_components = float(np.abs(rt) ** 2 + np.abs(rp) ** 2)
        gain_fast = float(element_gain_fast(np.asarray(0.0), np.asarray(0.0), np.asarray(1.0), elem_vector))
        close(gain_fast, gain_components, 1.0e-9, "vector ETheta/EPhi element gain")
        uv_vector = calc_uv_pattern(p_vector, d_vector, elem_vector, 61, -40.0)
        if not np.any(np.isfinite(uv_vector["pattern_db"])):
            raise AssertionError("Vector imported element pattern produced an empty u-v pattern.")

        ffd_path = Path(tmpdir) / "component_pattern.ffd"
        theta_values = np.linspace(0.0, 80.0, 17)
        phi_values = np.linspace(-180.0, 180.0, 37)
        rows = [
            "0 80 17",
            "-180 180 37",
            "1",
            "10e9",
        ]
        for theta in theta_values:
            for phi in phi_values:
                amp_theta = max(0.05, math.cos(math.radians(theta)) ** 1.18)
                amp_phi = 0.42 * max(0.05, math.cos(math.radians(theta)) ** 1.08) * (1.0 + 0.10 * math.sin(math.radians(phi)))
                phase_theta = math.radians(18.0 * math.sin(math.radians(phi)))
                phase_phi = math.radians(-30.0 + 11.0 * math.cos(math.radians(theta)))
                etheta = amp_theta * complex(math.cos(phase_theta), math.sin(phase_theta))
                ephi = amp_phi * complex(math.cos(phase_phi), math.sin(phase_phi))
                rows.append(f"{etheta.real:.10e} {etheta.imag:.10e} {ephi.real:.10e} {ephi.imag:.10e}")
        ffd_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
        p_ffd = BeamParams(nx=3, ny=3, element_pattern_file=str(ffd_path), frequency_ghz=9.9, scan_x_deg=5.0)
        p_ffd, d_ffd = derive_params(p_ffd)
        elem_ffd = make_element_pattern(p_ffd, d_ffd.wavelength_m)
        if not elem_ffd.table_has_vector_components or not elem_ffd.table_has_phase:
            raise AssertionError("FFD imported far-field pattern was not marked as vector phase data.")
        if not math.isclose(float(elem_ffd.table_selected_frequency_ghz or -1.0), 10.0, abs_tol=1.0e-12):
            raise AssertionError(f"FFD frequency selection failed: {elem_ffd.table_selected_frequency_ghz}")
        uv_ffd = calc_uv_pattern(p_ffd, d_ffd, elem_ffd, 61, -40.0)
        if not np.any(np.isfinite(uv_ffd["pattern_db"])):
            raise AssertionError("FFD imported element pattern produced an empty u-v pattern.")

        vector_basis_path = Path(tmpdir) / "vector_basis_global_x_pattern.csv"
        rows = ["Theta [deg],Phi [deg],Real(ETheta),Imag(ETheta),Real(EPhi),Imag(EPhi)"]
        for theta in np.linspace(0.0, 80.0, 17):
            for phi in np.linspace(-180.0, 180.0, 37):
                th = math.radians(theta)
                ph = math.radians(phi)
                etheta = math.cos(th) * math.cos(ph)
                ephi = -math.sin(ph)
                rows.append(f"{theta:.8f},{phi:.8f},{etheta:.8f},0.0,{ephi:.8f},0.0")
        vector_basis_path.write_text("\n".join(rows), encoding="utf-8")
        p_vector_basis = BeamParams(nx=3, ny=3, element_pattern_file=str(vector_basis_path))
        p_vector_basis, d_vector_basis = derive_params(p_vector_basis)
        elem_vector_basis = make_element_pattern(p_vector_basis, d_vector_basis.wavelength_m)
        if (
            not elem_vector_basis.table_has_vector_components
            or elem_vector_basis.table_field_x is None
            or elem_vector_basis.table_field_y is None
            or elem_vector_basis.table_field_z is None
        ):
            raise AssertionError("Imported vector basis regression pattern did not build Cartesian component grids.")
        center_u = int(np.argmin(np.abs(elem_vector_basis.table_u_axis)))
        center_v = int(np.argmin(np.abs(elem_vector_basis.table_v_axis)))
        broadside_ex = complex(elem_vector_basis.table_field_x[center_v, center_u])
        broadside_ey = complex(elem_vector_basis.table_field_y[center_v, center_u])
        broadside_ez = complex(elem_vector_basis.table_field_z[center_v, center_u])
        if abs(broadside_ex) < 0.85 or abs(broadside_ey) > 0.12 or abs(broadside_ez) > 0.12:
            raise AssertionError(
                "Vector ETheta/EPhi duplicate-direction merge is basis-dependent; "
                f"broadside Ex/Ey/Ez={broadside_ex}/{broadside_ey}/{broadside_ez}"
            )
        rt_basis, rp_basis = element_response_components_fast(
            np.asarray(0.0),
            np.asarray(0.0),
            np.asarray(1.0),
            elem_vector_basis,
        )
        if not (abs(complex(rt_basis)) > 0.85 * math.sqrt(elem_vector_basis.gain_norm) and abs(complex(rp_basis)) < 0.12):
            raise AssertionError("Vector Cartesian-basis interpolation did not preserve broadside x polarization.")

        vector_mag_phase_path = Path(tmpdir) / "vector_component_mag_phase_pattern.csv"
        rows = ["Theta [deg],Phi [deg],Abs(Theta),Phase(Theta),Abs(Phi),Phase(Phi)"]
        for theta in np.linspace(0.0, 80.0, 17):
            for phi in np.linspace(-180.0, 180.0, 37):
                amp_theta = max(0.04, math.cos(math.radians(theta)) ** 1.15)
                amp_phi = 0.36 * max(0.04, math.cos(math.radians(theta)) ** 1.05) * (
                    1.0 + 0.10 * math.sin(math.radians(phi))
                )
                phase_theta = 15.0 * math.sin(math.radians(phi))
                phase_phi = -28.0 + 10.0 * math.cos(math.radians(theta))
                rows.append(f"{theta:.8f},{phi:.8f},{amp_theta:.8f},{phase_theta:.8f},{amp_phi:.8f},{phase_phi:.8f}")
        vector_mag_phase_path.write_text("\n".join(rows), encoding="utf-8")
        p_vector_mag_phase = BeamParams(
            nx=3,
            ny=3,
            element_pattern_file=str(vector_mag_phase_path),
            scan_x_deg=5.0,
            scan_y_deg=3.0,
        )
        p_vector_mag_phase, d_vector_mag_phase = derive_params(p_vector_mag_phase)
        elem_vector_mag_phase = make_element_pattern(p_vector_mag_phase, d_vector_mag_phase.wavelength_m)
        if not elem_vector_mag_phase.table_has_vector_components or not elem_vector_mag_phase.table_has_phase:
            raise AssertionError("Imported Abs(Theta)/Phase(Theta) component pattern was not marked as vector phase data.")
        rt2, rp2 = element_response_components_fast(
            np.asarray(0.0),
            np.asarray(0.0),
            np.asarray(1.0),
            elem_vector_mag_phase,
        )
        if not (np.isfinite(rt2) and np.isfinite(rp2) and abs(complex(rt2)) > 0.0):
            raise AssertionError("Imported Abs/Phase ETheta/EPhi component pattern produced invalid broadside response.")
        uv_vector_mag_phase = calc_uv_pattern(p_vector_mag_phase, d_vector_mag_phase, elem_vector_mag_phase, 61, -40.0)
        if not np.any(np.isfinite(uv_vector_mag_phase["pattern_db"])):
            raise AssertionError("Abs/Phase vector imported element pattern produced an empty u-v pattern.")

        metadata_vector_path = Path(tmpdir) / "solver_metadata_vector_pattern.csv"
        rows = [
            "# Far field export",
            "Project,BeamCoverage",
            "Frequency,10 GHz",
            "Setup,LastAdaptive",
            "Theta [deg],Phi [deg],Abs(Theta),Phase(Theta),Abs(Phi),Phase(Phi)",
        ]
        for theta in np.linspace(0.0, 80.0, 17):
            for phi in np.linspace(-180.0, 180.0, 37):
                amp_theta = max(0.04, math.cos(math.radians(theta)) ** 1.12)
                amp_phi = 0.33 * max(0.04, math.cos(math.radians(theta)) ** 1.04) * (
                    1.0 + 0.08 * math.cos(math.radians(phi))
                )
                phase_theta = 13.0 * math.sin(math.radians(phi))
                phase_phi = -22.0 + 9.0 * math.cos(math.radians(theta))
                rows.append(f"{theta:.8f},{phi:.8f},{amp_theta:.8f},{phase_theta:.8f},{amp_phi:.8f},{phase_phi:.8f}")
        metadata_vector_path.write_text("\n".join(rows), encoding="utf-8")
        p_metadata_vector = BeamParams(nx=3, ny=3, element_pattern_file=str(metadata_vector_path), scan_x_deg=4.0)
        p_metadata_vector, d_metadata_vector = derive_params(p_metadata_vector)
        elem_metadata_vector = make_element_pattern(p_metadata_vector, d_metadata_vector.wavelength_m)
        if not elem_metadata_vector.table_has_vector_components or not elem_metadata_vector.table_has_phase:
            raise AssertionError("Solver metadata-prefixed Abs/Phase vector CSV was not parsed as vector phase data.")
        close(float(elem_metadata_vector.table_theta_max_deg), 80.0, 0.2, "metadata-prefixed theta_max")

        multifrequency_path = Path(tmpdir) / "multifrequency_vector_pattern.csv"
        rows = ["Frequency [GHz],Theta [deg],Phi [deg],Abs(Theta),Phase(Theta),Abs(Phi),Phase(Phi)"]
        for frequency_ghz, scale in ((9.0, 0.70), (11.0, 1.20)):
            for theta in np.linspace(0.0, 80.0, 17):
                for phi in np.linspace(-180.0, 180.0, 37):
                    amp_theta = scale * max(0.04, math.cos(math.radians(theta)) ** 1.11)
                    amp_phi = 0.32 * scale * max(0.04, math.cos(math.radians(theta)) ** 1.03)
                    phase_theta = 10.0 * math.sin(math.radians(phi))
                    phase_phi = -18.0 + 8.0 * math.cos(math.radians(theta))
                    rows.append(f"{frequency_ghz:.8f},{theta:.8f},{phi:.8f},{amp_theta:.8f},{phase_theta:.8f},{amp_phi:.8f},{phase_phi:.8f}")
        multifrequency_path.write_text("\n".join(rows), encoding="utf-8")
        p_multi_low = BeamParams(frequency_ghz=9.1, nx=3, ny=3, element_pattern_file=str(multifrequency_path))
        p_multi_low, d_multi_low = derive_params(p_multi_low)
        elem_multi_low = make_element_pattern(p_multi_low, d_multi_low.wavelength_m)
        p_multi_high = BeamParams(frequency_ghz=10.9, nx=3, ny=3, element_pattern_file=str(multifrequency_path))
        p_multi_high, d_multi_high = derive_params(p_multi_high)
        elem_multi_high = make_element_pattern(p_multi_high, d_multi_high.wavelength_m)
        if not math.isclose(float(elem_multi_low.table_selected_frequency_ghz or -1.0), 9.0, abs_tol=1.0e-12):
            raise AssertionError("Multi-frequency element pattern did not select the nearest 9 GHz slice.")
        if not math.isclose(float(elem_multi_high.table_selected_frequency_ghz or -1.0), 11.0, abs_tol=1.0e-12):
            raise AssertionError("Multi-frequency element pattern did not select the nearest 11 GHz slice.")
        expected_unique_theta_phi_count = 1 + 16 * 36
        if elem_multi_low.table_point_count != expected_unique_theta_phi_count or elem_multi_high.table_point_count != expected_unique_theta_phi_count:
            raise AssertionError("Multi-frequency element pattern mixed frequency slices instead of filtering one slice.")
        if math.isclose(float(elem_multi_low.gain_norm), float(elem_multi_high.gain_norm), rel_tol=1.0e-6):
            raise AssertionError("Multi-frequency element pattern cache reused the wrong frequency slice.")

        unit_cases = [
            ("freq_hz_numeric.csv", "Freq", (9.0e9, 11.0e9), 10.8, 11.0),
            ("freq_mhz_numeric.csv", "Freq", (9000.0, 11000.0), 10.8, 11.0),
            ("freq_low_mhz_numeric.csv", "Freq", (433.0, 915.0), 0.9, 0.915),
            ("freq_string_units.csv", "Freq", ("9 GHz", "11000 MHz"), 10.8, 11.0),
        ]
        for filename, frequency_header, frequencies, target_frequency, expected_selected in unit_cases:
            unit_path = Path(tmpdir) / filename
            rows = [f"{frequency_header},Theta [deg],Phi [deg],Abs(Theta),Phase(Theta),Abs(Phi),Phase(Phi)"]
            for freq_index, frequency in enumerate(frequencies):
                scale = 0.75 + 0.30 * freq_index
                for theta in np.linspace(0.0, 80.0, 17):
                    for phi in np.linspace(-180.0, 180.0, 37):
                        amp_theta = scale * max(0.04, math.cos(math.radians(theta)) ** 1.09)
                        amp_phi = 0.30 * scale * max(0.04, math.cos(math.radians(theta)) ** 1.01)
                        rows.append(f"{frequency},{theta:.8f},{phi:.8f},{amp_theta:.8f},0.0,{amp_phi:.8f},0.0")
            unit_path.write_text("\n".join(rows), encoding="utf-8")
            p_unit = BeamParams(frequency_ghz=target_frequency, nx=3, ny=3, element_pattern_file=str(unit_path))
            p_unit, d_unit = derive_params(p_unit)
            elem_unit = make_element_pattern(p_unit, d_unit.wavelength_m)
            if not math.isclose(float(elem_unit.table_selected_frequency_ghz or -1.0), expected_selected, abs_tol=1.0e-9):
                raise AssertionError(f"Frequency-unit inference failed for {filename}: selected {elem_unit.table_selected_frequency_ghz}")
            if elem_unit.table_point_count != expected_unique_theta_phi_count:
                raise AssertionError(f"Frequency-unit inference mixed slices for {filename}.")

    try:
        derive_params(BeamParams(element_pattern_file=str(Path(tmpdir) / "missing.csv")))
    except ValueError as exc:
        if "element pattern file does not exist" not in str(exc):
            raise AssertionError(f"Unexpected missing imported pattern error: {exc}") from exc
    else:
        raise AssertionError("Missing imported element pattern file was not rejected.")


def imported_near_field_checks() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        near_path = tmp / "element_near_field.csv"
        rows = [
            "# Solver export metadata",
            "Project,BeamCoverageNearField",
            "Frequency [GHz],x_m,y_m,z_m,Real(Ex),Imag(Ex),Real(Ey),Imag(Ey),Real(Ez),Imag(Ez)",
        ]
        for frequency, scale in ((9.0, 0.75), (11.0, 1.25)):
            for z in (0.04, 0.08):
                for y in (-0.02, 0.02):
                    for x in (-0.03, 0.0, 0.03):
                        phase = math.radians(30.0 * x - 18.0 * y + 2.0 * frequency)
                        ex = scale * (1.0 + x) * np.exp(1j * phase)
                        ey = 0.30 * scale * np.exp(1j * (phase - 0.4))
                        ez = 0.05 * scale * np.exp(1j * (phase + 0.2))
                        rows.append(
                            f"{frequency:.8g},{x:.8g},{y:.8g},{z:.8g},"
                            f"{ex.real:.12g},{ex.imag:.12g},{ey.real:.12g},{ey.imag:.12g},{ez.real:.12g},{ez.imag:.12g}"
                        )
        near_path.write_text("\n".join(rows), encoding="utf-8")
        table = load_imported_element_near_field(near_path, 10.8)
        if table.point_count != 12:
            raise AssertionError(f"Near-field frequency filtering mixed slices: point_count={table.point_count}")
        close(float(table.selected_frequency_ghz or -1.0), 11.0, 1.0e-12, "near-field selected frequency")
        if not table.has_phase or not table.has_vector_components:
            raise AssertionError("Near-field complex vector data was not detected.")
        summary = near_field_summary(table)
        if not (summary["x_span_m"] > 0.0 and summary["y_span_m"] > 0.0 and summary["z_max_m"] > summary["z_min_m"]):
            raise AssertionError(f"Near-field summary did not expose spatial extents: {summary}")

        mm_path = tmp / "element_near_field_mm_power.csv"
        rows = ["x_mm,y_mm,z_mm,S [W/cm^2]"]
        for z in (20.0, 40.0):
            for y in (-5.0, 5.0):
                for x in (-5.0, 5.0):
                    rows.append(f"{x},{y},{z},0.002")
        mm_path.write_text("\n".join(rows), encoding="utf-8")
        scalar = load_imported_element_near_field(mm_path)
        if scalar.has_vector_components or not scalar.has_power_density:
            raise AssertionError("Scalar near-field power-density CSV was not classified correctly.")
        close(float(np.nanmax(scalar.z_m)), 0.04, 1.0e-12, "near-field z mm conversion")
        if scalar.s_w_m2 is None or not math.isclose(float(scalar.s_w_m2[0]), 20.0, rel_tol=0.0, abs_tol=1.0e-12):
            raise AssertionError("Near-field S W/cm2 was not converted to W/m2.")

        template_path = tmp / "near_field_template.csv"
        export_element_near_field_vector_template(template_path)
        template = load_imported_element_near_field(template_path, 10.0)
        if not template.has_vector_components or template.point_count <= 0:
            raise AssertionError("Exported near-field template could not be imported.")
        projected_pattern_path = tmp / "projected_far_field.csv"
        projection_info = export_near_field_projected_far_field_pattern(
            template_path,
            projected_pattern_path,
            frequency_ghz=10.0,
            theta_deg=np.linspace(0.0, 80.0, 9),
            phi_deg=np.linspace(-180.0, 180.0, 19),
        )
        if int(projection_info["output_rows"]) != 171:
            raise AssertionError(f"Unexpected near-field projection row count: {projection_info}")
        projected_params = BeamParams(nx=3, ny=3, element_pattern_file=str(projected_pattern_path))
        projected_params, projected_derived, projected_elem = derive_params_with_element(projected_params)
        if projected_elem.mode != "table" or not projected_elem.table_has_vector_components:
            raise AssertionError("Projected near-field far-field CSV did not load as a vector table element pattern.")
        projected_uv = calc_uv_pattern(projected_params, projected_derived, projected_elem, 31, -40.0)
        if not np.any(np.isfinite(projected_uv["pattern_db"])):
            raise AssertionError("Projected near-field far-field CSV produced an empty u-v pattern.")

        params = BeamParams(element_near_field_file=str(near_path))
        restored = BeamParams.from_dict(params.to_dict())
        if restored.element_near_field_file != str(near_path):
            raise AssertionError("BeamParams did not preserve element_near_field_file.")

        bad_path = tmp / "bad_near_field.csv"
        bad_path.write_text("x_m,y_m,Real(Ex),Imag(Ex)\n0,0,1,0\n", encoding="utf-8")
        try:
            load_imported_element_near_field(bad_path)
        except ValueError as exc:
            if "x/y/z" not in str(exc):
                raise AssertionError(f"Unexpected near-field validation error: {exc}") from exc
        else:
            raise AssertionError("Near-field CSV missing z coordinate was not rejected.")


def imported_array_layout_checks() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        layout_path = Path(tmpdir) / "array_layout.csv"
        rows = [
            "x_m,y_m,power_w,phase_deg,enabled",
            "-0.18,-0.10,1000,0,1",
            "0.00,-0.12,2000,10,1",
            "0.16,-0.08,1500,-20,1",
            "-0.08,0.08,1200,30,1",
            "0.12,0.10,1800,-15,1",
            "0.60,0.60,9999,0,0",
        ]
        layout_path.write_text("\n".join(rows), encoding="utf-8")
        p = BeamParams(
            frequency_ghz=10.0,
            nx=2,
            ny=2,
            dx_m=0.05,
            dy_m=0.05,
            ax_m=0.12,
            ay_m=0.08,
            array_layout="custom",
            array_layout_file=str(layout_path),
            element_power_w=777.0,
            scan_x_deg=7.0,
            scan_y_deg=3.0,
            s0_w_cm2=0.02,
            calc_mode="fast",
        )
        p, d = derive_params(p)
        if not d.imported_array_layout:
            raise AssertionError("Imported array layout was not marked on DerivedParams.")
        if d.element_x_m.size != 5:
            raise AssertionError(f"Imported array layout active element count changed: {d.element_x_m.size}")
        close(d.total_input_power_w, 7500.0, 1.0e-9, "imported array total power")
        if not math.isclose(float(d.element_phase_offset_rad[1]), math.radians(10.0), rel_tol=0.0, abs_tol=1.0e-12):
            raise AssertionError("Imported array phase_deg was not converted to radians.")
        if not (d.dx_aperture_m > 0.3 and d.dy_aperture_m > 0.2):
            raise AssertionError("Imported array aperture extents were not derived from element coordinates.")

        elem = make_element_pattern(p, d.wavelength_m)
        settings0 = get_mode_settings("fast")
        settings = settings0.__class__(
            **{
                **settings0.__dict__,
                "n_alpha_2d": 41,
                "n_range_2d": 24,
                "theta_3d_n": 12,
                "phi_3d_n": 18,
                "n_range_3d": 16,
                "n_uv": 61,
                "scan_union_step_deg": 6.0,
                "scan_union_theta_n": 12,
                "scan_union_phi_n": 18,
                "chunk_size": 512,
                "scan_block_size": 8,
            }
        )
        cuts = compute_2d_cuts(p, d, elem, settings)
        if not all(np.any(np.isfinite(cut["r_env_m"])) for cut in cuts):
            raise AssertionError("Imported array layout produced an empty 2D envelope.")
        uv = calc_uv_pattern(p, d, elem, settings.n_uv, settings.pattern_floor_db)
        if not np.any(np.isfinite(uv["pattern_db"])):
            raise AssertionError("Imported array layout produced an empty u-v pattern.")
        union = compute_scan_union_envelope_3d(p, d, elem, settings)
        if not np.nanmax(union["Rsurf"]) > 0.0:
            raise AssertionError("Imported array layout produced an empty scan union.")

    try:
        derive_params(BeamParams(array_layout="custom", array_layout_file=str(Path(tmpdir) / "missing_array.csv")))
    except ValueError as exc:
        if "array layout file does not exist" not in str(exc):
            raise AssertionError(f"Unexpected missing imported array layout error: {exc}") from exc
    else:
        raise AssertionError("Missing imported array layout file was not rejected.")


def _assert_imported_pattern_loads(path: Path, label: str) -> None:
    p = BeamParams(
        nx=3,
        ny=3,
        dx_m=0.1,
        dy_m=0.1,
        ax_m=0.08,
        ay_m=0.08,
        element_pattern_file=str(path),
        scan_x_deg=10.0,
        scan_y_deg=3.0,
    )
    p, d = derive_params(p)
    elem = make_element_pattern(p, d.wavelength_m)
    if elem.mode != "table":
        raise AssertionError(f"{label} did not load as table pattern")
    broadside = float(element_gain_fast(np.asarray(0.0), np.asarray(0.0), np.asarray(1.0), elem))
    off_axis = float(element_gain_fast(np.asarray(0.55), np.asarray(0.0), np.asarray(math.sqrt(1.0 - 0.55 * 0.55)), elem))
    if not (broadside > 0.0 and off_axis > 0.0):
        raise AssertionError(f"{label} produced non-positive interpolated gains")


def invalid_geometry_checks() -> None:
    bad_cases = [
        BeamParams(nx=2, ny=2, dx_m=0.1, dy_m=0.1, ax_m=0.2, ay_m=0.1),
        BeamParams(nx=2, ny=2, dx_m=0.1, dy_m=0.1, ax_m=0.1, ay_m=0.2),
        BeamParams(
            nx=2,
            ny=2,
            dx_m=0.1,
            dy_m=0.1,
            ax_m=0.2,
            ay_m=0.1,
            array_layout="ellipse",
            element_shape="ellipse",
        ),
        BeamParams(
            nx=2,
            ny=2,
            dx_m=0.1,
            dy_m=0.1,
            ax_m=0.2,
            ay_m=0.1,
            array_layout="diamond",
            element_shape="diamond",
        ),
    ]
    for params in bad_cases:
        try:
            derive_params(params)
        except ValueError as exc:
            message = str(exc)
            if "参数无效" not in message or "口径重叠" not in message:
                raise AssertionError(f"Unexpected geometry error text: {exc}") from exc
        else:
            raise AssertionError("Invalid overlapping element geometry was not rejected.")

    single_axis_cases = [
        BeamParams(nx=1, ny=2, dx_m=0.1, dy_m=0.1, ax_m=0.2, ay_m=0.1),
        BeamParams(nx=2, ny=1, dx_m=0.1, dy_m=0.1, ax_m=0.1, ay_m=0.2),
    ]
    for params in single_axis_cases:
        _, derived = derive_params(params)
        if derived.element_x_m.size != params.nx * params.ny:
            raise AssertionError("Single-row/single-column geometry was incorrectly rejected or clipped.")

    clipped_layout_case = BeamParams(
        nx=3,
        ny=2,
        dx_m=0.1,
        dy_m=0.5,
        ax_m=0.15,
        ay_m=0.05,
        array_layout="ellipse",
        element_shape="rectangular",
    )
    _, clipped = derive_params(clipped_layout_case)
    if clipped.element_x_m.size != 2:
        raise AssertionError("Generated clipped layout did not preserve the expected active column.")

    with tempfile.TemporaryDirectory() as tmpdir:
        overlap_path = Path(tmpdir) / "overlap_array.csv"
        overlap_path.write_text("x_m,y_m,power_w\n0,0,1000\n0.08,0,1000\n", encoding="utf-8")
        for shape in ("rectangular", "ellipse", "diamond"):
            try:
                derive_params(
                    BeamParams(
                        array_layout="custom",
                        array_layout_file=str(overlap_path),
                        element_shape=shape,
                        ax_m=0.1,
                        ay_m=0.04,
                    )
                )
            except ValueError as exc:
                if "口径重叠" not in str(exc):
                    raise AssertionError(f"Unexpected imported overlap error text for {shape}: {exc}") from exc
            else:
                raise AssertionError(f"Imported overlapping {shape} element coordinates were not rejected.")

        touch_path = Path(tmpdir) / "touch_array.csv"
        touch_path.write_text("x_m,y_m,power_w\n0,0,1000\n0.1,0,1000\n", encoding="utf-8")
        derive_params(
            BeamParams(
                array_layout="custom",
                array_layout_file=str(touch_path),
                element_shape="rectangular",
                ax_m=0.1,
                ay_m=0.04,
            )
        )


def invalid_parameter_checks() -> None:
    bad_cases = [
        BeamParams(frequency_ghz=0.0),
        BeamParams(nx=0),
        BeamParams(dx_m=0.0),
        BeamParams(efficiency=1.2),
        BeamParams(element_power_w=-1.0),
        BeamParams(s0_w_cm2=0.0),
        BeamParams(scan_x_deg=90.0, scan_y_deg=0.0),
        BeamParams(scan_x_deg=70.0, scan_y_deg=70.0),
        BeamParams(scan_limit_mode="manual", manual_scan_limit_x_deg=120.0),
        BeamParams(scan_limit_mode="manual", manual_scan_limit_y_deg=-1.0),
    ]
    for params in bad_cases:
        try:
            derive_params(params)
        except ValueError as exc:
            if "参数无效" not in str(exc):
                raise AssertionError(f"Unexpected parameter error text: {exc}") from exc
        else:
            raise AssertionError(f"Invalid parameter set was not rejected: {params.to_dict()}")


def reported_bug_regression_checks() -> None:
    settings0 = get_mode_settings("fast")
    settings = settings0.__class__(
        **{
            **settings0.__dict__,
            "n_alpha_2d": 121,
            "n_range_2d": 64,
            "scan_union_step_deg": 6.0,
            "scan_union_theta_n": 13,
            "scan_union_phi_n": 17,
            "scan_block_size": 12,
        }
    )
    params = BeamParams(
        frequency_ghz=9.5,
        nx=9,
        ny=9,
        dx_m=0.3,
        dy_m=0.3,
        ax_m=0.3,
        ay_m=0.3,
        element_power_w=1.0e6,
        s0_w_cm2=100.0,
        scan_x_deg=0.0,
        scan_y_deg=0.0,
        use_element_pattern=True,
        calc_mode="fast",
    )
    params, derived = derive_params(params)
    elem = make_element_pattern(params, derived.wavelength_m)
    cuts = compute_2d_cuts(params, derived, elem, settings)
    if len(cuts) != 2:
        raise AssertionError("9x9 regression case did not produce two fixed 2D cuts.")
    max_ranges = []
    for cut in cuts:
        r = np.asarray(cut["r_env_m"], dtype=float)
        if not np.all(np.isfinite(r)):
            raise AssertionError(f"9x9 regression fixed cut has missing envelope directions: {cut['short_name']}")
        if int(cut.get("finite_direction_count", 0)) != int(cut.get("total_direction_count", -1)):
            raise AssertionError("9x9 regression fixed cut summary did not count all finite directions.")
        max_ranges.append(float(cut["max_range_m"]))
    close(max_ranges[0], max_ranges[1], 1.0e-6, "symmetric 9x9 x/y fixed-cut max range")

    union = compute_scan_union_envelope_3d(params, derived, elem, settings)
    if "maxRangeNearFieldCuts_m" not in union:
        raise AssertionError("Scan union did not expose the near-field fixed-cut maximum range.")
    cut_max = float(union["maxRangeNearFieldCuts_m"])
    if not math.isfinite(cut_max) or cut_max <= 1.0:
        raise AssertionError(f"Invalid scan-union near-field fixed-cut max range: {cut_max}")
    cut_values = [float(cut.get("max_range_m", float("nan"))) for cut in union["unionCuts"]]
    if not math.isclose(cut_max, max(v for v in cut_values if math.isfinite(v)), rel_tol=1.0e-12, abs_tol=1.0e-9):
        raise AssertionError("Scan-union near-field cut max does not match unionCuts summaries.")


def main() -> int:
    checks = [
        ("scan definitions", scan_definition_checks),
        ("auto results", auto_result_checks),
        ("fixed 2D cuts", fixed_cut_checks),
        ("u-v extreme spacing", uv_pattern_extreme_spacing_checks),
        ("scan union spatial envelope", scan_union_checks),
        ("scan union cache decomposition", scan_union_cache_decomposition_checks),
        ("cache key invariants", cache_key_checks),
        ("custom sampling settings", custom_sampling_checks),
        ("shape model invariants", shape_model_checks),
        ("shape combination matrix", shape_combination_matrix_checks),
        ("imported element pattern", imported_element_pattern_checks),
        ("imported near field", imported_near_field_checks),
        ("imported array layout", imported_array_layout_checks),
        ("invalid geometry rejection", invalid_geometry_checks),
        ("invalid parameter rejection", invalid_parameter_checks),
        ("reported bug regressions", reported_bug_regression_checks),
    ]
    started = time.perf_counter()
    for name, func in checks:
        t0 = time.perf_counter()
        func()
        print(f"PASS {name}: {time.perf_counter() - t0:.2f} s")
    print(f"All acceptance checks passed in {time.perf_counter() - started:.2f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
