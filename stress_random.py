from __future__ import annotations

import argparse
import math
from pathlib import Path
import random
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.array_factor import calc_uv_pattern
from core.analysis import derive_params_with_element
from core.envelope import compute_2d_cuts, compute_current_3d_envelope
from core.geometry import BeamParams, derive_params, get_mode_settings, make_pin_matrix
from core.scan_union import compute_scan_union_envelope_3d


def _random_scan_pair(rng: random.Random) -> tuple[float, float]:
    for _ in range(1000):
        scan_x = rng.uniform(-25.0, 25.0)
        scan_y = rng.uniform(-25.0, 25.0)
        u = math.sin(math.radians(scan_x))
        v = math.sin(math.radians(scan_y))
        if u * u + v * v < 0.92:
            return scan_x, scan_y
    return 0.0, 0.0


def _random_params(rng: random.Random) -> BeamParams:
    nx = rng.randint(2, 12)
    ny = rng.randint(2, 12)
    frequency = rng.uniform(2.0, 20.0)
    wavelength = 0.3 / frequency
    dx = rng.uniform(0.35 * wavelength, 4.0 * wavelength)
    dy = rng.uniform(0.35 * wavelength, 4.0 * wavelength)
    ax = rng.uniform(0.25 * wavelength, max(0.3 * wavelength, min(dx, 2.5 * wavelength)))
    ay = rng.uniform(0.25 * wavelength, max(0.3 * wavelength, min(dy, 2.5 * wavelength)))
    scan_x, scan_y = _random_scan_pair(rng)
    manual = rng.random() < 0.35
    array_layout = rng.choice(["rectangular", "ellipse", "diamond"])
    element_shape = rng.choice(["rectangular", "ellipse", "diamond"])
    return BeamParams(
        frequency_ghz=frequency,
        nx=nx,
        ny=ny,
        dx_m=dx,
        dy_m=dy,
        ax_m=ax,
        ay_m=ay,
        efficiency=rng.uniform(0.35, 0.95),
        element_power_w=10.0 ** rng.uniform(2.0, 6.2),
        s0_w_cm2=10.0 ** rng.uniform(-1.0, 2.0),
        scan_x_deg=scan_x,
        scan_y_deg=scan_y,
        scan_limit_mode="manual" if manual else "auto",
        manual_scan_limit_x_deg=rng.uniform(1.0, 18.0),
        manual_scan_limit_y_deg=rng.uniform(1.0, 18.0),
        calc_mode="fast",
        use_element_pattern=rng.random() < 0.8,
        array_layout=array_layout,
        element_shape=element_shape,
    )


def _stress_settings():
    settings0 = get_mode_settings("fast")
    return settings0.__class__(
        **{
            **settings0.__dict__,
            "n_alpha_2d": 31,
            "n_range_2d": 20,
            "theta_3d_n": 10,
            "phi_3d_n": 16,
            "n_range_3d": 14,
            "n_uv": 51,
            "scan_union_step_deg": 9.0,
            "scan_union_theta_n": 10,
            "scan_union_phi_n": 18,
            "chunk_size": 512,
            "scan_block_size": 16,
        }
    )


def _assert_finite(name: str, value: np.ndarray, allow_nan: bool = True) -> None:
    arr = np.asarray(value)
    finite_or_nan = np.isfinite(arr) | np.isnan(arr)
    if allow_nan:
        if not np.all(finite_or_nan):
            raise AssertionError(f"{name} contains inf values")
    elif not np.all(np.isfinite(arr)):
            raise AssertionError(f"{name} contains non-finite values")


def _odd_count(value: int) -> int:
    n = max(3, int(value))
    return n if n % 2 == 1 else n + 1


def _run_one(index: int, params: BeamParams, settings) -> dict[str, float | int]:
    params, derived, elem = derive_params_with_element(params)
    if not (0.0 <= derived.u0 * derived.u0 + derived.v0 * derived.v0 < 1.0):
        raise AssertionError("invalid scan direction after derivation")
    pin = make_pin_matrix(params)
    active_count = int(np.count_nonzero(pin))
    if active_count <= 0:
        raise AssertionError("layout has no active elements")
    expected_power = active_count * params.element_power_w
    if not math.isclose(derived.total_input_power_w, expected_power, rel_tol=1.0e-12, abs_tol=1.0e-6):
        raise AssertionError("total input power is not based on active element count")

    t0 = time.perf_counter()
    cuts = compute_2d_cuts(params, derived, elem, settings)
    t_cuts = time.perf_counter() - t0
    if len(cuts) != 2 or cuts[0]["phi_cut_deg"] != 0.0 or cuts[1]["phi_cut_deg"] != 90.0:
        raise AssertionError("fixed cut metadata changed")
    for cut in cuts:
        _assert_finite(f"cut {cut['short_name']} r_env", cut["r_env_m"])
        _assert_finite(f"cut {cut['short_name']} s_near", cut["s_near_w_m2"], allow_nan=False)
        if cut.get("envelope_method") != "near_field_sampled_far_field_extrapolated":
            raise AssertionError("2D cut method metadata is missing or incorrect")

    t0 = time.perf_counter()
    env = compute_current_3d_envelope(params, derived, elem, settings)
    t_3d = time.perf_counter() - t0
    _assert_finite("current 3d r_env", env["r_env_m"])
    if env.get("envelopeMethod") != "near_field_sampled_far_field_extrapolated":
        raise AssertionError("current 3D method metadata is missing or incorrect")
    if env["Xsurf"].shape != (_odd_count(settings.theta_3d_n), _odd_count(settings.phi_3d_n)):
        raise AssertionError("current 3D grid shape mismatch")

    t0 = time.perf_counter()
    uv = calc_uv_pattern(params, derived, elem, settings.n_uv, settings.pattern_floor_db)
    t_uv = time.perf_counter() - t0
    _assert_finite("uv pattern", uv["pattern_db"])
    if uv["pattern_db"].shape != (settings.n_uv, settings.n_uv):
        raise AssertionError("u-v grid shape mismatch")
    if not np.nanmax(uv["pattern_db"]) <= 1.0e-9:
        raise AssertionError("u-v pattern is not normalized to <= 0 dB")

    t0 = time.perf_counter()
    union = compute_scan_union_envelope_3d(params, derived, elem, settings)
    t_union = time.perf_counter() - t0
    _assert_finite("scan union Rsurf", union["Rsurf"])
    if union["Rsurf"].shape != (_odd_count(settings.scan_union_theta_n), _odd_count(settings.scan_union_phi_n)):
        raise AssertionError("scan union grid shape mismatch")
    if union.get("envelopeMethod") != "far_field_coefficient_union":
        raise AssertionError("scan union method metadata is missing or incorrect")
    timing = union.get("timings", {})
    for timing_key in ("scan_union_3d_s", "scan_union_2d_cuts_s", "scan_union_compute_s"):
        timing_value = float(timing.get(timing_key, float("nan"))) if isinstance(timing, dict) else float("nan")
        if not math.isfinite(timing_value) or timing_value < 0.0:
            raise AssertionError(f"scan union timing {timing_key} is missing or invalid")
    if union["numScanCenters"] <= 0:
        raise AssertionError("scan union has no valid scan centers")
    has_zero_scan = any(
        abs(float(x)) < 1.0e-12 and abs(float(y)) < 1.0e-12
        for x, y in zip(union["scanXList_deg"], union["scanYList_deg"])
    )
    if not has_zero_scan:
        raise AssertionError("scan union centers do not include boresight")
    if not np.allclose(union["Xsurf"], union["Rsurf"] * union["sx"], equal_nan=True):
        raise AssertionError("scan union Xsurf != R*u")
    cuts_union = union.get("unionCuts")
    if not isinstance(cuts_union, list) or len(cuts_union) != 2:
        raise AssertionError("scan union fixed 2D cuts are missing")
    if float(cuts_union[0]["phi_cut_deg"]) != 0.0 or float(cuts_union[1]["phi_cut_deg"]) != 90.0:
        raise AssertionError("scan union fixed 2D cuts have wrong phi metadata")
    cut0_y = np.asarray(cuts_union[0]["y_env_m"], dtype=float)
    if not np.allclose(cut0_y[np.isfinite(cut0_y)], 0.0, atol=1.0e-12):
        raise AssertionError("scan union x-z cut has nonzero y")
    cut1_x = np.asarray(cuts_union[1]["x_env_m"], dtype=float)
    if not np.allclose(cut1_x[np.isfinite(cut1_x)], 0.0, atol=1.0e-12):
        raise AssertionError("scan union y-z cut has nonzero x")
    for cut in cuts_union:
        _assert_finite(f"scan union cut {cut['short_name']} r_env", cut["r_env_m"])
        if np.asarray(cut["r_env_m"]).size != settings.n_alpha_2d:
            raise AssertionError("scan union fixed 2D cut resolution does not match n_alpha_2d")
        if cut.get("envelope_method") != "near_field_sampled_far_field_scan_union":
            raise AssertionError("scan union fixed 2D cut method metadata is missing or incorrect")
    current_within_limits = (
        abs(float(params.scan_x_deg)) <= abs(float(derived.scan_limit_x_deg)) + 1.0e-9
        and abs(float(params.scan_y_deg)) <= abs(float(derived.scan_limit_y_deg)) + 1.0e-9
    )
    if current_within_limits:
        includes_current = any(
            abs(float(x) - float(params.scan_x_deg)) < 1.0e-9 and abs(float(y) - float(params.scan_y_deg)) < 1.0e-9
            for x, y in zip(union["scanXList_deg"], union["scanYList_deg"])
        )
        if not includes_current:
            raise AssertionError("scan union centers missed the current scan point inside the allowed scan limits")

    return {
        "index": index,
        "nx": params.nx,
        "ny": params.ny,
        "active": active_count,
        "cuts_s": t_cuts,
        "current_3d_s": t_3d,
        "uv_s": t_uv,
        "scan_union_s": t_union,
        "total_s": t_cuts + t_3d + t_uv + t_union,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Random stress test for BeamCoverage core calculations.")
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260611)
    args = parser.parse_args(argv)

    rng = random.Random(args.seed)
    settings = _stress_settings()
    totals: list[float] = []
    invalid_geometry_rejected = 0
    invalid_parameter_rejected = 0
    started = time.perf_counter()
    for idx in range(1, args.iterations + 1):
        params = _random_params(rng)
        try:
            result = _run_one(idx, params, settings)
        except Exception as exc:
            print(f"FAIL iteration={idx} seed={args.seed} params={params.to_dict()} error={exc}", file=sys.stderr)
            raise
        totals.append(float(result["total_s"]))
        print(
            "PASS "
            f"#{idx:03d} {int(result['nx'])}x{int(result['ny'])} active={int(result['active'])} "
            f"total={result['total_s']:.3f}s union={result['scan_union_s']:.3f}s"
        )

        if idx <= max(5, args.iterations // 5):
            bad = BeamParams.from_dict(params.to_dict())
            bad.array_layout = "rectangular"
            bad.ax_m = bad.dx_m * rng.uniform(1.05, 2.0)
            try:
                derive_params(bad)
            except ValueError:
                invalid_geometry_rejected += 1
            else:
                raise AssertionError("random invalid ax>dx geometry was not rejected")

            bad_limit = BeamParams.from_dict(params.to_dict())
            bad_limit.scan_limit_mode = "manual"
            bad_limit.manual_scan_limit_x_deg = rng.uniform(90.1, 180.0)
            try:
                derive_params(bad_limit)
            except ValueError:
                invalid_parameter_rejected += 1
            else:
                raise AssertionError("random invalid manual scan limit was not rejected")

            bad_scan = BeamParams.from_dict(params.to_dict())
            bad_scan.scan_x_deg = 70.0
            bad_scan.scan_y_deg = 70.0
            try:
                derive_params(bad_scan)
            except ValueError:
                invalid_parameter_rejected += 1
            else:
                raise AssertionError("random outside-visible-hemisphere scan was not rejected")

    elapsed = time.perf_counter() - started
    print(
        f"Random stress passed: iterations={args.iterations}, seed={args.seed}, "
        f"invalid_geometry_rejected={invalid_geometry_rejected}, "
        f"invalid_parameter_rejected={invalid_parameter_rejected}, "
        f"elapsed={elapsed:.2f}s, avg={np.mean(totals):.3f}s, p95={np.percentile(totals, 95):.3f}s"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
