from __future__ import annotations

import argparse
import math
from pathlib import Path
import random
import sys
import tempfile
import time

import numpy as np

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.array_factor import calc_uv_pattern
from core.analysis import derive_params_with_element
from core.envelope import compute_2d_cuts, compute_current_3d_envelope
from core.geometry import BeamParams, get_mode_settings
from core.scan_union import compute_scan_union_envelope_3d


def _settings():
    settings0 = get_mode_settings("fast")
    return settings0.__class__(
        **{
            **settings0.__dict__,
            "n_alpha_2d": 31,
            "n_range_2d": 18,
            "theta_3d_n": 9,
            "phi_3d_n": 15,
            "n_range_3d": 12,
            "n_uv": 51,
            "scan_union_step_deg": 10.0,
            "scan_union_theta_n": 9,
            "scan_union_phi_n": 15,
            "chunk_size": 512,
            "scan_block_size": 12,
        }
    )


def _field_from_uv(u: float, v: float, rng: random.Random, phase_scale: float = 1.0) -> complex:
    radius2 = u * u + v * v
    amp = max(0.015, (1.0 - 0.58 * radius2) * (1.0 - 0.10 * math.sin(4.0 * u + 1.7 * v)))
    phase = math.radians(35.0 * phase_scale * u - 24.0 * phase_scale * v + rng.uniform(-3.0, 3.0))
    return amp * complex(math.cos(phase), math.sin(phase))


def _uv_samples(rng: random.Random, n_axis: int) -> list[tuple[float, float, complex]]:
    samples: list[tuple[float, float, complex]] = []
    for u in np.linspace(-0.94, 0.94, n_axis):
        for v in np.linspace(-0.94, 0.94, n_axis):
            if u * u + v * v <= 0.96:
                field = _field_from_uv(float(u), float(v), rng)
                samples.append((float(u), float(v), field))
    samples.append((1.15, 0.0, 1.0 + 0.0j))
    samples.append((0.0, -1.18, 1.0 + 0.0j))
    return samples


def _theta_phi_samples(rng: random.Random, n_theta: int, n_phi: int) -> list[tuple[float, float, complex]]:
    samples: list[tuple[float, float, complex]] = []
    for theta in np.linspace(0.0, 82.0, n_theta):
        for phi in np.linspace(-180.0, 180.0, n_phi):
            th = math.radians(float(theta))
            ph = math.radians(float(phi))
            u = math.sin(th) * math.cos(ph)
            v = math.sin(th) * math.sin(ph)
            field = _field_from_uv(u, v, rng, phase_scale=1.3)
            samples.append((float(theta), float(phi), field))
    return samples


def _unique_uv_count(samples: list[tuple[float, float, complex]]) -> int:
    keys = {
        (round(float(u), 12), round(float(v), 12))
        for u, v, _field in samples
        if u * u + v * v <= 1.0 + 1.0e-9
    }
    return len(keys)


def _unique_theta_phi_count(samples: list[tuple[float, float, complex]]) -> int:
    keys = set()
    for theta, phi, _field in samples:
        th = math.radians(float(theta))
        ph = math.radians(float(phi))
        u = math.sin(th) * math.cos(ph)
        v = math.sin(th) * math.sin(ph)
        if u * u + v * v <= 1.0 + 1.0e-9:
            keys.add((round(u, 12), round(v, 12)))
    return len(keys)


def _write_pattern(path: Path, rng: random.Random, variant: str) -> tuple[bool, int]:
    if variant == "uv_gain_db_phase":
        samples = _uv_samples(rng, rng.choice([11, 13, 15]))
        rows = ["u,v,dB(GainTotal),Phase(GainTotal)"]
        for u, v, field in samples:
            gain_db = 10.0 * math.log10(max(abs(field) ** 2, 1.0e-12))
            phase_deg = math.degrees(math.atan2(field.imag, field.real))
            rows.append(f"{u:.9f},{v:.9f},{gain_db:.9f},{phase_deg:.9f}")
        path.write_text("\n".join(rows), encoding="utf-8")
        return True, _unique_uv_count(samples)

    if variant == "uv_complex_tab":
        samples = _uv_samples(rng, rng.choice([9, 11, 13]))
        rows = ["u\tv\tReal(ETotal)\tImag(ETotal)"]
        for u, v, field in samples:
            rows.append(f"{u:.9f}\t{v:.9f}\t{field.real:.9f}\t{field.imag:.9f}")
        path.write_text("\n".join(rows), encoding="utf-8")
        return True, _unique_uv_count(samples)

    if variant == "uv_gain_linear":
        samples = _uv_samples(rng, rng.choice([11, 13, 15]))
        rows = ["ux,uy,gain_linear"]
        for u, v, field in samples:
            rows.append(f"{u:.9f},{v:.9f},{abs(field) ** 2:.9f}")
        path.write_text("\n".join(rows), encoding="utf-8")
        return False, _unique_uv_count(samples)

    if variant == "theta_phi_linear_semicolon":
        samples = _theta_phi_samples(rng, rng.choice([11, 13, 15]), rng.choice([19, 25, 31]))
        rows = ["Theta [deg];Phi [deg];Abs(GainTotal)"]
        for theta, phi, field in samples:
            rows.append(f"{theta:.9f};{phi:.9f};{abs(field) ** 2:.9f}")
        path.write_text("\n".join(rows), encoding="utf-8")
        return False, _unique_theta_phi_count(samples)

    if variant == "theta_phi_radian_gain_db":
        samples = _theta_phi_samples(rng, rng.choice([11, 13, 15]), rng.choice([19, 25, 31]))
        rows = ["Theta [rad],Phi [rad],dB(GainTotal)"]
        for theta, phi, field in samples:
            gain_db = 10.0 * math.log10(max(abs(field) ** 2, 1.0e-12))
            rows.append(f"{math.radians(theta):.9f},{math.radians(phi):.9f},{gain_db:.9f}")
        path.write_text("\n".join(rows), encoding="utf-8")
        return False, _unique_theta_phi_count(samples)

    if variant == "theta_phi_mag_phase_space":
        samples = _theta_phi_samples(rng, rng.choice([11, 13, 15]), rng.choice([19, 25, 31]))
        rows = ["theta_deg phi_deg field_mag phase_rad"]
        for theta, phi, field in samples:
            rows.append(f"{theta:.9f} {phi:.9f} {abs(field):.9f} {math.atan2(field.imag, field.real):.9f}")
        path.write_text("\n".join(rows), encoding="utf-8")
        return True, _unique_theta_phi_count(samples)

    if variant == "theta_phi_complex":
        samples = _theta_phi_samples(rng, rng.choice([11, 13, 15]), rng.choice([19, 25, 31]))
        rows = ["theta_deg,phi_deg,field_real,field_imag"]
        for theta, phi, field in samples:
            rows.append(f"{theta:.9f},{phi:.9f},{field.real:.9f},{field.imag:.9f}")
        path.write_text("\n".join(rows), encoding="utf-8")
        return True, _unique_theta_phi_count(samples)

    if variant == "theta_phi_vector_components":
        samples = _theta_phi_samples(rng, rng.choice([11, 13, 15]), rng.choice([19, 25, 31]))
        rows = ["Theta [deg],Phi [deg],Real(ETheta),Imag(ETheta),Real(EPhi),Imag(EPhi)"]
        for theta, phi, field in samples:
            th = math.radians(theta)
            ph = math.radians(phi)
            etheta = field * (0.82 + 0.08 * math.cos(ph))
            ephi_phase = complex(math.cos(0.35 + 0.1 * math.sin(th)), math.sin(0.35 + 0.1 * math.sin(th)))
            ephi = field * (0.36 + 0.05 * math.sin(ph)) * ephi_phase
            rows.append(f"{theta:.9f},{phi:.9f},{etheta.real:.9f},{etheta.imag:.9f},{ephi.real:.9f},{ephi.imag:.9f}")
        path.write_text("\n".join(rows), encoding="utf-8")
        return True, _unique_theta_phi_count(samples)

    if variant == "theta_phi_vector_mag_phase":
        samples = _theta_phi_samples(rng, rng.choice([11, 13, 15]), rng.choice([19, 25, 31]))
        rows = ["Theta [deg],Phi [deg],Abs(Theta),Phase(Theta),Abs(Phi),Phase(Phi)"]
        for theta, phi, field in samples:
            th = math.radians(theta)
            ph = math.radians(phi)
            etheta = field * (0.78 + 0.10 * math.cos(ph))
            ephi_phase = complex(math.cos(-0.42 + 0.12 * math.sin(th)), math.sin(-0.42 + 0.12 * math.sin(th)))
            ephi = field * (0.34 + 0.06 * math.sin(ph)) * ephi_phase
            rows.append(
                f"{theta:.9f},{phi:.9f},{abs(etheta):.9f},{math.degrees(math.atan2(etheta.imag, etheta.real)):.9f},"
                f"{abs(ephi):.9f},{math.degrees(math.atan2(ephi.imag, ephi.real)):.9f}"
            )
        path.write_text("\n".join(rows), encoding="utf-8")
        return True, _unique_theta_phi_count(samples)

    if variant == "theta_phi_vector_mag_phase_metadata":
        samples = _theta_phi_samples(rng, rng.choice([11, 13, 15]), rng.choice([19, 25, 31]))
        rows = [
            "# Solver far-field export",
            f"Frequency,{rng.uniform(4.0, 18.0):.6f} GHz",
            "Setup,LastAdaptive",
            "Quantity,Farfield",
            "Theta [deg],Phi [deg],Abs(Theta),Phase(Theta),Abs(Phi),Phase(Phi)",
        ]
        for theta, phi, field in samples:
            th = math.radians(theta)
            ph = math.radians(phi)
            etheta = field * (0.76 + 0.11 * math.cos(ph))
            ephi_phase = complex(math.cos(-0.36 + 0.10 * math.sin(th)), math.sin(-0.36 + 0.10 * math.sin(th)))
            ephi = field * (0.31 + 0.06 * math.sin(ph)) * ephi_phase
            rows.append(
                f"{theta:.9f},{phi:.9f},{abs(etheta):.9f},{math.degrees(math.atan2(etheta.imag, etheta.real)):.9f},"
                f"{abs(ephi):.9f},{math.degrees(math.atan2(ephi.imag, ephi.real)):.9f}"
            )
        path.write_text("\n".join(rows), encoding="utf-8")
        return True, _unique_theta_phi_count(samples)

    if variant == "theta_phi_vector_mag_phase_multifrequency":
        samples = _theta_phi_samples(rng, rng.choice([11, 13, 15]), rng.choice([19, 25, 31]))
        unit_mode = rng.choice(["ghz_header", "hz_numeric", "mhz_numeric", "string_units"])
        if unit_mode == "ghz_header":
            frequency_header = "Frequency [GHz]"
            frequencies = (6.0, 10.0, 14.0)
        elif unit_mode == "hz_numeric":
            frequency_header = "Freq"
            frequencies = (6.0e9, 10.0e9, 14.0e9)
        elif unit_mode == "mhz_numeric":
            frequency_header = "Freq"
            frequencies = (6000.0, 10000.0, 14000.0)
        else:
            frequency_header = "Freq"
            frequencies = ("6 GHz", "10000 MHz", "14000000000 Hz")
        rows = [f"{frequency_header},Theta [deg],Phi [deg],Abs(Theta),Phase(Theta),Abs(Phi),Phase(Phi)"]
        for freq_index, frequency_ghz in enumerate(frequencies):
            scale = 0.68 + 0.21 * freq_index
            for theta, phi, field in samples:
                th = math.radians(theta)
                ph = math.radians(phi)
                etheta = field * scale * (0.74 + 0.12 * math.cos(ph))
                ephi_phase = complex(math.cos(-0.32 + 0.11 * math.sin(th)), math.sin(-0.32 + 0.11 * math.sin(th)))
                ephi = field * scale * (0.30 + 0.07 * math.sin(ph)) * ephi_phase
                rows.append(
                    f"{frequency_ghz},{theta:.9f},{phi:.9f},"
                    f"{abs(etheta):.9f},{math.degrees(math.atan2(etheta.imag, etheta.real)):.9f},"
                    f"{abs(ephi):.9f},{math.degrees(math.atan2(ephi.imag, ephi.real)):.9f}"
                )
        path.write_text("\n".join(rows), encoding="utf-8")
        return True, _unique_theta_phi_count(samples)

    if variant == "theta_phi_vector_global_x_basis":
        n_theta = rng.choice([11, 13, 15])
        n_phi = rng.choice([19, 25, 31, 37])
        samples = [(float(theta), float(phi), 1.0 + 0.0j) for theta in np.linspace(0.0, 82.0, n_theta) for phi in np.linspace(-180.0, 180.0, n_phi)]
        rows = ["Theta [deg],Phi [deg],Real(ETheta),Imag(ETheta),Real(EPhi),Imag(EPhi)"]
        for theta, phi, _field in samples:
            th = math.radians(theta)
            ph = math.radians(phi)
            etheta = math.cos(th) * math.cos(ph)
            ephi = -math.sin(ph)
            rows.append(f"{theta:.9f},{phi:.9f},{etheta:.9f},0.0,{ephi:.9f},0.0")
        path.write_text("\n".join(rows), encoding="utf-8")
        return True, _unique_theta_phi_count(samples)

    raise ValueError(f"Unknown pattern variant: {variant}")


def _scan_pair(rng: random.Random) -> tuple[float, float]:
    for _ in range(100):
        sx = rng.uniform(-16.0, 16.0)
        sy = rng.uniform(-16.0, 16.0)
        u = math.sin(math.radians(sx))
        v = math.sin(math.radians(sy))
        if u * u + v * v < 0.8:
            return sx, sy
    return 0.0, 0.0


def _assert_no_inf(name: str, arr: np.ndarray) -> None:
    values = np.asarray(arr)
    if np.any(np.isinf(values)):
        raise AssertionError(f"{name} contains inf")


def _assert_some_finite(name: str, arr: np.ndarray) -> None:
    _assert_no_inf(name, arr)
    if not np.any(np.isfinite(arr)):
        raise AssertionError(f"{name} has no finite values")


def _assert_2d_cut_consistent(cut: dict[str, np.ndarray], s0_w_m2: float) -> None:
    _assert_no_inf("2D imported element r_env", cut["r_env_m"])
    if np.any(np.isfinite(np.asarray(cut["r_env_m"], dtype=float))):
        return
    s_near = np.asarray(cut["s_near_w_m2"], dtype=float)
    if np.nanmax(s_near) >= s0_w_m2:
        raise AssertionError(
            f"2D imported element r_env is empty even though max S={np.nanmax(s_near):.6g} W/m^2 >= S0={s0_w_m2:.6g} W/m^2"
        )


def _run_one(index: int, root: Path, rng: random.Random, settings) -> dict[str, object]:
    variants = [
        "uv_gain_db_phase",
        "uv_complex_tab",
        "uv_gain_linear",
        "theta_phi_linear_semicolon",
        "theta_phi_radian_gain_db",
        "theta_phi_mag_phase_space",
        "theta_phi_complex",
        "theta_phi_vector_components",
        "theta_phi_vector_mag_phase",
        "theta_phi_vector_mag_phase_metadata",
        "theta_phi_vector_mag_phase_multifrequency",
        "theta_phi_vector_global_x_basis",
    ]
    variant = rng.choice(variants)
    pattern_path = root / f"element_pattern_{index:03d}_{variant}.csv"
    expected_phase, min_points = _write_pattern(pattern_path, rng, variant)
    scan_x, scan_y = _scan_pair(rng)
    params = BeamParams(
        frequency_ghz=rng.uniform(4.0, 18.0),
        nx=rng.randint(2, 7),
        ny=rng.randint(2, 7),
        dx_m=rng.uniform(0.05, 0.18),
        dy_m=rng.uniform(0.05, 0.18),
        ax_m=rng.uniform(0.015, 0.045),
        ay_m=rng.uniform(0.015, 0.045),
        efficiency=rng.uniform(0.45, 0.95),
        element_power_w=10.0 ** rng.uniform(4.5, 6.5),
        s0_w_cm2=10.0 ** rng.uniform(-2.0, 0.2),
        scan_x_deg=scan_x,
        scan_y_deg=scan_y,
        scan_limit_mode="manual",
        manual_scan_limit_x_deg=rng.uniform(2.0, 14.0),
        manual_scan_limit_y_deg=rng.uniform(2.0, 14.0),
        calc_mode="fast",
        use_element_pattern=True,
        element_pattern_file=str(pattern_path),
    )
    params, derived, elem = derive_params_with_element(params)
    if elem.mode != "table":
        raise AssertionError("Imported element pattern did not activate table mode.")
    if elem.table_has_phase != expected_phase:
        raise AssertionError(f"Phase metadata mismatch for {variant}: expected {expected_phase}, got {elem.table_has_phase}")
    expected_vector = variant in {
        "theta_phi_vector_components",
        "theta_phi_vector_mag_phase",
        "theta_phi_vector_mag_phase_metadata",
        "theta_phi_vector_mag_phase_multifrequency",
        "theta_phi_vector_global_x_basis",
    }
    if elem.table_has_vector_components != expected_vector:
        raise AssertionError(
            f"Vector metadata mismatch for {variant}: expected {expected_vector}, got {elem.table_has_vector_components}"
        )
    if elem.table_point_count < min(3, min_points):
        raise AssertionError(f"Imported point count is too small: {elem.table_point_count}")
    if variant == "theta_phi_vector_mag_phase_multifrequency":
        if elem.table_point_count != min_points:
            raise AssertionError("Multi-frequency imported pattern mixed more than one frequency slice.")
        if elem.table_selected_frequency_ghz not in {6.0, 10.0, 14.0}:
            raise AssertionError(f"Invalid selected imported frequency: {elem.table_selected_frequency_ghz}")
    if not math.isfinite(float(elem.gain_norm)) or float(elem.gain_norm) <= 0.0:
        raise AssertionError(f"Invalid imported gain normalization: {elem.gain_norm}")
    if not 0.0 < float(elem.table_theta_max_deg) <= 90.0:
        raise AssertionError(f"Invalid imported theta coverage: {elem.table_theta_max_deg}")
    if elem.table_covers_visible_edge != (float(elem.table_theta_max_deg) >= 89.0):
        raise AssertionError("Imported visible-edge coverage flag is inconsistent with theta_max.")

    cuts = compute_2d_cuts(params, derived, elem, settings)
    for cut in cuts:
        _assert_2d_cut_consistent(cut, derived.s0_w_m2)
    env = compute_current_3d_envelope(params, derived, elem, settings)
    _assert_some_finite("3D imported element r_env", env["r_env_m"])
    uv = calc_uv_pattern(params, derived, elem, settings.n_uv, settings.pattern_floor_db)
    _assert_some_finite("UV imported element pattern", uv["pattern_db"])
    if not np.nanmax(uv["pattern_db"]) <= 1.0e-9:
        raise AssertionError("Imported element u-v pattern is not normalized to <= 0 dB.")
    union = compute_scan_union_envelope_3d(params, derived, elem, settings)
    _assert_some_finite("Scan union imported element pattern", union["Rsurf"])
    return {
        "variant": variant,
        "phase": expected_phase,
        "vector": elem.table_has_vector_components,
        "points": elem.table_point_count,
        "gain_norm": elem.gain_norm,
        "theta_max": elem.table_theta_max_deg,
        "covers_edge": elem.table_covers_visible_edge,
        "selected_frequency_ghz": elem.table_selected_frequency_ghz,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Stress test imported element-pattern CSV parsing and calculations.")
    parser.add_argument("--iterations", type=int, default=80)
    parser.add_argument("--seed", type=int, default=20260617)
    args = parser.parse_args(argv)

    rng = random.Random(args.seed)
    settings = _settings()
    started = time.perf_counter()
    counts: dict[str, int] = {}
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        for idx in range(1, args.iterations + 1):
            try:
                result = _run_one(idx, root, rng, settings)
            except Exception as exc:
                candidates = sorted(root.glob(f"element_pattern_{idx:03d}_*.csv"))
                print(f"FAIL imported element #{idx:03d} seed={args.seed}: {exc}", file=sys.stderr)
                if candidates:
                    print(candidates[0].read_text(encoding="utf-8")[:4000], file=sys.stderr)
                raise
            variant = str(result["variant"])
            counts[variant] = counts.get(variant, 0) + 1
            print(
                f"PASS imported element #{idx:03d} {variant} "
                f"points={int(result['points'])} phase={bool(result['phase'])} vector={bool(result['vector'])} "
                f"theta_max={float(result['theta_max']):.1f} edge={bool(result['covers_edge'])} "
                f"f={result['selected_frequency_ghz']} Gnorm={float(result['gain_norm']):.4g}"
            )
    elapsed = time.perf_counter() - started
    summary = ", ".join(f"{key}={counts[key]}" for key in sorted(counts))
    print(f"Imported element pattern stress passed: iterations={args.iterations}, seed={args.seed}, elapsed={elapsed:.2f}s, {summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
