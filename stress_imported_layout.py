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
from core.element_pattern import make_element_pattern
from core.envelope import compute_2d_cuts, compute_current_3d_envelope
from core.geometry import BeamParams, derive_params, get_mode_settings
from core.scan_union import compute_scan_union_envelope_3d


def _settings():
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


def _write_layout(path: Path, rng: random.Random, count: int, ax_m: float, ay_m: float) -> tuple[int, float]:
    rows = ["x_m,y_m,power_w,phase_deg,enabled"]
    active = 0
    total_power = 0.0
    cols = max(1, math.ceil(math.sqrt(count)))
    rows_n = max(1, math.ceil(count / cols))
    spacing_x = max(1.35 * float(ax_m), 0.045)
    spacing_y = max(1.35 * float(ay_m), 0.045)
    jitter_x = 0.08 * spacing_x
    jitter_y = 0.08 * spacing_y
    cells = [(col, row) for row in range(rows_n) for col in range(cols)]
    rng.shuffle(cells)
    for idx, (col, row) in enumerate(cells[:count]):
        x = (col - (cols - 1) / 2.0) * spacing_x + rng.uniform(-jitter_x, jitter_x)
        y = (row - (rows_n - 1) / 2.0) * spacing_y + rng.uniform(-jitter_y, jitter_y)
        power = 10.0 ** rng.uniform(2.5, 6.0)
        phase = rng.uniform(-45.0, 45.0)
        enabled = 0 if rng.random() < 0.08 else 1
        if enabled:
            active += 1
            total_power += power
        rows.append(f"{x:.10f},{y:.10f},{power:.10f},{phase:.10f},{enabled}")
    if active == 0:
        parts = rows[1].split(",")
        parts[-1] = "1"
        rows[1] = ",".join(parts)
        active = 1
        total_power = float(parts[2])
    path.write_text("\n".join(rows), encoding="utf-8")
    return active, total_power


def _scan_pair(rng: random.Random) -> tuple[float, float]:
    for _ in range(100):
        sx = rng.uniform(-18.0, 18.0)
        sy = rng.uniform(-18.0, 18.0)
        u = math.sin(math.radians(sx))
        v = math.sin(math.radians(sy))
        if u * u + v * v < 0.85:
            return sx, sy
    return 0.0, 0.0


def _assert_finite(name: str, arr: np.ndarray) -> None:
    values = np.asarray(arr)
    if np.any(np.isinf(values)):
        raise AssertionError(f"{name} contains inf")
    if not np.any(np.isfinite(values)):
        raise AssertionError(f"{name} has no finite values")


def _assert_envelope_or_below_threshold(cut: dict[str, np.ndarray], s0_w_m2: float) -> None:
    r_env = np.asarray(cut["r_env_m"], dtype=float)
    if np.any(np.isinf(r_env)):
        raise AssertionError("2D imported r_env contains inf")
    if np.any(np.isfinite(r_env)):
        return
    s_near = np.asarray(cut["s_near_w_m2"], dtype=float)
    if np.nanmax(s_near) >= s0_w_m2:
        raise AssertionError(
            f"2D imported r_env has no finite values even though max S={np.nanmax(s_near):.6g} W/m^2 >= S0={s0_w_m2:.6g} W/m^2"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Stress test imported arbitrary array layout CSV calculations.")
    parser.add_argument("--iterations", type=int, default=80)
    parser.add_argument("--seed", type=int, default=20260617)
    args = parser.parse_args(argv)

    rng = random.Random(args.seed)
    settings = _settings()
    started = time.perf_counter()
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        for idx in range(1, args.iterations + 1):
            layout_path = root / f"layout_{idx:03d}.csv"
            scan_x, scan_y = _scan_pair(rng)
            ax_m = rng.uniform(0.02, 0.12)
            ay_m = rng.uniform(0.02, 0.12)
            active_expected, power_expected = _write_layout(layout_path, rng, rng.randint(3, 44), ax_m, ay_m)
            params = BeamParams(
                frequency_ghz=rng.uniform(3.0, 18.0),
                nx=2,
                ny=2,
                dx_m=0.05,
                dy_m=0.05,
                ax_m=ax_m,
                ay_m=ay_m,
                array_layout="custom",
                array_layout_file=str(layout_path),
                element_shape=rng.choice(["rectangular", "ellipse", "diamond"]),
                element_power_w=1000.0,
                s0_w_cm2=10.0 ** rng.uniform(-1.5, 1.2),
                scan_x_deg=scan_x,
                scan_y_deg=scan_y,
                scan_limit_mode="manual",
                manual_scan_limit_x_deg=rng.uniform(1.0, 12.0),
                manual_scan_limit_y_deg=rng.uniform(1.0, 12.0),
                calc_mode="fast",
                use_element_pattern=rng.random() < 0.8,
            )
            params, derived = derive_params(params)
            if not derived.imported_array_layout:
                raise AssertionError("Imported layout flag is false.")
            if derived.element_x_m.size != active_expected:
                raise AssertionError("Active imported element count mismatch.")
            if not math.isclose(derived.total_input_power_w, power_expected, rel_tol=1.0e-10, abs_tol=1.0e-5):
                raise AssertionError("Imported layout total power mismatch.")
            elem = make_element_pattern(params, derived.wavelength_m)
            cuts = compute_2d_cuts(params, derived, elem, settings)
            for cut in cuts:
                _assert_envelope_or_below_threshold(cut, derived.s0_w_m2)
            env = compute_current_3d_envelope(params, derived, elem, settings)
            _assert_finite("3D imported r_env", env["r_env_m"])
            uv = calc_uv_pattern(params, derived, elem, settings.n_uv, settings.pattern_floor_db)
            _assert_finite("UV imported pattern", uv["pattern_db"])
            union = compute_scan_union_envelope_3d(params, derived, elem, settings)
            _assert_finite("Imported scan union", union["Rsurf"])
            print(f"PASS imported #{idx:03d} active={active_expected} total_power={power_expected:.4g}")
    elapsed = time.perf_counter() - started
    print(f"Imported layout stress passed: iterations={args.iterations}, seed={args.seed}, elapsed={elapsed:.2f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
