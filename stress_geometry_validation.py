from __future__ import annotations

import argparse
from pathlib import Path
import random
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.geometry import (
    BeamParams,
    _element_overlap_metric,
    _normalize_shape_name,
    make_element_positions,
    make_layout_mask,
    validate_generated_layout_spacing,
    validate_imported_layout_spacing,
)


def _active_generated_points(params: BeamParams) -> tuple[np.ndarray, np.ndarray]:
    x_elem, y_elem, xe, ye = make_element_positions(params.nx, params.ny, params.dx_m, params.dy_m)
    active = make_layout_mask(params, x_elem, y_elem)
    return xe[active].astype(float), ye[active].astype(float)


def _has_bruteforce_overlap(params: BeamParams, x_m: np.ndarray, y_m: np.ndarray) -> bool:
    x = np.asarray(x_m, dtype=float).ravel()
    y = np.asarray(y_m, dtype=float).ravel()
    shape = _normalize_shape_name(params.element_shape, default="rectangular")
    ax = max(float(params.ax_m), 1.0e-12)
    ay = max(float(params.ay_m), 1.0e-12)
    for i in range(x.size):
        for j in range(i + 1, x.size):
            dx = abs(float(x[i]) - float(x[j]))
            dy = abs(float(y[i]) - float(y[j]))
            if _element_overlap_metric(shape, dx, dy, ax, ay) < 1.0 - 1.0e-12:
                return True
    return False


def _random_params(rng: random.Random) -> BeamParams:
    dx = 10.0 ** rng.uniform(-2.0, -0.2)
    dy = 10.0 ** rng.uniform(-2.0, -0.2)
    return BeamParams(
        nx=rng.randint(1, 14),
        ny=rng.randint(1, 14),
        dx_m=dx,
        dy_m=dy,
        ax_m=dx * rng.uniform(0.15, 1.85),
        ay_m=dy * rng.uniform(0.15, 1.85),
        array_layout=rng.choice(["rectangular", "ellipse", "diamond"]),
        element_shape=rng.choice(["rectangular", "ellipse", "diamond"]),
    )


def _assert_validator_matches_bruteforce(params: BeamParams, x_m: np.ndarray, y_m: np.ndarray, imported: bool) -> None:
    expected_reject = _has_bruteforce_overlap(params, x_m, y_m)
    try:
        if imported:
            validate_imported_layout_spacing(params, x_m, y_m)
        else:
            validate_generated_layout_spacing(params)
        actual_reject = False
    except ValueError as exc:
        actual_reject = True
        if "口径重叠" not in str(exc):
            raise AssertionError(f"Unexpected validation message: {exc}") from exc
    if actual_reject != expected_reject:
        kind = "imported" if imported else "generated"
        raise AssertionError(
            f"{kind} overlap mismatch: expected_reject={expected_reject}, actual_reject={actual_reject}, "
            f"params={params.to_dict()}, x={x_m.tolist()}, y={y_m.tolist()}"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Bruteforce geometry-overlap validator stress test.")
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260617)
    args = parser.parse_args(argv)

    rng = random.Random(args.seed)
    generated_rejects = 0
    imported_rejects = 0
    for idx in range(1, args.iterations + 1):
        params = _random_params(rng)
        x_m, y_m = _active_generated_points(params)
        before = _has_bruteforce_overlap(params, x_m, y_m)
        _assert_validator_matches_bruteforce(params, x_m, y_m, imported=False)
        generated_rejects += int(before)

        point_count = rng.randint(1, 40)
        cloud_scale_x = max(params.dx_m, params.ax_m) * rng.uniform(0.5, 4.0)
        cloud_scale_y = max(params.dy_m, params.ay_m) * rng.uniform(0.5, 4.0)
        xi = np.array([rng.uniform(-cloud_scale_x, cloud_scale_x) for _ in range(point_count)], dtype=float)
        yi = np.array([rng.uniform(-cloud_scale_y, cloud_scale_y) for _ in range(point_count)], dtype=float)
        imported_params = BeamParams.from_dict({**params.to_dict(), "array_layout": "custom"})
        before_imported = _has_bruteforce_overlap(imported_params, xi, yi)
        _assert_validator_matches_bruteforce(imported_params, xi, yi, imported=True)
        imported_rejects += int(before_imported)

        if idx % max(1, args.iterations // 10) == 0:
            print(f"PASS geometry #{idx:04d}")

    print(
        "Geometry validation stress passed: "
        f"iterations={args.iterations}, seed={args.seed}, "
        f"generated_rejects={generated_rejects}, imported_rejects={imported_rejects}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
