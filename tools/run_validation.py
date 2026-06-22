from __future__ import annotations

import argparse
import compileall
import os
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
EXCLUDED_COMPILE_DIRS = {
    ".pytest_cache",
    "__pycache__",
    "build",
    "dist",
    "export_smoke_output",
}


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    env = _validation_env()
    steps = _validation_steps(args)
    started = time.perf_counter()

    print(f"BeamCoverage validation root: {ROOT}", flush=True)
    print("Numerical thread limits: OPENBLAS=1 OMP=1 MKL=1 NUMEXPR=1", flush=True)
    print(f"Mode: {'quick' if args.quick else 'standard'}", flush=True)
    print(f"Release check: {'on' if args.release else 'off'}", flush=True)

    for index, step in enumerate(steps, start=1):
        _run_step(index, len(steps), step, env)

    elapsed = time.perf_counter() - started
    print(f"PASS full validation sequence in {elapsed:.2f}s", flush=True)
    return 0


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run BeamCoverage validation checks sequentially with stable BLAS threading.")
    parser.add_argument("--quick", action="store_true", help="Use shorter stress-test iteration counts for a fast local sanity run.")
    parser.add_argument("--release", action="store_true", help="Also validate the packaged release directory and ZIP.")
    parser.add_argument(
        "--smoke-output",
        type=Path,
        default=ROOT / "dist" / "_validation" / "smoke_validation_source.json",
        help="JSON output path for the source GUI smoke test.",
    )
    return parser.parse_args(argv)


def _validation_env() -> dict[str, str]:
    env = os.environ.copy()
    env["QT_QPA_PLATFORM"] = "offscreen"
    env["OPENBLAS_NUM_THREADS"] = "1"
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    env["NUMEXPR_NUM_THREADS"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUNBUFFERED"] = "1"
    return env


def _validation_steps(args: argparse.Namespace) -> list[tuple[str, list[str]]]:
    args.smoke_output.parent.mkdir(parents=True, exist_ok=True)
    random_iterations = "20" if args.quick else "80"
    imported_layout_iterations = "20" if args.quick else "80"
    imported_element_iterations = "15" if args.quick else "40"
    geometry_iterations = "150" if args.quick else "800"
    steps: list[tuple[str, list[str]]] = [
        ("compile Python sources", [sys.executable, "-c", _compile_command()]),
        ("source documentation checks", [sys.executable, "tools/check_docs.py"]),
        ("acceptance checks", [sys.executable, "acceptance_check.py"]),
        ("GUI check entrypoint", [sys.executable, "gui_check.py"]),
        (
            "geometry validator stress",
            [sys.executable, "stress_geometry_validation.py", "--iterations", geometry_iterations, "--seed", "20260617"],
        ),
        ("random core stress", [sys.executable, "stress_random.py", "--iterations", random_iterations, "--seed", "20260617"]),
        (
            "imported array-layout stress",
            [sys.executable, "stress_imported_layout.py", "--iterations", imported_layout_iterations, "--seed", "20260617"],
        ),
        (
            "imported element-pattern stress",
            [sys.executable, "stress_imported_element_pattern.py", "--iterations", imported_element_iterations, "--seed", "20260618"],
        ),
        ("export smoke", [sys.executable, "export_smoke.py"]),
        (
            "source GUI smoke",
            [sys.executable, "main.py", "--smoke-test", "--smoke-output", str(args.smoke_output)],
        ),
    ]
    if args.release:
        steps.append(("release package check", [sys.executable, "release_check.py"]))
    return steps


def _compile_command() -> str:
    return (
        "from tools.run_validation import compile_sources; "
        "raise SystemExit(0 if compile_sources() else 1)"
    )


def compile_sources() -> bool:
    ok = True
    for path in sorted(ROOT.iterdir()):
        if path.name in EXCLUDED_COMPILE_DIRS:
            continue
        if path.is_dir():
            ok = compileall.compile_dir(path, quiet=1, force=True) and ok
        elif path.suffix == ".py":
            ok = compileall.compile_file(path, quiet=1, force=True) and ok
    return ok


def _run_step(index: int, total: int, step: tuple[str, list[str]], env: dict[str, str]) -> None:
    name, command = step
    started = time.perf_counter()
    print(f"\n[{index}/{total}] {name}", flush=True)
    print(" ".join(_quote(part) for part in command), flush=True)
    proc = subprocess.run(command, cwd=ROOT, env=env, check=False)
    elapsed = time.perf_counter() - started
    if proc.returncode != 0:
        raise SystemExit(f"FAIL {name}: exit code {proc.returncode} after {elapsed:.2f}s")
    print(f"PASS {name}: {elapsed:.2f}s", flush=True)


def _quote(value: str) -> str:
    if not value or any(ch.isspace() for ch in value):
        return '"' + value.replace('"', '\\"') + '"'
    return value


if __name__ == "__main__":
    raise SystemExit(main())
