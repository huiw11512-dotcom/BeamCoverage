from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import lru_cache
import math
from pathlib import Path
from typing import Any

import numpy as np

from core.aperture_shapes import (
    analytic_scan_loss_db,
    element_overlap_metric,
    element_shape_to_mode,
    normalize_shape_name,
    shape_label,
)


C0 = 3.0e8
X3_DB = 0.442946


def sind(x: Any) -> Any:
    return np.sin(np.deg2rad(x))


def cosd(x: Any) -> Any:
    return np.cos(np.deg2rad(x))


def asind(x: Any) -> Any:
    return np.rad2deg(np.arcsin(x))


def atan2d(y: Any, x: Any) -> Any:
    return np.rad2deg(np.arctan2(y, x))


def wrap_to_180(angle_deg: Any) -> Any:
    return (np.asarray(angle_deg) + 180.0) % 360.0 - 180.0


@dataclass
class BeamParams:
    frequency_ghz: float = 10.0
    nx: int = 8
    ny: int = 8
    dx_m: float = 0.10
    dy_m: float = 0.10
    ax_m: float = 0.10
    ay_m: float = 0.10
    efficiency: float = 0.70
    element_power_w: float = 1.0e6
    s0_w_cm2: float = 20.0
    scan_x_deg: float = 8.0
    scan_y_deg: float = 8.0
    scan_limit_mode: str = "auto"
    manual_scan_limit_x_deg: float = 8.0
    manual_scan_limit_y_deg: float = 8.0
    calc_mode: str = "fast"
    use_element_pattern: bool = True
    array_layout: str = "rectangular"
    element_shape: str = "rectangular"
    array_layout_file: str = ""
    element_pattern_file: str = ""
    element_near_field_file: str = ""
    custom_sampling_enabled: bool = False
    sample_2d_alpha_n: int = 181
    sample_2d_range_n: int = 110
    sample_3d_theta_n: int = 128
    sample_3d_phi_n: int = 256
    sample_3d_range_n: int = 140
    sample_uv_n: int = 301
    sample_scan_union_step_deg: float = 1.0
    sample_scan_union_theta_n: int = 96
    sample_scan_union_phi_n: int = 181
    display_3d_grid_n: int = 180

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BeamParams":
        defaults = cls()
        cleaned = defaults.to_dict()
        cleaned.update({k: v for k, v in data.items() if k in cleaned})
        if "array_layout" not in data and cleaned.get("array_layout_file"):
            cleaned["array_layout"] = "custom"
        cleaned["nx"] = int(cleaned["nx"])
        cleaned["ny"] = int(cleaned["ny"])
        for key in (
            "sample_2d_alpha_n",
            "sample_2d_range_n",
            "sample_3d_theta_n",
            "sample_3d_phi_n",
            "sample_3d_range_n",
            "sample_uv_n",
            "sample_scan_union_theta_n",
            "sample_scan_union_phi_n",
            "display_3d_grid_n",
        ):
            cleaned[key] = int(cleaned[key])
        return cls(**cleaned)


@dataclass(frozen=True)
class ModeSettings:
    name: str
    cut_half_width_deg: float
    n_alpha_2d: int
    n_range_2d: int
    n_far_plot_2d: int
    theta_3d_n: int
    phi_3d_n: int
    n_range_3d: int
    n_uv: int
    pattern_floor_db: float
    scan_union_step_deg: float
    scan_union_theta_n: int
    scan_union_phi_n: int
    chunk_size: int
    scan_block_size: int


MODE_SETTINGS: dict[str, ModeSettings] = {
    "fast": ModeSettings(
        name="fast",
        cut_half_width_deg=10.0,
        n_alpha_2d=121,
        n_range_2d=72,
        n_far_plot_2d=64,
        theta_3d_n=36,
        phi_3d_n=72,
        n_range_3d=52,
        n_uv=121,
        pattern_floor_db=-40.0,
        scan_union_step_deg=2.0,
        scan_union_theta_n=32,
        scan_union_phi_n=61,
        chunk_size=6000,
        scan_block_size=96,
    ),
    "standard": ModeSettings(
        name="standard",
        cut_half_width_deg=10.0,
        n_alpha_2d=181,
        n_range_2d=110,
        n_far_plot_2d=90,
        theta_3d_n=64,
        phi_3d_n=128,
        n_range_3d=80,
        n_uv=201,
        pattern_floor_db=-40.0,
        scan_union_step_deg=1.5,
        scan_union_theta_n=50,
        scan_union_phi_n=97,
        chunk_size=8000,
        scan_block_size=80,
    ),
    "fine": ModeSettings(
        name="fine",
        cut_half_width_deg=10.0,
        n_alpha_2d=361,
        n_range_2d=160,
        n_far_plot_2d=120,
        theta_3d_n=96,
        phi_3d_n=192,
        n_range_3d=120,
        n_uv=301,
        pattern_floor_db=-40.0,
        scan_union_step_deg=1.0,
        scan_union_theta_n=75,
        scan_union_phi_n=145,
        chunk_size=10000,
        scan_block_size=64,
    ),
}


def get_mode_settings(name: str) -> ModeSettings:
    return MODE_SETTINGS.get(str(name).lower(), MODE_SETTINGS["fast"])


def get_effective_mode_settings(params: BeamParams) -> ModeSettings:
    p = sanitized_params(params)
    settings = get_mode_settings(p.calc_mode)
    if not p.custom_sampling_enabled:
        return _auto_mode_settings(p, settings)
    return ModeSettings(
        name=f"{settings.name}+custom",
        cut_half_width_deg=settings.cut_half_width_deg,
        n_alpha_2d=p.sample_2d_alpha_n,
        n_range_2d=p.sample_2d_range_n,
        n_far_plot_2d=max(settings.n_far_plot_2d, min(240, p.sample_2d_alpha_n)),
        theta_3d_n=p.sample_3d_theta_n,
        phi_3d_n=p.sample_3d_phi_n,
        n_range_3d=p.sample_3d_range_n,
        n_uv=p.sample_uv_n,
        pattern_floor_db=settings.pattern_floor_db,
        scan_union_step_deg=p.sample_scan_union_step_deg,
        scan_union_theta_n=p.sample_scan_union_theta_n,
        scan_union_phi_n=p.sample_scan_union_phi_n,
        chunk_size=settings.chunk_size,
        scan_block_size=settings.scan_block_size,
    )


def _auto_mode_settings(params: BeamParams, base: ModeSettings) -> ModeSettings:
    wavelength = C0 / max(float(params.frequency_ghz) * 1.0e9, np.finfo(float).tiny)
    aperture_x = max(float(params.nx) * float(params.dx_m), float(params.ax_m), wavelength)
    aperture_y = max(float(params.ny) * float(params.dy_m), float(params.ay_m), wavelength)
    electrical_x = aperture_x / wavelength
    electrical_y = aperture_y / wavelength
    electrical_aperture = max(electrical_x, electrical_y)
    spacing_ratio = max(float(params.dx_m), float(params.dy_m)) / wavelength
    aspect_ratio = max(
        float(params.dx_m) / max(float(params.dy_m), 1.0e-12),
        float(params.dy_m) / max(float(params.dx_m), 1.0e-12),
    )
    mode_boost = {"fast": 0.72, "standard": 1.0, "fine": 1.32}.get(base.name, 1.0)
    grating_boost = max(0.0, spacing_ratio - 0.55)
    aspect_boost = max(0.0, min(aspect_ratio, 40.0) - 1.0)

    alpha_n = _odd_clamped(
        max(
            base.n_alpha_2d,
            161 + mode_boost * (5.0 * electrical_aperture + 18.0 * grating_boost + 3.0 * aspect_boost),
        ),
        121,
        1201,
    )
    range_n = int(
        _clamped(
            max(base.n_range_2d, 96 + mode_boost * (1.5 * math.sqrt(max(electrical_aperture, 1.0)) + 0.15 * electrical_aperture)),
            72,
            360,
        )
    )
    theta_3d_n = _odd_clamped(
        max(base.theta_3d_n, 61 + mode_boost * (1.2 * electrical_aperture + 4.5 * grating_boost)),
        41,
        181,
    )
    phi_3d_n = _odd_clamped(
        max(base.phi_3d_n, 121 + mode_boost * (2.4 * electrical_aperture + 10.0 * grating_boost + 2.5 * aspect_boost)),
        81,
        361,
    )
    range_3d_n = int(
        _clamped(
            max(base.n_range_3d, 72 + mode_boost * (1.2 * math.sqrt(max(electrical_aperture, 1.0)) + 0.08 * electrical_aperture)),
            52,
            260,
        )
    )
    uv_n = _odd_clamped(
        max(base.n_uv, 181 + mode_boost * (4.5 * electrical_aperture + 18.0 * grating_boost + 2.0 * aspect_boost)),
        121,
        701,
    )

    hpbw_x = 50.8 / max(electrical_x, 1.0e-9)
    hpbw_y = 50.8 / max(electrical_y, 1.0e-9)
    smallest_hpbw = max(0.05, min(hpbw_x, hpbw_y))
    union_step = min(base.scan_union_step_deg, max(0.35, 0.75 * smallest_hpbw))
    union_theta_n = _odd_clamped(max(base.scan_union_theta_n, min(theta_3d_n, 121)), 31, 181)
    union_phi_n = _odd_clamped(max(base.scan_union_phi_n, min(phi_3d_n, 241)), 61, 361)

    return ModeSettings(
        name=f"{base.name}+auto",
        cut_half_width_deg=base.cut_half_width_deg,
        n_alpha_2d=alpha_n,
        n_range_2d=range_n,
        n_far_plot_2d=max(base.n_far_plot_2d, min(240, alpha_n)),
        theta_3d_n=theta_3d_n,
        phi_3d_n=phi_3d_n,
        n_range_3d=range_3d_n,
        n_uv=uv_n,
        pattern_floor_db=base.pattern_floor_db,
        scan_union_step_deg=union_step,
        scan_union_theta_n=union_theta_n,
        scan_union_phi_n=union_phi_n,
        chunk_size=base.chunk_size,
        scan_block_size=base.scan_block_size,
    )


def _clamped(value: float, minimum: float, maximum: float) -> float:
    return min(max(float(value), float(minimum)), float(maximum))


def _odd_clamped(value: float, minimum: int, maximum: int) -> int:
    n = int(round(_clamped(value, minimum, maximum)))
    if n % 2 == 0:
        n += 1
    if n > maximum:
        n -= 2
    if n < minimum:
        n = minimum if minimum % 2 == 1 else minimum + 1
    return int(n)


@dataclass
class DerivedParams:
    frequency_hz: float
    wavelength_m: float
    k_rad_m: float
    u0: float
    v0: float
    w0: float
    theta_deg: float
    phi_deg: float
    x_elem_m: np.ndarray
    y_elem_m: np.ndarray
    xe_m: np.ndarray
    ye_m: np.ndarray
    feed_phase_rad: np.ndarray
    element_x_m: np.ndarray
    element_y_m: np.ndarray
    element_power_w: np.ndarray
    element_phase_offset_rad: np.ndarray
    element_feed_phase_rad: np.ndarray
    imported_array_layout: bool
    array_layout_source: str
    dx_aperture_m: float
    dy_aperture_m: float
    aperture_m: float
    rff_m: float
    hpbw_x_deg: float
    hpbw_y_deg: float
    scan_limit_x_deg: float
    scan_limit_y_deg: float
    phase_step_x_deg: float
    phase_step_y_deg: float
    phase_step_x_mod_deg: float
    phase_step_y_mod_deg: float
    total_input_power_w: float
    scan_loss_db: float
    s0_w_m2: float


@dataclass(frozen=True)
class ImportedArrayLayout:
    x_m: np.ndarray
    y_m: np.ndarray
    power_w: np.ndarray
    phase_rad: np.ndarray
    source_path: str
    point_count: int


def sanitized_params(params: BeamParams, *, validate_external: bool = True) -> BeamParams:
    p = BeamParams.from_dict(params.to_dict())
    p.frequency_ghz = float(p.frequency_ghz)
    p.nx = int(p.nx)
    p.ny = int(p.ny)
    p.dx_m = float(p.dx_m)
    p.dy_m = float(p.dy_m)
    p.ax_m = float(p.ax_m)
    p.ay_m = float(p.ay_m)
    p.efficiency = float(p.efficiency)
    p.element_power_w = float(p.element_power_w)
    p.s0_w_cm2 = float(p.s0_w_cm2)
    p.scan_x_deg = float(p.scan_x_deg)
    p.scan_y_deg = float(p.scan_y_deg)
    p.manual_scan_limit_x_deg = float(p.manual_scan_limit_x_deg)
    p.manual_scan_limit_y_deg = float(p.manual_scan_limit_y_deg)
    p.scan_limit_mode = "manual" if str(p.scan_limit_mode).lower() == "manual" else "auto"
    p.calc_mode = str(p.calc_mode).lower()
    if p.calc_mode not in MODE_SETTINGS:
        p.calc_mode = "fast"
    p.use_element_pattern = _as_bool(p.use_element_pattern)
    p.array_layout = _normalize_shape_name(str(p.array_layout), default="rectangular")
    p.element_shape = _normalize_shape_name(str(p.element_shape), default="rectangular")
    p.array_layout_file = str(getattr(p, "array_layout_file", "") or "").strip()
    p.element_pattern_file = str(getattr(p, "element_pattern_file", "") or "").strip()
    p.element_near_field_file = str(getattr(p, "element_near_field_file", "") or "").strip()
    p.custom_sampling_enabled = _as_bool(getattr(p, "custom_sampling_enabled", False))
    p.sample_2d_alpha_n = int(p.sample_2d_alpha_n)
    p.sample_2d_range_n = int(p.sample_2d_range_n)
    p.sample_3d_theta_n = int(p.sample_3d_theta_n)
    p.sample_3d_phi_n = int(p.sample_3d_phi_n)
    p.sample_3d_range_n = int(p.sample_3d_range_n)
    p.sample_uv_n = int(p.sample_uv_n)
    p.sample_scan_union_step_deg = float(p.sample_scan_union_step_deg)
    p.sample_scan_union_theta_n = int(p.sample_scan_union_theta_n)
    p.sample_scan_union_phi_n = int(p.sample_scan_union_phi_n)
    p.display_3d_grid_n = int(p.display_3d_grid_n)
    validate_params(p, validate_external=validate_external)
    return p


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"0", "false", "no", "off", "disabled", "禁用", "否"}:
        return False
    if text in {"1", "true", "yes", "on", "enabled", "启用", "是"}:
        return True
    return bool(value)


def _normalize_shape_name(value: str, default: str = "rectangular") -> str:
    return normalize_shape_name(value, default=default, allow_custom=True)


def validate_params(p: BeamParams, *, validate_external: bool = True) -> None:
    errors: list[str] = []
    _positive(errors, "frequency f", p.frequency_ghz, unit="GHz")
    _integer_at_least(errors, "Nx", p.nx, 1)
    _integer_at_least(errors, "Ny", p.ny, 1)
    _positive(errors, "dx", p.dx_m, unit="m")
    _positive(errors, "dy", p.dy_m, unit="m")
    _positive(errors, "ax", p.ax_m, unit="m")
    _positive(errors, "ay", p.ay_m, unit="m")
    _range(errors, "efficiency eta", p.efficiency, 0.0, 1.0)
    _nonnegative(errors, "element power", p.element_power_w, unit="W")
    _positive(errors, "S0", p.s0_w_cm2, unit="W/cm^2")
    _int_range(errors, "2D angle samples", p.sample_2d_alpha_n, 21, 1441)
    _int_range(errors, "2D range samples", p.sample_2d_range_n, 8, 800)
    _int_range(errors, "3D theta samples", p.sample_3d_theta_n, 8, 361)
    _int_range(errors, "3D phi samples", p.sample_3d_phi_n, 16, 721)
    _int_range(errors, "3D range samples", p.sample_3d_range_n, 8, 800)
    _int_range(errors, "u-v samples", p.sample_uv_n, 31, 801)
    _positive(errors, "scan-union step", p.sample_scan_union_step_deg, unit="deg")
    if not 0.1 <= p.sample_scan_union_step_deg <= 20.0:
        errors.append("scan-union step must be in [0.1, 20] deg.")
    _int_range(errors, "scan-union theta samples", p.sample_scan_union_theta_n, 8, 361)
    _int_range(errors, "scan-union phi samples", p.sample_scan_union_phi_n, 16, 721)
    _int_range(errors, "3D display grid", p.display_3d_grid_n, 32, 500)
    _range(errors, "scanX", p.scan_x_deg, -89.0, 89.0, unit="deg")
    _range(errors, "scanY", p.scan_y_deg, -89.0, 89.0, unit="deg")
    if _is_finite(p.scan_x_deg) and _is_finite(p.scan_y_deg):
        u0 = math.sin(math.radians(float(p.scan_x_deg)))
        v0 = math.sin(math.radians(float(p.scan_y_deg)))
        if u0 * u0 + v0 * v0 >= 1.0:
            errors.append(
                f"scanX={p.scan_x_deg:g} deg, scanY={p.scan_y_deg:g} deg maps outside visible hemisphere (u^2+v^2 >= 1)"
            )
    if p.scan_limit_mode == "manual":
        _range(errors, "manual scan limit X", p.manual_scan_limit_x_deg, 0.0, 89.0, unit="deg")
        _range(errors, "manual scan limit Y", p.manual_scan_limit_y_deg, 0.0, 89.0, unit="deg")
    imported_layout = _uses_imported_array_layout(p)
    if not imported_layout and not errors:
        try:
            validate_generated_layout_spacing(p)
        except ValueError as exc:
            errors.append(str(exc))
    if validate_external and imported_layout and not p.array_layout_file:
        errors.append("导入坐标CSV排布需要先加载阵元坐标文件。")
    if validate_external and imported_layout:
        if p.array_layout_file and not Path(p.array_layout_file).expanduser().is_file():
            errors.append(f"array layout file does not exist: {p.array_layout_file}")
        if p.array_layout_file and Path(p.array_layout_file).expanduser().is_file():
            try:
                table = load_imported_array_layout(p.array_layout_file, p.element_power_w)
                validate_imported_layout_spacing(p, table.x_m, table.y_m)
            except ValueError as exc:
                errors.append(str(exc))
    if validate_external and _uses_imported_element_pattern(p) and not Path(p.element_pattern_file).expanduser().is_file():
        errors.append(f"element pattern file does not exist: {p.element_pattern_file}")
    if errors:
        raise ValueError("参数无效：\n" + "\n".join(errors) + "\n请修正左侧输入后重新计算。")


def validate_geometry(p: BeamParams) -> None:
    validate_params(p)


def _uses_imported_array_layout(p: BeamParams) -> bool:
    return str(getattr(p, "array_layout", "")) == "custom"


def _uses_imported_element_pattern(p: BeamParams) -> bool:
    return bool(getattr(p, "use_element_pattern", True)) and bool(getattr(p, "element_pattern_file", ""))


def _is_finite(value: float) -> bool:
    return math.isfinite(float(value))


def _positive(errors: list[str], label: str, value: float, unit: str = "") -> None:
    if not _is_finite(value) or float(value) <= 0.0:
        suffix = f" {unit}" if unit else ""
        errors.append(f"{label} must be finite and > 0, got {value:g}{suffix}")


def _nonnegative(errors: list[str], label: str, value: float, unit: str = "") -> None:
    if not _is_finite(value) or float(value) < 0.0:
        suffix = f" {unit}" if unit else ""
        errors.append(f"{label} must be finite and >= 0, got {value:g}{suffix}")


def _range(errors: list[str], label: str, value: float, minimum: float, maximum: float, unit: str = "") -> None:
    if not _is_finite(value) or not (minimum <= float(value) <= maximum):
        suffix = f" {unit}" if unit else ""
        errors.append(f"{label} must be in [{minimum:g}, {maximum:g}], got {value:g}{suffix}")


def _integer_at_least(errors: list[str], label: str, value: int, minimum: int) -> None:
    if int(value) < minimum:
        errors.append(f"{label} must be >= {minimum}, got {value}")


def _int_range(errors: list[str], label: str, value: int, minimum: int, maximum: int) -> None:
    ivalue = int(value)
    if ivalue < minimum or ivalue > maximum:
        errors.append(f"{label} must be in [{minimum}, {maximum}], got {value}")


def make_element_positions(nx: int, ny: int, dx_m: float, dy_m: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x_elem = (np.arange(nx, dtype=float) - (nx - 1.0) / 2.0) * dx_m
    y_elem = (np.arange(ny, dtype=float) - (ny - 1.0) / 2.0) * dy_m
    xe, ye = np.meshgrid(x_elem, y_elem)
    return x_elem, y_elem, xe, ye


def make_layout_mask(params: BeamParams, x_elem: np.ndarray | None = None, y_elem: np.ndarray | None = None) -> np.ndarray:
    p = BeamParams.from_dict(params.to_dict())
    if x_elem is None or y_elem is None:
        x_elem, y_elem, _, _ = make_element_positions(p.nx, p.ny, p.dx_m, p.dy_m)
    x_elem = np.asarray(x_elem, dtype=float)
    y_elem = np.asarray(y_elem, dtype=float)
    xe, ye = np.meshgrid(x_elem, y_elem)
    layout = _normalize_shape_name(p.array_layout, default="rectangular")
    if layout == "rectangular":
        return np.ones_like(xe, dtype=bool)

    half_x = max((float(np.nanmax(x_elem)) - float(np.nanmin(x_elem))) / 2.0 + p.ax_m / 2.0, p.ax_m / 2.0, 1.0e-12)
    half_y = max((float(np.nanmax(y_elem)) - float(np.nanmin(y_elem))) / 2.0 + p.ay_m / 2.0, p.ay_m / 2.0, 1.0e-12)
    if layout == "ellipse":
        metric = (xe / half_x) ** 2 + (ye / half_y) ** 2
        mask = metric <= 1.0 + 1.0e-12
        return _ensure_nonempty_mask(mask, metric)
    if layout == "diamond":
        metric = np.abs(xe) / half_x + np.abs(ye) / half_y
        mask = metric <= 1.0 + 1.0e-12
        return _ensure_nonempty_mask(mask, metric)
    return np.ones_like(xe, dtype=bool)


def _ensure_nonempty_mask(mask: np.ndarray, metric: np.ndarray) -> np.ndarray:
    if np.any(mask):
        return mask
    fallback = np.zeros_like(mask, dtype=bool)
    finite_metric = np.where(np.isfinite(metric), metric, np.inf)
    min_metric = float(np.nanmin(finite_metric))
    fallback[np.isclose(finite_metric, min_metric, rtol=0.0, atol=1.0e-12)] = True
    return fallback


def load_imported_array_layout(path: str | Path, default_power_w: float) -> ImportedArrayLayout:
    source = Path(path).expanduser()
    stat = source.stat()
    return _load_imported_array_layout_cached(
        str(source.resolve()),
        int(stat.st_mtime_ns),
        int(stat.st_size),
        float(default_power_w),
    )


@lru_cache(maxsize=16)
def _load_imported_array_layout_cached(path: str, mtime_ns: int, size: int, default_power_w: float) -> ImportedArrayLayout:
    del mtime_ns, size
    return _read_imported_array_layout_csv(Path(path), default_power_w)


def _read_imported_array_layout_csv(path: Path, default_power_w: float) -> ImportedArrayLayout:
    data = _read_named_csv_table(path, "array layout")
    if data.dtype.names is None:
        raise ValueError("Array layout CSV must include a header row.")
    if data.shape == ():
        data = np.asarray([data], dtype=data.dtype)

    columns = _column_map(data.dtype.names)
    x_col = _find_column(columns, ("x_m", "x", "xm", "x[m]", "x mm", "x_mm", "x[mm]"))
    y_col = _find_column(columns, ("y_m", "y", "ym", "y[m]", "y mm", "y_mm", "y[mm]"))
    if not x_col or not y_col:
        available = ", ".join(data.dtype.names or ())
        raise ValueError(f"Array layout CSV must contain x_m/y_m columns. Available columns: {available}")

    x = np.asarray(data[x_col], dtype=float).ravel()
    y = np.asarray(data[y_col], dtype=float).ravel()
    if _normalize_column_name(x_col) in {"xmm", "xmm"} or "mm" in str(x_col).lower():
        x = x / 1000.0
    if _normalize_column_name(y_col) in {"ymm", "ymm"} or "mm" in str(y_col).lower():
        y = y / 1000.0

    power_col = _find_column(columns, ("power_w", "power", "pin_w", "element_power_w", "p_w", "p"))
    power_scale_col = _find_column(columns, ("power_scale", "weight_power", "power_weight"))
    amp_col = _find_column(columns, ("amp", "amplitude", "weight", "excitation", "mag"))
    if power_col:
        power = np.asarray(data[power_col], dtype=float).ravel()
    elif power_scale_col:
        power = float(default_power_w) * np.asarray(data[power_scale_col], dtype=float).ravel()
    elif amp_col:
        amp = np.asarray(data[amp_col], dtype=float).ravel()
        power = float(default_power_w) * amp * amp
    else:
        power = np.full_like(x, float(default_power_w), dtype=float)

    phase_deg_col = _find_column(columns, ("phase_deg", "phase", "phase[deg]", "phase_deg_init", "phase_init_deg"))
    phase_rad_col = _find_column(columns, ("phase_rad", "phase[rad]", "phase_init_rad"))
    if phase_rad_col:
        phase = np.asarray(data[phase_rad_col], dtype=float).ravel()
    elif phase_deg_col:
        phase = np.deg2rad(np.asarray(data[phase_deg_col], dtype=float).ravel())
    else:
        phase = np.zeros_like(x, dtype=float)

    enabled_col = _find_column(columns, ("enabled", "active", "on", "used", "valid"))
    if enabled_col:
        enabled = _parse_enabled_column(np.asarray(data[enabled_col]).ravel())
    else:
        enabled = np.ones_like(x, dtype=bool)

    valid = enabled & np.isfinite(x) & np.isfinite(y) & np.isfinite(power) & np.isfinite(phase) & (power > 0.0)
    if int(np.count_nonzero(valid)) < 1:
        raise ValueError("Array layout CSV has no active finite elements with positive power.")
    x = x[valid]
    y = y[valid]
    power = power[valid]
    phase = phase[valid]

    return ImportedArrayLayout(
        x_m=x.astype(float),
        y_m=y.astype(float),
        power_w=power.astype(float),
        phase_rad=phase.astype(float),
        source_path=str(path),
        point_count=int(x.size),
    )


def _parse_enabled_column(values: np.ndarray) -> np.ndarray:
    out = np.zeros(values.size, dtype=bool)
    for idx, value in enumerate(values):
        if isinstance(value, (bytes, bytearray)):
            text = value.decode("utf-8", errors="ignore").strip().lower()
        else:
            text = str(value).strip().lower()
        if text in {"", "nan", "none"}:
            out[idx] = False
        elif text in {"1", "true", "yes", "y", "on", "active", "enabled"}:
            out[idx] = True
        elif text in {"0", "false", "no", "n", "off", "inactive", "disabled"}:
            out[idx] = False
        else:
            try:
                out[idx] = float(text) > 0.0
            except ValueError:
                out[idx] = True
    return out


def _read_named_csv_table(path: Path, kind: str) -> np.ndarray:
    last_error: Exception | None = None
    for delimiter in (",", ";", "\t", None):
        try:
            data = np.genfromtxt(path, delimiter=delimiter, names=True, dtype=None, encoding="utf-8-sig", invalid_raise=False)
            if data.dtype.names is not None and len(data.dtype.names) >= 2:
                return data
        except Exception as exc:
            last_error = exc
    if last_error is not None:
        raise ValueError(f"Could not read {kind} CSV: {last_error}") from last_error
    raise ValueError(f"{kind.title()} CSV must include a header row with at least two columns.")


def _column_map(names: tuple[str, ...]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for name in names:
        key = _normalize_column_name(name)
        if key and key not in mapping:
            mapping[key] = name
    return mapping


def _normalize_column_name(name: str) -> str:
    return "".join(ch for ch in str(name).lower().strip() if ch.isalnum())


def _find_column(columns: dict[str, str], candidates: tuple[str, ...]) -> str | None:
    for candidate in candidates:
        key = _normalize_column_name(candidate)
        if key in columns:
            return columns[key]
    return None


def active_aperture_extents(params: BeamParams, x_elem: np.ndarray, y_elem: np.ndarray, active_mask: np.ndarray) -> tuple[float, float]:
    active = np.asarray(active_mask, dtype=bool)
    xe, ye = np.meshgrid(np.asarray(x_elem, dtype=float), np.asarray(y_elem, dtype=float))
    if not np.any(active):
        return params.ax_m, params.ay_m
    active_x = xe[active]
    active_y = ye[active]
    dx_ap = float(np.nanmax(active_x) - np.nanmin(active_x) + params.ax_m)
    dy_ap = float(np.nanmax(active_y) - np.nanmin(active_y) + params.ay_m)
    return max(dx_ap, params.ax_m), max(dy_ap, params.ay_m)


def active_aperture_extents_from_points(params: BeamParams, x_m: np.ndarray, y_m: np.ndarray) -> tuple[float, float]:
    x = np.asarray(x_m, dtype=float)
    y = np.asarray(y_m, dtype=float)
    if x.size == 0 or y.size == 0:
        return params.ax_m, params.ay_m
    dx_ap = float(np.nanmax(x) - np.nanmin(x) + params.ax_m)
    dy_ap = float(np.nanmax(y) - np.nanmin(y) + params.ay_m)
    return max(dx_ap, params.ax_m), max(dy_ap, params.ay_m)


def validate_imported_layout_spacing(params: BeamParams, x_m: np.ndarray, y_m: np.ndarray) -> None:
    overlaps = _find_element_footprint_overlaps(params, x_m, y_m)
    if not overlaps:
        return
    raise ValueError(_format_element_footprint_overlap_error("导入阵元坐标", params, overlaps))


def validate_generated_layout_spacing(params: BeamParams) -> None:
    p = BeamParams.from_dict(params.to_dict())
    p.nx = int(p.nx)
    p.ny = int(p.ny)
    p.dx_m = float(p.dx_m)
    p.dy_m = float(p.dy_m)
    p.ax_m = float(p.ax_m)
    p.ay_m = float(p.ay_m)
    p.array_layout = _normalize_shape_name(p.array_layout, default="rectangular")
    p.element_shape = _normalize_shape_name(p.element_shape, default="rectangular")
    if _uses_imported_array_layout(p):
        return
    x_elem, y_elem, xe, ye = make_element_positions(p.nx, p.ny, p.dx_m, p.dy_m)
    active_mask = make_layout_mask(p, x_elem, y_elem)
    if not np.any(active_mask):
        return
    overlaps = _find_element_footprint_overlaps(p, xe[active_mask], ye[active_mask])
    if not overlaps:
        return
    raise ValueError(_format_element_footprint_overlap_error("参数生成阵列", p, overlaps))


def _format_element_footprint_overlap_error(
    context: str,
    params: BeamParams,
    overlaps: list[tuple[int, int, float, float, float]],
) -> str:
    layout_label = shape_label(params.array_layout, kind="array", default=str(params.array_layout))
    element_label = shape_label(params.element_shape, kind="element", default=str(params.element_shape))
    examples = []
    for first, second, dx, dy, metric in overlaps:
        examples.append(
            f"#{first + 1} 与 #{second + 1}: Δx={dx:.6g} m, Δy={dy:.6g} m, overlap_metric={metric:.4g}"
        )
    return (
        f"{context}存在单元口径重叠：{layout_label} / {element_label}, "
        f"ax={params.ax_m:g} m, ay={params.ay_m:g} m。"
        "请减小单元尺寸、增大单元间距，或调整排布/导入坐标。"
        "示例：" + "；".join(examples)
    )


def _find_element_footprint_overlaps(
    params: BeamParams,
    x_m: np.ndarray,
    y_m: np.ndarray,
    *,
    max_report: int = 8,
) -> list[tuple[int, int, float, float, float]]:
    x = np.asarray(x_m, dtype=float).ravel()
    y = np.asarray(y_m, dtype=float).ravel()
    if x.size < 2 or y.size < 2:
        return []
    ax = max(float(params.ax_m), 1.0e-12)
    ay = max(float(params.ay_m), 1.0e-12)
    shape = _normalize_shape_name(params.element_shape, default="rectangular")
    cell = max(ax, ay, 1.0e-12)
    buckets: dict[tuple[int, int], list[int]] = {}
    overlaps: list[tuple[int, int, float, float, float]] = []
    tol = 1.0e-12
    for idx, (xi, yi) in enumerate(zip(x, y)):
        cx = int(math.floor(float(xi) / cell))
        cy = int(math.floor(float(yi) / cell))
        for bx in range(cx - 1, cx + 2):
            for by in range(cy - 1, cy + 2):
                for prev in buckets.get((bx, by), []):
                    dx = abs(float(xi) - float(x[prev]))
                    dy = abs(float(yi) - float(y[prev]))
                    metric = _element_overlap_metric(shape, dx, dy, ax, ay)
                    if metric < 1.0 - tol:
                        overlaps.append((prev, idx, dx, dy, metric))
                        if len(overlaps) >= max_report:
                            return overlaps
        buckets.setdefault((cx, cy), []).append(idx)
    return overlaps


def _element_overlap_metric(shape: str, dx: float, dy: float, ax: float, ay: float) -> float:
    return element_overlap_metric(shape, dx, dy, ax, ay)


def estimate_scan_limit_deg(wavelength_m: float, element_size_m: float) -> float:
    return float(asind(min(1.0, X3_DB * wavelength_m / max(element_size_m, 1.0e-12))))


def estimate_hpbw_deg(wavelength_m: float, aperture_m: float) -> float:
    return float(2.0 * asind(min(1.0, X3_DB * wavelength_m / max(aperture_m, 1.0e-12))))


def derive_params(params: BeamParams) -> tuple[BeamParams, DerivedParams]:
    p = sanitized_params(params)
    frequency_hz = p.frequency_ghz * 1.0e9
    wavelength = C0 / frequency_hz
    k = 2.0 * math.pi / wavelength

    u0 = float(sind(p.scan_x_deg))
    v0 = float(sind(p.scan_y_deg))
    uv2 = u0 * u0 + v0 * v0
    if uv2 >= 1.0:
        raise ValueError("scanX/scanY map to u^2+v^2 >= 1; reduce scan angles.")

    w0 = math.sqrt(max(1.0 - uv2, 0.0))
    theta = float(asind(math.sqrt(uv2)))
    phi = float(atan2d(v0, u0))

    imported_layout = _uses_imported_array_layout(p)
    if imported_layout:
        table = load_imported_array_layout(p.array_layout_file, p.element_power_w)
        element_x = table.x_m
        element_y = table.y_m
        element_power = table.power_w
        element_phase_offset = table.phase_rad
        element_feed_phase = element_phase_offset - k * (element_x * u0 + element_y * v0)
        x_elem = np.arange(table.point_count, dtype=float)
        y_elem = np.array([0.0], dtype=float)
        xe = element_x.reshape(1, -1)
        ye = element_y.reshape(1, -1)
        feed_phase = element_feed_phase.reshape(1, -1)
        dap_x, dap_y = active_aperture_extents_from_points(p, element_x, element_y)
        array_layout_source = table.source_path
    else:
        x_elem, y_elem, xe, ye = make_element_positions(p.nx, p.ny, p.dx_m, p.dy_m)
        active_mask = make_layout_mask(p, x_elem, y_elem)
        feed_phase = -k * (xe * u0 + ye * v0)
        element_x = xe[active_mask].astype(float)
        element_y = ye[active_mask].astype(float)
        element_power = np.full(element_x.shape, p.element_power_w, dtype=float)
        element_phase_offset = np.zeros(element_x.shape, dtype=float)
        element_feed_phase = feed_phase[active_mask].astype(float)
        dap_x, dap_y = active_aperture_extents(p, x_elem, y_elem, active_mask)
        array_layout_source = ""
    dap = max(dap_x, dap_y)
    rff = 2.0 * dap * dap / wavelength

    hpbw_x = estimate_hpbw_deg(wavelength, dap_x)
    hpbw_y = estimate_hpbw_deg(wavelength, dap_y)

    if p.scan_limit_mode == "manual":
        scan_limit_x = abs(float(p.manual_scan_limit_x_deg))
        scan_limit_y = abs(float(p.manual_scan_limit_y_deg))
    else:
        scan_limit_x = estimate_scan_limit_deg(wavelength, p.ax_m)
        scan_limit_y = estimate_scan_limit_deg(wavelength, p.ay_m)

    phase_step_x = -360.0 * p.dx_m / wavelength * u0
    phase_step_y = -360.0 * p.dy_m / wavelength * v0
    phase_step_x_mod = float(wrap_to_180(phase_step_x))
    phase_step_y_mod = float(wrap_to_180(phase_step_y))

    scan_loss_mode = "isotropic" if not p.use_element_pattern else element_shape_to_mode(p.element_shape, default="rect")
    scan_loss_db = analytic_scan_loss_db(
        scan_loss_mode,
        p.ax_m,
        p.ay_m,
        wavelength,
        u0,
        v0,
        w0,
        q=1.0,
    )

    derived = DerivedParams(
        frequency_hz=frequency_hz,
        wavelength_m=wavelength,
        k_rad_m=k,
        u0=u0,
        v0=v0,
        w0=w0,
        theta_deg=theta,
        phi_deg=phi,
        x_elem_m=x_elem,
        y_elem_m=y_elem,
        xe_m=xe,
        ye_m=ye,
        feed_phase_rad=feed_phase,
        element_x_m=element_x,
        element_y_m=element_y,
        element_power_w=element_power,
        element_phase_offset_rad=element_phase_offset,
        element_feed_phase_rad=element_feed_phase,
        imported_array_layout=imported_layout,
        array_layout_source=array_layout_source,
        dx_aperture_m=dap_x,
        dy_aperture_m=dap_y,
        aperture_m=dap,
        rff_m=rff,
        hpbw_x_deg=hpbw_x,
        hpbw_y_deg=hpbw_y,
        scan_limit_x_deg=scan_limit_x,
        scan_limit_y_deg=scan_limit_y,
        phase_step_x_deg=phase_step_x,
        phase_step_y_deg=phase_step_y,
        phase_step_x_mod_deg=phase_step_x_mod,
        phase_step_y_mod_deg=phase_step_y_mod,
        total_input_power_w=float(np.sum(element_power)),
        scan_loss_db=scan_loss_db,
        s0_w_m2=p.s0_w_cm2 * 1.0e4,
    )
    return p, derived


def make_pin_matrix(params: BeamParams) -> np.ndarray:
    p = sanitized_params(params)
    if _uses_imported_array_layout(p):
        table = load_imported_array_layout(p.array_layout_file, p.element_power_w)
        return table.power_w.reshape(1, -1).astype(float)
    x_elem, y_elem, _, _ = make_element_positions(p.nx, p.ny, p.dx_m, p.dy_m)
    active_mask = make_layout_mask(p, x_elem, y_elem)
    return np.where(active_mask, p.element_power_w, 0.0).astype(float)


def make_range_grid(r_min: float, r_max: float, n: int) -> np.ndarray:
    n = max(2, int(n))
    if r_max <= r_min:
        return np.linspace(r_min, r_max, n)
    if r_max / max(r_min, 1.0e-12) > 80.0:
        return np.logspace(math.log10(r_min), math.log10(r_max), n)
    return np.linspace(r_min, r_max, n)


def make_vertical_cut_directions(alpha_deg: np.ndarray, phi_cut_deg: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    alpha = np.asarray(alpha_deg, dtype=float)
    sx = sind(alpha) * cosd(phi_cut_deg)
    sy = sind(alpha) * sind(phi_cut_deg)
    sz = cosd(alpha)
    return sx, sy, sz


def make_hemisphere_directions(theta_n: int, phi_n: int, theta_max_deg: float = 89.0) -> dict[str, np.ndarray]:
    theta = np.linspace(0.0, theta_max_deg, int(theta_n))
    phi = np.linspace(-180.0, 180.0, int(phi_n), endpoint=False)
    phi_grid, theta_grid = np.meshgrid(phi, theta)
    sx = sind(theta_grid) * cosd(phi_grid)
    sy = sind(theta_grid) * sind(phi_grid)
    sz = cosd(theta_grid)
    return {
        "theta": theta,
        "phi": phi,
        "THETA": theta_grid,
        "PHI": phi_grid,
        "sx": sx,
        "sy": sy,
        "sz": sz,
    }


def base_cache_key(params: BeamParams, settings: ModeSettings) -> tuple[Any, ...]:
    p = sanitized_params(params)
    pattern_file_token: tuple[str, int, int] | str = ""
    if _uses_imported_element_pattern(p):
        pattern_path = Path(p.element_pattern_file).expanduser()
        try:
            stat = pattern_path.stat()
            pattern_file_token = (str(pattern_path.resolve()), int(stat.st_mtime_ns), int(stat.st_size))
        except OSError:
            pattern_file_token = p.element_pattern_file
    layout_file_token: tuple[str, int, int] | str = ""
    if _uses_imported_array_layout(p) and p.array_layout_file:
        layout_path = Path(p.array_layout_file).expanduser()
        try:
            stat = layout_path.stat()
            layout_file_token = (str(layout_path.resolve()), int(stat.st_mtime_ns), int(stat.st_size))
        except OSError:
            layout_file_token = p.array_layout_file
    return (
        round(p.frequency_ghz, 12),
        p.nx,
        p.ny,
        round(p.dx_m, 12),
        round(p.dy_m, 12),
        round(p.ax_m, 12),
        round(p.ay_m, 12),
        round(p.efficiency, 12),
        round(p.element_power_w, 6),
        round(p.s0_w_cm2, 12),
        p.scan_limit_mode,
        round(p.manual_scan_limit_x_deg, 9),
        round(p.manual_scan_limit_y_deg, 9),
        p.use_element_pattern,
        p.array_layout,
        p.element_shape,
        layout_file_token,
        pattern_file_token,
        _settings_cache_token(settings),
    )


def _settings_cache_token(settings: ModeSettings) -> tuple[tuple[str, Any], ...]:
    return tuple((key, value) for key, value in asdict(settings).items())
