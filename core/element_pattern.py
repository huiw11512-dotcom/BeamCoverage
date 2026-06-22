from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import math
from pathlib import Path
import re
import warnings

import numpy as np
from numpy.lib._iotools import ConversionWarning

from core.aperture_shapes import element_shape_to_mode, j1_approx, jinc2, raw_aperture_gain, sinc_local
from core.geometry import BeamParams, cosd, sind


def _trapz(y: np.ndarray, x: np.ndarray, axis: int = -1) -> np.ndarray | float:
    integrator = getattr(np, "trapezoid", None)
    if integrator is None:
        integrator = np.trapz
    return integrator(y, x, axis=axis)


@dataclass(frozen=True)
class ElementPattern:
    mode: str
    size_x_m: float
    size_y_m: float
    wavelength_m: float
    obliquity_q: float
    eta_rad: float
    use_element_pattern: bool
    gain_norm: float
    table_u_axis: np.ndarray | None = None
    table_v_axis: np.ndarray | None = None
    table_gain: np.ndarray | None = None
    table_field: np.ndarray | None = None
    table_field_theta: np.ndarray | None = None
    table_field_phi: np.ndarray | None = None
    table_field_x: np.ndarray | None = None
    table_field_y: np.ndarray | None = None
    table_field_z: np.ndarray | None = None
    table_has_vector_components: bool = False
    table_point_count: int = 0
    table_has_phase: bool = False
    table_theta_max_deg: float = 0.0
    table_covers_visible_edge: bool = False
    table_selected_frequency_ghz: float | None = None
    source_path: str = ""


@dataclass(frozen=True)
class ImportedPatternTable:
    u_axis: np.ndarray
    v_axis: np.ndarray
    gain: np.ndarray
    field: np.ndarray
    field_theta: np.ndarray | None
    field_phi: np.ndarray | None
    field_x: np.ndarray | None
    field_y: np.ndarray | None
    field_z: np.ndarray | None
    has_vector_components: bool
    gain_norm: float
    source_path: str
    point_count: int
    has_phase: bool
    theta_max_deg: float
    covers_visible_edge: bool
    selected_frequency_ghz: float | None = None
    available_frequencies_ghz: tuple[float, ...] = ()


def _j1_approx(x: np.ndarray) -> np.ndarray:
    return j1_approx(x)


def _jinc2(x: np.ndarray) -> np.ndarray:
    return jinc2(x)


def _raw_gain(ux: np.ndarray, uy: np.ndarray, uz: np.ndarray, mode: str, size_x_m: float, size_y_m: float, wavelength_m: float, q: float) -> np.ndarray:
    return raw_aperture_gain(ux, uy, uz, mode, size_x_m, size_y_m, wavelength_m, q)


def _interp_gain_grid(
    u: np.ndarray,
    v: np.ndarray,
    u_axis: np.ndarray,
    v_axis: np.ndarray,
    gain_grid: np.ndarray,
) -> np.ndarray:
    u_arr = np.asarray(u, dtype=float)
    v_arr = np.asarray(v, dtype=float)
    out = np.zeros(np.broadcast_shapes(u_arr.shape, v_arr.shape), dtype=float)
    u_b = np.broadcast_to(u_arr, out.shape)
    v_b = np.broadcast_to(v_arr, out.shape)
    valid = (
        np.isfinite(u_b)
        & np.isfinite(v_b)
        & (u_b >= float(u_axis[0]))
        & (u_b <= float(u_axis[-1]))
        & (v_b >= float(v_axis[0]))
        & (v_b <= float(v_axis[-1]))
    )
    if not np.any(valid):
        return out
    ui = np.searchsorted(u_axis, u_b[valid], side="right") - 1
    vi = np.searchsorted(v_axis, v_b[valid], side="right") - 1
    ui = np.clip(ui, 0, u_axis.size - 2)
    vi = np.clip(vi, 0, v_axis.size - 2)
    u0 = u_axis[ui]
    u1 = u_axis[ui + 1]
    v0 = v_axis[vi]
    v1 = v_axis[vi + 1]
    tx = np.divide(u_b[valid] - u0, u1 - u0, out=np.zeros_like(u0), where=(u1 != u0))
    ty = np.divide(v_b[valid] - v0, v1 - v0, out=np.zeros_like(v0), where=(v1 != v0))
    g00 = gain_grid[vi, ui]
    g10 = gain_grid[vi, ui + 1]
    g01 = gain_grid[vi + 1, ui]
    g11 = gain_grid[vi + 1, ui + 1]
    out[valid] = (
        (1.0 - tx) * (1.0 - ty) * g00
        + tx * (1.0 - ty) * g10
        + (1.0 - tx) * ty * g01
        + tx * ty * g11
    )
    return np.maximum(np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0), 0.0)


def _interp_field_grid(
    u: np.ndarray,
    v: np.ndarray,
    u_axis: np.ndarray,
    v_axis: np.ndarray,
    field_grid: np.ndarray,
) -> np.ndarray:
    real = _interp_signed_grid(u, v, u_axis, v_axis, np.real(field_grid))
    imag = _interp_signed_grid(u, v, u_axis, v_axis, np.imag(field_grid))
    return real + 1j * imag


def _interp_signed_grid(
    u: np.ndarray,
    v: np.ndarray,
    u_axis: np.ndarray,
    v_axis: np.ndarray,
    value_grid: np.ndarray,
) -> np.ndarray:
    u_arr = np.asarray(u, dtype=float)
    v_arr = np.asarray(v, dtype=float)
    out = np.zeros(np.broadcast_shapes(u_arr.shape, v_arr.shape), dtype=float)
    u_b = np.broadcast_to(u_arr, out.shape)
    v_b = np.broadcast_to(v_arr, out.shape)
    valid = (
        np.isfinite(u_b)
        & np.isfinite(v_b)
        & (u_b >= float(u_axis[0]))
        & (u_b <= float(u_axis[-1]))
        & (v_b >= float(v_axis[0]))
        & (v_b <= float(v_axis[-1]))
    )
    if not np.any(valid):
        return out
    ui = np.searchsorted(u_axis, u_b[valid], side="right") - 1
    vi = np.searchsorted(v_axis, v_b[valid], side="right") - 1
    ui = np.clip(ui, 0, u_axis.size - 2)
    vi = np.clip(vi, 0, v_axis.size - 2)
    u0 = u_axis[ui]
    u1 = u_axis[ui + 1]
    v0 = v_axis[vi]
    v1 = v_axis[vi + 1]
    tx = np.divide(u_b[valid] - u0, u1 - u0, out=np.zeros_like(u0), where=(u1 != u0))
    ty = np.divide(v_b[valid] - v0, v1 - v0, out=np.zeros_like(v0), where=(v1 != v0))
    g00 = value_grid[vi, ui]
    g10 = value_grid[vi, ui + 1]
    g01 = value_grid[vi + 1, ui]
    g11 = value_grid[vi + 1, ui + 1]
    out[valid] = (
        (1.0 - tx) * (1.0 - ty) * g00
        + tx * (1.0 - ty) * g10
        + (1.0 - tx) * ty * g01
        + tx * ty * g11
    )
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def _calc_table_gain_norm(u_axis: np.ndarray, v_axis: np.ndarray, gain_grid: np.ndarray) -> float:
    theta = np.linspace(0.0, math.pi / 2.0, 181)
    phi = np.linspace(-math.pi, math.pi, 361)
    phi_grid, theta_grid = np.meshgrid(phi, theta)
    u = np.sin(theta_grid) * np.cos(phi_grid)
    v = np.sin(theta_grid) * np.sin(phi_grid)
    raw = _interp_gain_grid(u, v, u_axis, v_axis, gain_grid)
    integrand = raw * np.sin(theta_grid)
    int_theta = _trapz(integrand, theta, axis=0)
    int_all = _trapz(int_theta, phi)
    return float(4.0 * math.pi / max(float(int_all), np.finfo(float).tiny))


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
        key = "".join(ch for ch in candidate.lower().strip() if ch.isalnum())
        if key in columns:
            return columns[key]
    return None


U_COLUMN_CANDIDATES = ("u", "ux", "dir_x", "dirx", "xdir", "l", "sin_theta_cos_phi")
V_COLUMN_CANDIDATES = ("v", "uy", "dir_y", "diry", "ydir", "m", "sin_theta_sin_phi")
THETA_RAD_COLUMN_CANDIDATES = ("theta_rad", "theta_radians", "theta[rad]", "theta rad", "theta(rad)")
PHI_RAD_COLUMN_CANDIDATES = ("phi_rad", "phi_radians", "phi[rad]", "phi rad", "phi(rad)")
THETA_DEG_COLUMN_CANDIDATES = ("theta_deg", "theta", "theta_degrees", "theta[deg]", "theta deg", "theta(deg)", "th", "theta_scan")
PHI_DEG_COLUMN_CANDIDATES = ("phi_deg", "phi", "phi_degrees", "phi[deg]", "phi deg", "phi(deg)", "ph", "phi_scan")
FREQUENCY_COLUMN_CANDIDATES = (
    "frequency_ghz",
    "freq_ghz",
    "f_ghz",
    "frequency[ghz]",
    "frequency ghz",
    "frequency(ghz)",
    "freq[ghz]",
    "freq ghz",
    "freq(ghz)",
    "frequency_hz",
    "freq_hz",
    "f_hz",
    "frequency[hz]",
    "freq[hz]",
    "frequency_mhz",
    "freq_mhz",
    "f_mhz",
    "frequency[mhz]",
    "freq[mhz]",
    "frequency_khz",
    "freq_khz",
    "frequency[khz]",
    "freq[khz]",
    "frequency",
    "freq",
    "f",
)


def _named_table_has_pattern_coordinates(names: tuple[str, ...]) -> bool:
    columns = _column_map(names)
    return bool(
        (_find_column(columns, U_COLUMN_CANDIDATES) and _find_column(columns, V_COLUMN_CANDIDATES))
        or (_find_column(columns, THETA_RAD_COLUMN_CANDIDATES) and _find_column(columns, PHI_RAD_COLUMN_CANDIDATES))
        or (_find_column(columns, THETA_DEG_COLUMN_CANDIDATES) and _find_column(columns, PHI_DEG_COLUMN_CANDIDATES))
    )


def _component_aliases(component: str) -> tuple[str, ...]:
    compact = "".join(ch for ch in component.lower() if ch.isalnum())
    if compact == "etheta":
        return ("etheta", "e_theta", "theta")
    return ("ephi", "e_phi", "phi")


def _component_display_aliases(component: str) -> tuple[str, ...]:
    compact = "".join(ch for ch in component.lower() if ch.isalnum())
    if compact == "etheta":
        return ("ETheta", "E_Theta", "Theta")
    return ("EPhi", "E_Phi", "Phi")


def _component_name_candidates(component: str, suffixes: tuple[str, ...], prefixes: tuple[str, ...] = ()) -> tuple[str, ...]:
    candidates: list[str] = []
    for alias in _component_aliases(component):
        for suffix in suffixes:
            candidates.append(f"{alias}_{suffix}")
            candidates.append(f"{alias}{suffix}")
        for prefix in prefixes:
            candidates.append(f"{prefix}_{alias}")
            candidates.append(f"{prefix}{alias}")
    return tuple(candidates)


def _component_wrapped_candidates(component: str, wrappers: tuple[str, ...]) -> tuple[str, ...]:
    candidates: list[str] = []
    for label in _component_display_aliases(component):
        for wrapper in wrappers:
            candidates.append(f"{wrapper}({label})")
    return tuple(candidates)


def _component_field_from_columns(
    data: np.ndarray,
    columns: dict[str, str],
    component: str,
) -> tuple[np.ndarray | None, bool]:
    compact = "".join(ch for ch in component.lower() if ch.isalnum())
    pretty = "ETheta" if compact == "etheta" else "EPhi"
    real_col = _find_column(
        columns,
        (
            f"{compact}_real",
            f"{compact}real",
            f"real_{compact}",
            f"real{compact}",
            f"Re({pretty})",
            f"Real({pretty})",
        )
        + _component_name_candidates(component, ("real", "re"), ("real", "re"))
        + _component_wrapped_candidates(component, ("Re", "Real")),
    )
    imag_col = _find_column(
        columns,
        (
            f"{compact}_imag",
            f"{compact}imag",
            f"{compact}_im",
            f"imag_{compact}",
            f"imag{compact}",
            f"im_{compact}",
            f"Im({pretty})",
            f"Imag({pretty})",
        )
        + _component_name_candidates(component, ("imag", "im", "imaginary"), ("imag", "im"))
        + _component_wrapped_candidates(component, ("Im", "Imag")),
    )
    if real_col and imag_col:
        return np.asarray(data[real_col], dtype=float).ravel() + 1j * np.asarray(data[imag_col], dtype=float).ravel(), True

    mag_col = _find_column(
        columns,
        (
            f"{compact}_mag",
            f"{compact}mag",
            f"{compact}_magnitude",
            f"mag_{compact}",
            f"abs_{compact}",
            f"Abs({pretty})",
            f"Mag({pretty})",
        )
        + _component_name_candidates(component, ("mag", "magnitude", "abs", "amp", "amplitude"), ("mag", "abs"))
        + _component_wrapped_candidates(component, ("Abs", "Mag", "Magnitude")),
    )
    db_mag_col = _find_column(
        columns,
        (
            f"{compact}_db",
            f"{compact}db",
            f"db_{compact}",
            f"db{compact}",
            f"dB({pretty})",
            f"{pretty}[dB]",
            f"{pretty} [dB]",
        )
        + _component_name_candidates(component, ("db", "db20"), ("db",))
        + _component_wrapped_candidates(component, ("dB", "DB")),
    )
    phase_deg_col = _find_column(
        columns,
        (
            f"{compact}_phase_deg",
            f"{compact}phase_deg",
            f"phase_{compact}_deg",
            f"phase{compact}",
            f"Phase({pretty})",
        )
        + _component_name_candidates(component, ("phase_deg", "phase_degrees", "phase"), ("phase",))
        + _component_wrapped_candidates(component, ("Phase", "Ang", "Angle", "Arg")),
    )
    phase_rad_col = _find_column(
        columns,
        (
            f"{compact}_phase_rad",
            f"{compact}phase_rad",
            f"phase_{compact}_rad",
        )
        + _component_name_candidates(component, ("phase_rad", "phase_radians"), ())
        + tuple(f"{wrapper}Rad({label})" for label in _component_display_aliases(component) for wrapper in ("Phase", "Arg")),
    )
    if mag_col or db_mag_col:
        if mag_col:
            mag = np.maximum(np.asarray(data[mag_col], dtype=float).ravel(), 0.0)
        else:
            # dB(ETheta)/dB(EPhi) are field-magnitude exports, so convert with 20*log10.
            mag = 10.0 ** (np.asarray(data[db_mag_col], dtype=float).ravel() / 20.0)
            mag = np.maximum(mag, 0.0)
        if phase_rad_col:
            return mag * np.exp(1j * np.asarray(data[phase_rad_col], dtype=float).ravel()), True
        if phase_deg_col:
            return mag * np.exp(1j * np.deg2rad(np.asarray(data[phase_deg_col], dtype=float).ravel())), True
        return mag.astype(complex), False
    return None, False


def _read_named_table(path: Path) -> np.ndarray:
    last_error: Exception | None = None
    try:
        max_skip = min(80, max(1, len(path.read_text(encoding="utf-8-sig", errors="replace").splitlines())))
    except Exception:
        max_skip = 1
    for skip_header in range(max_skip):
        for delimiter in (",", ";", "\t", None):
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", ConversionWarning)
                    data = np.genfromtxt(
                        path,
                        delimiter=delimiter,
                        names=True,
                        dtype=None,
                        encoding="utf-8-sig",
                        invalid_raise=False,
                        skip_header=skip_header,
                    )
                if (
                    data.dtype.names is not None
                    and len(data.dtype.names) >= 2
                    and _named_table_has_pattern_coordinates(tuple(str(name) for name in data.dtype.names))
                ):
                    return data
            except Exception as exc:
                last_error = exc
    if last_error is not None:
        raise ValueError(f"Could not read element pattern CSV: {last_error}") from last_error
    raise ValueError("Element pattern CSV must include a header row with u/v or theta/phi columns.")


def _float_array_from_column(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values)
    if np.issubdtype(arr.dtype, np.number):
        return np.asarray(arr, dtype=float).ravel()
    parsed: list[float] = []
    for value in arr.ravel():
        if isinstance(value, bytes):
            text = value.decode("utf-8", errors="ignore")
        else:
            text = str(value)
        match = re.search(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?", text)
        parsed.append(float(match.group(0)) if match else float("nan"))
    return np.asarray(parsed, dtype=float)


def _frequency_scale_to_ghz(column_name: str) -> float | None:
    key = _normalize_column_name(column_name)
    if "ghz" in key:
        return 1.0
    if "mhz" in key:
        return 1.0e-3
    if "khz" in key:
        return 1.0e-6
    if "hz" in key:
        return 1.0e-9
    return None


def _frequency_scale_from_text(text: str) -> float | None:
    key = _normalize_column_name(text)
    if "ghz" in key:
        return 1.0
    if "mhz" in key:
        return 1.0e-3
    if "khz" in key:
        return 1.0e-6
    if "hz" in key:
        return 1.0e-9
    return None


def _infer_frequency_scale_to_ghz(values: np.ndarray, target_frequency_ghz: float | None = None) -> float:
    finite = values[np.isfinite(values) & (values > 0.0)]
    if finite.size == 0:
        return 1.0
    token = _frequency_cache_token(target_frequency_ghz)
    if token is not None:
        candidates = (1.0, 1.0e-3, 1.0e-6, 1.0e-9)
        return min(candidates, key=lambda scale: float(np.nanmin(np.abs(finite * scale - token))))
    median = float(np.nanmedian(finite))
    if median >= 1.0e6:
        return 1.0e-9
    if median >= 1.0e3:
        return 1.0e-3
    return 1.0


def _frequency_values_ghz(data: np.ndarray, column_name: str, target_frequency_ghz: float | None = None) -> np.ndarray:
    raw = np.asarray(data[column_name])
    column_scale = _frequency_scale_to_ghz(column_name)
    if not np.issubdtype(raw.dtype, np.number):
        parsed: list[float] = []
        scales: list[float | None] = []
        for value in raw.ravel():
            if isinstance(value, bytes):
                text = value.decode("utf-8", errors="ignore")
            else:
                text = str(value)
            match = re.search(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?", text)
            parsed.append(float(match.group(0)) if match else float("nan"))
            scales.append(_frequency_scale_from_text(text))
        values = np.asarray(parsed, dtype=float)
        if column_scale is not None:
            return values * column_scale
        explicit = np.asarray([scale if scale is not None else np.nan for scale in scales], dtype=float)
        if np.any(np.isfinite(explicit)):
            fallback = _infer_frequency_scale_to_ghz(values, target_frequency_ghz)
            scale_values = np.where(np.isfinite(explicit), explicit, fallback)
            return values * scale_values
        return values * _infer_frequency_scale_to_ghz(values, target_frequency_ghz)

    values = np.asarray(raw, dtype=float).ravel()
    if column_scale is not None:
        return values * column_scale
    return values * _infer_frequency_scale_to_ghz(values, target_frequency_ghz)


def _frequency_cache_token(target_frequency_ghz: float | None) -> float | None:
    if target_frequency_ghz is None:
        return None
    try:
        value = float(target_frequency_ghz)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(value) or value <= 0.0:
        return None
    return round(value, 12)


def _filter_table_by_frequency(
    data: np.ndarray,
    columns: dict[str, str],
    target_frequency_ghz: float | None,
) -> tuple[np.ndarray, float | None, tuple[float, ...]]:
    freq_col = _find_column(columns, FREQUENCY_COLUMN_CANDIDATES)
    if not freq_col:
        return data, None, ()

    values = _frequency_values_ghz(data, freq_col, target_frequency_ghz)
    finite = np.isfinite(values) & (values > 0.0)
    if not np.any(finite):
        return data, None, ()

    unique = np.unique(np.round(values[finite], 12))
    available = tuple(float(value) for value in unique)
    token = _frequency_cache_token(target_frequency_ghz)
    if token is None:
        selected = float(unique[0])
    else:
        selected = float(unique[int(np.argmin(np.abs(unique - token)))])

    tolerance = max(1.0e-9, abs(selected) * 1.0e-9)
    keep = finite & np.isclose(values, selected, rtol=1.0e-9, atol=tolerance)
    if int(np.count_nonzero(keep)) < 3:
        raise ValueError(f"Element pattern CSV has fewer than three samples at selected frequency {selected:.12g} GHz.")
    return data[keep], selected, available


def _regularize_imported_pattern(u: np.ndarray, v: np.ndarray, field: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_points = int(u.size)
    n_axis = int(max(81, min(241, round(math.sqrt(max(n_points, 1)) * 1.35))))
    if n_axis % 2 == 0:
        n_axis += 1
    u_axis = np.linspace(-1.0, 1.0, n_axis)
    v_axis = np.linspace(-1.0, 1.0, n_axis)
    u_grid, v_grid = np.meshgrid(u_axis, v_axis)
    inside = u_grid * u_grid + v_grid * v_grid <= 1.0 + 1.0e-12

    try:
        from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator

        points = np.column_stack([u, v])
        linear_real = LinearNDInterpolator(points, np.real(field), fill_value=np.nan)
        linear_imag = LinearNDInterpolator(points, np.imag(field), fill_value=np.nan)
        real_grid = np.asarray(linear_real(u_grid, v_grid), dtype=float)
        imag_grid = np.asarray(linear_imag(u_grid, v_grid), dtype=float)
        field_grid = real_grid + 1j * imag_grid
        if np.any(inside & ~np.isfinite(field_grid)):
            missing = inside & ~np.isfinite(field_grid)
            nearest_real = NearestNDInterpolator(points, np.real(field))
            nearest_imag = NearestNDInterpolator(points, np.imag(field))
            field_grid[missing] = nearest_real(u_grid[missing], v_grid[missing]) + 1j * nearest_imag(u_grid[missing], v_grid[missing])
    except Exception:
        field_grid = _nearest_complex_grid_fallback(u, v, field, u_grid, v_grid, inside)

    field_grid = np.asarray(field_grid, dtype=complex)
    clean = np.nan_to_num(np.real(field_grid), nan=0.0) + 1j * np.nan_to_num(np.imag(field_grid), nan=0.0)
    field_grid = np.where(inside, clean, 0.0 + 0.0j)
    return u_axis, v_axis, field_grid.astype(complex)


def _nearest_grid_fallback(
    u: np.ndarray,
    v: np.ndarray,
    gain: np.ndarray,
    u_grid: np.ndarray,
    v_grid: np.ndarray,
    inside: np.ndarray,
) -> np.ndarray:
    grid = np.zeros_like(u_grid, dtype=float)
    targets = np.column_stack([u_grid[inside], v_grid[inside]])
    points = np.column_stack([u, v])
    values = np.zeros(targets.shape[0], dtype=float)
    block = 2048
    for start in range(0, targets.shape[0], block):
        stop = min(start + block, targets.shape[0])
        diff = targets[start:stop, None, :] - points[None, :, :]
        idx = np.argmin(np.sum(diff * diff, axis=2), axis=1)
        values[start:stop] = gain[idx]
    grid[inside] = values
    return grid


def _nearest_complex_grid_fallback(
    u: np.ndarray,
    v: np.ndarray,
    field: np.ndarray,
    u_grid: np.ndarray,
    v_grid: np.ndarray,
    inside: np.ndarray,
) -> np.ndarray:
    grid = np.zeros_like(u_grid, dtype=complex)
    targets = np.column_stack([u_grid[inside], v_grid[inside]])
    points = np.column_stack([u, v])
    values = np.zeros(targets.shape[0], dtype=complex)
    block = 2048
    for start in range(0, targets.shape[0], block):
        stop = min(start + block, targets.shape[0])
        diff = targets[start:stop, None, :] - points[None, :, :]
        idx = np.argmin(np.sum(diff * diff, axis=2), axis=1)
        values[start:stop] = field[idx]
    grid[inside] = values
    return grid


def _basis_from_uv(u: np.ndarray, v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    u_arr = np.asarray(u, dtype=float)
    v_arr = np.asarray(v, dtype=float)
    rho = np.sqrt(np.maximum(u_arr * u_arr + v_arr * v_arr, 0.0))
    theta = np.arcsin(np.clip(rho, 0.0, 1.0))
    phi = np.where(rho > 1.0e-12, np.arctan2(v_arr, u_arr), 0.0)
    return theta, phi


def _vector_components_to_cartesian(
    theta: np.ndarray,
    phi: np.ndarray,
    field_theta: np.ndarray,
    field_phi: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    theta_arr = np.asarray(theta, dtype=float)
    phi_arr = np.asarray(phi, dtype=float)
    etheta = np.asarray(field_theta, dtype=complex)
    ephi = np.asarray(field_phi, dtype=complex)
    sin_theta = np.sin(theta_arr)
    cos_theta = np.cos(theta_arr)
    sin_phi = np.sin(phi_arr)
    cos_phi = np.cos(phi_arr)
    field_x = etheta * cos_theta * cos_phi - ephi * sin_phi
    field_y = etheta * cos_theta * sin_phi + ephi * cos_phi
    field_z = -etheta * sin_theta
    return field_x, field_y, field_z


def _cartesian_to_local_components(
    ux: np.ndarray,
    uy: np.ndarray,
    uz: np.ndarray,
    field_x: np.ndarray,
    field_y: np.ndarray,
    field_z: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    ux_arr = np.asarray(ux, dtype=float)
    uy_arr = np.asarray(uy, dtype=float)
    uz_arr = np.asarray(uz, dtype=float)
    fx = np.asarray(field_x, dtype=complex)
    fy = np.asarray(field_y, dtype=complex)
    fz = np.asarray(field_z, dtype=complex)
    rho = np.sqrt(np.maximum(ux_arr * ux_arr + uy_arr * uy_arr, 0.0))
    cos_phi = np.divide(ux_arr, rho, out=np.ones_like(rho, dtype=float), where=rho > 1.0e-12)
    sin_phi = np.divide(uy_arr, rho, out=np.zeros_like(rho, dtype=float), where=rho > 1.0e-12)
    sin_theta = np.clip(rho, 0.0, 1.0)
    cos_theta = uz_arr
    field_theta = fx * cos_theta * cos_phi + fy * cos_theta * sin_phi - fz * sin_theta
    field_phi = -fx * sin_phi + fy * cos_phi
    return field_theta, field_phi


def _merge_duplicate_uv_samples(
    u: np.ndarray,
    v: np.ndarray,
    field: np.ndarray,
    field_theta: np.ndarray | None,
    field_phi: np.ndarray | None,
    field_x: np.ndarray | None = None,
    field_y: np.ndarray | None = None,
    field_z: np.ndarray | None = None,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray | None,
    np.ndarray | None,
    np.ndarray | None,
    np.ndarray | None,
    np.ndarray | None,
]:
    keys = np.column_stack([np.round(u, 12), np.round(v, 12)])
    unique_keys, inverse, counts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
    if unique_keys.shape[0] == u.size:
        return u, v, field, field_theta, field_phi, field_x, field_y, field_z

    n_unique = int(unique_keys.shape[0])
    counts_f = counts.astype(float)
    merged_u = np.zeros(n_unique, dtype=float)
    merged_v = np.zeros(n_unique, dtype=float)
    np.add.at(merged_u, inverse, u)
    np.add.at(merged_v, inverse, v)
    merged_u /= counts_f
    merged_v /= counts_f

    def merge_complex(values: np.ndarray | None) -> np.ndarray | None:
        if values is None:
            return None
        merged = np.zeros(n_unique, dtype=complex)
        np.add.at(merged, inverse, values)
        return merged / counts_f

    return (
        merged_u,
        merged_v,
        merge_complex(field),
        merge_complex(field_theta),
        merge_complex(field_phi),
        merge_complex(field_x),
        merge_complex(field_y),
        merge_complex(field_z),
    )


def _read_imported_pattern_csv(path: Path, target_frequency_ghz: float | None = None) -> ImportedPatternTable:
    data = _read_named_table(path)
    if data.dtype.names is None:
        raise ValueError("Element pattern CSV must include a header row.")
    if data.shape == ():
        data = np.asarray([data], dtype=data.dtype)

    columns = _column_map(data.dtype.names)
    data, selected_frequency_ghz, available_frequencies_ghz = _filter_table_by_frequency(data, columns, target_frequency_ghz)
    u_col = _find_column(columns, U_COLUMN_CANDIDATES)
    v_col = _find_column(columns, V_COLUMN_CANDIDATES)
    theta_rad_col = _find_column(columns, THETA_RAD_COLUMN_CANDIDATES)
    phi_rad_col = _find_column(columns, PHI_RAD_COLUMN_CANDIDATES)
    theta_col = _find_column(columns, THETA_DEG_COLUMN_CANDIDATES)
    phi_col = _find_column(columns, PHI_DEG_COLUMN_CANDIDATES)
    db_col = _find_column(
        columns,
        (
            "gain_db",
            "gain_dbi",
            "pattern_db",
            "directivity_db",
            "realized_gain_db",
            "realized_gain_dbi",
            "value_db",
            "db",
            "db_gain",
            "dbi",
            "dB(GainTotal)",
            "dB(RealizedGainTotal)",
            "dB(Realized Gain Total)",
            "dB(Gain)",
            "dB(Directivity)",
            "dB(DirTotal)",
            "dB(GainTheta)",
            "dB(GainPhi)",
            "dB(GainRHCP)",
            "dB(GainLHCP)",
            "GainTotal[dB]",
            "GainTotal [dB]",
            "RealizedGainTotal[dB]",
            "Realized Gain Total [dB]",
            "Directivity[dB]",
            "DirTotal[dB]",
        ),
    )
    linear_col = _find_column(
        columns,
        (
            "gain_linear",
            "gain",
            "gain_total",
            "gaintotal",
            "realized_gain",
            "realized_gain_total",
            "realizedgaintotal",
            "pattern",
            "directivity",
            "dir_total",
            "dirtotal",
            "power",
            "value",
            "Abs(GainTotal)",
            "Abs(RealizedGainTotal)",
            "Abs(Directivity)",
            "Abs(DirTotal)",
            "Mag(GainTotal)",
            "Mag(RealizedGainTotal)",
            "GainTotal",
            "RealizedGainTotal",
            "DirectivityTotal",
        ),
    )
    mag_col = _find_column(
        columns,
        (
            "field_mag",
            "field_magnitude",
            "e_mag",
            "emag",
            "e_total_mag",
            "etotal_mag",
            "amplitude",
            "amp",
            "magnitude",
            "mag",
            "Abs(ETotal)",
            "Mag(ETotal)",
            "Abs(ETheta)",
            "Abs(EPhi)",
            "Mag(ETheta)",
            "Mag(EPhi)",
        ),
    )
    real_col = _find_column(
        columns,
        (
            "real",
            "re",
            "field_real",
            "e_real",
            "etotal_real",
            "etheta_real",
            "ephi_real",
            "Real(ETotal)",
            "Re(ETotal)",
            "Real(ETheta)",
            "Re(ETheta)",
            "Real(EPhi)",
            "Re(EPhi)",
            "Real(GainTotal)",
            "Re(GainTotal)",
        ),
    )
    imag_col = _find_column(
        columns,
        (
            "imag",
            "im",
            "imaginary",
            "field_imag",
            "e_imag",
            "etotal_imag",
            "etheta_imag",
            "ephi_imag",
            "Imag(ETotal)",
            "Im(ETotal)",
            "Imag(ETheta)",
            "Im(ETheta)",
            "Imag(EPhi)",
            "Im(EPhi)",
            "Imag(GainTotal)",
            "Im(GainTotal)",
        ),
    )
    phase_deg_col = _find_column(
        columns,
        (
            "phase_deg",
            "phase",
            "phase_degrees",
            "phase[deg]",
            "phase deg",
            "phase_total_deg",
            "phase_total",
            "Phase(GainTotal)",
            "Phase(RealizedGainTotal)",
            "Phase(ETotal)",
            "Phase(ETheta)",
            "Phase(EPhi)",
            "Ang(GainTotal)",
            "Angle(GainTotal)",
            "arg",
            "arg_deg",
        ),
    )
    phase_rad_col = _find_column(
        columns,
        (
            "phase_rad",
            "phase[rad]",
            "phase rad",
            "arg_rad",
            "phase_radians",
        ),
    )

    theta_from_columns = None
    phi_from_columns = None
    if theta_rad_col and phi_rad_col:
        theta_from_columns = np.asarray(data[theta_rad_col], dtype=float).ravel()
        phi_from_columns = np.asarray(data[phi_rad_col], dtype=float).ravel()
    elif theta_col and phi_col:
        theta_from_columns = np.deg2rad(np.asarray(data[theta_col], dtype=float).ravel())
        phi_from_columns = np.deg2rad(np.asarray(data[phi_col], dtype=float).ravel())

    if u_col and v_col:
        u = np.asarray(data[u_col], dtype=float).ravel()
        v = np.asarray(data[v_col], dtype=float).ravel()
        if theta_from_columns is not None and phi_from_columns is not None:
            theta_basis = theta_from_columns
            phi_basis = phi_from_columns
        else:
            theta_basis, phi_basis = _basis_from_uv(u, v)
    elif theta_from_columns is not None and phi_from_columns is not None:
        theta_basis = theta_from_columns
        phi_basis = phi_from_columns
        u = np.sin(theta_basis) * np.cos(phi_basis)
        v = np.sin(theta_basis) * np.sin(phi_basis)
    else:
        available = ", ".join(data.dtype.names or ())
        raise ValueError(
            "Element pattern CSV must contain u/v columns or theta/phi columns. "
            f"Available columns: {available}"
        )

    field_theta, theta_has_phase = _component_field_from_columns(data, columns, "etheta")
    field_phi, phi_has_phase = _component_field_from_columns(data, columns, "ephi")
    has_vector_components = field_theta is not None and field_phi is not None
    field_x = None
    field_y = None
    field_z = None
    has_phase = bool(phase_deg_col or phase_rad_col or (real_col and imag_col) or theta_has_phase or phi_has_phase)
    if phase_rad_col:
        phase = np.asarray(data[phase_rad_col], dtype=float).ravel()
    elif phase_deg_col:
        phase = np.deg2rad(np.asarray(data[phase_deg_col], dtype=float).ravel())
    else:
        phase = np.zeros_like(u, dtype=float)

    if has_vector_components:
        field_x, field_y, field_z = _vector_components_to_cartesian(theta_basis, phi_basis, field_theta, field_phi)
        field = np.sqrt(np.maximum(np.abs(field_x) ** 2 + np.abs(field_y) ** 2 + np.abs(field_z) ** 2, 0.0)).astype(complex)
    elif real_col and imag_col:
        field = np.asarray(data[real_col], dtype=float).ravel() + 1j * np.asarray(data[imag_col], dtype=float).ravel()
    elif db_col:
        gain = 10.0 ** (np.asarray(data[db_col], dtype=float).ravel() / 10.0)
        field = np.sqrt(np.maximum(gain, 0.0)) * np.exp(1j * phase)
    elif linear_col:
        gain = np.asarray(data[linear_col], dtype=float).ravel()
        field = np.sqrt(np.maximum(gain, 0.0)) * np.exp(1j * phase)
    elif mag_col:
        mag = np.asarray(data[mag_col], dtype=float).ravel()
        field = np.maximum(mag, 0.0) * np.exp(1j * phase)
    else:
        available = ", ".join(data.dtype.names or ())
        raise ValueError(
            "Element pattern CSV must contain gain, field magnitude/phase, or real/imag columns. "
            f"Available columns: {available}"
        )

    gain = np.abs(field) ** 2
    valid = (
        np.isfinite(u)
        & np.isfinite(v)
        & np.isfinite(np.real(field))
        & np.isfinite(np.imag(field))
        & np.isfinite(gain)
        & (gain >= 0.0)
        & (u * u + v * v <= 1.0 + 1.0e-9)
    )
    if has_vector_components:
        valid &= (
            np.isfinite(np.real(field_theta))
            & np.isfinite(np.imag(field_theta))
            & np.isfinite(np.real(field_phi))
            & np.isfinite(np.imag(field_phi))
            & np.isfinite(np.real(field_x))
            & np.isfinite(np.imag(field_x))
            & np.isfinite(np.real(field_y))
            & np.isfinite(np.imag(field_y))
            & np.isfinite(np.real(field_z))
            & np.isfinite(np.imag(field_z))
        )
    if int(np.count_nonzero(valid)) < 3:
        raise ValueError("Element pattern CSV has fewer than three valid visible-hemisphere samples.")
    u = u[valid]
    v = v[valid]
    field = field[valid]
    if has_vector_components:
        field_theta = field_theta[valid]
        field_phi = field_phi[valid]
        field_x = field_x[valid]
        field_y = field_y[valid]
        field_z = field_z[valid]
    u, v, field, field_theta, field_phi, field_x, field_y, field_z = _merge_duplicate_uv_samples(
        u,
        v,
        field,
        field_theta,
        field_phi,
        field_x,
        field_y,
        field_z,
    )
    if int(u.size) < 3:
        raise ValueError("Element pattern CSV has fewer than three unique visible-hemisphere directions.")
    if has_vector_components:
        field = np.sqrt(np.maximum(np.abs(field_x) ** 2 + np.abs(field_y) ** 2 + np.abs(field_z) ** 2, 0.0)).astype(complex)
        uz_samples = np.sqrt(np.maximum(1.0 - u * u - v * v, 0.0))
        field_theta, field_phi = _cartesian_to_local_components(u, v, uz_samples, field_x, field_y, field_z)
    gain = np.maximum(np.abs(field) ** 2, 0.0)
    if not float(np.nanmax(gain)) > 0.0:
        raise ValueError("Element pattern CSV field/gain data is all zero.")
    rho = np.sqrt(np.maximum(u * u + v * v, 0.0))
    theta_max_deg = float(math.degrees(math.asin(min(1.0, float(np.nanmax(rho))))))
    covers_visible_edge = theta_max_deg >= 89.0

    field_theta_grid = None
    field_phi_grid = None
    field_x_grid = None
    field_y_grid = None
    field_z_grid = None
    if has_vector_components:
        u_axis, v_axis, field_x_grid = _regularize_imported_pattern(u, v, field_x)
        y_u_axis, y_v_axis, field_y_grid = _regularize_imported_pattern(u, v, field_y)
        z_u_axis, z_v_axis, field_z_grid = _regularize_imported_pattern(u, v, field_z)
        if not (np.array_equal(u_axis, y_u_axis) and np.array_equal(v_axis, y_v_axis)):
            raise ValueError("Internal imported Ey grid regularization mismatch.")
        if not (np.array_equal(u_axis, z_u_axis) and np.array_equal(v_axis, z_v_axis)):
            raise ValueError("Internal imported Ez grid regularization mismatch.")
        u_grid, v_grid = np.meshgrid(u_axis, v_axis)
        uz_grid = np.sqrt(np.maximum(1.0 - u_grid * u_grid - v_grid * v_grid, 0.0))
        field_theta_grid, field_phi_grid = _cartesian_to_local_components(
            u_grid,
            v_grid,
            uz_grid,
            field_x_grid,
            field_y_grid,
            field_z_grid,
        )
        field_grid = np.sqrt(
            np.maximum(np.abs(field_x_grid) ** 2 + np.abs(field_y_grid) ** 2 + np.abs(field_z_grid) ** 2, 0.0)
        ).astype(complex)
        gain_grid = np.maximum(np.abs(field_x_grid) ** 2 + np.abs(field_y_grid) ** 2 + np.abs(field_z_grid) ** 2, 0.0)
    else:
        u_axis, v_axis, field_grid = _regularize_imported_pattern(u, v, field)
        gain_grid = np.maximum(np.abs(field_grid) ** 2, 0.0)
    gain_norm = _calc_table_gain_norm(u_axis, v_axis, gain_grid)
    return ImportedPatternTable(
        u_axis=u_axis,
        v_axis=v_axis,
        gain=gain_grid,
        field=field_grid,
        field_theta=field_theta_grid,
        field_phi=field_phi_grid,
        field_x=field_x_grid,
        field_y=field_y_grid,
        field_z=field_z_grid,
        has_vector_components=has_vector_components,
        gain_norm=gain_norm,
        source_path=str(path),
        point_count=int(u.size),
        has_phase=has_phase,
        theta_max_deg=theta_max_deg,
        covers_visible_edge=covers_visible_edge,
        selected_frequency_ghz=selected_frequency_ghz,
        available_frequencies_ghz=available_frequencies_ghz,
    )


def _read_imported_pattern_ffd(path: Path, target_frequency_ghz: float | None = None) -> ImportedPatternTable:
    text = path.read_text(encoding="utf-8-sig", errors="replace")
    numeric_lines = _numeric_lines(text)
    try:
        theta_spec_idx, theta_spec = _find_grid_spec_line(numeric_lines, start=0)
        phi_spec_idx, phi_spec = _find_grid_spec_line(numeric_lines, start=theta_spec_idx + 1)
    except Exception:
        explicit = _try_explicit_numeric_far_field_table(path, numeric_lines, target_frequency_ghz)
        if explicit is not None:
            return explicit
        raise
    theta_start, theta_stop, theta_count = theta_spec
    phi_start, phi_stop, phi_count = phi_spec
    theta_count_i = int(round(theta_count))
    phi_count_i = int(round(phi_count))
    if theta_count_i < 2 or phi_count_i < 2:
        raise ValueError("FFD/FFS theta/phi grid counts must both be >= 2.")
    theta_deg = np.linspace(theta_start, theta_stop, theta_count_i)
    phi_deg = np.linspace(phi_start, phi_stop, phi_count_i)
    point_count = theta_count_i * phi_count_i

    cursor = phi_spec_idx + 1
    freq_count = 1
    if cursor < len(numeric_lines) and len(numeric_lines[cursor][1]) == 1:
        candidate = numeric_lines[cursor][1][0]
        if 1 <= int(round(candidate)) <= 256 and abs(candidate - round(candidate)) < 1.0e-9:
            freq_count = int(round(candidate))
            cursor += 1

    first_data_idx = _find_first_data_line(numeric_lines, cursor, point_count)
    frequency_values = _frequency_values_from_numeric_header(numeric_lines[cursor:first_data_idx], target_frequency_ghz)
    data_rows = [values[:4] for _line_idx, values in numeric_lines[first_data_idx:] if len(values) >= 4]
    if len(data_rows) < point_count:
        raise ValueError(f"FFD/FFS file has {len(data_rows)} field rows, expected at least {point_count}.")
    data = np.asarray(data_rows, dtype=float)
    available_frequencies_ghz: tuple[float, ...] = tuple(float(v) for v in frequency_values[:freq_count]) if frequency_values.size else ()
    selected_frequency_ghz = None
    block_index = 0
    if available_frequencies_ghz:
        target = _frequency_cache_token(target_frequency_ghz)
        if target is not None:
            block_index = int(np.argmin(np.abs(np.asarray(available_frequencies_ghz, dtype=float) - target)))
        selected_frequency_ghz = float(available_frequencies_ghz[block_index])
    max_block = max(1, len(data_rows) // point_count)
    block_index = min(block_index, max_block - 1)
    block = data[block_index * point_count : (block_index + 1) * point_count]
    if block.shape[0] != point_count:
        raise ValueError("FFD/FFS selected frequency block is incomplete.")

    theta_grid, phi_grid = np.meshgrid(theta_deg, phi_deg, indexing="ij")
    theta_rad = np.deg2rad(theta_grid.ravel())
    phi_rad = np.deg2rad(phi_grid.ravel())
    field_theta = block[:, 0] + 1j * block[:, 1]
    field_phi = block[:, 2] + 1j * block[:, 3]
    return _imported_pattern_table_from_far_field_components(
        path=path,
        theta_rad=theta_rad,
        phi_rad=phi_rad,
        field_theta=field_theta,
        field_phi=field_phi,
        has_phase=True,
        selected_frequency_ghz=selected_frequency_ghz,
        available_frequencies_ghz=available_frequencies_ghz,
    )


def _imported_pattern_table_from_far_field_components(
    *,
    path: Path,
    theta_rad: np.ndarray,
    phi_rad: np.ndarray,
    field_theta: np.ndarray,
    field_phi: np.ndarray,
    has_phase: bool,
    selected_frequency_ghz: float | None,
    available_frequencies_ghz: tuple[float, ...],
) -> ImportedPatternTable:
    theta = np.asarray(theta_rad, dtype=float).ravel()
    phi = np.asarray(phi_rad, dtype=float).ravel()
    etheta = np.asarray(field_theta, dtype=complex).ravel()
    ephi = np.asarray(field_phi, dtype=complex).ravel()
    if not (theta.size == phi.size == etheta.size == ephi.size):
        raise ValueError("Far-field component arrays must have the same length.")
    u = np.sin(theta) * np.cos(phi)
    v = np.sin(theta) * np.sin(phi)
    field_x, field_y, field_z = _vector_components_to_cartesian(theta, phi, etheta, ephi)
    valid = (
        np.isfinite(u)
        & np.isfinite(v)
        & np.isfinite(np.real(etheta))
        & np.isfinite(np.imag(etheta))
        & np.isfinite(np.real(ephi))
        & np.isfinite(np.imag(ephi))
        & np.isfinite(np.real(field_x))
        & np.isfinite(np.imag(field_x))
        & np.isfinite(np.real(field_y))
        & np.isfinite(np.imag(field_y))
        & np.isfinite(np.real(field_z))
        & np.isfinite(np.imag(field_z))
        & (u * u + v * v <= 1.0 + 1.0e-9)
    )
    if int(np.count_nonzero(valid)) < 3:
        raise ValueError("FFD/FFS far-field file has fewer than three valid visible-hemisphere samples.")
    u = u[valid]
    v = v[valid]
    field_x = field_x[valid]
    field_y = field_y[valid]
    field_z = field_z[valid]
    field = np.sqrt(np.maximum(np.abs(field_x) ** 2 + np.abs(field_y) ** 2 + np.abs(field_z) ** 2, 0.0)).astype(complex)
    u, v, field, field_theta, field_phi, field_x, field_y, field_z = _merge_duplicate_uv_samples(
        u,
        v,
        field,
        etheta[valid],
        ephi[valid],
        field_x,
        field_y,
        field_z,
    )
    if int(u.size) < 3:
        raise ValueError("FFD/FFS far-field file has fewer than three unique visible-hemisphere directions.")
    field = np.sqrt(np.maximum(np.abs(field_x) ** 2 + np.abs(field_y) ** 2 + np.abs(field_z) ** 2, 0.0)).astype(complex)
    uz_samples = np.sqrt(np.maximum(1.0 - u * u - v * v, 0.0))
    field_theta, field_phi = _cartesian_to_local_components(u, v, uz_samples, field_x, field_y, field_z)
    u_axis, v_axis, field_x_grid = _regularize_imported_pattern(u, v, field_x)
    y_u_axis, y_v_axis, field_y_grid = _regularize_imported_pattern(u, v, field_y)
    z_u_axis, z_v_axis, field_z_grid = _regularize_imported_pattern(u, v, field_z)
    if not (np.array_equal(u_axis, y_u_axis) and np.array_equal(v_axis, y_v_axis)):
        raise ValueError("Internal imported FFD Ey grid regularization mismatch.")
    if not (np.array_equal(u_axis, z_u_axis) and np.array_equal(v_axis, z_v_axis)):
        raise ValueError("Internal imported FFD Ez grid regularization mismatch.")
    u_grid, v_grid = np.meshgrid(u_axis, v_axis)
    uz_grid = np.sqrt(np.maximum(1.0 - u_grid * u_grid - v_grid * v_grid, 0.0))
    field_theta_grid, field_phi_grid = _cartesian_to_local_components(
        u_grid,
        v_grid,
        uz_grid,
        field_x_grid,
        field_y_grid,
        field_z_grid,
    )
    field_grid = np.sqrt(
        np.maximum(np.abs(field_x_grid) ** 2 + np.abs(field_y_grid) ** 2 + np.abs(field_z_grid) ** 2, 0.0)
    ).astype(complex)
    gain_grid = np.maximum(np.abs(field_x_grid) ** 2 + np.abs(field_y_grid) ** 2 + np.abs(field_z_grid) ** 2, 0.0)
    gain_norm = _calc_table_gain_norm(u_axis, v_axis, gain_grid)
    rho = np.sqrt(np.maximum(u * u + v * v, 0.0))
    theta_max_deg = float(math.degrees(math.asin(min(1.0, float(np.nanmax(rho))))))
    return ImportedPatternTable(
        u_axis=u_axis,
        v_axis=v_axis,
        gain=gain_grid,
        field=field_grid,
        field_theta=field_theta_grid,
        field_phi=field_phi_grid,
        field_x=field_x_grid,
        field_y=field_y_grid,
        field_z=field_z_grid,
        has_vector_components=True,
        gain_norm=gain_norm,
        source_path=str(path),
        point_count=int(u.size),
        has_phase=has_phase,
        theta_max_deg=theta_max_deg,
        covers_visible_edge=theta_max_deg >= 89.0,
        selected_frequency_ghz=selected_frequency_ghz,
        available_frequencies_ghz=available_frequencies_ghz,
    )


_NUMERIC_TOKEN_RE = re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?")


def _numeric_lines(text: str) -> list[tuple[int, list[float]]]:
    out: list[tuple[int, list[float]]] = []
    for idx, line in enumerate(text.splitlines()):
        stripped = line.strip()
        if not stripped or stripped.startswith(("#", "!", "//")):
            continue
        values = [float(match.group(0)) for match in _NUMERIC_TOKEN_RE.finditer(stripped)]
        if values:
            out.append((idx, values))
    if not out:
        raise ValueError("FFD/FFS file contains no numeric data.")
    return out


def _find_grid_spec_line(numeric_lines: list[tuple[int, list[float]]], start: int) -> tuple[int, tuple[float, float, float]]:
    for idx in range(start, len(numeric_lines)):
        values = numeric_lines[idx][1]
        if len(values) < 3:
            continue
        count = values[2]
        if abs(count - round(count)) < 1.0e-9 and 2 <= int(round(count)) <= 10000:
            if math.isfinite(values[0]) and math.isfinite(values[1]) and values[0] != values[1]:
                return idx, (float(values[0]), float(values[1]), float(count))
    raise ValueError("FFD/FFS file must include theta and phi grid spec lines: start stop count.")


def _find_first_data_line(numeric_lines: list[tuple[int, list[float]]], start: int, point_count: int) -> int:
    for idx in range(start, len(numeric_lines)):
        if len(numeric_lines[idx][1]) >= 4:
            remaining = sum(1 for _line_idx, values in numeric_lines[idx:] if len(values) >= 4)
            if remaining >= point_count:
                return idx
    raise ValueError("FFD/FFS file does not contain enough Etheta/Ephi field rows.")


def _frequency_values_from_numeric_header(header_lines: list[tuple[int, list[float]]], target_frequency_ghz: float | None) -> np.ndarray:
    values: list[float] = []
    for _line_idx, numbers in header_lines:
        if len(numbers) == 1 and numbers[0] > 0.0:
            values.append(float(numbers[0]))
    if not values:
        return np.asarray([], dtype=float)
    raw = np.asarray(values, dtype=float)
    return raw * _infer_frequency_scale_to_ghz(raw, target_frequency_ghz)


def _try_explicit_numeric_far_field_table(
    path: Path,
    numeric_lines: list[tuple[int, list[float]]],
    target_frequency_ghz: float | None,
) -> ImportedPatternTable | None:
    rows = [values for _line_idx, values in numeric_lines if len(values) >= 6]
    if len(rows) < 3:
        return None
    data = np.asarray([row[:6] for row in rows], dtype=float)
    theta = data[:, 0]
    phi = data[:, 1]
    if not (
        np.nanmin(theta) >= -1.0e-9
        and np.nanmax(theta) <= 180.0 + 1.0e-9
        and np.nanmin(phi) >= -360.0 - 1.0e-9
        and np.nanmax(phi) <= 360.0 + 1.0e-9
    ):
        return None
    del target_frequency_ghz
    return _imported_pattern_table_from_far_field_components(
        path=path,
        theta_rad=np.deg2rad(theta),
        phi_rad=np.deg2rad(phi),
        field_theta=data[:, 2] + 1j * data[:, 3],
        field_phi=data[:, 4] + 1j * data[:, 5],
        has_phase=True,
        selected_frequency_ghz=None,
        available_frequencies_ghz=(),
    )


def load_imported_element_pattern(path: str | Path, target_frequency_ghz: float | None = None) -> ImportedPatternTable:
    source = Path(path).expanduser()
    stat = source.stat()
    return _load_imported_element_pattern_cached(
        str(source.resolve()),
        int(stat.st_mtime_ns),
        int(stat.st_size),
        _frequency_cache_token(target_frequency_ghz),
    )


@lru_cache(maxsize=16)
def _load_imported_element_pattern_cached(path: str, mtime_ns: int, size: int, target_frequency_ghz: float | None) -> ImportedPatternTable:
    del mtime_ns, size
    return _read_imported_pattern_file(Path(path), target_frequency_ghz)


def _read_imported_pattern_file(path: Path, target_frequency_ghz: float | None = None) -> ImportedPatternTable:
    suffix = path.suffix.lower()
    if suffix == ".ffd":
        try:
            return _read_imported_pattern_ffd(path, target_frequency_ghz)
        except Exception as ffd_exc:
            try:
                return _read_imported_pattern_csv(path, target_frequency_ghz)
            except Exception as csv_exc:
                raise ValueError(f"Could not read far-field .ffd file. FFD parser: {ffd_exc}; table parser: {csv_exc}") from ffd_exc
    try:
        return _read_imported_pattern_csv(path, target_frequency_ghz)
    except Exception as csv_exc:
        if suffix in {".ffs", ".ffd", ".txt", ".dat"}:
            try:
                return _read_imported_pattern_ffd(path, target_frequency_ghz)
            except Exception as ffd_exc:
                raise ValueError(f"Could not read far-field file. Table parser: {csv_exc}; FFD/FFS parser: {ffd_exc}") from csv_exc
        raise


@lru_cache(maxsize=64)
def calc_element_gain_norm(mode: str, size_x_m: float, size_y_m: float, wavelength_m: float, q: float) -> float:
    theta = np.linspace(0.0, math.pi / 2.0, 241)
    phi = np.linspace(-math.pi, math.pi, 361)
    phi_grid, theta_grid = np.meshgrid(phi, theta)
    ux = np.sin(theta_grid) * np.cos(phi_grid)
    uy = np.sin(theta_grid) * np.sin(phi_grid)
    uz = np.cos(theta_grid)
    raw = _raw_gain(ux, uy, uz, mode, size_x_m, size_y_m, wavelength_m, q)
    integrand = raw * np.sin(theta_grid)
    int_theta = _trapz(integrand, theta, axis=0)
    int_all = _trapz(int_theta, phi)
    return float(4.0 * math.pi / max(int_all, np.finfo(float).tiny))


def make_element_pattern(params: BeamParams, wavelength_m: float) -> ElementPattern:
    if params.use_element_pattern:
        if getattr(params, "element_pattern_file", ""):
            table = load_imported_element_pattern(params.element_pattern_file, params.frequency_ghz)
            mode = "table"
            gain_norm = table.gain_norm
            table_u_axis = table.u_axis
            table_v_axis = table.v_axis
            table_gain = table.gain
            table_field = table.field
            table_field_theta = table.field_theta
            table_field_phi = table.field_phi
            table_field_x = table.field_x
            table_field_y = table.field_y
            table_field_z = table.field_z
            table_has_vector_components = table.has_vector_components
            table_point_count = table.point_count
            table_has_phase = table.has_phase
            table_theta_max_deg = table.theta_max_deg
            table_covers_visible_edge = table.covers_visible_edge
            table_selected_frequency_ghz = table.selected_frequency_ghz
            source_path = table.source_path
        else:
            mode = element_shape_to_mode(getattr(params, "element_shape", "rectangular"), default="rect")
            gain_norm = calc_element_gain_norm(mode, params.ax_m, params.ay_m, wavelength_m, 1.0)
            table_u_axis = None
            table_v_axis = None
            table_gain = None
            table_field = None
            table_field_theta = None
            table_field_phi = None
            table_field_x = None
            table_field_y = None
            table_field_z = None
            table_has_vector_components = False
            table_point_count = 0
            table_has_phase = False
            table_theta_max_deg = 0.0
            table_covers_visible_edge = False
            table_selected_frequency_ghz = None
            source_path = ""
    else:
        mode = "isotropic"
        gain_norm = 2.0
        table_u_axis = None
        table_v_axis = None
        table_gain = None
        table_field = None
        table_field_theta = None
        table_field_phi = None
        table_field_x = None
        table_field_y = None
        table_field_z = None
        table_has_vector_components = False
        table_point_count = 0
        table_has_phase = False
        table_theta_max_deg = 0.0
        table_covers_visible_edge = False
        table_selected_frequency_ghz = None
        source_path = ""
    return ElementPattern(
        mode=mode,
        size_x_m=params.ax_m,
        size_y_m=params.ay_m,
        wavelength_m=wavelength_m,
        obliquity_q=1.0,
        eta_rad=params.efficiency,
        use_element_pattern=params.use_element_pattern,
        gain_norm=gain_norm,
        table_u_axis=table_u_axis,
        table_v_axis=table_v_axis,
        table_gain=table_gain,
        table_field=table_field,
        table_field_theta=table_field_theta,
        table_field_phi=table_field_phi,
        table_field_x=table_field_x,
        table_field_y=table_field_y,
        table_field_z=table_field_z,
        table_has_vector_components=table_has_vector_components,
        table_point_count=table_point_count,
        table_has_phase=table_has_phase,
        table_theta_max_deg=table_theta_max_deg,
        table_covers_visible_edge=table_covers_visible_edge,
        table_selected_frequency_ghz=table_selected_frequency_ghz,
        source_path=source_path,
    )


def element_response_fast(ux: np.ndarray, uy: np.ndarray, uz: np.ndarray, elem: ElementPattern) -> np.ndarray:
    if not elem.use_element_pattern:
        return np.sqrt(elem.gain_norm) * (np.asarray(uz) > 0.0).astype(complex)
    if elem.mode == "table":
        if elem.table_u_axis is None or elem.table_v_axis is None or elem.table_field is None:
            raise ValueError("Imported element pattern table is not initialized.")
        ux_arr = np.asarray(ux, dtype=float)
        uy_arr = np.asarray(uy, dtype=float)
        uz_arr = np.asarray(uz, dtype=float)
        if elem.table_has_vector_components:
            response_theta, response_phi = element_response_components_fast(ux_arr, uy_arr, uz_arr, elem)
            return np.sqrt(np.maximum(np.abs(response_theta) ** 2 + np.abs(response_phi) ** 2, 0.0)).astype(complex)
        else:
            raw = _interp_field_grid(ux_arr, uy_arr, elem.table_u_axis, elem.table_v_axis, elem.table_field)
        valid = (uz_arr > 0.0) & (ux_arr * ux_arr + uy_arr * uy_arr <= 1.0 + 1.0e-9)
        return np.sqrt(elem.gain_norm) * np.where(valid, raw, 0.0 + 0.0j)
    raw = _raw_gain(
        np.asarray(ux),
        np.asarray(uy),
        np.asarray(uz),
        elem.mode,
        elem.size_x_m,
        elem.size_y_m,
        elem.wavelength_m,
        elem.obliquity_q,
    )
    return np.sqrt(np.maximum(elem.gain_norm * raw, 0.0)).astype(complex)


def element_response_components_fast(ux: np.ndarray, uy: np.ndarray, uz: np.ndarray, elem: ElementPattern) -> tuple[np.ndarray, np.ndarray]:
    if elem.mode == "table" and elem.table_has_vector_components:
        if (
            elem.table_u_axis is None
            or elem.table_v_axis is None
        ):
            raise ValueError("Imported vector element pattern table is not initialized.")
        ux_arr = np.asarray(ux, dtype=float)
        uy_arr = np.asarray(uy, dtype=float)
        uz_arr = np.asarray(uz, dtype=float)
        if elem.table_field_x is not None and elem.table_field_y is not None and elem.table_field_z is not None:
            raw_x = _interp_field_grid(ux_arr, uy_arr, elem.table_u_axis, elem.table_v_axis, elem.table_field_x)
            raw_y = _interp_field_grid(ux_arr, uy_arr, elem.table_u_axis, elem.table_v_axis, elem.table_field_y)
            raw_z = _interp_field_grid(ux_arr, uy_arr, elem.table_u_axis, elem.table_v_axis, elem.table_field_z)
            raw_theta, raw_phi = _cartesian_to_local_components(ux_arr, uy_arr, uz_arr, raw_x, raw_y, raw_z)
        elif elem.table_field_theta is not None and elem.table_field_phi is not None:
            raw_theta = _interp_field_grid(ux_arr, uy_arr, elem.table_u_axis, elem.table_v_axis, elem.table_field_theta)
            raw_phi = _interp_field_grid(ux_arr, uy_arr, elem.table_u_axis, elem.table_v_axis, elem.table_field_phi)
        else:
            raise ValueError("Imported vector element pattern table is not initialized.")
        valid = (uz_arr > 0.0) & (ux_arr * ux_arr + uy_arr * uy_arr <= 1.0 + 1.0e-9)
        scale = np.sqrt(elem.gain_norm)
        return (
            scale * np.where(valid, raw_theta, 0.0 + 0.0j),
            scale * np.where(valid, raw_phi, 0.0 + 0.0j),
        )
    scalar = element_response_fast(ux, uy, uz, elem)
    return scalar, np.zeros_like(scalar, dtype=complex)


def element_gain_fast(ux: np.ndarray, uy: np.ndarray, uz: np.ndarray, elem: ElementPattern) -> np.ndarray:
    if elem.mode == "table" and elem.table_has_vector_components:
        response_theta, response_phi = element_response_components_fast(ux, uy, uz, elem)
        return np.maximum(np.abs(response_theta) ** 2 + np.abs(response_phi) ** 2, 0.0)
    response = element_response_fast(ux, uy, uz, elem)
    return np.maximum(np.abs(response) ** 2, 0.0)


def element_gain_for_angles(theta_deg: np.ndarray, phi_deg: np.ndarray, elem: ElementPattern) -> np.ndarray:
    ux = sind(theta_deg) * cosd(phi_deg)
    uy = sind(theta_deg) * sind(phi_deg)
    uz = cosd(theta_deg)
    return element_gain_fast(ux, uy, uz, elem)


def scan_loss_db_for_direction(u0: float, v0: float, w0: float, elem: ElementPattern) -> float:
    broadside = float(np.asarray(element_gain_fast(np.asarray(0.0), np.asarray(0.0), np.asarray(1.0), elem)))
    current = float(np.asarray(element_gain_fast(np.asarray(u0), np.asarray(v0), np.asarray(w0), elem)))
    return 10.0 * math.log10(max(current, np.finfo(float).tiny) / max(broadside, np.finfo(float).tiny))
