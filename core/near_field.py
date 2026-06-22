from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from io import StringIO
from pathlib import Path
from typing import Any
import warnings

import numpy as np
from numpy.lib._iotools import ConversionWarning

C0 = 3.0e8


@dataclass(frozen=True)
class ImportedNearFieldTable:
    x_m: np.ndarray
    y_m: np.ndarray
    z_m: np.ndarray
    ex: np.ndarray
    ey: np.ndarray
    ez: np.ndarray
    s_w_m2: np.ndarray | None
    source_path: str
    point_count: int
    has_phase: bool
    has_vector_components: bool
    has_power_density: bool
    selected_frequency_ghz: float | None = None
    available_frequencies_ghz: tuple[float, ...] = ()


def load_imported_element_near_field(
    path: str | Path,
    target_frequency_ghz: float | None = None,
) -> ImportedNearFieldTable:
    source = Path(path).expanduser()
    stat = source.stat()
    return _load_imported_element_near_field_cached(
        str(source.resolve()),
        int(stat.st_mtime_ns),
        int(stat.st_size),
        _frequency_cache_token(target_frequency_ghz),
    )


@lru_cache(maxsize=12)
def _load_imported_element_near_field_cached(
    path: str,
    mtime_ns: int,
    size: int,
    target_frequency_ghz: float | None,
) -> ImportedNearFieldTable:
    del mtime_ns, size
    return _read_imported_near_field_csv(Path(path), target_frequency_ghz)


def _read_imported_near_field_csv(path: Path, target_frequency_ghz: float | None) -> ImportedNearFieldTable:
    data = _read_named_table_with_required_columns(path, required_any=("x", "xm", "xmm"))
    if data.shape == ():
        data = np.asarray([data], dtype=data.dtype)
    columns = _column_map(data.dtype.names or ())
    data, selected_frequency_ghz, available_frequencies_ghz = _filter_table_by_frequency(
        data,
        columns,
        target_frequency_ghz,
    )

    x_col = _find_column(columns, ("x_m", "x", "X [m]", "X(m)", "x_mm", "X [mm]", "x[mm]"))
    y_col = _find_column(columns, ("y_m", "y", "Y [m]", "Y(m)", "y_mm", "Y [mm]", "y[mm]"))
    z_col = _find_column(columns, ("z_m", "z", "Z [m]", "Z(m)", "z_mm", "Z [mm]", "z[mm]"))
    if not (x_col and y_col and z_col):
        available = ", ".join(data.dtype.names or ())
        raise ValueError(f"Near-field CSV must contain x/y/z coordinate columns. Available columns: {available}")

    x = _coordinate_values(data, x_col)
    y = _coordinate_values(data, y_col)
    z = _coordinate_values(data, z_col)

    ex, ex_phase = _component_complex(data, columns, "x")
    ey, ey_phase = _component_complex(data, columns, "y")
    ez, ez_phase = _component_complex(data, columns, "z")
    has_vector = ex is not None or ey is not None or ez is not None
    if ex is None:
        ex = np.zeros_like(x, dtype=complex)
    if ey is None:
        ey = np.zeros_like(x, dtype=complex)
    if ez is None:
        ez = np.zeros_like(x, dtype=complex)
    has_phase = bool(ex_phase or ey_phase or ez_phase)

    s_w_m2 = _power_density_values(data, columns)
    has_power_density = s_w_m2 is not None
    if not has_vector and not has_power_density:
        raise ValueError(
            "Near-field CSV must contain complex/vector E-field columns such as Real(Ex)/Imag(Ex), "
            "Abs(Ex)/Phase(Ex), or scalar power-density columns such as S_W_m2."
        )

    valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    valid &= np.isfinite(np.real(ex)) & np.isfinite(np.imag(ex))
    valid &= np.isfinite(np.real(ey)) & np.isfinite(np.imag(ey))
    valid &= np.isfinite(np.real(ez)) & np.isfinite(np.imag(ez))
    if s_w_m2 is not None:
        valid &= np.isfinite(s_w_m2) & (s_w_m2 >= 0.0)
    if int(np.count_nonzero(valid)) < 1:
        raise ValueError("Near-field CSV has no finite valid samples.")

    x = x[valid].astype(float)
    y = y[valid].astype(float)
    z = z[valid].astype(float)
    ex = ex[valid].astype(complex)
    ey = ey[valid].astype(complex)
    ez = ez[valid].astype(complex)
    if s_w_m2 is not None:
        s_w_m2 = s_w_m2[valid].astype(float)

    if not np.any(z > 0.0):
        raise ValueError("Near-field CSV samples must include z > 0 points in front of the element aperture.")
    if has_vector and not np.any(np.abs(ex) + np.abs(ey) + np.abs(ez) > 0.0):
        raise ValueError("Near-field CSV vector E-field columns are all zero.")
    if s_w_m2 is not None and not np.any(s_w_m2 > 0.0):
        raise ValueError("Near-field CSV power-density column is all zero.")

    return ImportedNearFieldTable(
        x_m=x,
        y_m=y,
        z_m=z,
        ex=ex,
        ey=ey,
        ez=ez,
        s_w_m2=s_w_m2,
        source_path=str(path),
        point_count=int(x.size),
        has_phase=has_phase,
        has_vector_components=has_vector,
        has_power_density=has_power_density,
        selected_frequency_ghz=selected_frequency_ghz,
        available_frequencies_ghz=available_frequencies_ghz,
    )


def near_field_summary(table: ImportedNearFieldTable) -> dict[str, float | int | bool | None]:
    return {
        "point_count": table.point_count,
        "has_phase": table.has_phase,
        "has_vector_components": table.has_vector_components,
        "has_power_density": table.has_power_density,
        "selected_frequency_ghz": table.selected_frequency_ghz,
        "x_span_m": _span(table.x_m),
        "y_span_m": _span(table.y_m),
        "z_min_m": float(np.nanmin(table.z_m)),
        "z_max_m": float(np.nanmax(table.z_m)),
    }


def export_element_near_field_vector_template(path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    rows = ["Frequency [GHz],x_m,y_m,z_m,Real(Ex),Imag(Ex),Real(Ey),Imag(Ey),Real(Ez),Imag(Ez)"]
    for z in (0.05, 0.10, 0.20):
        for y in (-0.05, 0.0, 0.05):
            for x in (-0.05, 0.0, 0.05):
                r2 = x * x + y * y + z * z
                amp = 1.0 / max(r2, 1.0e-6)
                phase = 20.0 * x - 12.0 * y
                phasor = np.exp(1j * np.deg2rad(phase))
                ex = amp * phasor
                ey = 0.25 * amp * np.exp(1j * np.deg2rad(phase - 35.0))
                ez = 0.05 * amp * np.exp(1j * np.deg2rad(phase + 18.0))
                rows.append(
                    f"10,{x:.8g},{y:.8g},{z:.8g},"
                    f"{ex.real:.12g},{ex.imag:.12g},{ey.real:.12g},{ey.imag:.12g},{ez.real:.12g},{ez.imag:.12g}"
                )
    target.write_text("\n".join(rows) + "\n", encoding="utf-8")


def export_near_field_projected_far_field_pattern(
    near_field_path: str | Path,
    output_path: str | Path,
    *,
    frequency_ghz: float,
    theta_deg: np.ndarray | None = None,
    phi_deg: np.ndarray | None = None,
) -> dict[str, float | int | str | None]:
    """Export an approximate far-field element-pattern CSV from sampled near-field data.

    The main solver consumes far-field element-pattern CSVs. This helper bridges
    imported full-wave near-field samples into that established path by applying a
    sampled Fraunhofer projection. It is intentionally an offline conversion tool:
    the resulting CSV should be reviewed and then imported as the active far-field
    element pattern.
    """

    table = load_imported_element_near_field(near_field_path, frequency_ghz)
    if not table.has_vector_components:
        raise ValueError("Near-field to far-field export requires vector E-field columns Ex/Ey/Ez.")
    if not float(frequency_ghz) > 0.0:
        raise ValueError("frequency_ghz must be > 0 for near-field to far-field projection.")

    if theta_deg is None:
        theta_deg = np.linspace(0.0, 85.0, 18)
    if phi_deg is None:
        phi_deg = np.linspace(-180.0, 180.0, 49)
    theta = np.asarray(theta_deg, dtype=float).ravel()
    phi = np.asarray(phi_deg, dtype=float).ravel()
    if theta.size < 2 or phi.size < 2:
        raise ValueError("Near-field to far-field export requires at least two theta and phi samples.")
    if not np.all(np.isfinite(theta)) or not np.all(np.isfinite(phi)):
        raise ValueError("Near-field to far-field theta/phi grids must be finite.")
    if np.any(theta < 0.0) or np.any(theta > 89.999):
        raise ValueError("Near-field to far-field theta grid must stay within [0, 89.999] degrees.")

    selected = _select_projection_plane(table)
    x = table.x_m[selected]
    y = table.y_m[selected]
    z = table.z_m[selected]
    ex = table.ex[selected]
    ey = table.ey[selected]
    ez = table.ez[selected]
    if x.size < 3:
        raise ValueError("Selected near-field plane has fewer than three vector samples.")

    weights = _sample_area_weights(x, y)
    k = 2.0 * np.pi * (float(frequency_ghz) * 1.0e9) / C0
    rows = ["Theta [deg],Phi [deg],Real(ETheta),Imag(ETheta),Real(EPhi),Imag(EPhi)"]
    max_abs = 0.0
    for theta_value in theta:
        th = np.deg2rad(float(theta_value))
        sin_th = float(np.sin(th))
        cos_th = float(np.cos(th))
        for phi_value in phi:
            ph = np.deg2rad(float(phi_value))
            cos_ph = float(np.cos(ph))
            sin_ph = float(np.sin(ph))
            u = sin_th * cos_ph
            v = sin_th * sin_ph
            w = cos_th
            phase = np.exp(1j * k * (x * u + y * v + z * w))
            fx = np.sum(ex * phase * weights)
            fy = np.sum(ey * phase * weights)
            fz = np.sum(ez * phase * weights)
            e_theta = fx * cos_th * cos_ph + fy * cos_th * sin_ph - fz * sin_th
            e_phi = -fx * sin_ph + fy * cos_ph
            max_abs = max(max_abs, float(abs(e_theta)), float(abs(e_phi)))
            rows.append(
                f"{theta_value:.10g},{phi_value:.10g},"
                f"{e_theta.real:.12g},{e_theta.imag:.12g},{e_phi.real:.12g},{e_phi.imag:.12g}"
            )
    if not max_abs > 0.0:
        raise ValueError("Near-field projection produced all-zero far-field samples.")

    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return {
        "source_path": str(Path(near_field_path).expanduser()),
        "output_path": str(target),
        "selected_frequency_ghz": table.selected_frequency_ghz,
        "frequency_ghz": float(frequency_ghz),
        "source_point_count": int(table.point_count),
        "projected_point_count": int(x.size),
        "theta_count": int(theta.size),
        "phi_count": int(phi.size),
        "output_rows": int(theta.size * phi.size),
        "plane_z_m": float(np.nanmedian(z)),
    }


def _select_projection_plane(table: ImportedNearFieldTable) -> np.ndarray:
    z = np.asarray(table.z_m, dtype=float).ravel()
    rounded = np.round(z, decimals=9)
    unique, counts = np.unique(rounded, return_counts=True)
    if unique.size == 0:
        raise ValueError("Near-field table has no z samples.")
    max_count = int(np.max(counts))
    candidates = unique[counts == max_count]
    selected_z = float(candidates[int(np.argmax(candidates))])
    return np.isclose(rounded, selected_z, rtol=0.0, atol=1.0e-12)


def _sample_area_weights(x_m: np.ndarray, y_m: np.ndarray) -> np.ndarray:
    x = np.asarray(x_m, dtype=float).ravel()
    y = np.asarray(y_m, dtype=float).ravel()
    dx = _median_positive_spacing(x)
    dy = _median_positive_spacing(y)
    area = max(dx * dy, 1.0)
    return np.full(x.shape, area, dtype=float)


def _median_positive_spacing(values: np.ndarray) -> float:
    unique = np.unique(np.round(np.asarray(values, dtype=float).ravel(), decimals=12))
    if unique.size < 2:
        return 1.0
    diff = np.diff(np.sort(unique))
    diff = diff[np.isfinite(diff) & (diff > 0.0)]
    if diff.size == 0:
        return 1.0
    return float(np.nanmedian(diff))


def _read_named_table_with_required_columns(path: Path, required_any: tuple[str, ...]) -> np.ndarray:
    text = path.read_text(encoding="utf-8-sig")
    lines = text.splitlines()
    last_error: Exception | None = None
    required = {key.lower() for key in required_any}
    for start in range(min(len(lines), 80)):
        candidate = "\n".join(lines[start:])
        if not candidate.strip():
            continue
        for delimiter in (",", ";", "\t", None):
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", ConversionWarning)
                    data = np.genfromtxt(
                        StringIO(candidate),
                        delimiter=delimiter,
                        names=True,
                        dtype=None,
                        encoding="utf-8-sig",
                        invalid_raise=False,
                    )
                names = data.dtype.names
                if names is None or len(names) < 4:
                    continue
                keys = set(_normalize_column_name(name) for name in names)
                if keys.intersection(required):
                    return data
            except Exception as exc:
                last_error = exc
    if last_error is not None:
        raise ValueError(f"Could not read near-field CSV: {last_error}") from last_error
    raise ValueError("Near-field CSV must include a header row with x/y/z and field columns.")


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


def _coordinate_values(data: np.ndarray, column: str) -> np.ndarray:
    values = np.asarray(data[column], dtype=float).ravel()
    name = str(column).lower()
    key = _normalize_column_name(column)
    if "mm" in name or key.endswith("mm"):
        return values / 1000.0
    return values


def _component_complex(data: np.ndarray, columns: dict[str, str], axis: str) -> tuple[np.ndarray | None, bool]:
    axis_upper = axis.upper()
    real_col = _find_column(
        columns,
        (
            f"real_e{axis}",
            f"e{axis}_real",
            f"re_e{axis}",
            f"Real(E{axis_upper})",
            f"Re(E{axis_upper})",
            f"RealE{axis_upper}",
        ),
    )
    imag_col = _find_column(
        columns,
        (
            f"imag_e{axis}",
            f"e{axis}_imag",
            f"im_e{axis}",
            f"Imag(E{axis_upper})",
            f"Im(E{axis_upper})",
            f"ImagE{axis_upper}",
        ),
    )
    if real_col and imag_col:
        return (
            np.asarray(data[real_col], dtype=float).ravel() + 1j * np.asarray(data[imag_col], dtype=float).ravel(),
            True,
        )

    mag_col = _find_column(
        columns,
        (
            f"abs_e{axis}",
            f"mag_e{axis}",
            f"e{axis}_abs",
            f"e{axis}_mag",
            f"Abs(E{axis_upper})",
            f"Mag(E{axis_upper})",
        ),
    )
    phase_deg_col = _find_column(
        columns,
        (
            f"phase_e{axis}",
            f"e{axis}_phase",
            f"phase_e{axis}_deg",
            f"Phase(E{axis_upper})",
            f"PhaseE{axis_upper}",
        ),
    )
    phase_rad_col = _find_column(
        columns,
        (
            f"phase_e{axis}_rad",
            f"e{axis}_phase_rad",
            f"PhaseRad(E{axis_upper})",
        ),
    )
    if mag_col:
        mag = np.asarray(data[mag_col], dtype=float).ravel()
        if phase_rad_col:
            phase = np.asarray(data[phase_rad_col], dtype=float).ravel()
            return mag * np.exp(1j * phase), True
        if phase_deg_col:
            phase = np.deg2rad(np.asarray(data[phase_deg_col], dtype=float).ravel())
            return mag * np.exp(1j * phase), True
        return mag.astype(complex), False
    return None, False


def _power_density_values(data: np.ndarray, columns: dict[str, str]) -> np.ndarray | None:
    s_col = _find_column(
        columns,
        (
            "s_w_m2",
            "s_wpm2",
            "power_density_w_m2",
            "powerdensitywm2",
            "S(W/m^2)",
            "S [W/m^2]",
            "poynting_w_m2",
        ),
    )
    if s_col:
        return np.asarray(data[s_col], dtype=float).ravel()
    s_cm2_col = _find_column(
        columns,
        (
            "s_w_cm2",
            "s_wpcm2",
            "power_density_w_cm2",
            "S(W/cm^2)",
            "S [W/cm^2]",
        ),
    )
    if s_cm2_col:
        return np.asarray(data[s_cm2_col], dtype=float).ravel() * 1.0e4
    return None


def _filter_table_by_frequency(
    data: np.ndarray,
    columns: dict[str, str],
    target_frequency_ghz: float | None,
) -> tuple[np.ndarray, float | None, tuple[float, ...]]:
    freq_col = _find_column(
        columns,
        (
            "Frequency [GHz]",
            "Frequency [MHz]",
            "Frequency [Hz]",
            "Freq [GHz]",
            "Freq [MHz]",
            "Freq [Hz]",
            "frequency",
            "freq",
        ),
    )
    if not freq_col:
        return data, None, ()
    freq = _parse_frequency_column(np.asarray(data[freq_col]).ravel(), freq_col)
    finite = np.isfinite(freq)
    if not np.any(finite):
        return data, None, ()
    available = np.unique(np.round(freq[finite], decimals=12))
    if available.size == 0:
        return data, None, ()
    if target_frequency_ghz is None:
        selected = float(available[0])
    else:
        selected = float(available[int(np.argmin(np.abs(available - float(target_frequency_ghz))))])
    mask = finite & np.isclose(freq, selected, rtol=0.0, atol=max(abs(selected) * 1.0e-9, 1.0e-12))
    filtered = data[mask]
    if filtered.shape == ():
        filtered = np.asarray([filtered], dtype=data.dtype)
    return filtered, selected, tuple(float(v) for v in available)


def _parse_frequency_column(values: np.ndarray, column_name: str) -> np.ndarray:
    parsed = np.full(values.size, np.nan, dtype=float)
    for idx, value in enumerate(values):
        parsed[idx] = _parse_frequency_value(value)
    key = _normalize_column_name(column_name)
    if "mhz" in key:
        return parsed / 1000.0
    if "hz" in key and "ghz" not in key and "mhz" not in key:
        return parsed / 1.0e9
    finite = np.isfinite(parsed)
    if not np.any(finite):
        return parsed
    median = float(np.nanmedian(np.abs(parsed[finite])))
    if median > 1.0e7:
        return parsed / 1.0e9
    if median > 1.0e3:
        return parsed / 1000.0
    return parsed


def _parse_frequency_value(value: Any) -> float:
    text = str(value).strip().lower()
    if not text:
        return float("nan")
    multiplier = 1.0
    if "ghz" in text:
        multiplier = 1.0
    elif "mhz" in text:
        multiplier = 1.0e-3
    elif "khz" in text:
        multiplier = 1.0e-6
    elif "hz" in text:
        multiplier = 1.0e-9
    clean = "".join(ch if (ch.isdigit() or ch in ".+-eE") else " " for ch in text)
    parts = clean.split()
    if not parts:
        return float("nan")
    try:
        return float(parts[0]) * multiplier
    except ValueError:
        return float("nan")


def _frequency_cache_token(value: float | None) -> float | None:
    if value is None:
        return None
    return round(float(value), 12)


def _span(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=float)
    return float(np.nanmax(arr) - np.nanmin(arr)) if arr.size else 0.0
