from __future__ import annotations

import math
from typing import Any

import numpy as np


CANONICAL_ELEMENT_SHAPES = ("rectangular", "ellipse", "diamond")
CANONICAL_ARRAY_LAYOUTS = (*CANONICAL_ELEMENT_SHAPES, "custom")

ARRAY_LAYOUT_CHOICES = (
    ("rectangular", "矩形排布"),
    ("ellipse", "圆形/椭圆排布"),
    ("diamond", "菱形排布"),
    ("custom", "导入坐标CSV"),
)
ELEMENT_SHAPE_CHOICES = (
    ("rectangular", "矩形单元"),
    ("ellipse", "圆形/椭圆单元"),
    ("diamond", "菱形单元"),
)

_SHAPE_ALIASES = {
    "rect": "rectangular",
    "rectangle": "rectangular",
    "rectangular": "rectangular",
    "矩形": "rectangular",
    "矩形网格": "rectangular",
    "矩形排布": "rectangular",
    "矩形单元": "rectangular",
    "circle": "ellipse",
    "circular": "ellipse",
    "ellipse": "ellipse",
    "elliptical": "ellipse",
    "圆形": "ellipse",
    "椭圆": "ellipse",
    "圆形/椭圆": "ellipse",
    "圆形/椭圆排布": "ellipse",
    "圆形/椭圆单元": "ellipse",
    "圆形/椭圆裁剪": "ellipse",
    "diamond": "diamond",
    "rhombus": "diamond",
    "菱形": "diamond",
    "菱形排布": "diamond",
    "菱形单元": "diamond",
    "菱形裁剪": "diamond",
    "custom": "custom",
    "imported": "custom",
    "csv": "custom",
    "导入": "custom",
    "导入坐标": "custom",
    "导入坐标csv": "custom",
    "导入坐标CSV": "custom",
    "导入阵元坐标": "custom",
}


def normalize_shape_name(value: str, default: str = "rectangular", *, allow_custom: bool = True) -> str:
    fallback = _SHAPE_ALIASES.get(str(default).strip().lower(), default)
    text = str(value or "").strip()
    key = _SHAPE_ALIASES.get(text.lower())
    if key is None:
        key = _SHAPE_ALIASES.get(text)
    if key is None:
        return fallback
    if key == "custom" and not allow_custom:
        return fallback
    return key


def shape_label(shape: str, *, kind: str = "generic", default: str | None = None) -> str:
    key = normalize_shape_name(shape, default=default or "rectangular", allow_custom=True)
    labels = {
        "generic": {
            "rectangular": "矩形",
            "ellipse": "圆形/椭圆",
            "diamond": "菱形",
            "custom": "导入坐标CSV",
        },
        "array": dict(ARRAY_LAYOUT_CHOICES),
        "element": dict(ELEMENT_SHAPE_CHOICES),
        "element_model": {
            "rectangular": "矩形解析口径",
            "ellipse": "圆形/椭圆解析口径",
            "diamond": "菱形解析口径",
            "custom": "解析口径",
        },
    }.get(kind, {})
    if key in labels:
        return labels[key]
    return str(default if default is not None else shape)


def element_overlap_metric(shape: str, dx: float, dy: float, ax: float, ay: float) -> float:
    key = normalize_shape_name(shape, default="rectangular", allow_custom=False)
    dx_abs = abs(float(dx))
    dy_abs = abs(float(dy))
    ax_safe = max(abs(float(ax)), 1.0e-12)
    ay_safe = max(abs(float(ay)), 1.0e-12)
    if key == "ellipse":
        return (dx_abs / ax_safe) ** 2 + (dy_abs / ay_safe) ** 2
    if key == "diamond":
        return dx_abs / ax_safe + dy_abs / ay_safe
    return max(dx_abs / ax_safe, dy_abs / ay_safe)


def sinc_local(x: Any) -> Any:
    return np.sinc(x)


def element_shape_to_mode(shape: str, default: str = "rect") -> str:
    text = str(shape or "").strip().lower()
    if text in {"isotropic", "iso", "各向同性"}:
        return "isotropic"
    if text in {"cosine", "cos"}:
        return "cosine"
    key = normalize_shape_name(shape, default="rectangular", allow_custom=False)
    if key == "rectangular":
        return "rect"
    if key == "ellipse":
        return "ellipse"
    if key == "diamond":
        return "diamond"
    return default


def j1_approx(x: np.ndarray) -> np.ndarray:
    x_arr = np.asarray(x, dtype=float)
    ax = np.abs(x_arr)
    out = np.empty_like(ax, dtype=float)

    small = ax < 1.0e-8
    mid = (ax < 8.0) & ~small
    large = ~small & ~mid
    out[small] = 0.0

    if np.any(mid):
        xm = ax[mid]
        y = xm * xm
        r = xm * (
            72362614232.0
            + y
            * (
                -7895059235.0
                + y
                * (
                    242396853.1
                    + y * (-2972611.439 + y * (15704.48260 + y * (-30.16036606)))
                )
            )
        )
        s = 144725228442.0 + y * (
            2300535178.0
            + y * (18583304.74 + y * (99447.43394 + y * (376.9991397 + y)))
        )
        out[mid] = r / s

    if np.any(large):
        xl = ax[large]
        z = 8.0 / xl
        y = z * z
        xx = xl - 2.356194491
        p = 1.0 + y * (
            0.00183105
            + y * (-0.00003516396496 + y * (0.000002457520174 + y * (-0.000000240337019)))
        )
        qv = 0.04687499995 + y * (
            -0.0002002690873
            + y * (0.000008449199096 + y * (-0.00000088228987 + y * 0.000000105787412))
        )
        out[large] = np.sqrt(0.636619772 / xl) * (np.cos(xx) * p - z * np.sin(xx) * qv)

    return np.where(x_arr < 0.0, -out, out)


def jinc2(x: np.ndarray) -> np.ndarray:
    x_arr = np.asarray(x, dtype=float)
    out = np.ones_like(x_arr, dtype=float)
    mask = np.abs(x_arr) > 1.0e-8
    out[mask] = 2.0 * j1_approx(x_arr[mask]) / x_arr[mask]
    return out


def raw_aperture_gain(
    ux: np.ndarray,
    uy: np.ndarray,
    uz: np.ndarray,
    mode: str,
    size_x_m: float,
    size_y_m: float,
    wavelength_m: float,
    q: float,
) -> np.ndarray:
    valid = uz > 0.0
    mode = mode.lower()
    if mode == "isotropic":
        raw = valid.astype(float)
    elif mode == "cosine":
        raw = np.maximum(uz, 0.0) ** q
    elif mode == "rect":
        fx = sinc_local(size_x_m / wavelength_m * ux)
        fy = sinc_local(size_y_m / wavelength_m * uy)
        raw = np.abs(fx * fy) ** 2 * np.maximum(uz, 0.0) ** q
        raw = np.where(valid, raw, 0.0)
    elif mode == "ellipse":
        arg = math.pi * np.sqrt((size_x_m / wavelength_m * ux) ** 2 + (size_y_m / wavelength_m * uy) ** 2)
        raw = np.abs(jinc2(arg)) ** 2 * np.maximum(uz, 0.0) ** q
        raw = np.where(valid, raw, 0.0)
    elif mode == "diamond":
        fp = sinc_local((size_x_m * ux + size_y_m * uy) / (2.0 * wavelength_m))
        fm = sinc_local((size_x_m * ux - size_y_m * uy) / (2.0 * wavelength_m))
        raw = np.abs(fp * fm) ** 2 * np.maximum(uz, 0.0) ** q
        raw = np.where(valid, raw, 0.0)
    else:
        raise ValueError(f"Unknown element pattern mode: {mode}")
    return raw


def analytic_scan_loss_db(
    mode: str,
    size_x_m: float,
    size_y_m: float,
    wavelength_m: float,
    u0: float,
    v0: float,
    w0: float,
    q: float = 1.0,
) -> float:
    broadside = float(
        np.asarray(
            raw_aperture_gain(
                np.asarray(0.0),
                np.asarray(0.0),
                np.asarray(1.0),
                mode,
                size_x_m,
                size_y_m,
                wavelength_m,
                q,
            )
        )
    )
    current = float(
        np.asarray(
            raw_aperture_gain(
                np.asarray(u0),
                np.asarray(v0),
                np.asarray(w0),
                mode,
                size_x_m,
                size_y_m,
                wavelength_m,
                q,
            )
        )
    )
    return 10.0 * math.log10(max(current, np.finfo(float).tiny) / max(broadside, np.finfo(float).tiny))
