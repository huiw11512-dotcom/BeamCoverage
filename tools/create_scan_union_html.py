from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import plotly.graph_objects as go


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app_info import APP_SCAN_UNION_HTML_NAME, APP_SCAN_UNION_HTML_TITLE

OUT = ROOT / "dist" / APP_SCAN_UNION_HTML_NAME


def main() -> None:
    from core.element_pattern import make_element_pattern
    from core.geometry import BeamParams, derive_params, get_mode_settings
    from core.scan_union import compute_scan_union_envelope_3d

    params = BeamParams(
        frequency_ghz=5.6,
        nx=2,
        ny=8,
        dx_m=0.7,
        dy_m=0.1,
        ax_m=0.7,
        ay_m=0.1,
        efficiency=0.7,
        element_power_w=1_000_000.0,
        s0_w_cm2=20.0,
        scan_x_deg=0.0,
        scan_y_deg=0.0,
        scan_limit_mode="auto",
        manual_scan_limit_x_deg=8.0,
        manual_scan_limit_y_deg=8.0,
        calc_mode="standard",
        use_element_pattern=True,
    )
    params, derived = derive_params(params)
    settings = get_mode_settings("standard")
    elem = make_element_pattern(params, derived.wavelength_m)
    union = compute_scan_union_envelope_3d(params, derived, elem, settings)

    x = np.asarray(union["Xsurf"], dtype=float)
    y = np.asarray(union["Ysurf"], dtype=float)
    z = np.asarray(union["Zsurf"], dtype=float)
    r = np.asarray(union["Rsurf"], dtype=float)
    finite = np.isfinite(x) & np.isfinite(y) & np.isfinite(z) & np.isfinite(r) & (r > 0.0)
    x = np.where(finite, x, np.nan)
    y = np.where(finite, y, np.nan)
    z = np.where(finite, z, np.nan)
    r = np.where(finite, r, np.nan)

    max_range = float(union["maxRange_m"])
    title = (
        APP_SCAN_UNION_HTML_TITLE
        +
        f"<br><sup>f={params.frequency_ghz:g} GHz, {params.nx}x{params.ny}, "
        f"scan X=+/-{union['scanLimitX_deg']:.3g} deg, "
        f"scan Y=+/-{union['scanLimitY_deg']:.3g} deg, "
        f"centers={union['numScanCenters']}, max r={max_range:.4g} m</sup>"
    )

    surface = go.Surface(
        x=x,
        y=y,
        z=z,
        surfacecolor=r,
        colorscale="Viridis",
        colorbar={"title": "r (m)"},
        cmin=0.0,
        cmax=max_range,
        hovertemplate=(
            "x=%{x:.3g} m<br>y=%{y:.3g} m<br>z=%{z:.3g} m<br>"
            "r=%{surfacecolor:.3g} m<extra></extra>"
        ),
        contours={
            "x": {"show": True, "color": "rgba(255,255,255,0.35)", "width": 1},
            "y": {"show": True, "color": "rgba(255,255,255,0.35)", "width": 1},
            "z": {"show": True, "color": "rgba(255,255,255,0.35)", "width": 1},
        },
    )
    fig = go.Figure(data=[surface])
    lim = max(1.0, max_range)
    fig.update_layout(
        title=title,
        margin={"l": 0, "r": 0, "t": 70, "b": 0},
        scene={
            "xaxis": {"title": "x (m)", "range": [-lim, lim], "backgroundcolor": "rgb(248,248,248)"},
            "yaxis": {"title": "y (m)", "range": [-lim, lim], "backgroundcolor": "rgb(248,248,248)"},
            "zaxis": {"title": "z (m)", "range": [0, lim], "backgroundcolor": "rgb(248,248,248)"},
            "aspectmode": "cube",
            "camera": {"eye": {"x": 1.55, "y": -1.55, "z": 1.05}},
        },
        annotations=[
            {
                "text": "Drag to rotate, mouse wheel to zoom, double click to reset.",
                "showarrow": False,
                "xref": "paper",
                "yref": "paper",
                "x": 0.01,
                "y": 0.01,
                "font": {"size": 12, "color": "#555"},
            }
        ],
    )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(OUT, include_plotlyjs=True, full_html=True)
    print(OUT)


if __name__ == "__main__":
    main()
