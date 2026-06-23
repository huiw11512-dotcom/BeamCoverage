from __future__ import annotations

import math
import os
from pathlib import Path

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QLabel, QSplitter, QTextBrowser, QTabWidget, QWidget, QVBoxLayout

from matplotlib import rcParams
from matplotlib import colormaps
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.patches import Ellipse, Polygon, Rectangle
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

from app_info import resource_path
from core.envelope import append_far_field_for_plot
from core.geometry import BeamParams, DerivedParams, cosd, sind

try:
    import pyqtgraph.opengl as gl
    from pyqtgraph.opengl import MeshData

    _HAS_OPENGL = True
except Exception:
    gl = None
    MeshData = None
    _HAS_OPENGL = False


rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Arial Unicode MS", "DejaVu Sans"]
rcParams["axes.unicode_minus"] = False


class FigureTab(QWidget):
    def __init__(self, projection: str | None = None, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.figure = Figure(figsize=(7, 5))
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)
        self.projection = projection
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)


class OpenGLSceneWidget(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.opengl_enabled = bool(
            _HAS_OPENGL
            and gl is not None
            and MeshData is not None
            and os.environ.get("QT_QPA_PLATFORM", "").lower() not in {"offscreen", "minimal"}
        )
        self.info = QLabel("")
        self.info.setWordWrap(True)
        self.info.setStyleSheet("QLabel { color: #26384f; padding: 6px 8px; font-weight: 600; }")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self.info)
        self.items: list[object] = []
        if self.opengl_enabled:
            self.view = gl.GLViewWidget()
            self.view.setBackgroundColor("w")
            layout.addWidget(self.view, 1)
        else:
            self.view = None
            fallback = QLabel("OpenGL view is unavailable in this environment.")
            fallback.setAlignment(Qt.AlignmentFlag.AlignCenter)
            fallback.setStyleSheet("QLabel { color: #607086; background: white; border: 1px solid #d8e0ea; }")
            layout.addWidget(fallback, 1)

    def clear_scene(self, text: str = "") -> bool:
        self.info.setText(text)
        if not self.opengl_enabled or self.view is None:
            return False
        for item in self.items:
            try:
                self.view.removeItem(item)
            except Exception:
                pass
        self.items.clear()
        return True

    def add_item(self, item: object) -> None:
        if not self.opengl_enabled or self.view is None:
            return
        self.view.addItem(item)
        self.items.append(item)

    def export_png(self, path: str | Path) -> None:
        if self.opengl_enabled and self.view is not None:
            pixmap = self.grab()
            if not pixmap.save(str(path)):
                raise RuntimeError(f"Failed to save OpenGL view to {path}")
            return
        raise RuntimeError("OpenGL view is unavailable.")


class GL3DTab(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.figure = Figure(figsize=(7, 5))
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)
        self.scene = OpenGLSceneWidget()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        if self.scene.opengl_enabled:
            self.toolbar.hide()
            self.canvas.hide()
            layout.addWidget(self.scene)
        else:
            layout.addWidget(self.toolbar)
            layout.addWidget(self.canvas)

    def set_placeholder(self, text: str) -> None:
        if self.scene.opengl_enabled:
            self.scene.clear_scene(text)
            return
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.axis("off")
        ax.text(0.5, 0.5, text, ha="center", va="center", fontsize=12)
        self.canvas.draw_idle()

    def export_png(self, path: str | Path) -> None:
        if self.scene.opengl_enabled:
            self.scene.export_png(path)
        else:
            export_figure = self.figure
            export_figure.savefig(path, dpi=220)


class ScanUnionTab(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.figure = Figure(figsize=(5, 5))
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.addWidget(self.toolbar)
        left_layout.addWidget(self.canvas)
        self.scene = OpenGLSceneWidget()
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(left)
        splitter.addWidget(self.scene)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([430, 900])
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(splitter)

    def set_placeholder(self, text: str) -> None:
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.axis("off")
        ax.text(0.5, 0.5, text, ha="center", va="center", fontsize=12)
        self.canvas.draw_idle()
        self.scene.clear_scene(text)

    def export_png(self, path: str | Path) -> None:
        if self.scene.opengl_enabled:
            self.scene.export_png(path)
        else:
            self.figure.savefig(path, dpi=220)


class ReleaseNotesTab(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.browser = QTextBrowser()
        self.browser.setOpenExternalLinks(True)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.addWidget(self.browser)
        self.reload()

    def reload(self) -> None:
        changelog = resource_path("CHANGELOG.md")
        if not changelog.exists():
            changelog = Path(__file__).resolve().parents[1] / "CHANGELOG.md"
        try:
            text = changelog.read_text(encoding="utf-8")
        except OSError:
            text = "# BeamCoverage 更新说明\n\n当前安装包未包含 CHANGELOG.md。"
        self.browser.setMarkdown(text)


class PlotPanel(QTabWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.structure_tab = FigureTab()
        self.cuts_tab = FigureTab()
        self.envelope_tab = GL3DTab()
        self.uv_tab = FigureTab()
        self.pattern_3d_tab = GL3DTab()
        self.union_tab = ScanUnionTab()
        self.release_notes_tab = ReleaseNotesTab()
        self.addTab(self.structure_tab, "结构示意")
        self.addTab(self.cuts_tab, "二维切面")
        self.addTab(self.envelope_tab, "三维包络")
        self.addTab(self.uv_tab, "u-v方向图")
        self.addTab(self.pattern_3d_tab, "三维方向图")
        self.addTab(self.union_tab, "扫描并集")
        self.addTab(self.release_notes_tab, "更新说明")
        self._show_placeholder(self.structure_tab, "点击“计算”更新结构示意。")
        self._show_placeholder(self.cuts_tab, "点击“计算”更新二维切面。")
        self._show_placeholder(self.envelope_tab, "点击“计算”更新当前三维包络。")
        self._show_placeholder(self.uv_tab, "点击“计算”更新 u-v 方向图。")
        self._show_placeholder(self.pattern_3d_tab, "点击“计算”更新三维方向图。")
        self._show_placeholder(self.union_tab, "勾选“扫描并集三维包络”后点击“计算”。")

    def current_figure(self) -> Figure:
        widget = self.widget(self.currentIndex())
        if not hasattr(widget, "figure"):
            raise RuntimeError("The current tab has no exportable figure.")
        return widget.figure

    def show_release_notes(self) -> None:
        self.release_notes_tab.reload()
        self.setCurrentWidget(self.release_notes_tab)

    def show_structure_placeholder(self, text: str = "未选择结构示意图。") -> None:
        self._show_placeholder(self.structure_tab, text)

    def clear_calculated_plots(self, reason: str = "参数已变化，请重新计算。") -> None:
        self._show_placeholder(self.cuts_tab, f"{reason}\n二维切面待更新。")
        self._show_placeholder(self.envelope_tab, f"{reason}\n当前三维包络待更新。")
        self._show_placeholder(self.uv_tab, f"{reason}\nu-v 方向图待更新。")
        self._show_placeholder(self.pattern_3d_tab, f"{reason}\n三维方向图待更新。")
        self._show_placeholder(self.union_tab, f"{reason}\n扫描并集待更新。")

    def show_invalid_parameter_state(self, reason: str) -> None:
        self.show_structure_placeholder(f"参数无效：{reason}")
        self.clear_calculated_plots(f"参数无效：{reason}")

    def show_calculation_selection_placeholders(self, selected: dict[str, bool]) -> None:
        for key in ("structure", "cuts", "current_3d", "uv", "scan_union"):
            self.show_calculation_item_placeholder(key, selected.get(key, key != "scan_union"))

    def show_calculation_item_placeholder(self, key: str, selected: bool) -> None:
        messages = {
            "structure": ("未选择结构示意图。", "已选择结构示意图，参数刷新后自动更新。"),
            "cuts": ("未选择二维切面计算。", "已选择二维切面，请点击“计算”更新。"),
            "current_3d": ("未选择当前三维包络计算。", "已选择当前三维包络，请点击“计算”更新。"),
            "uv": ("未选择方向图计算。", "已选择方向图，请点击“计算”更新。"),
            "scan_union": ("未选择扫描并集计算。", "已选择扫描并集，请点击“计算”更新。"),
        }
        off_text, on_text = messages[key]
        text = on_text if selected else off_text
        if key == "structure":
            self.show_structure_placeholder(text)
        elif key == "cuts":
            self._show_placeholder(self.cuts_tab, text)
        elif key == "current_3d":
            self._show_placeholder(self.envelope_tab, text)
        elif key == "uv":
            self._show_placeholder(self.uv_tab, text)
            self._show_placeholder(self.pattern_3d_tab, text)
        elif key == "scan_union":
            self._show_placeholder(self.union_tab, text)

    def _show_placeholder(self, tab: QWidget, text: str) -> None:
        if hasattr(tab, "set_placeholder"):
            tab.set_placeholder(text)
            return
        tab.figure.clear()
        ax = tab.figure.add_subplot(111)
        ax.axis("off")
        ax.text(0.5, 0.5, text, ha="center", va="center", fontsize=12)
        tab.canvas.draw_idle()

    def plot_structure(self, params: BeamParams, derived: DerivedParams) -> None:
        fig = self.structure_tab.figure
        fig.clear()
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122, projection="3d")
        element_x = np.asarray(derived.element_x_m, dtype=float).ravel()
        element_y = np.asarray(derived.element_y_m, dtype=float).ravel()
        active_count = int(element_x.size)
        show_index = active_count <= 64
        element_colors = ("#8ecae6", "#ffb703")
        edge_color = "#202020"

        for idx, (x, y) in enumerate(zip(element_x, element_y)):
            color = element_colors[idx % 2]
            _add_element_patch_2d(ax1, float(x), float(y), params, color, edge_color)
            if show_index:
                label = str(idx + 1) if derived.imported_array_layout else _grid_index_label(float(x), float(y), derived)
                ax1.text(x, y, label, ha="center", va="center", fontsize=8, color="#111111")
        ax1.set_aspect("equal", adjustable="box")
        ax1.grid(True, linewidth=0.4)
        ax1.set_xlabel("x 向距离 (m)")
        ax1.set_ylabel("y 向距离 (m)")
        if derived.imported_array_layout:
            title = f"导入阵元坐标：有效单元 {active_count}"
        else:
            title = f"阵列口面排布：{params.nx} × {params.ny}，有效单元 {active_count}"
        ax1.set_title(title, fontsize=10)
        pad_x = max(0.12 * derived.dx_aperture_m, 0.5 * params.ax_m, 0.05)
        pad_y = max(0.12 * derived.dy_aperture_m, 0.5 * params.ay_m, 0.05)
        ax1.set_xlim(-derived.dx_aperture_m / 2.0 - pad_x, derived.dx_aperture_m / 2.0 + pad_x)
        ax1.set_ylim(-derived.dy_aperture_m / 2.0 - pad_y, derived.dy_aperture_m / 2.0 + pad_y)

        thickness = max(0.03 * max(params.ax_m, params.ay_m), 0.01 * max(derived.aperture_m, 1.0))
        for idx, (x, y) in enumerate(zip(element_x, element_y)):
            color = element_colors[idx % 2]
            ax2.bar3d(
                float(x) - params.ax_m / 2.0,
                float(y) - params.ay_m / 2.0,
                -thickness / 2.0,
                params.ax_m,
                params.ay_m,
                thickness,
                color=color,
                edgecolor=edge_color,
                linewidth=0.5,
                shade=True,
                alpha=0.92,
            )
        arr_len = max(derived.aperture_m, 1.0)
        _draw_aperture_outline(ax2, derived, params, z=0.0, color="black", linewidth=1.2)
        _draw_direction_guides(ax2, derived, arr_len)
        ax2.set_xlabel("x (m)")
        ax2.set_ylabel("y (m)")
        ax2.set_zlabel("z (m)")
        ax2.set_title("三维结构与当前扫描方向", fontsize=10)
        ax2.set_xlim(-0.72 * max(derived.dx_aperture_m, 1.0), 0.72 * max(derived.dx_aperture_m, 1.0))
        ax2.set_ylim(-0.72 * max(derived.dy_aperture_m, 1.0), 0.72 * max(derived.dy_aperture_m, 1.0))
        ax2.set_zlim(-0.12 * arr_len, 1.10 * arr_len)
        ax2.set_box_aspect((1.25, 1.05, 0.8))
        ax2.view_init(24, 36)
        fig.suptitle(
            f"f={params.frequency_ghz:.4g} GHz，λ={derived.wavelength_m:.4g} m，"
            f"θ={derived.theta_deg:.3f}°，φ={derived.phi_deg:.3f}°",
            fontsize=10,
        )
        fig.subplots_adjust(left=0.06, right=0.98, top=0.88, bottom=0.08, wspace=0.28)
        self.structure_tab.canvas.draw_idle()

    def plot_2d_cuts(self, cuts: list[dict[str, np.ndarray | str | float]], derived: DerivedParams, params: BeamParams) -> None:
        fig = self.cuts_tab.figure
        fig.clear()
        gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.0, 0.055], wspace=0.28)
        axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])]
        cax = fig.add_subplot(gs[0, 2])
        global_max = max(3.0 * params.s0_w_cm2, 1.0e-12)
        for ax, cut in zip(axes, cuts):
            r_env = np.asarray(cut["r_env_m"], dtype=float)
            valid = np.isfinite(r_env)
            if np.any(valid):
                r_plot_max = 1.08 * float(np.nanmax(r_env[valid]))
            else:
                r_plot_max = min(derived.rff_m, 8.0 * derived.aperture_m)
            r_plot_max = max(r_plot_max, 6.0 * derived.aperture_m)
            r_show, s_show = append_far_field_for_plot(
                np.asarray(cut["r_near_m"], dtype=float),
                np.asarray(cut["s_near_w_m2"], dtype=float),
                r_plot_max,
                derived.rff_m,
                80,
            )
            alpha = np.asarray(cut["alpha_deg"], dtype=float)
            rho = r_show[:, None] * sind(alpha)[None, :]
            zz = r_show[:, None] * cosd(alpha)[None, :]
            s_cm2 = np.minimum(s_show / 1.0e4, global_max)
            levels = np.linspace(0.0, global_max, 36)
            cf = ax.contourf(rho, zz, s_cm2, levels=levels, cmap="viridis")
            if np.any(valid):
                ax.plot(cut["rho_env_m"], cut["z_env_m"], "w-", linewidth=2.6, label="S=S0 外包络")
                ax.plot(cut["rho_env_m"], cut["z_env_m"], "k-", linewidth=1.0)
            else:
                ax.text(
                    0.5,
                    0.92,
                    "该固定切面内未达到 S=S0",
                    transform=ax.transAxes,
                    ha="center",
                    va="top",
                    fontsize=9,
                    bbox={"facecolor": "white", "edgecolor": "#777777", "alpha": 0.82, "boxstyle": "round,pad=0.25"},
                )
            phi_cut = float(cut["phi_cut_deg"])
            aperture_width = derived.dx_aperture_m * abs(cosd(phi_cut)) + derived.dy_aperture_m * abs(sind(phi_cut))
            ax.plot([-aperture_width / 2.0, aperture_width / 2.0], [0, 0], "k-", linewidth=4)
            scan_angle = params.scan_x_deg if abs(phi_cut) < 45.0 else params.scan_y_deg
            rho_scan = np.array([0.0, r_plot_max * float(sind(scan_angle))])
            z_scan = np.array([0.0, r_plot_max * float(cosd(scan_angle))])
            ax.plot(rho_scan, z_scan, "r--", linewidth=1.8, label="当前扫描方向")
            rho_max = max(float(np.nanmax(np.abs(rho))), aperture_width, 1.0)
            rho_arc = np.linspace(-min(rho_max, derived.rff_m), min(rho_max, derived.rff_m), 400)
            z_arc = np.sqrt(np.maximum(derived.rff_m * derived.rff_m - rho_arc * rho_arc, 0.0))
            ax.plot(rho_arc, z_arc, "k--", linewidth=0.8)
            if np.any(valid):
                rho_lim = max(1.0, aperture_width, float(1.15 * np.nanmax(np.abs(np.asarray(cut["rho_env_m"])[valid]))))
            else:
                rho_lim = max(1.0, aperture_width, float(np.nanmax(np.abs(rho))))
            ax.set_xlim(-rho_lim, rho_lim)
            ax.set_ylim(0.0, r_plot_max)
            ax.set_xlabel("横向距离 ρ (m)")
            ax.set_ylabel("高度 z (m)")
            ax.grid(True, linewidth=0.35)
            title = "x-z固定切面：S = S0 外包络" if abs(phi_cut) < 45.0 else "y-z固定切面：S = S0 外包络"
            method_label = str(cut.get("envelope_method_label", "近场采样 + 远场外推"))
            max_line = _format_cut_max_line(cut)
            ax.set_title(f"{title}\nS0={params.s0_w_cm2:.4g} W/cm²，方法：{method_label}\n{max_line}", fontsize=9)
            ax.legend(loc="upper right", fontsize=8)
        fig.colorbar(cf, cax=cax, label="功率密度 S (W/cm²，色标截断)")
        fig.subplots_adjust(left=0.07, right=0.94, top=0.86, bottom=0.10)
        self.cuts_tab.canvas.draw_idle()

    def plot_current_3d(self, envelope: dict[str, np.ndarray | float], derived: DerivedParams, params: BeamParams) -> None:
        if self.envelope_tab.scene.opengl_enabled:
            _plot_current_3d_opengl(self.envelope_tab.scene, envelope, derived, params)
            _set_export_placeholder(self.envelope_tab.figure, self.envelope_tab.canvas, "Current S=S0 3D envelope is shown in the OpenGL view.")
            return
        fig = self.envelope_tab.figure
        fig.clear()
        ax = fig.add_subplot(111, projection="3d")
        _prepare_interactive_3d_axis(ax)
        u = np.asarray(envelope["u"], dtype=float)
        v = np.asarray(envelope["v"], dtype=float)
        x_raw = np.asarray(envelope["Xsurf"], dtype=float)
        y_raw = np.asarray(envelope["Ysurf"], dtype=float)
        z_raw = np.asarray(envelope["Zsurf"], dtype=float)
        r = np.asarray(envelope["r_env_m"], dtype=float)
        if not np.any(np.isfinite(r)):
            ax.text2D(0.5, 0.5, "当前窗口内没有 S=S0 外包络。", transform=ax.transAxes, ha="center")
            x_plot = y_plot = z_plot = None
        else:
            U, V = np.meshgrid(u, v)
            visible = np.isfinite(r) & ((U * U + V * V) <= 0.999 * 0.999)
            W = np.sqrt(np.maximum(1.0 - U * U - V * V, 0.0))
            r_plot = np.where(visible, r, np.nan)
            x_plot = np.where(visible, r_plot * U, np.nan)
            y_plot = np.where(visible, r_plot * V, np.nan)
            z_plot = np.where(visible, r_plot * W, np.nan)
            display_n = _display_grid_n(params)
            _, mappable = _plot_range_surface(
                ax,
                x_plot,
                y_plot,
                z_plot,
                r_plot,
                alpha=0.72,
                edgecolor=(0.0, 0.0, 0.0, 0.0),
                max_rows=display_n,
                max_cols=max(display_n, int(display_n * 4 / 3)),
            )
            fig.colorbar(mappable, ax=ax, shrink=0.75, label="包络距离 r (m)")
        _draw_aperture_outline(ax, derived, params, z=0.0, color="black", linewidth=1.2)
        max_range = envelope.get("max_range_m", np.nan)
        length = _direction_length(float(max_range), derived)
        _draw_direction_guides(ax, derived, length)
        if x_plot is not None and y_plot is not None and z_plot is not None:
            _set_3d_display_limits(ax, x_raw, y_raw, z_raw, derived)
        else:
            _set_3d_display_limits(ax, x_raw, y_raw, z_raw, derived)
        ax.set_box_aspect((1.35, 1.35, 0.55))
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_zlabel("z (m)")
        title_range = _format_range(float(max_range))
        method_label = str(envelope.get("envelopeMethodLabel", "近场采样 + 远场外推"))
        ax.set_title(f"当前扫描 S = S0 三维外包络\n最大包络距离 {title_range}，方法：{method_label}")
        ax.view_init(25, 38)
        fig.subplots_adjust(left=0.02, right=0.88, top=0.90, bottom=0.02)
        self.envelope_tab.canvas.draw_idle()

    def plot_uv_pattern(self, uv: dict[str, np.ndarray | float], params: BeamParams | None = None) -> None:
        fig = self.uv_tab.figure
        fig.clear()
        ax = fig.add_subplot(111)
        image = ax.imshow(
            uv["pattern_db"],
            extent=[uv["u"][0], uv["u"][-1], uv["v"][0], uv["v"][-1]],
            origin="lower",
            cmap="viridis",
            vmin=float(uv["floor_db"]),
            vmax=0.0,
            interpolation="nearest",
        )
        t = np.linspace(0.0, 2.0 * math.pi, 720)
        ax.plot(np.cos(t), np.sin(t), "k-", linewidth=1)
        ax.plot(float(uv["scan_u"]), float(uv["scan_v"]), "r*", markersize=12, markeredgecolor="white", markeredgewidth=0.8, label="当前扫描方向")
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.grid(True, linewidth=0.35)
        ax.set_xlabel("u = sin(theta) cos(phi)")
        ax.set_ylabel("v = sin(theta) sin(phi)")
        ax.set_title("u-v全半空间远场方向图")
        ax.legend(loc="upper right")
        fig.colorbar(image, ax=ax, label="归一化方向图 (dB)")
        fig.subplots_adjust(left=0.08, right=0.88, top=0.92, bottom=0.08)
        self.uv_tab.canvas.draw_idle()
        self.plot_pattern_3d(uv, params)

    def plot_pattern_3d(self, uv: dict[str, np.ndarray | float], params: BeamParams | None = None) -> None:
        if self.pattern_3d_tab.scene.opengl_enabled:
            _plot_pattern_3d_opengl(self.pattern_3d_tab.scene, uv, params)
            _set_export_placeholder(self.pattern_3d_tab.figure, self.pattern_3d_tab.canvas, "3D pattern is shown in the OpenGL view.")
            return
        fig = self.pattern_3d_tab.figure
        fig.clear()
        ax = fig.add_subplot(111, projection="3d")
        _prepare_interactive_3d_axis(ax)
        u = np.asarray(uv["U"], dtype=float)
        v = np.asarray(uv["V"], dtype=float)
        pattern_db = np.asarray(uv["pattern_db"], dtype=float)
        visible = np.asarray(uv["visible"], dtype=bool) & np.isfinite(pattern_db)
        floor_db = float(uv["floor_db"])

        theta_x = np.degrees(np.arcsin(np.clip(u, -1.0, 1.0)))
        theta_y = np.degrees(np.arcsin(np.clip(v, -1.0, 1.0)))
        z = np.where(visible, np.maximum(pattern_db, floor_db), np.nan)
        theta_x = np.where(visible, theta_x, np.nan)
        theta_y = np.where(visible, theta_y, np.nan)

        display_n = _display_grid_n(params)
        theta_x, theta_y, z = _limit_surface_grid(theta_x, theta_y, z, max_rows=display_n, max_cols=display_n)
        cmap = colormaps["viridis"]
        norm = Normalize(vmin=floor_db, vmax=0.0)
        color_values = np.nan_to_num(z, nan=floor_db, posinf=0.0, neginf=floor_db)
        facecolors = cmap(norm(color_values))
        facecolors[~np.isfinite(z), 3] = 0.0

        surf = ax.plot_surface(
            theta_x,
            theta_y,
            z,
            facecolors=facecolors,
            rstride=1,
            cstride=1,
            linewidth=0.0,
            antialiased=False,
            shade=False,
        )
        _rasterize_surface_if_possible(surf)
        mappable = ScalarMappable(norm=norm, cmap=cmap)
        mappable.set_array(np.asarray(pattern_db[np.isfinite(pattern_db)], dtype=float))
        fig.colorbar(mappable, ax=ax, shrink=0.68, pad=0.10, label="归一化方向图 (dB)")

        scan_x = math.degrees(math.asin(max(-1.0, min(1.0, float(uv["scan_u"])))))
        scan_y = math.degrees(math.asin(max(-1.0, min(1.0, float(uv["scan_v"])))))
        ax.plot([scan_x, scan_x], [scan_y, scan_y], [floor_db, 0.0], "r--", linewidth=1.4)
        ax.scatter([scan_x], [scan_y], [0.0], c="red", marker="*", s=72, depthshade=False, label="当前扫描方向")

        ax.set_xlim(-90.0, 90.0)
        ax.set_ylim(-90.0, 90.0)
        ax.set_zlim(floor_db, 0.0)
        ax.set_xticks(np.arange(-80.0, 81.0, 40.0))
        ax.set_yticks(np.arange(-80.0, 81.0, 40.0))
        ax.set_xlabel("θx (deg)", labelpad=5)
        ax.set_ylabel("θy (deg)", labelpad=5)
        ax.set_zlabel("")
        ax.tick_params(axis="both", which="major", labelsize=8, pad=1)
        ax.tick_params(axis="z", which="major", labelsize=8, pad=1)
        ax.set_title("三维方向图 Pattern (dB)", pad=10)
        ax.legend(loc="upper right")
        ax.view_init(26, -54)
        ax.set_box_aspect((1.35, 1.15, 0.62))
        fig.subplots_adjust(left=0.03, right=0.84, top=0.90, bottom=0.05)
        self.pattern_3d_tab.canvas.draw_idle()

    def plot_scan_union(self, info: dict[str, np.ndarray | float | int], derived: DerivedParams, params: BeamParams) -> None:
        if self.union_tab.scene.opengl_enabled:
            fig = self.union_tab.figure
            fig.clear()
            axes = [fig.add_subplot(211), fig.add_subplot(212)]
            for cut_ax, cut in zip(axes, info.get("unionCuts", [])):
                _plot_scan_union_cut_2d(cut_ax, cut, derived, params)
            if axes:
                axes[0].set_xlabel("")
            fig.subplots_adjust(left=0.16, right=0.96, top=0.93, bottom=0.10, hspace=0.46)
            self.union_tab.canvas.draw_idle()
            _plot_scan_union_opengl(self.union_tab.scene, info, derived, params)
            return
        fig = self.union_tab.figure
        fig.clear()
        gs = fig.add_gridspec(2, 3, width_ratios=[1.0, 1.55, 0.055], height_ratios=[1.0, 1.0], wspace=0.38, hspace=0.50)
        cut_axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[1, 0])]
        ax = fig.add_subplot(gs[:, 1], projection="3d")
        _prepare_interactive_3d_axis(ax)
        cax = fig.add_subplot(gs[:, 2])
        for cut_ax, cut in zip(cut_axes, info.get("unionCuts", [])):
            _plot_scan_union_cut_2d(cut_ax, cut, derived, params)
        if cut_axes:
            cut_axes[0].set_xlabel("")
        u = np.asarray(info.get("u", []), dtype=float)
        v = np.asarray(info.get("v", []), dtype=float)
        r_raw = np.asarray(info["Rsurf"], dtype=float)
        if u.size >= 2 and v.size >= 2:
            U, V = np.meshgrid(u, v)
            visible = np.isfinite(r_raw) & ((U * U + V * V) <= 0.999 * 0.999)
            W = np.sqrt(np.maximum(1.0 - U * U - V * V, 0.0))
            r = np.where(visible, r_raw, np.nan)
            x = np.where(visible, r * U, np.nan)
            y = np.where(visible, r * V, np.nan)
            z = np.where(visible, r * W, np.nan)
        else:
            x = np.asarray(info["Xsurf"], dtype=float)
            y = np.asarray(info["Ysurf"], dtype=float)
            z = np.asarray(info["Zsurf"], dtype=float)
            r = r_raw
        display_n = _display_grid_n(params)
        _, mappable = _plot_range_surface(
            ax,
            x,
            y,
            z,
            r,
            alpha=0.66,
            edgecolor=(0.0, 0.0, 0.0, 0.0),
            max_rows=display_n,
            max_cols=max(display_n, int(display_n * 4 / 3)),
        )
        fig.colorbar(mappable, cax=cax, label="包络距离 r (m)")
        _draw_aperture_outline(ax, derived, params, z=0.0, color="black", linewidth=1.2)
        max_range = float(info["maxRange_m"]) if math.isfinite(float(info["maxRange_m"])) else derived.aperture_m
        cut_max_range = float(info.get("maxRangeNearFieldCuts_m", float("nan")))
        cut_max_text = _format_range(cut_max_range) if math.isfinite(cut_max_range) else "N/A"
        length = _direction_length(max_range, derived)
        _draw_direction_guides(ax, derived, length, show_labels=False)
        _set_3d_display_limits(ax, x, y, z, derived)
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_zlabel("z (m)")
        ax.set_title("扫描并集 S=S0 三维包络预览", fontsize=10, pad=8)
        fig.suptitle(
            f"扫描并集：3D预览最大 {max_range:.4g} m；固定切面最大 {cut_max_text}；"
            f"扫描中心 X=±{info['scanLimitX_deg']:.3g}°，Y=±{info['scanLimitY_deg']:.3g}°；中心数 {info['numScanCenters']}",
            fontsize=10,
            y=0.985,
        )
        ax.view_init(25, 38)
        fig.subplots_adjust(left=0.05, right=0.95, top=0.86, bottom=0.12)
        self.union_tab.canvas.draw_idle()


def _set_export_placeholder(figure: Figure, canvas: FigureCanvas, text: str) -> None:
    figure.clear()
    ax = figure.add_subplot(111)
    ax.axis("off")
    ax.text(0.5, 0.5, text, ha="center", va="center", fontsize=12)
    canvas.draw_idle()


def _plot_current_3d_opengl(
    scene: OpenGLSceneWidget,
    envelope: dict[str, np.ndarray | float],
    derived: DerivedParams,
    params: BeamParams,
) -> None:
    u = np.asarray(envelope["u"], dtype=float)
    v = np.asarray(envelope["v"], dtype=float)
    r = np.asarray(envelope["r_env_m"], dtype=float)
    if not np.any(np.isfinite(r)):
        scene.clear_scene("Current S=S0 envelope: no finite surface.")
        return
    U, V = np.meshgrid(u, v)
    visible = np.isfinite(r) & ((U * U + V * V) <= 0.999 * 0.999)
    W = np.sqrt(np.maximum(1.0 - U * U - V * V, 0.0))
    r_plot = np.where(visible, r, np.nan)
    x = np.where(visible, r_plot * U, np.nan)
    y = np.where(visible, r_plot * V, np.nan)
    z = np.where(visible, r_plot * W, np.nan)
    max_range = float(envelope.get("max_range_m", np.nan))
    method_label = str(envelope.get("envelopeMethodLabel", "near-field sampled + far-field extrapolated"))
    title = f"Current S=S0 3D envelope    max r={_format_range(max_range)}    method={method_label}"
    _plot_surface_opengl(scene, x, y, z, r_plot, title, derived=derived, params=params, show_direction=True)


def _plot_pattern_3d_opengl(scene: OpenGLSceneWidget, uv: dict[str, np.ndarray | float], params: BeamParams | None) -> None:
    u = np.asarray(uv["U"], dtype=float)
    v = np.asarray(uv["V"], dtype=float)
    pattern_db = np.asarray(uv["pattern_db"], dtype=float)
    visible = np.asarray(uv["visible"], dtype=bool) & np.isfinite(pattern_db)
    floor_db = float(uv["floor_db"])
    theta_x = np.where(visible, np.degrees(np.arcsin(np.clip(u, -1.0, 1.0))), np.nan)
    theta_y = np.where(visible, np.degrees(np.arcsin(np.clip(v, -1.0, 1.0))), np.nan)
    z = np.where(visible, np.maximum(pattern_db, floor_db), np.nan)
    title = "3D normalized pattern (dB)"
    _plot_surface_opengl(
        scene,
        theta_x,
        theta_y,
        z,
        z,
        title,
        derived=None,
        params=params,
        value_min=floor_db,
        value_max=0.0,
        show_direction=False,
    )
    scan_x = math.degrees(math.asin(max(-1.0, min(1.0, float(uv["scan_u"])))))
    scan_y = math.degrees(math.asin(max(-1.0, min(1.0, float(uv["scan_v"])))))
    _add_gl_line(scene, np.array([[scan_x, scan_y, floor_db], [scan_x, scan_y, 0.0]], dtype=float), (1.0, 0.0, 0.0, 1.0), 3.0)


def _plot_scan_union_opengl(
    scene: OpenGLSceneWidget,
    info: dict[str, np.ndarray | float | int],
    derived: DerivedParams,
    params: BeamParams,
) -> None:
    u = np.asarray(info.get("u", []), dtype=float)
    v = np.asarray(info.get("v", []), dtype=float)
    r_raw = np.asarray(info["Rsurf"], dtype=float)
    if u.size >= 2 and v.size >= 2:
        U, V = np.meshgrid(u, v)
        visible = np.isfinite(r_raw) & ((U * U + V * V) <= 0.999 * 0.999)
        W = np.sqrt(np.maximum(1.0 - U * U - V * V, 0.0))
        r = np.where(visible, r_raw, np.nan)
        x = np.where(visible, r * U, np.nan)
        y = np.where(visible, r * V, np.nan)
        z = np.where(visible, r * W, np.nan)
    else:
        x = np.asarray(info["Xsurf"], dtype=float)
        y = np.asarray(info["Ysurf"], dtype=float)
        z = np.asarray(info["Zsurf"], dtype=float)
        r = r_raw
    max_range = float(info["maxRange_m"]) if math.isfinite(float(info["maxRange_m"])) else float("nan")
    title = (
        f"Scan-union S=S0 3D envelope    max r={_format_range(max_range)}    "
        f"centers={info.get('numScanCenters', 'n/a')}"
    )
    _plot_surface_opengl(scene, x, y, z, r, title, derived=derived, params=params, show_direction=True)


def _plot_surface_opengl(
    scene: OpenGLSceneWidget,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    values: np.ndarray,
    title: str,
    *,
    derived: DerivedParams | None,
    params: BeamParams | None,
    value_min: float | None = None,
    value_max: float | None = None,
    show_direction: bool = False,
) -> None:
    if not scene.clear_scene(title):
        return
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    z_arr = np.asarray(z, dtype=float)
    value_arr = np.asarray(values, dtype=float)
    display_n = _display_grid_n(params, default=180)
    x_arr, y_arr, z_arr, value_arr = _limit_surface_grid(
        x_arr,
        y_arr,
        z_arr,
        value_arr,
        max_rows=max(36, min(display_n, 220)),
        max_cols=max(48, min(int(display_n * 4 / 3), 280)),
    )
    finite = np.isfinite(x_arr) & np.isfinite(y_arr) & np.isfinite(z_arr) & np.isfinite(value_arr)
    if not np.any(finite):
        scene.clear_scene(title + "    no finite surface")
        return

    finite_values = value_arr[finite]
    vmin = float(value_min) if value_min is not None else float(np.nanpercentile(finite_values, 2.0))
    vmax = float(value_max) if value_max is not None else float(np.nanpercentile(finite_values, 98.0))
    if not math.isfinite(vmin):
        vmin = float(np.nanmin(finite_values))
    if not math.isfinite(vmax) or vmax <= vmin:
        vmax = vmin + 1.0

    mesh_item = _make_gl_mesh_item(x_arr, y_arr, z_arr, value_arr, vmin, vmax)
    if mesh_item is not None:
        scene.add_item(mesh_item)
    else:
        pos = np.column_stack([x_arr[finite], y_arr[finite], z_arr[finite]]).astype(float)
        colors = _rgba_for_values(value_arr[finite], vmin, vmax, alpha=0.80)
        scatter = gl.GLScatterPlotItem(pos=pos, color=colors, size=2.0, pxMode=True)
        scene.add_item(scatter)

    _add_gl_reference_items(scene, x_arr, y_arr, z_arr, derived, params, show_direction)
    _set_gl_camera(scene, x_arr, y_arr, z_arr)


def _make_gl_mesh_item(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    values: np.ndarray,
    vmin: float,
    vmax: float,
) -> object | None:
    if gl is None or MeshData is None:
        return None
    rows, cols = x.shape
    finite = np.isfinite(x) & np.isfinite(y) & np.isfinite(z) & np.isfinite(values)
    index = np.full((rows, cols), -1, dtype=np.int32)
    vertices = np.column_stack([x[finite], y[finite], z[finite]]).astype(np.float32)
    if vertices.size == 0:
        return None
    index[finite] = np.arange(vertices.shape[0], dtype=np.int32)
    faces: list[list[int]] = []
    face_values: list[float] = []
    for row in range(rows - 1):
        for col in range(cols - 1):
            a = int(index[row, col])
            b = int(index[row, col + 1])
            c = int(index[row + 1, col + 1])
            d = int(index[row + 1, col])
            if min(a, b, c, d) < 0:
                continue
            val = float(np.nanmean([values[row, col], values[row, col + 1], values[row + 1, col + 1], values[row + 1, col]]))
            faces.append([a, b, c])
            face_values.append(val)
            faces.append([a, c, d])
            face_values.append(val)
    if not faces:
        return None
    colors = _rgba_for_values(np.asarray(face_values, dtype=float), vmin, vmax, alpha=0.78)
    mesh = MeshData(vertexes=vertices, faces=np.asarray(faces, dtype=np.int32), faceColors=colors.astype(np.float32))
    return gl.GLMeshItem(meshdata=mesh, smooth=False, drawFaces=True, drawEdges=False, shader="shaded", glOptions="translucent")


def _rgba_for_values(values: np.ndarray, vmin: float, vmax: float, alpha: float) -> np.ndarray:
    norm = Normalize(vmin=vmin, vmax=vmax)
    colors = colormaps["viridis"](norm(np.asarray(values, dtype=float)))
    colors[:, 3] = alpha
    return colors.astype(np.float32)


def _add_gl_reference_items(
    scene: OpenGLSceneWidget,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    derived: DerivedParams | None,
    params: BeamParams | None,
    show_direction: bool,
) -> None:
    finite = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    max_x = max(float(np.nanmax(np.abs(x[finite]))), 1.0) if np.any(finite) else 1.0
    max_y = max(float(np.nanmax(np.abs(y[finite]))), 1.0) if np.any(finite) else 1.0
    max_z = max(float(np.nanmax(np.abs(z[finite]))), 1.0) if np.any(finite) else 1.0
    grid_size = max(max_x, max_y) * 2.1
    grid = gl.GLGridItem()
    grid.setSize(x=grid_size, y=grid_size)
    spacing = _nice_grid_spacing(grid_size)
    grid.setSpacing(x=spacing, y=spacing)
    scene.add_item(grid)
    _add_gl_line(scene, np.array([[-max_x, 0.0, 0.0], [max_x, 0.0, 0.0]], dtype=float), (0.8, 0.05, 0.05, 1.0), 2.0)
    _add_gl_line(scene, np.array([[0.0, -max_y, 0.0], [0.0, max_y, 0.0]], dtype=float), (0.05, 0.45, 0.9, 1.0), 2.0)
    _add_gl_line(scene, np.array([[0.0, 0.0, 0.0], [0.0, 0.0, max_z]], dtype=float), (0.05, 0.55, 0.15, 1.0), 2.0)
    if derived is None:
        return
    outline = _aperture_outline_points(derived, params, z=0.0)
    if outline.shape[0] >= 2:
        _add_gl_line(scene, outline, (0.0, 0.0, 0.0, 1.0), 2.5)
    if show_direction:
        length = max(0.35 * max_z, derived.aperture_m, 1.0)
        _add_gl_line(scene, np.array([[0.0, 0.0, 0.0], [0.0, 0.0, length]], dtype=float), (0.0, 0.0, 0.0, 1.0), 1.8)
        _add_gl_line(
            scene,
            np.array([[0.0, 0.0, 0.0], [length * derived.u0, length * derived.v0, length * derived.w0]], dtype=float),
            (1.0, 0.0, 0.0, 1.0),
            3.0,
        )


def _add_gl_line(scene: OpenGLSceneWidget, points: np.ndarray, color: tuple[float, float, float, float], width: float) -> None:
    if gl is None:
        return
    line = gl.GLLinePlotItem(pos=np.asarray(points, dtype=float), color=color, width=width, antialias=True, mode="line_strip")
    scene.add_item(line)


def _set_gl_camera(scene: OpenGLSceneWidget, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> None:
    if not scene.opengl_enabled or scene.view is None:
        return
    finite = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    if np.any(finite):
        span_x = max(float(np.nanmax(x[finite]) - np.nanmin(x[finite])), 1.0)
        span_y = max(float(np.nanmax(y[finite]) - np.nanmin(y[finite])), 1.0)
        span_z = max(float(np.nanmax(z[finite]) - np.nanmin(z[finite])), 1.0)
        distance = 1.9 * max(span_x, span_y, 0.8 * span_z)
    else:
        distance = 10.0
    scene.view.setCameraPosition(distance=distance, elevation=24.0, azimuth=-42.0)


def _nice_grid_spacing(size: float) -> float:
    if not math.isfinite(size) or size <= 0:
        return 1.0
    raw = size / 8.0
    exponent = math.floor(math.log10(raw))
    base = 10.0 ** exponent
    scaled = raw / base
    nice = 1.0 if scaled <= 1.0 else 2.0 if scaled <= 2.0 else 5.0 if scaled <= 5.0 else 10.0
    return nice * base


def _aperture_outline_points(derived: DerivedParams, params: BeamParams | None = None, z: float = 0.0) -> np.ndarray:
    shape = getattr(params, "array_layout", "rectangular") if params is not None else "rectangular"
    half_x = derived.dx_aperture_m / 2.0
    half_y = derived.dy_aperture_m / 2.0
    if shape == "custom":
        x, y = _convex_hull_outline(derived.element_x_m, derived.element_y_m)
    elif shape == "ellipse":
        t = np.linspace(0.0, 2.0 * math.pi, 145)
        x = half_x * np.cos(t)
        y = half_y * np.sin(t)
    elif shape == "diamond":
        x = np.array([0.0, half_x, 0.0, -half_x, 0.0])
        y = np.array([half_y, 0.0, -half_y, 0.0, half_y])
    else:
        x = np.array([-half_x, half_x, half_x, -half_x, -half_x])
        y = np.array([-half_y, -half_y, half_y, half_y, -half_y])
    return np.column_stack([x, y, np.full_like(x, z, dtype=float)]).astype(float)


def _plot_scan_union_cut_2d(ax, cut: dict[str, object], derived: DerivedParams, params: BeamParams) -> None:
    rho = np.asarray(cut["rho_env_m"], dtype=float)
    z = np.asarray(cut["z_env_m"], dtype=float)
    r = np.asarray(cut["r_env_m"], dtype=float)
    valid = np.isfinite(rho) & np.isfinite(z) & np.isfinite(r)
    phi_cut = float(cut["phi_cut_deg"])
    aperture_width = derived.dx_aperture_m if abs(phi_cut) < 45.0 else derived.dy_aperture_m
    title = "扫描并集 x-z固定切面" if abs(phi_cut) < 45.0 else "扫描并集 y-z固定切面"
    rho_lim = max(1.0, aperture_width)
    z_lim = max(1.0, derived.aperture_m)

    if np.any(valid):
        rho_v = rho[valid]
        z_v = z[valid]
        r_v = r[valid]
        ax.fill(np.r_[0.0, rho_v, 0.0], np.r_[0.0, z_v, 0.0], color="#9bd5ff", alpha=0.28, linewidth=0.0)
        ax.plot(rho_v, z_v, color="#0b66c3", linewidth=2.0, label="所有扫描 S=S0 外包络")
        max_idx = int(np.nanargmax(r_v))
        ax.plot(rho_v[max_idx], z_v[max_idx], "o", color="#d62728", markersize=4.5, label=f"最远 {r_v[max_idx]:.4g} m")
        rho_lim = max(rho_lim, float(np.nanmax(np.abs(rho_v))) * 1.10)
        z_lim = max(z_lim, float(np.nanmax(z_v)) * 1.08)
    else:
        ax.text(0.5, 0.5, "该固定切面内无有限并集包络", transform=ax.transAxes, ha="center", va="center", fontsize=9)

    ax.plot([-aperture_width / 2.0, aperture_width / 2.0], [0.0, 0.0], color="black", linewidth=3.0)
    ax.axvline(0.0, color="#777777", linewidth=0.8)
    ax.set_xlim(-rho_lim, rho_lim)
    ax.set_ylim(0.0, z_lim)
    ax.set_xlabel("横向距离 ρ (m)")
    ax.set_ylabel("高度 z (m)")
    ax.set_title(title, fontsize=9)
    ax.grid(True, linewidth=0.35)
    if np.any(valid):
        ax.legend(loc="upper right", fontsize=7)


def _add_element_patch_2d(ax, x: float, y: float, params: BeamParams, facecolor: str, edgecolor: str) -> None:
    shape = getattr(params, "element_shape", "rectangular")
    if shape == "ellipse":
        patch = Ellipse(
            (x, y),
            width=params.ax_m,
            height=params.ay_m,
            facecolor=facecolor,
            edgecolor=edgecolor,
            linewidth=1.6,
        )
    elif shape == "diamond":
        hx = params.ax_m / 2.0
        hy = params.ay_m / 2.0
        patch = Polygon(
            [(x, y + hy), (x + hx, y), (x, y - hy), (x - hx, y)],
            closed=True,
            facecolor=facecolor,
            edgecolor=edgecolor,
            linewidth=1.6,
        )
    else:
        patch = Rectangle(
            (x - params.ax_m / 2.0, y - params.ay_m / 2.0),
            params.ax_m,
            params.ay_m,
            facecolor=facecolor,
            edgecolor=edgecolor,
            linewidth=1.6,
        )
    ax.add_patch(patch)


def _grid_index_label(x: float, y: float, derived: DerivedParams) -> str:
    ix = int(np.argmin(np.abs(np.asarray(derived.x_elem_m, dtype=float) - x))) + 1
    iy = int(np.argmin(np.abs(np.asarray(derived.y_elem_m, dtype=float) - y))) + 1
    return f"{iy},{ix}"


def _set_3d_display_limits(ax, x: np.ndarray, y: np.ndarray, z: np.ndarray, derived: DerivedParams) -> None:
    finite = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    if np.any(finite):
        max_x = max(float(np.nanmax(np.abs(x[finite]))), derived.dx_aperture_m / 2.0, 1.0)
        max_y = max(float(np.nanmax(np.abs(y[finite]))), derived.dy_aperture_m / 2.0, 1.0)
        max_z = max(float(np.nanmax(z[finite])), derived.aperture_m, 1.0)
    else:
        max_x = max(derived.dx_aperture_m / 2.0, 1.0)
        max_y = max(derived.dy_aperture_m / 2.0, 1.0)
        max_z = max(derived.aperture_m, 1.0)
    ax.set_xlim(-1.03 * max_x, 1.03 * max_x)
    ax.set_ylim(-1.03 * max_y, 1.03 * max_y)
    ax.set_zlim(0.0, 1.03 * max_z)
    ax.set_box_aspect((1.3, 1.3, 1.0))


def _prepare_interactive_3d_axis(ax) -> None:
    try:
        ax.set_proj_type("ortho")
    except Exception:
        pass
    ax.format_coord = lambda _x, _y: ""


def _rasterize_surface_if_possible(surface) -> None:
    try:
        surface.set_rasterized(True)
    except Exception:
        pass


def _display_grid_n(params: BeamParams | None, default: int = 180) -> int:
    if params is None:
        return default
    try:
        return max(32, min(500, int(getattr(params, "display_3d_grid_n", default))))
    except (TypeError, ValueError):
        return default


def _plot_range_surface(
    ax,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    r: np.ndarray,
    alpha: float,
    edgecolor,
    max_rows: int = 90,
    max_cols: int = 120,
) -> tuple[object, ScalarMappable]:
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    z_arr = np.asarray(z, dtype=float)
    r_arr_full = np.asarray(r, dtype=float)
    finite_r = np.isfinite(r_arr_full)
    if np.any(finite_r):
        vmin = float(np.nanmin(r_arr_full[finite_r]))
        vmax = float(np.nanmax(r_arr_full[finite_r]))
    else:
        vmin = 0.0
        vmax = 1.0
    if vmax <= vmin:
        vmax = vmin + 1.0

    x_arr, y_arr, z_arr, r_arr = _limit_surface_grid(x_arr, y_arr, z_arr, r_arr_full, max_rows=max_rows, max_cols=max_cols)
    finite_plot = np.isfinite(r_arr)
    cmap = colormaps["viridis"]
    norm = Normalize(vmin=vmin, vmax=vmax)
    color_values = np.nan_to_num(r_arr, nan=vmin, posinf=vmax, neginf=vmin)
    facecolors = cmap(norm(color_values))
    facecolors[..., 3] = alpha
    facecolors[~finite_plot, 3] = 0.0

    surf = ax.plot_surface(
        x_arr,
        y_arr,
        z_arr,
        facecolors=facecolors,
        linewidth=0.0,
        edgecolor=edgecolor,
        rstride=1,
        cstride=1,
        antialiased=False,
        shade=False,
    )
    _rasterize_surface_if_possible(surf)
    mappable = ScalarMappable(norm=norm, cmap=cmap)
    mappable.set_array(r_arr_full[finite_r])
    return surf, mappable


def _upsample_uv_surface_for_display(
    u: np.ndarray,
    v: np.ndarray,
    r: np.ndarray,
    max_u: int,
    max_v: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    u_arr = np.asarray(u, dtype=float).ravel()
    v_arr = np.asarray(v, dtype=float).ravel()
    r_arr = np.asarray(r, dtype=float)
    if r_arr.ndim != 2 or min(r_arr.shape) < 2:
        return u_arr, v_arr, r_arr

    v_n, u_n = r_arr.shape
    target_u = min(max_u, max(u_n, u_n * 2 - 1))
    target_v = min(max_v, max(v_n, v_n * 2 - 1))
    if target_u <= u_n and target_v <= v_n:
        return u_arr, v_arr, r_arr

    u_new = np.linspace(float(u_arr[0]), float(u_arr[-1]), target_u)
    v_new = np.linspace(float(v_arr[0]), float(v_arr[-1]), target_v)
    filled = np.empty((v_n, target_u), dtype=float)
    for row_idx in range(v_n):
        filled[row_idx, :] = _interp_preserving_nan_gaps(u_arr, r_arr[row_idx, :], u_new)

    out = np.empty((target_v, target_u), dtype=float)
    for col_idx in range(target_u):
        out[:, col_idx] = _interp_preserving_nan_gaps(v_arr, filled[:, col_idx], v_new)

    U_new, V_new = np.meshgrid(u_new, v_new)
    out[(U_new * U_new + V_new * V_new) > 0.999 * 0.999] = np.nan
    return u_new, v_new, out


def _interp_preserving_nan_gaps(x_old: np.ndarray, y_old: np.ndarray, x_new: np.ndarray) -> np.ndarray:
    x = np.asarray(x_old, dtype=float).ravel()
    y = np.asarray(y_old, dtype=float).ravel()
    xp = np.asarray(x_new, dtype=float).ravel()
    out = np.full(xp.shape, np.nan, dtype=float)
    finite = np.isfinite(x) & np.isfinite(y)
    if not np.any(finite):
        return out

    finite_idx = np.where(finite)[0]
    breaks = np.where(np.diff(finite_idx) > 1)[0] + 1
    segments = np.split(finite_idx, breaks)
    for segment in segments:
        if segment.size == 0:
            continue
        x_seg = x[segment]
        y_seg = y[segment]
        if segment.size == 1:
            nearest = int(np.nanargmin(np.abs(xp - x_seg[0])))
            out[nearest] = y_seg[0]
            continue
        lo = float(min(x_seg[0], x_seg[-1]))
        hi = float(max(x_seg[0], x_seg[-1]))
        in_segment = (xp >= lo) & (xp <= hi)
        if np.any(in_segment):
            out[in_segment] = np.interp(xp[in_segment], x_seg, y_seg)
    return out


def _limit_surface_grid(*arrays: np.ndarray, max_rows: int, max_cols: int) -> tuple[np.ndarray, ...]:
    if not arrays:
        return ()
    first = np.asarray(arrays[0])
    if first.ndim != 2:
        return tuple(np.asarray(arr) for arr in arrays)
    rows, cols = first.shape
    row_step = max(1, math.ceil(rows / max(2, int(max_rows))))
    col_step = max(1, math.ceil(cols / max(2, int(max_cols))))
    return tuple(np.asarray(arr)[::row_step, ::col_step] for arr in arrays)


def _limit_surface_points(*arrays: np.ndarray, max_points: int) -> tuple[np.ndarray, ...]:
    max_side = max(2, int(math.sqrt(max(1, max_points))))
    return _limit_surface_grid(*arrays, max_rows=max_side, max_cols=max_side)


def _smooth_nan_surface_display(values: np.ndarray, passes: int = 1) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 2 or not np.any(np.isfinite(arr)):
        return arr

    kernel = np.array([1.0, 4.0, 6.0, 4.0, 1.0], dtype=float)
    kernel /= kernel.sum()
    out = arr.copy()
    for _ in range(max(1, int(passes))):
        out = _nan_normalized_convolution(out, kernel)
    return out


def _nan_normalized_convolution(arr: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    finite = np.isfinite(arr)
    data = np.where(finite, arr, 0.0)
    weight = finite.astype(float)

    data = _convolve_axis_edge(data, kernel, axis=1)
    weight = _convolve_axis_edge(weight, kernel, axis=1)
    data = _convolve_axis_edge(data, kernel, axis=0)
    weight = _convolve_axis_edge(weight, kernel, axis=0)

    out = np.full_like(arr, np.nan, dtype=float)
    valid = weight > 1.0e-12
    out[valid] = data[valid] / weight[valid]
    return out


def _convolve_axis_edge(arr: np.ndarray, kernel: np.ndarray, axis: int) -> np.ndarray:
    pad = kernel.size // 2
    if axis == 0:
        padded = np.pad(arr, ((pad, pad), (0, 0)), mode="edge")
        out = np.empty_like(arr, dtype=float)
        for col in range(arr.shape[1]):
            out[:, col] = np.convolve(padded[:, col], kernel, mode="valid")
        return out
    padded = np.pad(arr, ((0, 0), (pad, pad)), mode="edge")
    out = np.empty_like(arr, dtype=float)
    for row in range(arr.shape[0]):
        out[row, :] = np.convolve(padded[row, :], kernel, mode="valid")
    return out


def _draw_aperture_outline(
    ax,
    derived: DerivedParams,
    params: BeamParams | None = None,
    z: float = 0.0,
    color: str = "black",
    linewidth: float = 1.0,
) -> None:
    shape = getattr(params, "array_layout", "rectangular") if params is not None else "rectangular"
    half_x = derived.dx_aperture_m / 2.0
    half_y = derived.dy_aperture_m / 2.0
    if shape == "custom":
        x, y = _convex_hull_outline(derived.element_x_m, derived.element_y_m)
    elif shape == "ellipse":
        t = np.linspace(0.0, 2.0 * math.pi, 145)
        x = half_x * np.cos(t)
        y = half_y * np.sin(t)
    elif shape == "diamond":
        x = np.array([0.0, half_x, 0.0, -half_x, 0.0])
        y = np.array([half_y, 0.0, -half_y, 0.0, half_y])
    else:
        x = np.array([-half_x, half_x, half_x, -half_x, -half_x])
        y = np.array([-half_y, -half_y, half_y, half_y, -half_y])
    ax.plot(x, y, np.full_like(x, z, dtype=float), color=color, linewidth=linewidth)


def _convex_hull_outline(x_m: np.ndarray, y_m: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    points = sorted({(float(x), float(y)) for x, y in zip(np.asarray(x_m).ravel(), np.asarray(y_m).ravel())})
    if len(points) == 0:
        return np.array([0.0]), np.array([0.0])
    if len(points) == 1:
        x, y = points[0]
        pad = 0.05
        return np.array([x - pad, x + pad, x + pad, x - pad, x - pad]), np.array([y - pad, y - pad, y + pad, y + pad, y - pad])

    def cross(o, a, b) -> float:
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower: list[tuple[float, float]] = []
    for point in points:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0.0:
            lower.pop()
        lower.append(point)
    upper: list[tuple[float, float]] = []
    for point in reversed(points):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0.0:
            upper.pop()
        upper.append(point)
    hull = lower[:-1] + upper[:-1]
    hull.append(hull[0])
    return np.asarray([p[0] for p in hull], dtype=float), np.asarray([p[1] for p in hull], dtype=float)


def _draw_direction_guides(ax, derived: DerivedParams, length: float, show_labels: bool = True) -> None:
    normal_len = max(length * 0.82, derived.aperture_m, 1.0)
    ax.plot([0.0, 0.0], [0.0, 0.0], [0.0, normal_len], "k--", linewidth=1.2)
    ax.quiver(0, 0, 0, 0, 0, normal_len, color="black", linewidth=1.0, arrow_length_ratio=0.08)
    ax.quiver(0, 0, 0, length * derived.u0, length * derived.v0, length * derived.w0, color="red", linewidth=2.2, arrow_length_ratio=0.08)
    if show_labels:
        ax.text(0.0, 0.0, normal_len * 1.04, "法向", color="black", fontsize=9)
        ax.text(length * derived.u0 * 1.04, length * derived.v0 * 1.04, length * derived.w0 * 1.04, "当前扫描方向", color="red", fontsize=9)


def _direction_length(max_range: float, derived: DerivedParams) -> float:
    if math.isfinite(max_range):
        return 0.38 * max(max_range, derived.aperture_m, 1.0)
    return max(derived.aperture_m, 1.0)


def _format_range(value: float) -> str:
    if not math.isfinite(value):
        return "n/a"
    return f"{value:.4g} m"


def _format_cut_max_line(cut: dict[str, object]) -> str:
    max_range = float(cut.get("max_range_m", float("nan")))
    max_alpha = float(cut.get("max_alpha_deg", float("nan")))
    finite_count = int(cut.get("finite_direction_count", 0))
    total_count = int(cut.get("total_direction_count", 0))
    if not math.isfinite(max_range):
        return f"max r=N/A, finite dirs={finite_count}/{total_count}"
    alpha_text = f"{max_alpha:.3g} deg" if math.isfinite(max_alpha) else "N/A"
    return f"max r={max_range:.4g} m @ alpha={alpha_text}, finite dirs={finite_count}/{total_count}"
