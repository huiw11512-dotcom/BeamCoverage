from __future__ import annotations

import math

from PySide6.QtWidgets import QFormLayout, QLabel, QGroupBox, QWidget

from core.aperture_shapes import shape_label
from core.element_pattern import ElementPattern
from core.geometry import BeamParams, DerivedParams


def _fmt(value: float, unit: str = "", precision: int = 6) -> str:
    if value is None or not math.isfinite(float(value)):
        return "n/a"
    text = f"{value:.{precision}g}"
    return f"{text} {unit}".strip()


class ResultPanel(QGroupBox):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__("自动计算结果", parent)
        self._labels: dict[str, QLabel] = {}
        layout = QFormLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setHorizontalSpacing(10)
        layout.setVerticalSpacing(4)
        rows = [
            ("array_source", "阵列坐标来源"),
            ("active_elements", "有效阵元数"),
            ("element_model", "单元远场模型"),
            ("current_envelope_method", "当前包络方法"),
            ("scan_union_method", "扫描并集方法"),
            ("lambda", "波长 λ"),
            ("dx_ap", "阵列口径 Dx"),
            ("dy_ap", "阵列口径 Dy"),
            ("rff", "远场边界 Rff"),
            ("hpbw_x", "HPBWx"),
            ("hpbw_y", "HPBWy"),
            ("limit_x", "经验最大扫描角 X"),
            ("limit_y", "经验最大扫描角 Y"),
            ("theta", "当前等效 θ"),
            ("phi", "当前等效 φ"),
            ("scan_loss", "当前扫描损失"),
            ("phase_x", "x向相邻单元相位差"),
            ("phase_y", "y向相邻单元相位差"),
            ("total_power", "总输入功率"),
            ("calc_mode", "当前计算模式"),
            ("time_total", "总计算时间"),
            ("time_cuts", "二维切面耗时"),
            ("time_3d", "三维包络耗时"),
            ("time_uv", "方向图耗时"),
            ("time_union", "扫描并集耗时"),
            ("time_union_3d", "扫描并集3D预览"),
            ("time_union_2d", "扫描并集2D切面"),
            ("time_union_cache", "扫描并集缓存"),
        ]
        for key, title in rows:
            label = QLabel("n/a")
            label.setTextInteractionFlags(label.textInteractionFlags())
            self._labels[key] = label
            layout.addRow(title, label)

    def update_results(self, d: DerivedParams, params: BeamParams | None = None, elem: ElementPattern | None = None) -> None:
        self._labels["array_source"].setText(_array_source_text(d, params))
        self._labels["active_elements"].setText(str(int(d.element_x_m.size)))
        self._labels["element_model"].setText(_element_model_text(params, elem))
        self._labels["current_envelope_method"].setText("近场采样+远场外推")
        self._labels["scan_union_method"].setText("3D远场；2D近场")
        self._labels["lambda"].setText(_fmt(d.wavelength_m, "m"))
        self._labels["dx_ap"].setText(_fmt(d.dx_aperture_m, "m"))
        self._labels["dy_ap"].setText(_fmt(d.dy_aperture_m, "m"))
        self._labels["rff"].setText(f"{_fmt(d.rff_m, 'm')} / {_fmt(d.rff_m / 1000.0, 'km', 4)}")
        self._labels["hpbw_x"].setText(_fmt(d.hpbw_x_deg, "deg", 5))
        self._labels["hpbw_y"].setText(_fmt(d.hpbw_y_deg, "deg", 5))
        self._labels["limit_x"].setText(_fmt(d.scan_limit_x_deg, "deg", 5))
        self._labels["limit_y"].setText(_fmt(d.scan_limit_y_deg, "deg", 5))
        self._labels["theta"].setText(_fmt(d.theta_deg, "deg", 5))
        self._labels["phi"].setText(_fmt(d.phi_deg, "deg", 5))
        self._labels["scan_loss"].setText(_fmt(d.scan_loss_db, "dB", 5))
        self._labels["phase_x"].setText(f"{_fmt(d.phase_step_x_deg, 'deg', 5)} ({_fmt(d.phase_step_x_mod_deg, 'deg 等效', 5)})")
        self._labels["phase_y"].setText(f"{_fmt(d.phase_step_y_deg, 'deg', 5)} ({_fmt(d.phase_step_y_mod_deg, 'deg 等效', 5)})")
        self._labels["total_power"].setText(_fmt(d.total_input_power_w, "W"))

    def clear_results(self) -> None:
        for key in (
            "lambda",
            "array_source",
            "active_elements",
            "element_model",
            "current_envelope_method",
            "scan_union_method",
            "dx_ap",
            "dy_ap",
            "rff",
            "hpbw_x",
            "hpbw_y",
            "limit_x",
            "limit_y",
            "theta",
            "phi",
            "scan_loss",
            "phase_x",
            "phase_y",
            "total_power",
        ):
            self._labels[key].setText("n/a")

    def update_timings(self, mode: str, timings: dict[str, float | bool] | None) -> None:
        mode_key = str(mode).lower()
        mode_text = {"fast": "快速", "standard": "标准", "fine": "精细"}.get(mode_key, str(mode))
        if mode_key.endswith("+custom"):
            base = mode_key.split("+", 1)[0]
            mode_text = {"fast": "快速", "standard": "标准", "fine": "精细"}.get(base, base) + " + 自定义采样"
        elif mode_key.endswith("+auto"):
            base = mode_key.split("+", 1)[0]
            mode_text = {"fast": "快速", "standard": "标准", "fine": "精细"}.get(base, base) + " + 自动采样"
        self._labels["calc_mode"].setText(mode_text)
        if not timings:
            for key in ("time_total", "time_cuts", "time_3d", "time_uv", "time_union", "time_union_3d", "time_union_2d", "time_union_cache"):
                self._labels[key].setText("未计算")
            return
        self._labels["time_total"].setText(_fmt(float(timings.get("total_s", float("nan"))), "s", 4))
        self._labels["time_cuts"].setText(_fmt(float(timings.get("cuts_s", float("nan"))), "s", 4))
        self._labels["time_3d"].setText(_fmt(float(timings.get("current_3d_s", float("nan"))), "s", 4))
        self._labels["time_uv"].setText(_fmt(float(timings.get("uv_s", float("nan"))), "s", 4))
        self._labels["time_union"].setText(_fmt(float(timings.get("scan_union_s", float("nan"))), "s", 4))
        self._labels["time_union_3d"].setText(_fmt(float(timings.get("scan_union_3d_s", float("nan"))), "s", 4))
        self._labels["time_union_2d"].setText(_fmt(float(timings.get("scan_union_2d_cuts_s", float("nan"))), "s", 4))
        if timings.get("scan_union_skipped"):
            self._labels["time_union"].setText("未勾选")
            self._labels["time_union_3d"].setText("未计算")
            self._labels["time_union_2d"].setText("未计算")
            self._labels["time_union_cache"].setText("未计算")
        else:
            if timings.get("scan_union_cache_hit") and timings.get("scan_union_current_cache_hit"):
                self._labels["time_union_cache"].setText("完全命中")
            elif timings.get("scan_union_cache_hit") and timings.get("scan_union_current_overlay"):
                self._labels["time_union_cache"].setText("基础命中+当前增量")
            else:
                self._labels["time_union_cache"].setText("命中" if timings.get("scan_union_cache_hit") else "未命中")
            if timings.get("scan_union_cache_hit") and (not timings.get("scan_union_current_overlay") or timings.get("scan_union_current_cache_hit")):
                self._labels["time_union_3d"].setText("缓存命中")
                self._labels["time_union_2d"].setText("缓存命中")


def _array_source_text(d: DerivedParams, params: BeamParams | None) -> str:
    if d.imported_array_layout:
        return "导入坐标CSV"
    if params is None:
        return "参数生成"
    return shape_label(getattr(params, "array_layout", "rectangular"), kind="array", default="参数生成")


def _element_model_text(params: BeamParams | None, elem: ElementPattern | None) -> str:
    if elem is not None and elem.mode == "table":
        return "导入远场矢量Eθ/Eφ" if elem.table_has_vector_components else "导入远场方向图"
    if params is not None and not getattr(params, "use_element_pattern", True):
        return "各向同性"
    if params is None:
        return "解析远场口径"
    return shape_label(getattr(params, "element_shape", "rectangular"), kind="element_model", default="解析远场口径")
