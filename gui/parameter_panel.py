from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
    QGroupBox,
    QLabel,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from core.aperture_shapes import ARRAY_LAYOUT_CHOICES, ELEMENT_SHAPE_CHOICES
from core.geometry import (
    BeamParams,
    get_effective_mode_settings,
    get_mode_settings,
    load_imported_array_layout,
    validate_generated_layout_spacing,
    validate_imported_layout_spacing,
)
from gui.result_panel import ResultPanel


class ParameterPanel(QScrollArea):
    params_changed = Signal()
    calculations_changed = Signal()
    refresh_requested = Signal()
    calculate_requested = Signal()
    export_png_requested = Signal()
    export_csv_requested = Signal()
    save_project_requested = Signal()
    load_project_requested = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWidgetResizable(True)
        self._updating = False
        self._calculation_busy = False
        self._results_dirty = False
        self._parameter_error_text = ""
        body = QWidget()
        self.setWidget(body)
        root = QVBoxLayout(body)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(8)

        self.frequency = self._double(0.001, 300.0, 10.0, 4, 0.1)
        self.nx = self._int(1, 512, 8)
        self.ny = self._int(1, 512, 8)
        self.dx = self._double(1.0e-6, 100.0, 0.10, 6, 0.01)
        self.dy = self._double(1.0e-6, 100.0, 0.10, 6, 0.01)
        self.ax = self._double(1.0e-6, 100.0, 0.10, 6, 0.01)
        self.ay = self._double(1.0e-6, 100.0, 0.10, 6, 0.01)
        self.array_layout = QComboBox()
        for key, label in ARRAY_LAYOUT_CHOICES:
            self.array_layout.addItem(label, key)
        self.element_shape = QComboBox()
        for key, label in ELEMENT_SHAPE_CHOICES:
            self.element_shape.addItem(label, key)
        self.eta = self._double(0.0, 1.0, 0.70, 4, 0.05)
        self.power = self._double(0.0, 1.0e12, 1.0e6, 4, 1000.0)
        self.s0 = self._double(1.0e-12, 1.0e12, 20.0, 4, 1.0)
        self.scan_x = self._double(-89.0, 89.0, 8.0, 4, 0.5)
        self.scan_y = self._double(-89.0, 89.0, 8.0, 4, 0.5)
        self.manual_limit_x = self._double(0.0, 89.0, 8.0, 4, 0.5)
        self.manual_limit_y = self._double(0.0, 89.0, 8.0, 4, 0.5)
        self.limit_mode = QComboBox()
        self.limit_mode.addItems(["自动", "手动"])
        self.calc_mode = QComboBox()
        self.calc_mode.addItems(["快速", "标准", "精细"])
        self.calc_mode.setCurrentIndex(1)
        self.custom_sampling = QCheckBox("自定义计算/显示采样")
        self.custom_sampling.setChecked(False)
        self.sample_2d_alpha = self._int(21, 1441, 181)
        self.sample_2d_range = self._int(8, 800, 110)
        self.sample_3d_theta = self._int(8, 361, 128)
        self.sample_3d_phi = self._int(16, 721, 256)
        self.sample_3d_range = self._int(8, 800, 140)
        self.sample_uv = self._int(31, 801, 301)
        self.sample_union_step = self._double(0.1, 20.0, 1.0, 3, 0.1)
        self.sample_union_theta = self._int(8, 361, 96)
        self.sample_union_phi = self._int(16, 721, 181)
        self.display_3d_grid = self._int(32, 500, 180)
        self.use_element_pattern = QCheckBox("使用单元远场方向图")
        self.use_element_pattern.setChecked(True)
        self.element_pattern_status = QLabel()
        self.element_pattern_status.setWordWrap(True)
        self._update_element_pattern_status("")
        self.element_near_field_status = QLabel()
        self.element_near_field_status.setWordWrap(True)
        self._update_element_near_field_status("")
        self.array_layout_status = QLabel()
        self.array_layout_status.setWordWrap(True)
        self._update_array_layout_status("")
        self.geometry_warning = QLabel()
        self.geometry_warning.setWordWrap(True)
        self.geometry_warning.setStyleSheet("QLabel { color: #b00020; font-weight: 600; }")
        self.geometry_warning.setVisible(False)

        self.calculate_button = QPushButton("计算")
        self.calculate_button.setMinimumHeight(44)
        self.calculate_button.setStyleSheet(
            "QPushButton { background-color: #1769aa; color: white; font-size: 18px; font-weight: 600; border-radius: 4px; }"
            "QPushButton:disabled { background-color: #8aa9c2; }"
            "QPushButton:hover { background-color: #145a92; }"
        )
        root.addWidget(self.calculate_button)

        calc_box = QGroupBox("计算内容")
        calc_layout = QVBoxLayout(calc_box)
        calc_layout.setContentsMargins(8, 8, 8, 8)
        calc_layout.setSpacing(3)
        self.calc_structure = QCheckBox("结构示意图")
        self.calc_cuts = QCheckBox("二维切面")
        self.calc_current_3d = QCheckBox("当前三维包络")
        self.calc_uv = QCheckBox("方向图（u-v/三维）")
        self.calc_scan_union = QCheckBox("扫描并集三维包络")
        for checkbox in (self.calc_structure, self.calc_cuts, self.calc_current_3d, self.calc_uv):
            checkbox.setChecked(True)
        self.calc_scan_union.setChecked(False)
        for checkbox in (self.calc_structure, self.calc_cuts, self.calc_current_3d, self.calc_uv, self.calc_scan_union):
            calc_layout.addWidget(checkbox)
        root.addWidget(calc_box)

        array_box = QGroupBox("阵列参数")
        array_layout = QFormLayout(array_box)
        array_layout.setVerticalSpacing(4)
        array_layout.addRow("频率 f (GHz)", self.frequency)
        array_layout.addRow("x向阵元数 Nx", self.nx)
        array_layout.addRow("y向阵元数 Ny", self.ny)
        array_layout.addRow("x向间距 dx (m)", self.dx)
        array_layout.addRow("y向间距 dy (m)", self.dy)
        array_layout.addRow("单元尺寸 ax (m)", self.ax)
        array_layout.addRow("单元尺寸 ay (m)", self.ay)
        array_layout.addRow("", self.geometry_warning)
        array_layout.addRow("阵列排布", self.array_layout)
        array_layout.addRow("阵元坐标文件", self.array_layout_status)
        array_layout.addRow("单元口径", self.element_shape)
        array_layout.addRow("辐射效率 η", self.eta)
        array_layout.addRow("每单元输入功率 (W)", self.power)
        array_layout.addRow("目标功率密度 S0 (W/cm²)", self.s0)
        array_layout.addRow("单元远场方向图文件", self.element_pattern_status)
        array_layout.addRow("单元近场文件", self.element_near_field_status)
        array_layout.addRow("", self.use_element_pattern)
        root.addWidget(array_box)

        scan_box = QGroupBox("扫描参数")
        scan_layout = QFormLayout(scan_box)
        scan_layout.setVerticalSpacing(4)
        scan_layout.addRow("x-z切面扫描角 (°)", self.scan_x)
        scan_layout.addRow("y-z切面扫描角 (°)", self.scan_y)
        scan_layout.addRow("最大扫描角模式", self.limit_mode)
        scan_layout.addRow("手动X最大扫描角 (°)", self.manual_limit_x)
        scan_layout.addRow("手动Y最大扫描角 (°)", self.manual_limit_y)
        scan_layout.addRow("计算精度", self.calc_mode)
        root.addWidget(scan_box)

        sampling_box = QGroupBox("采样设置")
        sampling_layout = QFormLayout(sampling_box)
        sampling_layout.setVerticalSpacing(4)
        sampling_layout.addRow("", self.custom_sampling)
        sampling_layout.addRow("二维角度采样", self.sample_2d_alpha)
        sampling_layout.addRow("二维径向采样", self.sample_2d_range)
        sampling_layout.addRow("三维θ采样", self.sample_3d_theta)
        sampling_layout.addRow("三维φ采样", self.sample_3d_phi)
        sampling_layout.addRow("三维径向采样", self.sample_3d_range)
        sampling_layout.addRow("u-v边长采样", self.sample_uv)
        sampling_layout.addRow("扫描中心步进 (°)", self.sample_union_step)
        sampling_layout.addRow("并集θ采样", self.sample_union_theta)
        sampling_layout.addRow("并集φ采样", self.sample_union_phi)
        sampling_layout.addRow("3D显示网格上限", self.display_3d_grid)
        root.addWidget(sampling_box)

        self.results = ResultPanel()
        root.addWidget(self.results)

        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        root.addWidget(line)
        root.addStretch(1)

        self._wire_signals()
        self._sync_sampling_from_mode()
        self._update_limit_enabled()
        self._update_sampling_enabled()
        self._update_geometry_warning()

    def _wire_signals(self) -> None:
        widgets = [
            self.frequency,
            self.nx,
            self.ny,
            self.dx,
            self.dy,
            self.ax,
            self.ay,
            self.eta,
            self.power,
            self.s0,
            self.scan_x,
            self.scan_y,
            self.manual_limit_x,
            self.manual_limit_y,
        ]
        for widget in widgets:
            widget.valueChanged.connect(self._emit_changed)
        self.array_layout.currentIndexChanged.connect(self._emit_changed)
        self.element_shape.currentIndexChanged.connect(self._emit_changed)
        self.limit_mode.currentIndexChanged.connect(self._on_limit_mode_changed)
        self.calc_mode.currentIndexChanged.connect(self._on_calc_mode_changed)
        self.custom_sampling.toggled.connect(self._on_custom_sampling_changed)
        for widget in (
            self.sample_2d_alpha,
            self.sample_2d_range,
            self.sample_3d_theta,
            self.sample_3d_phi,
            self.sample_3d_range,
            self.sample_uv,
            self.sample_union_step,
            self.sample_union_theta,
            self.sample_union_phi,
            self.display_3d_grid,
        ):
            widget.valueChanged.connect(self._emit_changed)
        self.use_element_pattern.toggled.connect(self._emit_changed)
        for checkbox in (self.calc_structure, self.calc_cuts, self.calc_current_3d, self.calc_uv, self.calc_scan_union):
            checkbox.toggled.connect(self._emit_calculations_changed)
        self.calculate_button.clicked.connect(self.calculate_requested)

    def _emit_changed(self) -> None:
        if not self._updating and not self.custom_sampling.isChecked():
            self._sync_sampling_from_mode()
        self._update_geometry_warning()
        if not self._updating:
            self.params_changed.emit()

    def _emit_calculations_changed(self) -> None:
        if not self._updating:
            self.calculations_changed.emit()

    def _on_limit_mode_changed(self) -> None:
        self._update_limit_enabled()
        self._emit_changed()

    def _on_calc_mode_changed(self) -> None:
        if not self.custom_sampling.isChecked():
            self._sync_sampling_from_mode()
        self._emit_changed()

    def _on_custom_sampling_changed(self) -> None:
        self._update_sampling_enabled()
        self._emit_changed()

    def _update_limit_enabled(self) -> None:
        manual = self.limit_mode.currentIndex() == 1
        self.manual_limit_x.setEnabled(manual)
        self.manual_limit_y.setEnabled(manual)

    def _update_sampling_enabled(self) -> None:
        enabled = self.custom_sampling.isChecked()
        for widget in (
            self.sample_2d_alpha,
            self.sample_2d_range,
            self.sample_3d_theta,
            self.sample_3d_phi,
            self.sample_3d_range,
            self.sample_uv,
            self.sample_union_step,
            self.sample_union_theta,
            self.sample_union_phi,
            self.display_3d_grid,
        ):
            widget.setEnabled(enabled)

    def _sync_sampling_from_mode(self) -> None:
        try:
            preview_params = self.params()
            preview_params.custom_sampling_enabled = False
            settings = get_effective_mode_settings(preview_params)
        except Exception:
            settings = get_mode_settings(["fast", "standard", "fine"][self.calc_mode.currentIndex()])
        blocked = [
            self.sample_2d_alpha,
            self.sample_2d_range,
            self.sample_3d_theta,
            self.sample_3d_phi,
            self.sample_3d_range,
            self.sample_uv,
            self.sample_union_step,
            self.sample_union_theta,
            self.sample_union_phi,
            self.display_3d_grid,
        ]
        old_states = [widget.blockSignals(True) for widget in blocked]
        try:
            self.sample_2d_alpha.setValue(settings.n_alpha_2d)
            self.sample_2d_range.setValue(settings.n_range_2d)
            self.sample_3d_theta.setValue(settings.theta_3d_n)
            self.sample_3d_phi.setValue(settings.phi_3d_n)
            self.sample_3d_range.setValue(settings.n_range_3d)
            self.sample_uv.setValue(settings.n_uv)
            self.sample_union_step.setValue(settings.scan_union_step_deg)
            self.sample_union_theta.setValue(settings.scan_union_theta_n)
            self.sample_union_phi.setValue(settings.scan_union_phi_n)
            self.display_3d_grid.setValue(min(260, max(180, settings.phi_3d_n, settings.scan_union_phi_n)))
        finally:
            for widget, old_state in zip(blocked, old_states):
                widget.blockSignals(old_state)

    def _update_geometry_warning(self) -> None:
        errors: list[str] = []
        array_layout_key = str(self.array_layout.currentData() or "rectangular")
        element_shape_key = str(self.element_shape.currentData() or "rectangular")
        imported_layout = array_layout_key == "custom"
        geometry_params = BeamParams(
            nx=self.nx.value(),
            ny=self.ny.value(),
            dx_m=self.dx.value(),
            dy_m=self.dy.value(),
            ax_m=self.ax.value(),
            ay_m=self.ay.value(),
            array_layout=array_layout_key,
            element_shape=element_shape_key,
            element_power_w=self.power.value(),
        )
        if not imported_layout:
            try:
                validate_generated_layout_spacing(geometry_params)
            except Exception as exc:
                errors.append(str(exc).splitlines()[0])
        if imported_layout and not (self.array_layout_status.property("layout_path") or ""):
            errors.append("导入坐标CSV排布需要先通过 File 菜单加载阵元坐标文件。")
        elif imported_layout:
            path = str(self.array_layout_status.property("layout_path") or "")
            try:
                table = load_imported_array_layout(path, self.power.value())
                validate_imported_layout_spacing(
                    geometry_params,
                    table.x_m,
                    table.y_m,
                )
            except Exception as exc:
                errors.append(str(exc).splitlines()[0])
        self.geometry_warning.setText(" ".join(errors))
        self.geometry_warning.setVisible(bool(errors))
        self._parameter_error_text = "\n".join(errors)
        self._sync_calculate_button_state()

    def has_parameter_errors(self) -> bool:
        return bool(self._parameter_error_text)

    def parameter_error_text(self) -> str:
        return self._parameter_error_text

    def set_calculation_busy(self, busy: bool) -> None:
        self._calculation_busy = bool(busy)
        self._sync_calculate_button_state()

    def set_results_dirty(self, dirty: bool) -> None:
        self._results_dirty = bool(dirty)
        self._sync_calculate_button_state()

    def _sync_calculate_button_state(self) -> None:
        enabled = (not self._calculation_busy) and (not self._parameter_error_text)
        self.calculate_button.setEnabled(enabled)
        if self._calculation_busy:
            self.calculate_button.setText("正在计算...")
            self.calculate_button.setToolTip("Calculation is running.")
        elif self._parameter_error_text:
            self.calculate_button.setText("计算")
            self.calculate_button.setToolTip(self._parameter_error_text)
        else:
            self.calculate_button.setText("计算（参数已修改）" if self._results_dirty else "计算")
            self.calculate_button.setToolTip("右侧仍为上次计算结果，点击后刷新。 " if self._results_dirty else "")

    def _double(self, minimum: float, maximum: float, value: float, decimals: int, step: float) -> QDoubleSpinBox:
        box = QDoubleSpinBox()
        box.setRange(minimum, maximum)
        box.setDecimals(decimals)
        box.setSingleStep(step)
        box.setValue(value)
        box.setKeyboardTracking(False)
        return box

    def _int(self, minimum: int, maximum: int, value: int) -> QSpinBox:
        box = QSpinBox()
        box.setRange(minimum, maximum)
        box.setValue(value)
        box.setKeyboardTracking(False)
        return box

    def params(self) -> BeamParams:
        array_layout_key = str(self.array_layout.currentData() or "rectangular")
        element_shape_key = str(self.element_shape.currentData() or "rectangular")
        return BeamParams(
            frequency_ghz=self.frequency.value(),
            nx=self.nx.value(),
            ny=self.ny.value(),
            dx_m=self.dx.value(),
            dy_m=self.dy.value(),
            ax_m=self.ax.value(),
            ay_m=self.ay.value(),
            efficiency=self.eta.value(),
            element_power_w=self.power.value(),
            s0_w_cm2=self.s0.value(),
            scan_x_deg=self.scan_x.value(),
            scan_y_deg=self.scan_y.value(),
            scan_limit_mode="manual" if self.limit_mode.currentIndex() == 1 else "auto",
            manual_scan_limit_x_deg=self.manual_limit_x.value(),
            manual_scan_limit_y_deg=self.manual_limit_y.value(),
            calc_mode=["fast", "standard", "fine"][self.calc_mode.currentIndex()],
            use_element_pattern=self.use_element_pattern.isChecked(),
            array_layout=array_layout_key,
            element_shape=element_shape_key,
            array_layout_file=self.array_layout_status.property("layout_path") or "",
            element_pattern_file=self.element_pattern_status.property("pattern_path") or "",
            element_near_field_file=self.element_near_field_status.property("near_field_path") or "",
            custom_sampling_enabled=self.custom_sampling.isChecked(),
            sample_2d_alpha_n=self.sample_2d_alpha.value(),
            sample_2d_range_n=self.sample_2d_range.value(),
            sample_3d_theta_n=self.sample_3d_theta.value(),
            sample_3d_phi_n=self.sample_3d_phi.value(),
            sample_3d_range_n=self.sample_3d_range.value(),
            sample_uv_n=self.sample_uv.value(),
            sample_scan_union_step_deg=self.sample_union_step.value(),
            sample_scan_union_theta_n=self.sample_union_theta.value(),
            sample_scan_union_phi_n=self.sample_union_phi.value(),
            display_3d_grid_n=self.display_3d_grid.value(),
        )

    def selected_calculations(self) -> dict[str, bool]:
        return {
            "structure": self.calc_structure.isChecked(),
            "cuts": self.calc_cuts.isChecked(),
            "current_3d": self.calc_current_3d.isChecked(),
            "uv": self.calc_uv.isChecked(),
            "scan_union": self.calc_scan_union.isChecked(),
        }

    def set_selected_calculations(self, selected: dict[str, bool]) -> None:
        self._updating = True
        changed = False
        boxes = {
            "structure": self.calc_structure,
            "cuts": self.calc_cuts,
            "current_3d": self.calc_current_3d,
            "uv": self.calc_uv,
            "scan_union": self.calc_scan_union,
        }
        try:
            for key, box in boxes.items():
                if key in selected:
                    checked = bool(selected[key])
                    changed = changed or box.isChecked() != checked
                    box.setChecked(checked)
        finally:
            self._updating = False
        if changed:
            self.calculations_changed.emit()

    def set_params(self, params: BeamParams) -> None:
        self._validate_set_params_ranges(params)
        self._updating = True
        try:
            self.frequency.setValue(params.frequency_ghz)
            self.nx.setValue(params.nx)
            self.ny.setValue(params.ny)
            self.dx.setValue(params.dx_m)
            self.dy.setValue(params.dy_m)
            self.ax.setValue(params.ax_m)
            self.ay.setValue(params.ay_m)
            self._set_combo_by_data(self.array_layout, getattr(params, "array_layout", "rectangular"))
            self._set_combo_by_data(self.element_shape, getattr(params, "element_shape", "rectangular"))
            self.eta.setValue(params.efficiency)
            self.power.setValue(params.element_power_w)
            self.s0.setValue(params.s0_w_cm2)
            self.scan_x.setValue(params.scan_x_deg)
            self.scan_y.setValue(params.scan_y_deg)
            self.limit_mode.setCurrentIndex(1 if params.scan_limit_mode == "manual" else 0)
            self.manual_limit_x.setValue(params.manual_scan_limit_x_deg)
            self.manual_limit_y.setValue(params.manual_scan_limit_y_deg)
            mode_map = {"fast": 0, "standard": 1, "fine": 2}
            self.calc_mode.setCurrentIndex(mode_map.get(params.calc_mode, 1))
            self.custom_sampling.setChecked(bool(getattr(params, "custom_sampling_enabled", False)))
            self.sample_2d_alpha.setValue(int(getattr(params, "sample_2d_alpha_n", self.sample_2d_alpha.value())))
            self.sample_2d_range.setValue(int(getattr(params, "sample_2d_range_n", self.sample_2d_range.value())))
            self.sample_3d_theta.setValue(int(getattr(params, "sample_3d_theta_n", self.sample_3d_theta.value())))
            self.sample_3d_phi.setValue(int(getattr(params, "sample_3d_phi_n", self.sample_3d_phi.value())))
            self.sample_3d_range.setValue(int(getattr(params, "sample_3d_range_n", self.sample_3d_range.value())))
            self.sample_uv.setValue(int(getattr(params, "sample_uv_n", self.sample_uv.value())))
            self.sample_union_step.setValue(float(getattr(params, "sample_scan_union_step_deg", self.sample_union_step.value())))
            self.sample_union_theta.setValue(int(getattr(params, "sample_scan_union_theta_n", self.sample_union_theta.value())))
            self.sample_union_phi.setValue(int(getattr(params, "sample_scan_union_phi_n", self.sample_union_phi.value())))
            self.display_3d_grid.setValue(int(getattr(params, "display_3d_grid_n", self.display_3d_grid.value())))
            self.use_element_pattern.setChecked(params.use_element_pattern)
            self._update_array_layout_status(getattr(params, "array_layout_file", ""))
            self._update_element_pattern_status(getattr(params, "element_pattern_file", ""))
            self._update_element_near_field_status(getattr(params, "element_near_field_file", ""))
            if not self.custom_sampling.isChecked():
                self._sync_sampling_from_mode()
        finally:
            self._updating = False
        self._update_limit_enabled()
        self._update_sampling_enabled()
        self._update_geometry_warning()
        self.params_changed.emit()

    def set_element_pattern_summary(
        self,
        path: str,
        *,
        point_count: int | None = None,
        has_phase: bool | None = None,
        has_vector_components: bool | None = None,
        gain_norm: float | None = None,
        theta_max_deg: float | None = None,
        covers_visible_edge: bool | None = None,
        selected_frequency_ghz: float | None = None,
        active: bool | None = None,
        error: str | None = None,
    ) -> None:
        self._update_element_pattern_status(
            path,
            point_count=point_count,
            has_phase=has_phase,
            has_vector_components=has_vector_components,
            gain_norm=gain_norm,
            theta_max_deg=theta_max_deg,
            covers_visible_edge=covers_visible_edge,
            selected_frequency_ghz=selected_frequency_ghz,
            active=active,
            error=error,
        )

    def set_array_layout_summary(
        self,
        path: str,
        *,
        point_count: int | None = None,
        total_power_w: float | None = None,
        aperture_x_m: float | None = None,
        aperture_y_m: float | None = None,
        active: bool | None = None,
        error: str | None = None,
    ) -> None:
        self._update_array_layout_status(
            path,
            point_count=point_count,
            total_power_w=total_power_w,
            aperture_x_m=aperture_x_m,
            aperture_y_m=aperture_y_m,
            active=active,
            error=error,
        )

    def set_element_near_field_summary(
        self,
        path: str,
        *,
        point_count: int | None = None,
        has_phase: bool | None = None,
        has_vector_components: bool | None = None,
        has_power_density: bool | None = None,
        selected_frequency_ghz: float | None = None,
        x_span_m: float | None = None,
        y_span_m: float | None = None,
        z_min_m: float | None = None,
        z_max_m: float | None = None,
        active: bool | None = None,
        error: str | None = None,
    ) -> None:
        self._update_element_near_field_status(
            path,
            point_count=point_count,
            has_phase=has_phase,
            has_vector_components=has_vector_components,
            has_power_density=has_power_density,
            selected_frequency_ghz=selected_frequency_ghz,
            x_span_m=x_span_m,
            y_span_m=y_span_m,
            z_min_m=z_min_m,
            z_max_m=z_max_m,
            active=active,
            error=error,
        )

    def _update_element_pattern_status(
        self,
        path: str,
        *,
        point_count: int | None = None,
        has_phase: bool | None = None,
        has_vector_components: bool | None = None,
        gain_norm: float | None = None,
        theta_max_deg: float | None = None,
        covers_visible_edge: bool | None = None,
        selected_frequency_ghz: float | None = None,
        active: bool | None = None,
        error: str | None = None,
    ) -> None:
        path = str(path or "")
        self.element_pattern_status.setProperty("pattern_path", path)
        if path:
            details: list[str] = []
            if error:
                details.append(error)
            if active is False:
                details.append("未启用")
            if point_count is not None:
                details.append(f"{int(point_count)} 点")
            if selected_frequency_ghz is not None:
                details.append(f"f={float(selected_frequency_ghz):.6g} GHz")
            if has_phase is not None:
                details.append("含相位/复数场" if has_phase else "纯增益")
            if has_vector_components:
                details.append("矢量Eθ/Eφ")
            if theta_max_deg is not None:
                details.append(f"θmax={float(theta_max_deg):.1f}°")
            if covers_visible_edge is False:
                details.append("未覆盖90°边缘")
            if gain_norm is not None:
                details.append(f"Gnorm={float(gain_norm):.4g}")
            details.append("参与方向图/包络/扫描并集计算")
            text = f"远场文件: {Path(path).name}"
            if details:
                text += "\n" + " | ".join(details)
            self.element_pattern_status.setText(text)
            tooltip = [str(Path(path).resolve())]
            if details:
                tooltip.append("；".join(details))
            self.element_pattern_status.setToolTip("\n".join(tooltip))
        else:
            self.element_pattern_status.setText("内置解析远场模型")
            self.element_pattern_status.setToolTip("未导入实际单元远场方向图文件；主计算使用当前单元口径解析模型。")

    def _update_element_near_field_status(
        self,
        path: str,
        *,
        point_count: int | None = None,
        has_phase: bool | None = None,
        has_vector_components: bool | None = None,
        has_power_density: bool | None = None,
        selected_frequency_ghz: float | None = None,
        x_span_m: float | None = None,
        y_span_m: float | None = None,
        z_min_m: float | None = None,
        z_max_m: float | None = None,
        active: bool | None = None,
        error: str | None = None,
    ) -> None:
        path = str(path or "")
        self.element_near_field_status.setProperty("near_field_path", path)
        if path:
            details: list[str] = []
            if error:
                details.append(error)
            if active is False:
                details.append("未启用")
            if point_count is not None:
                details.append(f"{int(point_count)} 点")
            if selected_frequency_ghz is not None:
                details.append(f"f={float(selected_frequency_ghz):.6g} GHz")
            if has_vector_components:
                details.append("矢量Ex/Ey/Ez")
            if has_power_density:
                details.append("含S")
            if has_phase is not None:
                details.append("含相位" if has_phase else "幅度/实值")
            if x_span_m is not None and y_span_m is not None:
                details.append(f"xy范围 {float(x_span_m):.4g}×{float(y_span_m):.4g} m")
            if z_min_m is not None and z_max_m is not None:
                details.append(f"z={float(z_min_m):.4g}..{float(z_max_m):.4g} m")
            details.append("实验接口：当前不参与主包络计算")
            text = f"近场文件: {Path(path).name}"
            if details:
                text += "\n" + " | ".join(details)
            self.element_near_field_status.setText(text)
            tooltip = [str(Path(path).resolve()), "近场导入为实验接口，当前版本只校验/保存数据，不改变主包络计算。"]
            if details:
                tooltip.append(" | ".join(details))
            self.element_near_field_status.setToolTip("\n".join(tooltip))
        else:
            self.element_near_field_status.setText("未导入（实验接口）")
            self.element_near_field_status.setToolTip("单元近场文件入口已预留；当前主计算仍使用解析/远场单元模型。")

    def _update_array_layout_status(
        self,
        path: str,
        *,
        point_count: int | None = None,
        total_power_w: float | None = None,
        aperture_x_m: float | None = None,
        aperture_y_m: float | None = None,
        active: bool | None = None,
        error: str | None = None,
    ) -> None:
        path = str(path or "")
        self.array_layout_status.setProperty("layout_path", path)
        if path:
            details: list[str] = []
            if error:
                details.append(error)
            if active is False:
                details.append("未启用")
            if point_count is not None:
                details.append(f"有效单元 {int(point_count)}")
            if total_power_w is not None:
                details.append(f"总功率 {float(total_power_w):.4g} W")
            if aperture_x_m is not None and aperture_y_m is not None:
                details.append(f"口径 {float(aperture_x_m):.4g}×{float(aperture_y_m):.4g} m")
            text = f"CSV: {Path(path).name}"
            if details:
                text += "\n" + " | ".join(details)
            self.array_layout_status.setText(text)
            tooltip = [str(Path(path).resolve())]
            if details:
                tooltip.append(" | ".join(details))
            self.array_layout_status.setToolTip("\n".join(tooltip))
        else:
            self.array_layout_status.setText("使用参数生成")
            self.array_layout_status.setToolTip("未导入阵元坐标 CSV")

    def _validate_set_params_ranges(self, params: BeamParams) -> None:
        checks = [
            ("frequency f", self.frequency, float(params.frequency_ghz)),
            ("Nx", self.nx, int(params.nx)),
            ("Ny", self.ny, int(params.ny)),
            ("dx", self.dx, float(params.dx_m)),
            ("dy", self.dy, float(params.dy_m)),
            ("ax", self.ax, float(params.ax_m)),
            ("ay", self.ay, float(params.ay_m)),
            ("efficiency eta", self.eta, float(params.efficiency)),
            ("element power", self.power, float(params.element_power_w)),
            ("S0", self.s0, float(params.s0_w_cm2)),
            ("scanX", self.scan_x, float(params.scan_x_deg)),
            ("scanY", self.scan_y, float(params.scan_y_deg)),
            ("manual scan limit X", self.manual_limit_x, float(params.manual_scan_limit_x_deg)),
            ("manual scan limit Y", self.manual_limit_y, float(params.manual_scan_limit_y_deg)),
            ("2D angle samples", self.sample_2d_alpha, int(params.sample_2d_alpha_n)),
            ("2D range samples", self.sample_2d_range, int(params.sample_2d_range_n)),
            ("3D theta samples", self.sample_3d_theta, int(params.sample_3d_theta_n)),
            ("3D phi samples", self.sample_3d_phi, int(params.sample_3d_phi_n)),
            ("3D range samples", self.sample_3d_range, int(params.sample_3d_range_n)),
            ("u-v samples", self.sample_uv, int(params.sample_uv_n)),
            ("scan union step", self.sample_union_step, float(params.sample_scan_union_step_deg)),
            ("scan union theta samples", self.sample_union_theta, int(params.sample_scan_union_theta_n)),
            ("scan union phi samples", self.sample_union_phi, int(params.sample_scan_union_phi_n)),
            ("3D display grid", self.display_3d_grid, int(params.display_3d_grid_n)),
        ]
        errors: list[str] = []
        for label, widget, value in checks:
            minimum = float(widget.minimum())
            maximum = float(widget.maximum())
            if not minimum <= value <= maximum:
                errors.append(f"{label}={value:g}, allowed [{minimum:g}, {maximum:g}]")
        if errors:
            raise ValueError("Project parameters exceed UI limits:\n" + "\n".join(errors))

    @staticmethod
    def _set_combo_by_data(combo: QComboBox, value: str) -> None:
        index = combo.findData(str(value))
        combo.setCurrentIndex(index if index >= 0 else 0)
