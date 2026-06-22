from __future__ import annotations

from copy import deepcopy
from datetime import date
import json
import os
from pathlib import Path
from time import perf_counter
import traceback
from typing import Any

import numpy as np
from PySide6.QtCore import QSettings, QTimer, Qt, QUrl
from PySide6.QtGui import QAction, QDesktopServices, QGuiApplication, QIcon
from PySide6.QtWidgets import (
    QFileDialog,
    QMainWindow,
    QMessageBox,
    QSplitter,
    QStatusBar,
    QWidget,
)

from app_info import APP_TITLE, APP_VERSION, resource_path
from core.array_factor import calc_uv_pattern
from core.analysis import derive_params_with_element
from core.element_pattern import load_imported_element_pattern
from core.envelope import compute_2d_cuts, compute_current_3d_envelope
from core.geometry import BeamParams, base_cache_key, derive_params, get_effective_mode_settings, get_mode_settings, load_imported_array_layout, sanitized_params
from core.near_field import export_near_field_projected_far_field_pattern, load_imported_element_near_field, near_field_summary
from core.scan_union import compute_scan_union_base_envelope_3d, merge_scan_union_with_current_scan
from core.update_checker import check_for_update, format_update_message
from export.csv_export import export_2d_cuts, export_current_3d, export_scan_union, export_structure, export_uv_pattern
from export.csv_templates import (
    export_current_array_layout_template,
    export_element_near_field_vector_template,
    export_element_pattern_abs_phase_template,
    export_element_pattern_vector_template,
)
from export.excel_export import export_excel_report
from export.png_export import export_figure_png
from gui.parameter_panel import ParameterPanel
from gui.plot_panel import PlotPanel


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle(APP_TITLE)
        icon_path = resource_path("resources", "beamcoverage_icon.png")
        if icon_path.exists():
            self.setWindowIcon(QIcon(str(icon_path)))
        self.resize(1400, 860)

        self.parameter_panel = ParameterPanel()
        self.plot_panel = PlotPanel()
        splitter = QSplitter(Qt.Orientation.Horizontal)
        self.splitter = splitter
        splitter.addWidget(self.parameter_panel)
        splitter.addWidget(self.plot_panel)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        self.parameter_panel.setMinimumWidth(360)
        self.parameter_panel.setMaximumWidth(420)
        splitter.setSizes([390, 1010])
        self.setCentralWidget(splitter)

        self.setStatusBar(QStatusBar(self))
        self._pending_timer = QTimer(self)
        self._pending_timer.setSingleShot(True)
        self._pending_timer.setInterval(150)
        self._pending_timer.timeout.connect(self.refresh_auto)
        self._session_settings = QSettings("BeamCoverage", "BeamCoverage")
        self._results_dirty = False
        self._load_last_session()

        self.params: BeamParams = self.parameter_panel.params()
        self.derived = None
        self.elem = None
        self.settings = get_effective_mode_settings(self.params)
        self.cuts: list[dict[str, Any]] | None = None
        self.current_3d: dict[str, Any] | None = None
        self.uv_pattern: dict[str, Any] | None = None
        self.scan_union: dict[str, Any] | None = None
        self._scan_union_cache: dict[tuple[Any, ...], dict[str, Any]] = {}
        self._scan_union_current_cache: dict[tuple[Any, ...], dict[str, Any]] = {}
        self.last_timings: dict[str, float | bool] | None = None
        self._last_selected_calculations = self.parameter_panel.selected_calculations()

        self._build_menu()
        self._wire_signals()
        self.refresh_auto()
        QTimer.singleShot(2000, self._check_updates_on_startup)

    def _build_menu(self) -> None:
        file_menu = self.menuBar().addMenu("文件")
        save_action = QAction("保存工程", self)
        load_action = QAction("加载工程", self)
        load_array_layout_action = QAction("导入阵元坐标 CSV", self)
        clear_array_layout_action = QAction("清除阵元坐标 CSV", self)
        export_array_layout_template_action = QAction("导出当前阵元坐标 CSV", self)
        load_element_pattern_action = QAction("导入单元远场方向图文件", self)
        clear_element_pattern_action = QAction("清除单元远场方向图文件", self)
        export_element_pattern_template_action = QAction("导出单元远场 Real/Imag 模板", self)
        export_element_pattern_abs_phase_template_action = QAction("导出单元远场幅相模板", self)
        load_element_near_field_action = QAction("导入单元近场文件", self)
        clear_element_near_field_action = QAction("清除单元近场文件", self)
        export_element_near_field_template_action = QAction("导出单元近场 Ex/Ey/Ez 模板", self)
        export_near_field_as_far_field_action = QAction("从单元近场导出远场方向图 CSV", self)
        png_action = QAction("导出当前 PNG", self)
        csv_action = QAction("导出当前 CSV", self)
        excel_action = QAction("导出 Excel 报告", self)
        exit_action = QAction("退出", self)
        save_action.triggered.connect(self.save_project)
        load_action.triggered.connect(self.load_project)
        load_array_layout_action.triggered.connect(self.load_array_layout_csv)
        clear_array_layout_action.triggered.connect(self.clear_array_layout_csv)
        export_array_layout_template_action.triggered.connect(self.export_current_array_layout_template)
        load_element_pattern_action.triggered.connect(self.load_element_pattern_csv)
        clear_element_pattern_action.triggered.connect(self.clear_element_pattern_csv)
        export_element_pattern_template_action.triggered.connect(self.export_element_pattern_template)
        export_element_pattern_abs_phase_template_action.triggered.connect(self.export_element_pattern_abs_phase_template)
        load_element_near_field_action.triggered.connect(self.load_element_near_field_csv)
        clear_element_near_field_action.triggered.connect(self.clear_element_near_field_csv)
        export_element_near_field_template_action.triggered.connect(self.export_element_near_field_template)
        export_near_field_as_far_field_action.triggered.connect(self.export_near_field_as_far_field_pattern)
        png_action.triggered.connect(self.export_png)
        csv_action.triggered.connect(self.export_csv)
        excel_action.triggered.connect(self.export_excel_report)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(save_action)
        file_menu.addAction(load_action)
        file_menu.addSeparator()
        file_menu.addAction(load_array_layout_action)
        file_menu.addAction(clear_array_layout_action)
        file_menu.addAction(export_array_layout_template_action)
        file_menu.addSeparator()
        file_menu.addAction(load_element_pattern_action)
        file_menu.addAction(clear_element_pattern_action)
        file_menu.addAction(export_element_pattern_template_action)
        file_menu.addAction(export_element_pattern_abs_phase_template_action)
        file_menu.addAction(load_element_near_field_action)
        file_menu.addAction(clear_element_near_field_action)
        file_menu.addAction(export_element_near_field_template_action)
        file_menu.addAction(export_near_field_as_far_field_action)
        file_menu.addSeparator()
        file_menu.addAction(png_action)
        file_menu.addAction(csv_action)
        file_menu.addAction(excel_action)
        file_menu.addSeparator()
        file_menu.addAction(exit_action)

        help_menu = self.menuBar().addMenu("帮助")
        release_notes_action = QAction("更新说明", self)
        check_update_action = QAction("检查更新", self)
        release_notes_action.triggered.connect(self.show_release_notes)
        check_update_action.triggered.connect(self.check_updates_interactive)
        help_menu.addAction(release_notes_action)
        help_menu.addAction(check_update_action)

    def _wire_signals(self) -> None:
        self.parameter_panel.params_changed.connect(self._schedule_refresh)
        self.parameter_panel.calculations_changed.connect(self._on_calculation_selection_changed)
        self.parameter_panel.refresh_requested.connect(self.refresh_auto)
        self.parameter_panel.calculate_requested.connect(self.calculate_all)
        self.parameter_panel.export_png_requested.connect(self.export_png)
        self.parameter_panel.export_csv_requested.connect(self.export_csv)
        self.parameter_panel.save_project_requested.connect(self.save_project)
        self.parameter_panel.load_project_requested.connect(self.load_project)

    def _load_last_session(self) -> None:
        if os.environ.get("BEAMCOVERAGE_DISABLE_SESSION_RESTORE") == "1":
            return
        payload_text = str(self._session_settings.value("last_project_payload", "") or "")
        if payload_text:
            try:
                payload = json.loads(payload_text)
                params = sanitized_params(BeamParams.from_dict(payload.get("params", payload)), validate_external=False)
                self.parameter_panel.set_params(params)
                selected = payload.get("selected_calculations")
                if isinstance(selected, dict):
                    self.parameter_panel.set_selected_calculations(selected)
            except Exception:
                pass
        geometry = self._session_settings.value("window_geometry")
        if geometry:
            try:
                self.restoreGeometry(geometry)
            except Exception:
                pass
        splitter_state = self._session_settings.value("splitter_state")
        if splitter_state:
            try:
                self.splitter.restoreState(splitter_state)
            except Exception:
                pass
        try:
            tab_index = int(self._session_settings.value("current_tab", 0) or 0)
            if 0 <= tab_index < self.plot_panel.count():
                self.plot_panel.setCurrentIndex(tab_index)
        except Exception:
            pass

    def _save_last_session(self) -> None:
        if os.environ.get("BEAMCOVERAGE_DISABLE_SESSION_RESTORE") == "1":
            return
        try:
            self._session_settings.setValue("last_project_payload", json.dumps(self.project_payload(), ensure_ascii=False))
            self._session_settings.setValue("window_geometry", self.saveGeometry())
            self._session_settings.setValue("splitter_state", self.splitter.saveState())
            self._session_settings.setValue("current_tab", self.plot_panel.currentIndex())
            self._session_settings.sync()
        except Exception:
            pass

    def closeEvent(self, event) -> None:  # type: ignore[override]
        self._pending_timer.stop()
        self._save_last_session()
        super().closeEvent(event)

    def show_release_notes(self) -> None:
        self.plot_panel.show_release_notes()

    def check_updates_interactive(self) -> None:
        self._run_update_check(interactive=True)

    def _check_updates_on_startup(self) -> None:
        if os.environ.get("BEAMCOVERAGE_DISABLE_SESSION_RESTORE") == "1":
            return
        today = date.today().isoformat()
        if str(self._session_settings.value("updates/last_check_date", "") or "") == today:
            return
        self._session_settings.setValue("updates/last_check_date", today)
        self._run_update_check(interactive=False)

    def _run_update_check(self, *, interactive: bool) -> None:
        try:
            info = check_for_update(APP_VERSION, timeout_s=4.0)
        except Exception as exc:
            if interactive:
                QMessageBox.warning(self, "检查更新失败", f"无法连接 GitHub 更新源：{exc}")
            return
        if not info.is_newer:
            if interactive:
                QMessageBox.information(self, "检查更新", f"当前已是最新版本：{APP_VERSION}")
            return
        message = format_update_message(info)
        buttons = QMessageBox.StandardButton.Open | QMessageBox.StandardButton.Cancel
        choice = QMessageBox.information(self, "发现新版本", message, buttons, QMessageBox.StandardButton.Open)
        if choice == QMessageBox.StandardButton.Open:
            QDesktopServices.openUrl(QUrl(info.html_url))

    def _schedule_refresh(self) -> None:
        self._mark_results_dirty()
        self._pending_timer.start()

    def _on_calculation_selection_changed(self) -> None:
        selected = self.parameter_panel.selected_calculations()
        previous = self._last_selected_calculations
        self._last_selected_calculations = selected
        if selected == previous:
            return
        if previous.get("structure", True) != selected.get("structure", True):
            self.plot_panel.show_calculation_item_placeholder("structure", selected.get("structure", True))
        if previous.get("cuts", True) != selected.get("cuts", True):
            self.cuts = None
            self.plot_panel.show_calculation_item_placeholder("cuts", selected.get("cuts", True))
        if previous.get("current_3d", True) != selected.get("current_3d", True):
            self.current_3d = None
            self.plot_panel.show_calculation_item_placeholder("current_3d", selected.get("current_3d", True))
        if previous.get("uv", True) != selected.get("uv", True):
            self.uv_pattern = None
            self.plot_panel.show_calculation_item_placeholder("uv", selected.get("uv", True))
        if previous.get("scan_union", False) != selected.get("scan_union", False):
            self.scan_union = None
            self.plot_panel.show_calculation_item_placeholder("scan_union", selected.get("scan_union", False))
        self._mark_results_dirty()
        self.refresh_auto()
        self.statusBar().showMessage("计算内容已更新。右侧仍保留上次结果，点击“计算”后刷新。", 5000)

    def _mark_results_dirty(self) -> None:
        self._results_dirty = True
        self.last_timings = None
        self.parameter_panel.set_results_dirty(True)
        try:
            self.parameter_panel.results.update_timings(self.settings.name, None)
        except Exception:
            pass

    def _clear_calculated_results(
        self,
        *,
        clear_timings: bool = True,
        clear_cache: bool = True,
        clear_plots: bool = True,
        reason: str = "参数已变化，请重新计算。",
    ) -> None:
        self.cuts = None
        self.current_3d = None
        self.uv_pattern = None
        self.scan_union = None
        if clear_timings:
            self.last_timings = None
        if clear_cache:
            self._scan_union_cache.clear()
            self._scan_union_current_cache.clear()
        if clear_plots:
            self.plot_panel.clear_calculated_plots(reason)

    def _prune_scan_union_cache(self, max_entries: int = 8) -> None:
        while len(self._scan_union_cache) > max_entries:
            oldest_key = next(iter(self._scan_union_cache))
            self._scan_union_cache.pop(oldest_key, None)
        while len(self._scan_union_current_cache) > max_entries * 4:
            oldest_key = next(iter(self._scan_union_current_cache))
            self._scan_union_current_cache.pop(oldest_key, None)

    def refresh_auto(self) -> None:
        try:
            self.params = self.parameter_panel.params()
            self.settings = get_effective_mode_settings(self.params)
            self.params, self.derived, self.elem = derive_params_with_element(self.params)
            self._refresh_array_layout_status()
            self._refresh_element_pattern_status()
            self._refresh_element_near_field_status()
            self.parameter_panel.results.update_results(self.derived, self.params, self.elem)
            self.parameter_panel.results.update_timings(self.settings.name, self.last_timings)
            if self._results_dirty:
                self.statusBar().showMessage("参数已更新，右侧图仍为上次计算结果。点击“计算”刷新。", 5000)
            else:
                self.statusBar().showMessage("参数已刷新。", 4000)
        except Exception as exc:
            self.derived = None
            self.elem = None
            try:
                current_params = self.parameter_panel.params()
                layout_path = current_params.array_layout_file
                if layout_path and current_params.array_layout == "custom":
                    self.parameter_panel.set_array_layout_summary(
                        layout_path,
                        error="文件缺失" if not Path(layout_path).expanduser().is_file() else "文件不可用",
                    )
                elif layout_path:
                    self.parameter_panel.set_array_layout_summary(layout_path, active=False)
                pattern_path = current_params.element_pattern_file
                if pattern_path and current_params.use_element_pattern:
                    self.parameter_panel.set_element_pattern_summary(
                        pattern_path,
                        error="文件缺失" if not Path(pattern_path).expanduser().is_file() else "文件不可用",
                    )
                elif pattern_path:
                    self.parameter_panel.set_element_pattern_summary(pattern_path, active=False)
                near_field_path = current_params.element_near_field_file
                if near_field_path:
                    self.parameter_panel.set_element_near_field_summary(
                        near_field_path,
                        error="文件缺失" if not Path(near_field_path).expanduser().is_file() else "文件不可用",
                    )
            except Exception:
                pass
            self.parameter_panel.results.clear_results()
            self.parameter_panel.results.update_timings(self.settings.name, None)
            self.statusBar().showMessage(str(exc), 8000)

    def _refresh_element_pattern_status(self) -> None:
        path = getattr(self.params, "element_pattern_file", "") or ""
        if not path:
            self.parameter_panel.set_element_pattern_summary("")
            return
        if self.elem is None or self.elem.mode != "table":
            self.parameter_panel.set_element_pattern_summary(path, active=False)
            return
        self.parameter_panel.set_element_pattern_summary(
            path,
            point_count=self.elem.table_point_count,
            has_phase=self.elem.table_has_phase,
            has_vector_components=self.elem.table_has_vector_components,
            gain_norm=self.elem.gain_norm,
            theta_max_deg=self.elem.table_theta_max_deg,
            covers_visible_edge=self.elem.table_covers_visible_edge,
            selected_frequency_ghz=self.elem.table_selected_frequency_ghz,
            active=True,
        )

    def _refresh_array_layout_status(self) -> None:
        path = getattr(self.params, "array_layout_file", "") or ""
        if not path:
            self.parameter_panel.set_array_layout_summary("")
            return
        if self.derived is None or not self.derived.imported_array_layout:
            self.parameter_panel.set_array_layout_summary(path, active=False)
            return
        self.parameter_panel.set_array_layout_summary(
            path,
            point_count=int(self.derived.element_x_m.size),
            total_power_w=float(self.derived.total_input_power_w),
            aperture_x_m=float(self.derived.dx_aperture_m),
            aperture_y_m=float(self.derived.dy_aperture_m),
            active=True,
        )

    def _refresh_element_near_field_status(self) -> None:
        path = getattr(self.params, "element_near_field_file", "") or ""
        if not path:
            self.parameter_panel.set_element_near_field_summary("")
            return
        if not Path(path).expanduser().is_file():
            self.parameter_panel.set_element_near_field_summary(path, error="文件缺失")
            return
        try:
            table = load_imported_element_near_field(path, self.params.frequency_ghz)
            summary = near_field_summary(table)
            self.parameter_panel.set_element_near_field_summary(
                path,
                point_count=int(summary["point_count"]),
                has_phase=bool(summary["has_phase"]),
                has_vector_components=bool(summary["has_vector_components"]),
                has_power_density=bool(summary["has_power_density"]),
                selected_frequency_ghz=summary["selected_frequency_ghz"],
                x_span_m=float(summary["x_span_m"]),
                y_span_m=float(summary["y_span_m"]),
                z_min_m=float(summary["z_min_m"]),
                z_max_m=float(summary["z_max_m"]),
                active=True,
            )
        except Exception as exc:
            self.parameter_panel.set_element_near_field_summary(path, error=str(exc).splitlines()[0])

    def calculate_all(self) -> None:
        self._pending_timer.stop()
        self.refresh_auto()
        if self.derived is None or self.elem is None:
            QMessageBox.warning(self, "参数无效", "当前参数无法计算，请先修正左侧输入。")
            return

        QGuiApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        self.parameter_panel.set_calculation_busy(True)
        self._clear_calculated_results(clear_timings=True, clear_cache=False, reason="正在重新计算。")
        timings: dict[str, float | bool] = {}
        total_t0 = perf_counter()
        selected = self.parameter_panel.selected_calculations()
        self._last_selected_calculations = selected
        try:
            if selected.get("structure", True):
                self.statusBar().showMessage("正在计算结构图...")
                QGuiApplication.processEvents()
                self.plot_panel.plot_structure(self.params, self.derived)

            if selected.get("cuts", True):
                self.statusBar().showMessage("正在计算二维切面...")
                QGuiApplication.processEvents()
                t0 = perf_counter()
                self.cuts = compute_2d_cuts(self.params, self.derived, self.elem, self.settings)
                timings["cuts_s"] = perf_counter() - t0
                self.plot_panel.plot_2d_cuts(self.cuts, self.derived, self.params)
            else:
                timings["cuts_s"] = float("nan")

            if selected.get("current_3d", True):
                self.statusBar().showMessage("正在计算三维包络...")
                QGuiApplication.processEvents()
                t0 = perf_counter()
                self.current_3d = compute_current_3d_envelope(self.params, self.derived, self.elem, self.settings)
                timings["current_3d_s"] = perf_counter() - t0
                self.plot_panel.plot_current_3d(self.current_3d, self.derived, self.params)
            else:
                timings["current_3d_s"] = float("nan")

            if selected.get("uv", True):
                self.statusBar().showMessage("正在计算u-v方向图...")
                QGuiApplication.processEvents()
                t0 = perf_counter()
                self.uv_pattern = calc_uv_pattern(self.params, self.derived, self.elem, self.settings.n_uv, self.settings.pattern_floor_db)
                timings["uv_s"] = perf_counter() - t0
                self.plot_panel.plot_uv_pattern(self.uv_pattern, self.params)
            else:
                timings["uv_s"] = float("nan")

            if selected.get("scan_union", False):
                self.statusBar().showMessage("正在计算扫描并集...")
                QGuiApplication.processEvents()
                t0 = perf_counter()
                key = base_cache_key(self.params, self.settings)
                cache_hit = key in self._scan_union_cache
                if key not in self._scan_union_cache:
                    self._scan_union_cache[key] = compute_scan_union_base_envelope_3d(self.params, self.derived, self.elem, self.settings)
                    self._prune_scan_union_cache()
                base_union = self._scan_union_cache[key]
                current_key = _scan_union_current_cache_key(key, self.params)
                current_cache_hit = current_key in self._scan_union_current_cache
                if current_cache_hit:
                    self.scan_union = self._scan_union_current_cache[current_key]
                else:
                    self.scan_union = merge_scan_union_with_current_scan(base_union, self.params, self.derived, self.elem, self.settings)
                    if bool(self.scan_union.get("currentScanOverlayApplied")):
                        self._scan_union_current_cache[current_key] = self.scan_union
                        self._prune_scan_union_cache()
                timings["scan_union_s"] = perf_counter() - t0
                timings["scan_union_cache_hit"] = cache_hit
                timings["scan_union_current_cache_hit"] = current_cache_hit
                timings["scan_union_skipped"] = False
                union_internal_timings = self.scan_union.get("timings", {}) if isinstance(self.scan_union, dict) else {}
                base_internal_timings = base_union.get("timings", {}) if isinstance(base_union, dict) else {}
                current_overlay = bool(self.scan_union.get("currentScanOverlayApplied")) if isinstance(self.scan_union, dict) else False
                timings["scan_union_current_overlay"] = current_overlay
                if cache_hit and (not current_overlay or current_cache_hit):
                    timings["scan_union_3d_s"] = float("nan")
                    timings["scan_union_2d_cuts_s"] = float("nan")
                    timings["scan_union_compute_s"] = float("nan")
                else:
                    base_3d_s = 0.0 if cache_hit else float(base_internal_timings.get("scan_union_3d_s", 0.0))
                    base_2d_s = 0.0 if cache_hit else float(base_internal_timings.get("scan_union_2d_cuts_s", 0.0))
                    base_compute_s = 0.0 if cache_hit else float(base_internal_timings.get("scan_union_compute_s", 0.0))
                    overlay_3d_s = float(union_internal_timings.get("scan_union_3d_s", 0.0)) if current_overlay else 0.0
                    overlay_2d_s = float(union_internal_timings.get("scan_union_2d_cuts_s", 0.0)) if current_overlay else 0.0
                    overlay_compute_s = float(union_internal_timings.get("scan_union_compute_s", 0.0)) if current_overlay else 0.0
                    timings["scan_union_3d_s"] = base_3d_s + overlay_3d_s
                    timings["scan_union_2d_cuts_s"] = base_2d_s + overlay_2d_s
                    timings["scan_union_compute_s"] = base_compute_s + overlay_compute_s
                if current_overlay and not current_cache_hit:
                    timings["scan_union_current_overlay_s"] = float(union_internal_timings.get("scan_union_current_overlay_s", float("nan")))
                elif current_overlay:
                    timings["scan_union_current_overlay_s"] = float("nan")
                self.plot_panel.plot_scan_union(self.scan_union, self.derived, self.params)
            else:
                timings["scan_union_s"] = float("nan")
                timings["scan_union_3d_s"] = float("nan")
                timings["scan_union_2d_cuts_s"] = float("nan")
                timings["scan_union_compute_s"] = float("nan")
                timings["scan_union_cache_hit"] = False
                timings["scan_union_current_cache_hit"] = False
                timings["scan_union_current_overlay"] = False
                timings["scan_union_skipped"] = True

            timings["total_s"] = perf_counter() - total_t0
            self.last_timings = timings
            self._results_dirty = False
            self.parameter_panel.set_results_dirty(False)
            self.parameter_panel.results.update_timings(self.settings.name, timings)
            self.statusBar().showMessage(f"计算完成，用时 {timings['total_s']:.2f} s。", 8000)
        except Exception as exc:
            traceback.print_exc()
            self._results_dirty = True
            self.parameter_panel.set_results_dirty(True)
            self._clear_calculated_results(
                clear_timings=True,
                clear_cache=True,
                reason=f"计算失败：{exc}",
            )
            self.parameter_panel.results.update_timings(self.settings.name, None)
            QMessageBox.critical(self, "计算失败", str(exc))
            self.statusBar().showMessage(f"计算失败：{exc}", 10000)
        finally:
            self.parameter_panel.set_calculation_busy(False)
            QGuiApplication.restoreOverrideCursor()

    def export_png(self) -> None:
        idx = self.plot_panel.currentIndex()
        try:
            self._validate_current_plot_ready(idx)
        except Exception as exc:
            QMessageBox.warning(self, "PNG 导出不可用", str(exc))
            return
        figure = self.plot_panel.current_figure()
        path, _ = QFileDialog.getSaveFileName(self, "导出当前图形", "beamcoverage_plot.png", "PNG 文件 (*.png)")
        if not path:
            return
        try:
            export_figure_png(figure, path)
            self.statusBar().showMessage(f"PNG 已导出：{path}", 8000)
        except Exception as exc:
            QMessageBox.critical(self, "PNG 导出失败", str(exc))

    def export_csv(self) -> None:
        idx = self.plot_panel.currentIndex()
        try:
            self._validate_current_csv_ready(idx)
        except Exception as exc:
            QMessageBox.warning(self, "CSV 导出不可用", str(exc))
            return
        default_name = [
            "structure.csv",
            "2d_cuts.csv",
            "current_3d_envelope.csv",
            "uv_pattern.csv",
            "pattern_3d.csv",
            "scan_union.csv",
        ][idx]
        path, _ = QFileDialog.getSaveFileName(self, "导出当前图页数据", default_name, "CSV 文件 (*.csv)")
        if not path:
            return
        try:
            if idx == 1:
                if self.cuts is None:
                    raise RuntimeError("No 2D cut data. Run Calculate first.")
                export_2d_cuts(self.cuts, path, self.params.s0_w_cm2)
            elif idx == 2:
                if self.current_3d is None:
                    raise RuntimeError("No current 3D envelope data. Run Calculate first.")
                export_current_3d(self.current_3d, path, self.params.s0_w_cm2)
            elif idx == 3:
                if self.uv_pattern is None:
                    raise RuntimeError("No u-v pattern data. Run Calculate first.")
                export_uv_pattern(self.uv_pattern, path)
            elif idx == 4:
                if self.uv_pattern is None:
                    raise RuntimeError("No pattern data. Run Calculate first.")
                export_uv_pattern(self.uv_pattern, path)
            elif idx == 5:
                if self.scan_union is None:
                    raise RuntimeError("No scan union data. Run Calculate first.")
                export_scan_union(self.scan_union, path)
            else:
                self._export_structure_csv(path)
            self.statusBar().showMessage(f"CSV 已导出：{path}", 8000)
        except Exception as exc:
            QMessageBox.critical(self, "CSV 导出失败", str(exc))

    def export_excel_report(self) -> None:
        try:
            self._validate_excel_report_ready()
        except Exception as exc:
            QMessageBox.warning(self, "Excel 报告导出不可用", str(exc))
            return
        path, _ = QFileDialog.getSaveFileName(self, "导出 Excel 报告", "BeamCoverage_report.xlsx", "Excel 文件 (*.xlsx)")
        if not path:
            return
        try:
            figures = self._report_figures()
            export_excel_report(
                path,
                params=self.params,
                derived=self.derived,
                settings=self.settings,
                timings=self.last_timings,
                figures=figures,
                cuts=self.cuts,
                current_3d=self.current_3d,
                uv_pattern=self.uv_pattern,
                scan_union=self.scan_union,
            )
            self.statusBar().showMessage(f"Excel 报告已导出：{path}", 8000)
        except Exception as exc:
            QMessageBox.critical(self, "Excel 报告导出失败", str(exc))

    def _report_figures(self) -> dict[str, Any]:
        selected = self.parameter_panel.selected_calculations()
        figures: dict[str, Any] = {}
        if selected.get("structure", True) and self.derived is not None:
            figures[self.plot_panel.tabText(0)] = self.plot_panel.widget(0).figure
        if selected.get("cuts", True) and self.cuts is not None:
            figures[self.plot_panel.tabText(1)] = self.plot_panel.widget(1).figure
        if selected.get("current_3d", True) and self.current_3d is not None:
            figures[self.plot_panel.tabText(2)] = self.plot_panel.widget(2).figure
        if selected.get("uv", True) and self.uv_pattern is not None:
            figures[self.plot_panel.tabText(3)] = self.plot_panel.widget(3).figure
            figures[self.plot_panel.tabText(4)] = self.plot_panel.widget(4).figure
        if selected.get("scan_union", False) and self.scan_union is not None:
            figures[self.plot_panel.tabText(5)] = self.plot_panel.widget(5).figure
        return figures

    def _flush_pending_refresh(self) -> None:
        if self._pending_timer.isActive():
            self._pending_timer.stop()
            self.refresh_auto()

    def _validate_current_csv_ready(self, idx: int) -> None:
        self._validate_current_plot_ready(idx)

    def _validate_current_plot_ready(self, idx: int) -> None:
        self._flush_pending_refresh()
        if idx >= 6:
            raise RuntimeError("The release-notes tab does not contain exportable plot data.")
        if self._results_dirty:
            raise RuntimeError("Parameters changed after the last calculation. Run Calculate before exporting.")
        if idx == 0:
            if self.derived is None:
                raise RuntimeError("No structure data. Fix parameters first.")
            return
        if idx == 1 and self.cuts is None:
            raise RuntimeError("No 2D cut data. Run Calculate first.")
        if idx == 2 and self.current_3d is None:
            raise RuntimeError("No current 3D envelope data. Run Calculate first.")
        if idx in (3, 4) and self.uv_pattern is None:
            raise RuntimeError("No pattern data. Run Calculate first.")
        if idx == 5 and self.scan_union is None:
            raise RuntimeError("No scan union data. Run Calculate first.")

    def _validate_excel_report_ready(self) -> None:
        self._flush_pending_refresh()
        if self._results_dirty:
            raise RuntimeError("Parameters changed after the last calculation. Run Calculate before exporting a report.")
        if self.derived is None:
            raise RuntimeError("Parameters are invalid. Fix parameters before exporting a report.")
        if self.last_timings is None:
            raise RuntimeError("No complete calculation is available. Run Calculate before exporting a report.")
        selected = self.parameter_panel.selected_calculations()
        missing: list[str] = []
        if selected.get("cuts", True) and self.cuts is None:
            missing.append("2D cuts")
        if selected.get("current_3d", True) and self.current_3d is None:
            missing.append("current 3D envelope")
        if selected.get("uv", True) and self.uv_pattern is None:
            missing.append("u-v / 3D pattern")
        if selected.get("scan_union", False) and self.scan_union is None:
            missing.append("scan union")
        if missing:
            raise RuntimeError("Selected calculation data is missing: " + ", ".join(missing))

    def _export_structure_csv(self, path: str | Path) -> None:
        if self.derived is None:
            raise RuntimeError("No structure data.")
        export_structure(
            {
                "x_m": self.derived.element_x_m,
                "y_m": self.derived.element_y_m,
                "z_m": np.zeros(int(self.derived.element_x_m.size)),
                "power_w": self.derived.element_power_w,
            },
            path,
        )

    def save_project(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "保存 BeamCoverage 工程", "beamcoverage.project", "工程文件 (*.project)")
        if not path:
            return
        try:
            Path(path).write_text(json.dumps(self.project_payload_for_file(path), indent=2, ensure_ascii=False), encoding="utf-8")
            self.statusBar().showMessage(f"工程已保存：{path}", 8000)
        except Exception as exc:
            QMessageBox.critical(self, "保存失败", str(exc))

    def load_project(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "加载 BeamCoverage 工程", "", "工程文件 (*.project);;JSON 文件 (*.json)")
        if not path:
            return
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
            self.apply_project_payload(data, base_dir=Path(path).resolve().parent)
            self.statusBar().showMessage(f"工程已加载：{path}", 8000)
        except Exception as exc:
            QMessageBox.critical(self, "加载失败", str(exc))

    def load_element_pattern_csv(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "导入单元远场方向图文件",
            "",
            "远场文件 (*.csv *.txt *.dat *.ffd *.ffs);;CST/HFSS FFD/FFS (*.ffd *.ffs);;CSV 文件 (*.csv);;文本文件 (*.txt *.dat);;所有文件 (*.*)",
        )
        if not path:
            return
        try:
            params = self.parameter_panel.params()
            table = load_imported_element_pattern(path, params.frequency_ghz)
            params.element_pattern_file = str(Path(path).resolve())
            self.parameter_panel.set_params(params)
            self.parameter_panel.set_element_pattern_summary(
                params.element_pattern_file,
                point_count=table.point_count,
                has_phase=table.has_phase,
                has_vector_components=table.has_vector_components,
                gain_norm=table.gain_norm,
                theta_max_deg=table.theta_max_deg,
                covers_visible_edge=table.covers_visible_edge,
                selected_frequency_ghz=table.selected_frequency_ghz,
                active=True,
            )
            self.statusBar().showMessage(
                f"单元远场方向图已导入并参与主计算：{Path(path).name}，样本 {table.point_count}，相位 {'有' if table.has_phase else '无'}，θmax={table.theta_max_deg:.1f}°",
                8000,
            )
        except Exception as exc:
            QMessageBox.critical(self, "单元远场方向图导入失败", str(exc))

    def clear_element_pattern_csv(self) -> None:
        params = self.parameter_panel.params()
        if not params.element_pattern_file:
            self.statusBar().showMessage("当前没有启用导入单元远场方向图。", 5000)
            return
        params.element_pattern_file = ""
        self.parameter_panel.set_params(params)
        self.statusBar().showMessage("已清除导入单元远场方向图，恢复内置解析远场单元模型。", 8000)

    def load_element_near_field_csv(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "导入单元近场文件",
            "",
            "近场表格 (*.csv *.txt *.dat);;CSV 文件 (*.csv);;文本文件 (*.txt *.dat);;所有文件 (*.*)",
        )
        if not path:
            return
        try:
            params = self.parameter_panel.params()
            table = load_imported_element_near_field(path, params.frequency_ghz)
            summary = near_field_summary(table)
            params.element_near_field_file = str(Path(path).resolve())
            self.parameter_panel.set_params(params)
            self.parameter_panel.set_element_near_field_summary(
                params.element_near_field_file,
                point_count=int(summary["point_count"]),
                has_phase=bool(summary["has_phase"]),
                has_vector_components=bool(summary["has_vector_components"]),
                has_power_density=bool(summary["has_power_density"]),
                selected_frequency_ghz=summary["selected_frequency_ghz"],
                x_span_m=float(summary["x_span_m"]),
                y_span_m=float(summary["y_span_m"]),
                z_min_m=float(summary["z_min_m"]),
                z_max_m=float(summary["z_max_m"]),
                active=True,
            )
            self.statusBar().showMessage(
                f"单元近场文件已导入并校验：{Path(path).name}，样本 {table.point_count}。当前版本仅保存/校验，主包络计算未改变。",
                10000,
            )
        except Exception as exc:
            QMessageBox.critical(self, "单元近场导入失败", str(exc))

    def clear_element_near_field_csv(self) -> None:
        params = self.parameter_panel.params()
        if not params.element_near_field_file:
            self.statusBar().showMessage("当前没有导入单元近场文件。", 5000)
            return
        params.element_near_field_file = ""
        self.parameter_panel.set_params(params)
        self.statusBar().showMessage("已清除导入单元近场文件。", 8000)

    def load_array_layout_csv(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "导入阵元坐标 CSV",
            "",
            "CSV 文件 (*.csv);;文本文件 (*.txt);;所有文件 (*.*)",
        )
        if not path:
            return
        try:
            table = load_imported_array_layout(path, self.parameter_panel.power.value())
            params = self.parameter_panel.params()
            params.array_layout = "custom"
            params.array_layout_file = str(Path(path).resolve())
            self.parameter_panel.set_params(params)
            self.parameter_panel.set_array_layout_summary(
                params.array_layout_file,
                point_count=table.point_count,
                total_power_w=float(np.sum(table.power_w)),
                active=True,
            )
            self.statusBar().showMessage(
                f"阵元坐标已导入：{Path(path).name}，有效单元 {table.point_count}",
                8000,
            )
        except Exception as exc:
            QMessageBox.critical(self, "阵元坐标导入失败", str(exc))

    def clear_array_layout_csv(self) -> None:
        params = self.parameter_panel.params()
        if not params.array_layout_file and params.array_layout != "custom":
            self.statusBar().showMessage("当前没有启用导入阵元坐标。", 5000)
            return
        params.array_layout = "rectangular"
        params.array_layout_file = ""
        self.parameter_panel.set_params(params)
        self.statusBar().showMessage("已清除导入阵元坐标，恢复参数生成的矩形排布。", 8000)

    def export_current_array_layout_template(self) -> None:
        self._flush_pending_refresh()
        if self.derived is None:
            QMessageBox.warning(self, "阵元坐标导出不可用", "当前参数无效，请先修正参数再导出阵元坐标。")
            return
        path, _ = QFileDialog.getSaveFileName(
            self,
            "导出当前阵元坐标 CSV",
            "BeamCoverage_array_layout.csv",
            "CSV 文件 (*.csv)",
        )
        if not path:
            return
        try:
            export_current_array_layout_template(self.derived, path)
            self.statusBar().showMessage(f"阵元坐标 CSV 已导出：{path}", 8000)
        except Exception as exc:
            QMessageBox.critical(self, "阵元坐标导出失败", str(exc))

    def export_element_pattern_template(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self,
            "导出单元远场 Real/Imag 模板",
            "BeamCoverage_element_far_field_vector_template.csv",
            "CSV 文件 (*.csv)",
        )
        if not path:
            return
        try:
            export_element_pattern_vector_template(path)
            self.statusBar().showMessage(f"单元远场 Real/Imag 模板已导出：{path}", 8000)
        except Exception as exc:
            QMessageBox.critical(self, "单元远场模板导出失败", str(exc))

    def export_element_pattern_abs_phase_template(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self,
            "导出单元远场幅相模板",
            "BeamCoverage_element_far_field_abs_phase_template.csv",
            "CSV 文件 (*.csv)",
        )
        if not path:
            return
        try:
            export_element_pattern_abs_phase_template(path)
            self.statusBar().showMessage(f"单元远场幅相模板已导出：{path}", 8000)
        except Exception as exc:
            QMessageBox.critical(self, "单元远场幅相模板导出失败", str(exc))

    def export_element_near_field_template(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self,
            "导出单元近场 Ex/Ey/Ez 模板",
            "BeamCoverage_element_near_field_template.csv",
            "CSV 文件 (*.csv)",
        )
        if not path:
            return
        try:
            export_element_near_field_vector_template(path)
            self.statusBar().showMessage(f"单元近场 Ex/Ey/Ez 模板已导出：{path}", 8000)
        except Exception as exc:
            QMessageBox.critical(self, "单元近场模板导出失败", str(exc))

    def export_near_field_as_far_field_pattern(self) -> None:
        params = self.parameter_panel.params()
        source_path = str(params.element_near_field_file or "")
        if not source_path:
            source_path, _ = QFileDialog.getOpenFileName(
                self,
                "选择单元近场文件",
                "",
                "近场表格 (*.csv *.txt *.dat);;CSV 文件 (*.csv);;文本文件 (*.txt *.dat);;所有文件 (*.*)",
            )
        if not source_path:
            return
        default_name = f"{Path(source_path).stem}_projected_far_field.csv"
        output_path, _ = QFileDialog.getSaveFileName(
            self,
            "导出近场投影远场方向图 CSV",
            default_name,
            "CSV 文件 (*.csv)",
        )
        if not output_path:
            return
        try:
            info = export_near_field_projected_far_field_pattern(
                source_path,
                output_path,
                frequency_ghz=params.frequency_ghz,
            )
            table = load_imported_element_pattern(output_path, params.frequency_ghz)
            params.element_near_field_file = str(Path(source_path).resolve())
            params.element_pattern_file = str(Path(output_path).resolve())
            params.use_element_pattern = True
            self.parameter_panel.set_params(params)
            self.parameter_panel.set_element_pattern_summary(
                params.element_pattern_file,
                point_count=table.point_count,
                has_phase=table.has_phase,
                has_vector_components=table.has_vector_components,
                gain_norm=table.gain_norm,
                theta_max_deg=table.theta_max_deg,
                covers_visible_edge=table.covers_visible_edge,
                selected_frequency_ghz=table.selected_frequency_ghz,
                active=True,
            )
            self.statusBar().showMessage(
                "近场已投影为远场方向图并设为当前单元远场模型："
                f"{Path(output_path).name}，近场平面样本 {int(info['projected_point_count'])}，"
                f"输出 {int(info['output_rows'])} 点。",
                10000,
            )
        except Exception as exc:
            QMessageBox.critical(self, "近场投影远场导出失败", str(exc))

    def project_payload(self) -> dict[str, Any]:
        return {
            "schema": "BeamCoverage.project",
            "schema_version": 2,
            "app_version": APP_VERSION,
            "params": self.parameter_panel.params().to_dict(),
            "selected_calculations": self.parameter_panel.selected_calculations(),
        }

    def project_payload_for_file(self, path: str | Path) -> dict[str, Any]:
        payload = self.project_payload()
        base_dir = Path(path).expanduser().resolve().parent
        params = payload["params"]
        for key in ("array_layout_file", "element_pattern_file", "element_near_field_file"):
            params[key] = _path_for_project_save(params.get(key, ""), base_dir)
        return payload

    def apply_project_payload(self, data: dict[str, Any], *, base_dir: Path | None = None) -> None:
        payload = _resolve_project_payload_paths(data, base_dir)
        params = sanitized_params(BeamParams.from_dict(payload.get("params", payload)), validate_external=False)
        self.parameter_panel.set_params(params)
        selected = payload.get("selected_calculations")
        if isinstance(selected, dict):
            self.parameter_panel.set_selected_calculations(selected)
        self._last_selected_calculations = self.parameter_panel.selected_calculations()
        self._mark_results_dirty()
        self.refresh_auto()


def _path_for_project_save(value: object, base_dir: Path) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    path = Path(text).expanduser()
    try:
        resolved = path.resolve()
        base = base_dir.resolve()
    except OSError:
        return str(path)
    try:
        return str(resolved.relative_to(base))
    except ValueError:
        return str(resolved)


def _scan_union_current_cache_key(base_key: tuple[Any, ...], params: BeamParams) -> tuple[Any, ...]:
    return (
        *base_key,
        "current_scan",
        round(float(params.scan_x_deg), 9),
        round(float(params.scan_y_deg), 9),
    )


def _resolve_project_payload_paths(data: dict[str, Any], base_dir: Path | None) -> dict[str, Any]:
    payload = deepcopy(data)
    if base_dir is None:
        return payload
    params = payload.get("params") if isinstance(payload.get("params"), dict) else payload
    if not isinstance(params, dict):
        return payload
    base = Path(base_dir).expanduser().resolve()
    for key in ("array_layout_file", "element_pattern_file", "element_near_field_file"):
        params[key] = _resolve_project_external_path(params.get(key, ""), base)
    return payload


def _resolve_project_external_path(value: object, base_dir: Path) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    path = Path(text).expanduser()
    if path.is_file():
        return str(path.resolve())
    candidates: list[Path] = []
    if not path.is_absolute():
        candidates.append(base_dir / path)
    if path.name:
        candidates.append(base_dir / path.name)
        candidates.append(base_dir / "data" / path.name)
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate.resolve())
    return text
