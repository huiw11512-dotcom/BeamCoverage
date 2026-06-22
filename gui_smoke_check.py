from __future__ import annotations

from copy import deepcopy
import os
from pathlib import Path
import sys
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> int:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    os.environ.setdefault("BEAMCOVERAGE_DISABLE_SESSION_RESTORE", "1")

    from PySide6.QtGui import QAction
    from PySide6.QtWidgets import QApplication, QMessageBox

    from app_info import APP_TITLE
    import gui.main_window as main_window_module
    from gui.main_window import MainWindow
    from gui.plot_panel import _limit_surface_grid, _upsample_uv_surface_for_display

    app = QApplication.instance() or QApplication(sys.argv)
    window = MainWindow()
    tempdir = tempfile.TemporaryDirectory()
    try:
        sample_pattern_path = _write_sample_element_pattern(Path(tempdir.name))
        sample_array_path = _write_sample_array_layout(Path(tempdir.name))
        sample_near_field_path = _write_sample_near_field(Path(tempdir.name))
        default_calc_mode = window.parameter_panel.calc_mode.currentText()
        default_scan_union_checked = window.parameter_panel.calc_scan_union.isChecked()
        menu_titles = [action.text() for action in window.menuBar().actions()]
        if "文件" not in menu_titles:
            raise AssertionError(f"Top menu is not localized: {menu_titles}")
        file_actions = [action.text() for action in window.findChildren(QAction)]
        for expected_action in (
            "导入单元远场方向图文件",
            "导入单元近场文件",
            "导出当前阵元坐标 CSV",
            "导出单元远场 Real/Imag 模板",
            "导出单元远场幅相模板",
            "导出单元近场 Ex/Ey/Ez 模板",
            "从单元近场导出远场方向图 CSV",
        ):
            if expected_action not in file_actions:
                raise AssertionError(f"文件菜单缺少动作：{expected_action}")
        _assert_surface_upsample_preserves_nan_gaps(_upsample_uv_surface_for_display)
        _assert_surface_grid_limits(_limit_surface_grid)
        panel = window.parameter_panel
        panel.frequency.setValue(5.6)
        panel.nx.setValue(2)
        panel.ny.setValue(8)
        panel.dx.setValue(0.7)
        panel.dy.setValue(0.1)
        panel.ax.setValue(0.7)
        panel.ay.setValue(0.1)
        panel.scan_x.setValue(1.0)
        panel.scan_y.setValue(13.0)
        panel.s0.setValue(20.0)
        panel.power.setValue(1.0e6)
        panel.calc_mode.setCurrentIndex(0)
        window.refresh_auto()
        _assert_result_label(window, "array_source", "矩形排布")
        _assert_result_label(window, "active_elements", "16")
        _assert_result_label(window, "element_model", "矩形解析口径")
        _assert_result_label(window, "current_envelope_method", "近场采样+远场外推")
        _assert_result_label(window, "scan_union_method", "3D远场；2D近场")

        panel.array_layout.setCurrentIndex(1)
        panel.element_shape.setCurrentIndex(2)
        imported_params = panel.params()
        imported_params.element_pattern_file = str(sample_pattern_path)
        imported_params.element_near_field_file = str(sample_near_field_path)
        panel.set_params(imported_params)
        window.refresh_auto()
        if not window.elem or window.elem.mode != "table":
            raise AssertionError("Imported element pattern file did not activate in the GUI.")
        _assert_result_label(window, "element_model", "导入远场方向图")
        pattern_status = panel.element_pattern_status.text()
        if "含相位/复数场" not in pattern_status or "点" not in pattern_status or "θmax=" not in pattern_status:
            raise AssertionError(f"Imported element pattern status did not show phase/sample metadata: {pattern_status!r}")
        near_field_status = panel.element_near_field_status.text()
        if "实验接口" not in near_field_status or "矢量Ex/Ey/Ez" not in near_field_status or "点" not in near_field_status:
            raise AssertionError(f"Imported near-field status did not show experimental/vector metadata: {near_field_status!r}")
        window.plot_panel.plot_structure(window.params, window.derived)
        _assert_structure_outline_shape(window.plot_panel.structure_tab.figure, "ellipse")
        panel.array_layout.setCurrentIndex(2)
        window.refresh_auto()
        window.plot_panel.plot_structure(window.params, window.derived)
        _assert_structure_outline_shape(window.plot_panel.structure_tab.figure, "diamond")
        panel.array_layout.setCurrentIndex(1)
        window.refresh_auto()
        panel.calc_scan_union.setChecked(True)
        panel.calc_current_3d.setChecked(False)
        payload = window.project_payload()
        if payload["params"].get("element_pattern_file") != str(sample_pattern_path):
            raise AssertionError("Project payload did not store imported element pattern path.")
        if payload["params"].get("element_near_field_file") != str(sample_near_field_path):
            raise AssertionError("Project payload did not store imported near-field path.")
        panel.array_layout.setCurrentIndex(0)
        panel.element_shape.setCurrentIndex(0)
        cleared_params = panel.params()
        cleared_params.element_pattern_file = ""
        panel.set_params(cleared_params)
        panel.calc_scan_union.setChecked(False)
        panel.calc_current_3d.setChecked(True)
        window.apply_project_payload(payload)
        restored = window.parameter_panel.params()
        restored_selected = window.parameter_panel.selected_calculations()
        if restored.array_layout != "ellipse" or restored.element_shape != "diamond":
            raise AssertionError("Project payload did not restore array layout and element shape.")
        if restored.element_pattern_file != str(sample_pattern_path):
            raise AssertionError("Project payload did not restore imported element pattern path.")
        if restored.element_near_field_file != str(sample_near_field_path):
            raise AssertionError("Project payload did not restore imported near-field path.")
        if not restored_selected["scan_union"] or restored_selected["current_3d"]:
            raise AssertionError("Project payload did not restore selected calculation checkboxes.")

        relative_payload = deepcopy(payload)
        relative_payload["params"]["array_layout"] = "custom"
        relative_payload["params"]["element_shape"] = "rectangular"
        relative_payload["params"]["ax_m"] = 0.08
        relative_payload["params"]["ay_m"] = 0.06
        relative_payload["params"]["array_layout_file"] = str(Path("old_location") / sample_array_path.name)
        relative_payload["params"]["element_pattern_file"] = str(Path("old_location") / sample_pattern_path.name)
        relative_payload["params"]["element_near_field_file"] = str(Path("old_location") / sample_near_field_path.name)
        window.apply_project_payload(relative_payload, base_dir=Path(tempdir.name))
        resolved_params = window.parameter_panel.params()
        if resolved_params.array_layout_file != str(sample_array_path.resolve()):
            raise AssertionError(
                "Project loader did not recover the array layout CSV from the project directory: "
                f"{resolved_params.array_layout_file!r}"
            )
        if resolved_params.element_pattern_file != str(sample_pattern_path.resolve()):
            raise AssertionError(
                "Project loader did not recover the element pattern CSV from the project directory: "
                f"{resolved_params.element_pattern_file!r}"
            )
        if resolved_params.element_near_field_file != str(sample_near_field_path.resolve()):
            raise AssertionError(
                "Project loader did not recover the near-field CSV from the project directory: "
                f"{resolved_params.element_near_field_file!r}"
            )
        saved_payload = window.project_payload_for_file(Path(tempdir.name) / "BeamCoverage.project")
        if saved_payload["params"]["array_layout_file"] != sample_array_path.name:
            raise AssertionError("Project save did not store colocated array layout CSV as a relative path.")
        if saved_payload["params"]["element_pattern_file"] != sample_pattern_path.name:
            raise AssertionError("Project save did not store colocated element pattern CSV as a relative path.")
        if saved_payload["params"]["element_near_field_file"] != sample_near_field_path.name:
            raise AssertionError("Project save did not store colocated near-field CSV as a relative path.")

        missing_import_payload = deepcopy(payload)
        missing_import_payload["params"]["array_layout"] = "custom"
        missing_import_payload["params"]["array_layout_file"] = str(Path(tempdir.name) / "missing_layout.csv")
        missing_import_payload["params"]["use_element_pattern"] = True
        missing_import_payload["params"]["element_pattern_file"] = str(Path(tempdir.name) / "missing_pattern.csv")
        missing_import_payload["params"]["element_near_field_file"] = str(Path(tempdir.name) / "missing_near_field.csv")
        window.apply_project_payload(missing_import_payload, base_dir=Path(tempdir.name))
        window.refresh_auto()
        if window.derived is not None:
            raise AssertionError("Project with missing active import CSVs unexpectedly produced derived parameters.")
        if "文件缺失" not in window.parameter_panel.array_layout_status.text():
            raise AssertionError("Missing active array-layout CSV loaded from a project was not marked 文件缺失.")
        if "文件缺失" not in window.parameter_panel.element_pattern_status.text():
            raise AssertionError("Missing active element-pattern CSV loaded from a project was not marked 文件缺失.")
        if "文件缺失" not in window.parameter_panel.element_near_field_status.text():
            raise AssertionError("Missing near-field CSV loaded from a project was not marked 文件缺失.")
        window.apply_project_payload(relative_payload, base_dir=Path(tempdir.name))

        before_invalid_project = window.parameter_panel.params().to_dict()
        bad_core_payload = deepcopy(payload)
        bad_core_payload["params"]["nx"] = 0
        try:
            window.apply_project_payload(bad_core_payload)
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid project payload with nx=0 was not rejected before UI update.")
        if window.parameter_panel.params().to_dict() != before_invalid_project:
            raise AssertionError("Invalid core project payload changed the visible UI parameters.")

        bad_ui_payload = deepcopy(payload)
        bad_ui_payload["params"]["nx"] = 10000
        try:
            window.apply_project_payload(bad_ui_payload)
        except ValueError:
            pass
        else:
            raise AssertionError("Out-of-range project payload with nx=10000 was not rejected before UI clamp.")
        if window.parameter_panel.params().to_dict() != before_invalid_project:
            raise AssertionError("Out-of-range project payload was clamped into the visible UI.")

        panel.array_layout.setCurrentIndex(0)
        panel.element_shape.setCurrentIndex(0)
        panel.calc_scan_union.setChecked(False)
        panel.calc_current_3d.setChecked(True)

        settings = window.settings
        window.settings = settings.__class__(
            **{
                **settings.__dict__,
                "n_alpha_2d": 21,
                "n_range_2d": 16,
                "theta_3d_n": 8,
                "phi_3d_n": 12,
                "n_range_3d": 12,
                "n_uv": 31,
                "scan_union_step_deg": 12.0,
                "scan_union_theta_n": 8,
                "scan_union_phi_n": 12,
                "chunk_size": 256,
                "scan_block_size": 8,
            }
        )
        window._pending_timer.stop()

        if window.windowTitle() != APP_TITLE:
            raise AssertionError(f"Unexpected window title: {window.windowTitle()!r}")
        if window.windowIcon().isNull():
            raise AssertionError("Window icon was not loaded.")
        if default_calc_mode != "标准":
            raise AssertionError("Default calculation mode is not 标准.")
        if default_scan_union_checked:
            raise AssertionError("Scan union should be unchecked by default.")

        tabs = [window.plot_panel.tabText(i) for i in range(window.plot_panel.count())]
        expected_tabs = ["结构示意", "二维切面", "三维包络", "u-v方向图", "三维方向图", "扫描并集", "更新说明"]
        if tabs != expected_tabs:
            raise AssertionError(f"Unexpected tabs: {tabs}")
        _assert_plot_toolbars(window.plot_panel)
        window.show_release_notes()
        if window.plot_panel.tabText(window.plot_panel.currentIndex()) != "更新说明":
            raise AssertionError("Release notes tab did not open from the main window action.")

        window.calculate_all()
        if window.scan_union is not None:
            raise AssertionError("Default calculation should skip scan union.")
        if not window.last_timings or not window.last_timings.get("scan_union_skipped"):
            raise AssertionError("Skipped scan union timing/status was not recorded.")
        if not (window.cuts and window.current_3d and window.uv_pattern):
            raise AssertionError("Default selected plots did not calculate.")
        _assert_axes_cover_surface(window.plot_panel.envelope_tab.figure.axes[0], window.current_3d, "current 3D envelope")
        if not window.plot_panel.pattern_3d_tab.figure.axes:
            raise AssertionError("3D pattern tab did not render.")
        window._validate_excel_report_ready()
        window._validate_current_plot_ready(0)
        window._validate_current_plot_ready(1)
        window._validate_current_plot_ready(2)
        window._validate_current_plot_ready(3)
        window._validate_current_plot_ready(4)
        window._validate_current_csv_ready(1)
        window._validate_current_csv_ready(2)
        window._validate_current_csv_ready(3)
        _assert_report_figures(window, expect_scan_union=False)

        original_compute_current_3d = main_window_module.compute_current_3d_envelope
        original_critical = QMessageBox.critical
        original_print_exc = main_window_module.traceback.print_exc

        def _raise_current_3d(*args, **kwargs):
            raise RuntimeError("forced current_3d failure")

        try:
            main_window_module.compute_current_3d_envelope = _raise_current_3d
            QMessageBox.critical = lambda *args, **kwargs: QMessageBox.StandardButton.Ok
            main_window_module.traceback.print_exc = lambda *args, **kwargs: None
            window.calculate_all()
        finally:
            main_window_module.compute_current_3d_envelope = original_compute_current_3d
            QMessageBox.critical = original_critical
            main_window_module.traceback.print_exc = original_print_exc
        if any(value is not None for value in (window.cuts, window.current_3d, window.uv_pattern, window.scan_union)):
            raise AssertionError("Failed calculation left partial calculated data behind.")
        if window.last_timings is not None:
            raise AssertionError("Failed calculation left partial timing data behind.")
        _assert_placeholder(window.plot_panel.cuts_tab.figure, "forced current_3d failure")
        _assert_placeholder(window.plot_panel.envelope_tab.figure, "forced current_3d failure")
        _assert_placeholder(window.plot_panel.uv_tab.figure, "forced current_3d failure")

        window.calculate_all()
        if not (window.cuts and window.current_3d and window.uv_pattern):
            raise AssertionError("Calculation did not recover after a forced mid-run failure.")

        panel.calc_cuts.setChecked(False)
        if window.cuts is not None:
            raise AssertionError("2D cut data was not cleared when 2D cuts were unchecked.")
        _assert_placeholder(window.plot_panel.cuts_tab.figure, "未选择二维切面计算")
        panel.calc_cuts.setChecked(True)
        _assert_placeholder(window.plot_panel.cuts_tab.figure, "已选择二维切面")
        panel.calc_uv.setChecked(False)
        if window.uv_pattern is not None:
            raise AssertionError("Pattern data was not cleared when pattern calculation was unchecked.")
        _assert_placeholder(window.plot_panel.uv_tab.figure, "未选择方向图计算")
        _assert_placeholder(window.plot_panel.pattern_3d_tab.figure, "未选择方向图计算")
        panel.calc_uv.setChecked(True)
        _assert_placeholder(window.plot_panel.uv_tab.figure, "已选择方向图")

        window.calculate_all()
        if not (window.cuts and window.current_3d and window.uv_pattern):
            raise AssertionError("Calculation did not restore selected data before staged parameter-change test.")
        panel.scan_x.setValue(1.5)
        if not (window.cuts and window.current_3d and window.uv_pattern):
            raise AssertionError("Calculated plots were cleared after a staged parameter change.")
        if not window._results_dirty:
            raise AssertionError("Parameter change did not mark the calculation as dirty.")
        if window.last_timings is not None:
            raise AssertionError("Timing results were not invalidated after a parameter change.")
        _assert_raises_runtime_error(window._validate_excel_report_ready, "stale Excel export")
        _assert_raises_runtime_error(lambda: window._validate_current_plot_ready(1), "stale 2D PNG export")
        _assert_raises_runtime_error(lambda: window._validate_current_csv_ready(1), "stale 2D CSV export")

        panel.ax.setValue(0.8)
        window.refresh_auto()
        if not window.parameter_panel.geometry_warning.text():
            raise AssertionError("Invalid ax>dx geometry did not show the inline geometry warning.")
        if window.derived is not None or window.elem is not None:
            raise AssertionError("Invalid geometry did not clear derived parameters.")
        if window.parameter_panel.results._labels["lambda"].text() != "n/a":
            raise AssertionError("Invalid geometry did not clear automatic result labels.")
        if window.parameter_panel.calculate_button.isEnabled():
            raise AssertionError("Invalid geometry did not disable the calculate button.")
        panel.ax.setValue(0.7)
        window.refresh_auto()
        if window.derived is None or window.elem is None:
            raise AssertionError("Restoring valid geometry did not restore derived parameters.")
        if window.parameter_panel.geometry_warning.text():
            raise AssertionError("Geometry warning stayed visible after restoring valid geometry.")
        if not window.parameter_panel.calculate_button.isEnabled():
            raise AssertionError("Restoring valid geometry did not re-enable the calculate button.")

        custom_params = window.parameter_panel.params()
        custom_params.array_layout = "custom"
        custom_params.array_layout_file = str(sample_array_path)
        custom_params.ax_m = 0.08
        custom_params.ay_m = 0.06
        window.parameter_panel.set_params(custom_params)
        window.refresh_auto()
        if not window.derived or not window.derived.imported_array_layout or window.derived.element_x_m.size != 5:
            raise AssertionError("Imported array layout did not activate in the GUI.")
        _assert_result_label(window, "array_source", "导入坐标CSV")
        _assert_result_label(window, "active_elements", "5")
        custom_payload = window.project_payload()
        if custom_payload["params"].get("array_layout_file") != str(sample_array_path):
            raise AssertionError("Project payload did not store imported array layout path.")
        window.calculate_all()
        _assert_structure_outline_shape(window.plot_panel.structure_tab.figure, "custom")
        regular_params = window.parameter_panel.params()
        regular_params.array_layout = "rectangular"
        window.parameter_panel.set_params(regular_params)
        window.refresh_auto()
        if window.parameter_panel.params().array_layout_file != str(sample_array_path):
            raise AssertionError("Inactive imported array layout path was not preserved after switching to rectangular layout.")
        if not window.derived or window.derived.imported_array_layout:
            raise AssertionError("Inactive imported array layout path still activated after switching to rectangular layout.")
        _assert_result_label(window, "array_source", "矩形排布")
        if "未启用" not in window.parameter_panel.array_layout_status.text():
            raise AssertionError("Inactive imported array layout status did not show 未启用.")
        inactive_payload = window.project_payload()
        if inactive_payload["params"].get("array_layout_file") != str(sample_array_path):
            raise AssertionError("Project payload did not preserve inactive imported array layout path.")

        _assert_nine_by_nine_2d_plot(window.plot_panel)

        panel.calc_scan_union.setChecked(True)
        window.calculate_all()
        if not window.scan_union:
            raise AssertionError("Scan union did not calculate after being selected.")
        timing = window.last_timings or {}
        for key in ("scan_union_3d_s", "scan_union_2d_cuts_s", "scan_union_compute_s"):
            value = float(timing.get(key, float("nan")))
            if not np.isfinite(value) or value < 0.0:
                raise AssertionError(f"Scan union timing breakdown {key} was not recorded.")
        max_range = float(window.scan_union["maxRange_m"])
        if max_range <= 1.0:
            raise AssertionError(f"Scan union max range is not meter-scale: {max_range}")
        _assert_scan_union_composite_plot(window.plot_panel.union_tab.figure, window.scan_union)
        window._validate_current_plot_ready(5)
        window._validate_current_csv_ready(5)
        _assert_report_figures(window, expect_scan_union=True)

        print("PASS GUI title, Chinese tabs, default standard mode, default union skip, and selected union calculation.")
        print(f"PASS scan union max range: {max_range:.4g} m")
        return 0
    finally:
        window.close()
        app.quit()
        tempdir.cleanup()


def _assert_axes_cover_surface(ax, surface: dict[str, object], label: str) -> None:
    x = np.asarray(surface["Xsurf"], dtype=float)
    y = np.asarray(surface["Ysurf"], dtype=float)
    z = np.asarray(surface["Zsurf"], dtype=float)
    finite = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    if not np.any(finite):
        raise AssertionError(f"{label} has no finite surface points.")
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    zlim = ax.get_zlim()
    if np.nanmin(x[finite]) < xlim[0] or np.nanmax(x[finite]) > xlim[1]:
        raise AssertionError(f"{label} x-axis does not cover full surface: x={np.nanmin(x[finite])}..{np.nanmax(x[finite])}, lim={xlim}")
    if np.nanmin(y[finite]) < ylim[0] or np.nanmax(y[finite]) > ylim[1]:
        raise AssertionError(f"{label} y-axis does not cover full surface: y={np.nanmin(y[finite])}..{np.nanmax(y[finite])}, lim={ylim}")
    if np.nanmin(z[finite]) < zlim[0] or np.nanmax(z[finite]) > zlim[1]:
        raise AssertionError(f"{label} z-axis does not cover full surface: z={np.nanmin(z[finite])}..{np.nanmax(z[finite])}, lim={zlim}")


def _assert_scan_union_composite_plot(figure, surface: dict[str, object]) -> None:
    if "unionCuts" not in surface or len(surface["unionCuts"]) != 2:
        raise AssertionError("Scan union result does not include fixed 2D union cuts.")
    three_d_axes = [ax for ax in figure.axes if hasattr(ax, "get_zlim")]
    if len(three_d_axes) != 1:
        raise AssertionError(f"Scan union composite plot should contain exactly one 3D axis, got {len(three_d_axes)}.")
    if "\n" in three_d_axes[0].get_title():
        raise AssertionError("Scan union 3D title should stay compact; long summary belongs in the figure suptitle.")
    suptitle = getattr(figure, "_suptitle", None)
    if suptitle is None or "扫描并集：" not in suptitle.get_text():
        raise AssertionError("Scan union figure did not render the compact top summary.")
    two_d_axes = [ax for ax in figure.axes if not hasattr(ax, "get_zlim") and ax.has_data()]
    if len(two_d_axes) < 2:
        raise AssertionError("Scan union composite plot did not render both fixed 2D cut axes.")
    titles = "\n".join(ax.get_title() for ax in two_d_axes)
    if "x-z" not in titles or "y-z" not in titles:
        raise AssertionError(f"Scan union 2D cut titles are missing x-z/y-z labels: {titles!r}")
    _assert_axes_cover_surface(three_d_axes[0], surface, "scan union")


def _assert_structure_outline_shape(figure, expected: str) -> None:
    if len(figure.axes) < 2:
        raise AssertionError("Structure figure does not contain the 3D aperture view.")
    ax = figure.axes[1]
    outline = None
    for line in ax.lines:
        try:
            x, y, z = line.get_data_3d()
        except AttributeError:
            continue
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        z = np.asarray(z, dtype=float)
        if x.size >= 5 and np.allclose(z, 0.0):
            outline = (x, y)
            break
    if outline is None:
        raise AssertionError("Structure figure did not render an aperture outline.")
    x, y = outline
    if expected == "ellipse":
        if x.size < 80:
            raise AssertionError("Ellipse layout outline was not rendered as a smooth closed curve.")
    elif expected == "diamond":
        if x.size != 5 or y.size != 5:
            raise AssertionError("Diamond layout outline was not rendered with four vertices.")
        if not (np.isclose(x[0], 0.0) and np.isclose(x[2], 0.0) and np.isclose(y[1], 0.0) and np.isclose(y[3], 0.0)):
            raise AssertionError("Diamond layout outline vertices are not aligned to the expected diagonals.")
    elif expected == "custom":
        if x.size < 4:
            raise AssertionError("Custom imported layout outline was not rendered as a hull.")


def _assert_nine_by_nine_2d_plot(plot_panel) -> None:
    from core.element_pattern import make_element_pattern
    from core.envelope import compute_2d_cuts
    from core.geometry import BeamParams, derive_params, get_mode_settings

    params = BeamParams(
        frequency_ghz=9.5,
        nx=9,
        ny=9,
        dx_m=0.3,
        dy_m=0.3,
        ax_m=0.3,
        ay_m=0.3,
        s0_w_cm2=100.0,
        scan_x_deg=0.0,
        scan_y_deg=0.0,
        calc_mode="fast",
    )
    params, derived = derive_params(params)
    elem = make_element_pattern(params, derived.wavelength_m)
    settings0 = get_mode_settings("fast")
    settings = settings0.__class__(
        **{
            **settings0.__dict__,
            "n_alpha_2d": 121,
            "n_range_2d": 72,
            "chunk_size": 1024,
        }
    )
    cuts = compute_2d_cuts(params, derived, elem, settings)
    for cut in cuts:
        finite = np.isfinite(np.asarray(cut["r_env_m"], dtype=float))
        if int(np.count_nonzero(finite)) < 100:
            raise AssertionError("9x9 2D cut did not produce a dense finite envelope.")
    plot_panel.plot_2d_cuts(cuts, derived, params)
    axes = plot_panel.cuts_tab.figure.axes
    if len(axes) < 3:
        raise AssertionError("9x9 2D cut figure did not render both cuts and a colorbar.")
    for idx, ax in enumerate(axes[:2]):
        if len(ax.lines) < 4:
            raise AssertionError(f"9x9 2D cut axis {idx} did not render envelope/scan/aperture lines.")
        if not ax.collections:
            raise AssertionError(f"9x9 2D cut axis {idx} did not render a filled power-density map.")


def _assert_placeholder(figure, expected_text: str) -> None:
    texts = []
    for ax in figure.axes:
        texts.extend(text.get_text() for text in ax.texts)
    joined = "\n".join(texts)
    if expected_text not in joined:
        raise AssertionError(f"Expected placeholder text {expected_text!r}, got {joined!r}")


def _assert_plot_toolbars(plot_panel) -> None:
    for idx in range(plot_panel.count()):
        tab = plot_panel.widget(idx)
        title = plot_panel.tabText(idx)
        if title == "更新说明":
            continue
        toolbar = getattr(tab, "toolbar", None)
        canvas = getattr(tab, "canvas", None)
        if toolbar is None or canvas is None:
            raise AssertionError(f"Plot tab {title!r} does not expose a toolbar and canvas.")
        if getattr(toolbar, "canvas", None) is not canvas:
            raise AssertionError(f"Plot tab {title!r} toolbar is not attached to its canvas.")
        action_texts = [action.text() for action in toolbar.actions()]
        if len(action_texts) < 5:
            raise AssertionError(f"Plot tab {title!r} toolbar has too few actions: {action_texts}")


def _assert_surface_upsample_preserves_nan_gaps(upsample_func) -> None:
    u = np.linspace(-1.0, 1.0, 5)
    v = np.linspace(-1.0, 1.0, 3)
    row = np.array([1.0, 2.0, np.nan, 4.0, 5.0], dtype=float)
    r = np.vstack([row, row + 1.0, row + 2.0])
    u_new, _v_new, r_new = upsample_func(u, v, r, max_u=9, max_v=5)
    center_idx = int(np.argmin(np.abs(u_new - 0.0)))
    left_gap_idx = int(np.argmin(np.abs(u_new - -0.25)))
    right_gap_idx = int(np.argmin(np.abs(u_new - 0.25)))
    if np.any(np.isfinite(r_new[:, [left_gap_idx, center_idx, right_gap_idx]])):
        raise AssertionError("3D surface display upsampling filled a NaN gap between disconnected envelope lobes.")
    finite_idx = int(np.argmin(np.abs(u_new - -0.75)))
    if not np.any(np.isfinite(r_new[:, finite_idx])):
        raise AssertionError("3D surface display upsampling removed valid data inside a continuous envelope lobe.")


def _assert_surface_grid_limits(limit_func) -> None:
    base = np.arange(201 * 181, dtype=float).reshape(201, 181)
    limited, limited_nan = limit_func(base, np.where(base % 17 == 0, np.nan, base), max_rows=86, max_cols=86)
    if limited.shape[0] > 86 or limited.shape[1] > 86:
        raise AssertionError(f"3D display grid limiter did not cap shape: {limited.shape}")
    if not np.isnan(limited_nan).any():
        raise AssertionError("3D display grid limiter unexpectedly removed NaN mask information.")
    small = np.ones((5, 7), dtype=float)
    (small_limited,) = limit_func(small, max_rows=86, max_cols=86)
    if small_limited.shape != small.shape:
        raise AssertionError("3D display grid limiter changed an already-small surface.")


def _assert_report_figures(window, *, expect_scan_union: bool) -> None:
    figures = window._report_figures()
    expected = {"结构示意", "二维切面", "三维包络", "u-v方向图", "三维方向图"}
    if expect_scan_union:
        expected.add("扫描并集")
    if set(figures) != expected:
        raise AssertionError(f"Excel report figure set mismatch: expected {sorted(expected)}, got {sorted(figures)}")


def _assert_raises_runtime_error(func, label: str) -> None:
    try:
        func()
    except RuntimeError:
        return
    raise AssertionError(f"{label} did not raise RuntimeError.")


def _assert_result_label(window, key: str, expected: str) -> None:
    actual = window.parameter_panel.results._labels[key].text()
    if actual != expected:
        raise AssertionError(f"Result label {key!r}: expected {expected!r}, got {actual!r}")


def _write_sample_element_pattern(directory: Path) -> Path:
    path = directory / "gui_element_pattern.csv"
    rows = ["u,v,gain_db,phase_deg"]
    for u in np.linspace(-0.8, 0.8, 13):
        for v in np.linspace(-0.8, 0.8, 13):
            if u * u + v * v <= 1.0:
                gain = max(0.05, (1.0 - 0.55 * u * u) * (1.0 - 0.15 * v * v))
                phase = 35.0 + 12.0 * u - 7.0 * v
                rows.append(f"{u:.8f},{v:.8f},{10.0 * np.log10(gain):.8f},{phase:.8f}")
    path.write_text("\n".join(rows), encoding="utf-8")
    return path


def _write_sample_array_layout(directory: Path) -> Path:
    path = directory / "gui_array_layout.csv"
    rows = [
        "x_m,y_m,power_w,phase_deg,enabled",
        "-0.16,-0.10,1000000,0,1",
        "0.00,-0.12,900000,8,1",
        "0.18,-0.08,1100000,-6,1",
        "-0.08,0.09,950000,12,1",
        "0.12,0.11,1050000,-4,1",
        "0.80,0.80,1000000,0,0",
    ]
    path.write_text("\n".join(rows), encoding="utf-8")
    return path


def _write_sample_near_field(directory: Path) -> Path:
    path = directory / "gui_element_near_field.csv"
    rows = ["Frequency [GHz],x_m,y_m,z_m,Real(Ex),Imag(Ex),Real(Ey),Imag(Ey),Real(Ez),Imag(Ez)"]
    for z in (0.05, 0.10):
        for y in (-0.02, 0.02):
            for x in (-0.03, 0.0, 0.03):
                phase = 10.0 * x - 6.0 * y
                phasor = np.exp(1j * np.deg2rad(phase))
                ex = phasor
                ey = 0.25 * np.exp(1j * np.deg2rad(phase - 30.0))
                ez = 0.05 * np.exp(1j * np.deg2rad(phase + 15.0))
                rows.append(
                    f"10,{x:.8f},{y:.8f},{z:.8f},"
                    f"{ex.real:.8f},{ex.imag:.8f},{ey.real:.8f},{ey.imag:.8f},{ez.real:.8f},{ez.imag:.8f}"
                )
    path.write_text("\n".join(rows), encoding="utf-8")
    return path


if __name__ == "__main__":
    raise SystemExit(main())
