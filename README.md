# BeamCoverage

PySide6 desktop software for phased-array power-density envelopes and scan coverage.

BeamCoverage is a Windows-first engineering tool for large-aperture phased-array antenna structure checks, scan analysis, far-field pattern calculation, S=S0 power-density envelope prediction, and scan-union coverage review. The desktop EXE is the primary validated product; the Excel/WPS workbook and companion HTML are review aids.

Project homepage and update channel:

- GitHub repository: <https://github.com/huiw11512-dotcom/BeamCoverage>
- Release packages are expected to be published from GitHub Releases.
- `CHANGELOG.md` is included in source, release ZIPs, and the in-application update-notes page.

Run:

```powershell
cd BeamCoverage
python -m pip install -r requirements.txt
python main.py
```

GUI check:

```powershell
python gui_check.py
```

Acceptance check:

```powershell
python acceptance_check.py
```

Recommended validation sequence:

```powershell
python tools\run_validation.py --quick
python tools\run_validation.py --release
```

`tools\run_validation.py` runs the compile check, source documentation/text-integrity checks, acceptance checks, `gui_check.py`, geometry stress, random core stress, imported-layout stress, imported element-pattern stress, export smoke, GUI smoke, and optionally release-package validation in a fixed sequential order. `release_check.py` also validates the release README, EXE metadata, smoke-test payload, Excel companion HTML, APK signature when available, and both ZIP packages (`BeamCoverage_release.zip` and the user-facing `BeamCoverage.zip`). It rejects old verbose prototype branding in window titles, generated notes, workbook titles, companion HTML, and Windows version-resource sources. The validation runner pins OpenBLAS/OMP/MKL/NumExpr thread counts to 1 so Windows memory-pressure false failures from parallel test launches are avoided.

The source documentation check also validates the fallback release-README template used by `tools\build_release.py` and confirms `version_info.txt` matches `app_info.py`, so a clean release directory cannot generate a README that immediately fails `release_check.py` and a version bump cannot leave stale Windows EXE metadata behind. The release README in `dist` must exactly match the generated template, not just contain a few expected keywords. Release artifact names such as the EXE, APK, workbook, companion HTML, release directory, and ZIP files are derived from `app_info.py`.

`tools\build_release.py` rewrites `dist\BeamCoverage_release\README.txt` from the current source template on every package build. The release README is treated as a generated artifact, so stale notes from an older package cannot remain in the ZIP.

Random stress check:

```powershell
python stress_random.py --iterations 50 --seed 20260611
python stress_imported_layout.py --iterations 80 --seed 20260617
python stress_imported_element_pattern.py --iterations 100 --seed 20260617
```

The stress checks also inject invalid geometry, invalid scan-limit cases, arbitrary imported array-layout CSVs, and randomized imported element-pattern CSVs to verify that bad project/script inputs are rejected before expensive calculations start and that supported import formats remain connected to the full calculation chain.
The packaged smoke test also verifies that an imported array layout with overlapping element apertures is rejected and reported in the parameter panel.
The automatic-result panel reports the active element count, array-coordinate source, and active element-pattern model so imported layouts and imported full-wave element patterns are visible in the main calculation state.

Export check:

```powershell
python export_smoke.py
```

Smoke-test startup path, also used for packaged EXE validation:

```powershell
python main.py --smoke-test --smoke-output smoke_dev.json
```

The smoke test verifies the split scan-union cache path by changing only the current scan angle after a full union calculation and requiring a base-cache hit plus a current-angle overlay.

Build release package:

```powershell
python tools\build_release.py
```

For documentation or Excel-only packaging when `dist\BeamCoverage.exe` is already current:

```powershell
python tools\build_release.py --skip-exe-build
```

Build only the internal test EXE:

```powershell
pyinstaller --clean --noconfirm BeamCoverage.spec
```

Validate packaged EXE:

```powershell
$env:QT_QPA_PLATFORM='offscreen'
.\dist\BeamCoverage.exe --smoke-test --smoke-output .\dist\smoke_exe.json
Get-Content .\dist\smoke_exe.json
```

Validate release package:

```powershell
python release_check.py
```

The release directory contains `BeamCoverage.xlsx` plus `BeamCoverage_ScanUnion3D.html`; the workbook links to the companion HTML for interactive 3D scan-union viewing. `release_check.py` validates both files so an old workbook title or missing HTML companion cannot slip into the ZIP.

The GUI uses fixed global x-z and y-z cuts, the same scan-angle-to-direction-cosine convention as the MATLAB prototype, and a cached far-field scan-union calculation. Scan union is unchecked by default and is calculated only when selected.
Parameter edits are staged. Changing a number in the left panel updates only validation and derived text results; the right-hand plots keep showing the last completed calculation until the user clicks `计算`. Exports are blocked while parameters are staged so old plots cannot be mistaken for new results.
The last session is persisted with `QSettings`: parameters, selected calculation pages, window geometry, splitter position, and active tab are restored on the next launch. Use `.project` files when settings must be shared with another machine or archived with imported CSV/FFD/FFS data.
The fixed x-z and y-z 2D envelope cuts remain at phi=0 and phi=90, but their angular windows are sized independently from scanX and scanY so one scan component does not unnecessarily resize the other fixed cut.
The packaged Windows GUI uses `BeamCoverage` as the window title/product name and keeps the top menu/local import-export dialogs in Chinese professional wording. Every plot tab includes the Matplotlib navigation toolbar for home/back/forward, pan, zoom, save, and 3D drag rotation on 3D axes.
3D envelope previews upsample the u-v surface only within contiguous finite envelope regions; NaN gaps from nulls, no-envelope directions, or visible-hemisphere boundaries are preserved instead of being bridged by display interpolation.
Current 2D/3D envelopes are marked as `near_field_sampled_far_field_extrapolated` (`近场采样 + 远场外推`). The scan-union 3D preview remains `far_field_coefficient_union` (`远场系数扫描并集`) for performance, while the fixed x-z/y-z scan-union cuts are marked as `near_field_sampled_far_field_scan_union` (`近场采样 + 远场外推扫描并集`) so the dimension-reading 2D plots use the same envelope method as the current-scan 2D cuts. The scan-union data exposes both `maxRange_m` (3D far-field preview) and `maxRangeNearFieldCuts_m` (near-field fixed-cut maximum), and the GUI displays them separately.
Generated rectangular, circular/elliptical, and diamond array layouts now use the same physical footprint-overlap validator as imported coordinate CSV layouts. The GUI rejects only the active element centers that really overlap for the selected element aperture, so clipped non-rectangular layouts are not rejected just because `ax > dx` or `ay > dy` in an inactive direction.
Analytic rectangular, circular/elliptical, and diamond element apertures share one aperture-shape implementation for automatic scan-loss results and actual field/power-density calculations. This keeps the left-panel scan-loss value aligned with the element model used by 2D cuts, 3D envelopes, and u-v/3D pattern plots.
Shape and layout naming is centralized in `core/aperture_shapes.py`: `ARRAY_LAYOUT_CHOICES`, `ELEMENT_SHAPE_CHOICES`, `normalize_shape_name`, `shape_label`, `element_shape_to_mode`, and `element_overlap_metric` are the single source for GUI labels, project aliases, calculation modes, and invalid-geometry checks.
The GUI now uses the core `derive_params_with_element` entry point, which returns sanitized parameters, derived geometry, and the active element model together. CSV-imported element patterns therefore update `scan_loss_db` at the core workflow boundary rather than relying on a GUI-only correction after derivation.
Excel report export only embeds plot pages whose selected calculation has completed, so unchecked or uncomputed tabs do not appear as placeholder images in the report. The report labels remembered imported CSV files as active or inactive so preserved project paths are not confused with data that participated in the calculation.
When loading a project, imported array-layout and element-pattern CSV paths are resolved from the saved path first, then from the project file directory by the same relative path or same filename. This makes projects portable when the `.project` file and its CSV inputs are moved together. Projects with missing active CSV inputs still load into the GUI so the parameter panel can mark the missing files and the user can relink or switch modes without losing the rest of the settings.

Imported array layout CSV:

- Use `文件 -> 导入阵元坐标 CSV` in the Windows GUI.
- The imported layout overrides Nx/Ny/dx/dy placement and lets arbitrary sparse, missing-element, or non-rectangular arrays enter the same calculation chain.
- Imported coordinates are active only when the array layout selector is set to `导入坐标CSV`; switching back to rectangular/ellipse/diamond uses generated coordinates even if a CSV path was previously loaded.
- Inactive imported-coordinate paths are preserved in the project/UI but ignored by validation and cache keys, so stale project data cannot silently affect generated rectangular/ellipse/diamond arrays.
- If an active imported-coordinate CSV is missing or unreadable, the parameter panel marks that file status explicitly instead of leaving the stale file name ambiguous.
- Imported coordinates are still checked against the selected element aperture. Rectangular, circular/elliptical, and diamond elements are rejected if their physical footprints overlap in the imported x/y coordinates.
- Required columns: `x_m,y_m`.
- Optional columns: `power_w`, `power_scale`, `amplitude`, `phase_deg`, `phase_rad`, `enabled`.
- If no power column is present, each imported element uses the GUI's "每单元输入功率 (W)" value.
- `phase_deg`/`phase_rad` is treated as a fixed per-element initial phase and is added before scan phase steering.
- `enabled=0/false/no/off` disables a row. Disabled rows are ignored by structure, 2D cuts, 3D envelope, u-v pattern, and scan-union calculations.
- After loading, the parameter panel shows the active element count, total imported input power, and derived x/y aperture size for the imported layout.
- Use `文件 -> 导出当前阵元坐标 CSV` to export the currently active generated/imported array coordinates in a format that can be edited and re-imported.
- When a project is saved next to the imported CSV files, those CSV references are stored as relative paths so the project folder can be copied as a bundle.
- Comma, semicolon, tab, and whitespace-delimited text tables are accepted if they include a header row.

Imported element far-field pattern files:

- Use `文件 -> 导入单元远场方向图文件` in the Windows GUI.
- Use `文件 -> 导出单元远场 Real/Imag 模板` for complex component exports with `Real(ETheta),Imag(ETheta),Real(EPhi),Imag(EPhi)`.
- Use `文件 -> 导出单元远场幅相模板` for common solver exports with `Abs(Theta),Phase(Theta),Abs(Phi),Phase(Phi)`.
- Imported far-field element-pattern data participates in u-v pattern, 2D envelope, current 3D envelope, scan-union, and scan-loss calculation only when `使用单元远场方向图` is enabled. If that checkbox is off, any remembered path is treated as inactive and is ignored by validation and cache keys.
- If an enabled imported element-pattern file is missing or unreadable, the parameter panel marks that file status explicitly before calculation continues.
- CSV/TXT/DAT tables and CST/HFSS-style `.ffd`/`.ffs` far-field grids are accepted. FFD/FFS grid files are parsed as theta/phi samples with complex `ETheta/EPhi` fields and are converted through the same imported vector-element path as CSV tables.
- Supported columns are `u,v,gain_db` or `u,v,gain_dbi` for dB data.
- Also supported: `u,v,gain` or `u,v,gain_linear` for linear power gain.
- Theta/phi grids can use `theta_deg,phi_deg,gain_db` or the same linear gain columns. Explicit radian columns such as `theta_rad,phi_rad` or `Theta [rad],Phi [rad]` are also supported; unitless `theta,phi` is kept as degrees for backward compatibility.
- Common HFSS/CST-style columns such as `Theta [deg],Phi [deg],dB(GainTotal)`,
  `dB(RealizedGainTotal)`, `GainTotal [dB]`, and `Abs(GainTotal)` are also recognized.
- Optional phase columns such as `phase_deg`, `phase_rad`, `Phase(GainTotal)`, and `Phase(ETotal)` are recognized.
- Complex field columns such as `field_real,field_imag`, `Real(ETotal),Imag(ETotal)`, or `Re(GainTotal),Im(GainTotal)` are recognized.
- Vector component columns such as `Real(ETheta),Imag(ETheta),Real(EPhi),Imag(EPhi)` are recognized. Magnitude/phase component exports such as `Abs(Theta),Phase(Theta),Abs(Phi),Phase(Phi)` and field-dB component exports such as `dB(ETheta),Phase(ETheta),dB(EPhi),Phase(EPhi)` are also recognized. Imported vector components are converted to global Cartesian `Ex/Ey/Ez` before duplicate-direction merging and grid interpolation, then projected back to the local direction basis during calculation. This prevents broadside `theta=0` multi-phi rows and the `phi=-180/180` seam from canceling because their local spherical bases differ. When both components are present, BeamCoverage sums the array field separately for Etheta and Ephi and then combines power as `|sum(Etheta)|^2 + |sum(Ephi)|^2`.
- When phase or real/imaginary columns are present, the imported complex element response is used directly in 2D cuts, 3D envelopes, u-v patterns, 3D pattern plots, and scan union. Older gain-only CSV files remain compatible and are treated as zero-phase field magnitude.
- The parameter panel shows imported effective direction count, whether the file is gain-only or complex/phase-aware, the maximum imported visible-hemisphere angle `theta_max`, whether the data reaches the 90-degree visible edge, and the normalization factor after the pattern is loaded or restored from a project. Effective direction count is measured after frequency filtering and after merging duplicate u-v directions such as theta=0 rows or the phi=-180/180 seam.
- Comma, semicolon, tab, and whitespace-delimited text tables are accepted. Solver metadata lines before the real `u/v` or `theta/phi` header row are skipped automatically.
- If a solver export contains a frequency column such as `Frequency [GHz]`, `Freq [GHz]`, `Frequency [MHz]`, or `Frequency [Hz]`, BeamCoverage filters the table to the frequency slice nearest the current GUI `f (GHz)` before coordinate conversion and interpolation. Unitless `Freq`/`Frequency` columns are inferred from scale: values around `1e10` are treated as Hz, values around `10000` as MHz, and values around `10` as GHz. Cell values such as `10 GHz`, `10000 MHz`, or `1e10 Hz` are also accepted. The imported-pattern status line shows the selected frequency.
- Imported data is regularized onto a u-v grid and normalized over the visible hemisphere before being used by all envelope and pattern calculations.

Imported element near-field files:

- Use `文件 -> 导入单元近场文件` in the Windows GUI.
- Use `文件 -> 导出单元近场 Ex/Ey/Ez 模板` to create a complex vector near-field template with `Frequency [GHz],x_m,y_m,z_m,Real(Ex),Imag(Ex),Real(Ey),Imag(Ey),Real(Ez),Imag(Ez)`.
- The near-field importer is an experimental data interface: it validates, summarizes, saves, loads, and reports single-element near-field CSV/TXT/DAT tables. Use `文件 -> 从单元近场导出远场方向图 CSV` to create an approximate projected far-field CSV from vector Ex/Ey/Ez samples; the exported CSV is then loaded as the active far-field element model and participates in the main calculation.
- The near-field table itself still does not directly change the main S=S0 envelope calculation. The main calculation uses the active analytic or imported far-field element model, including a projected far-field CSV exported from near-field samples.
- Supported coordinate columns include `x_m,y_m,z_m`, `X [m],Y [m],Z [m]`, and millimeter variants such as `x_mm,y_mm,z_mm`.
- Supported complex vector fields include `Real(Ex),Imag(Ex)`, `Real(Ey),Imag(Ey)`, and `Real(Ez),Imag(Ez)`. Magnitude/phase variants such as `Abs(Ex),Phase(Ex)` are also accepted.
- Scalar near-field power-density columns such as `S_W_m2`, `S [W/m^2]`, `S_W_cm2`, or `S [W/cm^2]` are accepted for validation/reporting.
- If a frequency column is present, BeamCoverage selects the slice nearest the current GUI `f (GHz)` using the same unit inference style as imported far-field direction patterns.
The scan-union tab renders fixed x-z and y-z scan-union envelope cuts alongside the 3D preview. The two fixed cuts are calculated at the 2D-cut angular resolution, instead of being sampled from the coarser 3D preview grid. `scan_union.csv` contains both `3d_union` rows and high-resolution `2d_union_cut` rows for the fixed cuts.
Scan-union timing is reported as a breakdown: total scan-union wall time, 3D far-field preview time, and fixed 2D near-field cut time. Cache hits are shown explicitly so a fast repeated calculation is not confused with a fresh compute.
The fixed 2D scan-union near-field cuts reuse a blockwise observation-to-element propagation matrix across scan centers. This preserves the per-scan envelope-union definition while avoiding repeated distance/element-response calculations.
The scan-union cache is split into a current-angle-independent base scan grid plus a small current-scan overlay. Changing only the current x-z/y-z scan angle reuses the base scan-union cache and computes only the current-angle contribution when that exact angle is not already on the scan grid.
The GUI also caches that current-angle overlay. Repeating the same off-grid scan angle after a successful calculation is a full scan-union cache hit, and the acceptance suite compares the split-cache result against the original full scan-union computation element by element.
The cache key includes the full `ModeSettings` sampling token, not only the mode name. This prevents stale scan-union reuse if future UI or tests vary the sampling resolution under the same fast/standard/fine label.

Custom sampling:

- The `采样设置` panel exposes calculation samples for fixed 2D cuts, current 3D envelopes, u-v patterns, scan-union direction grids, scan-center step, and 3D display-grid limits.
- By default the controls follow `快速/标准/精细`. Enable `自定义计算/显示采样` to override the active mode for final plots or dense debugging cases.
- Calculation sampling changes participate in the cache key. The 3D display-grid limit affects rendering smoothness and drag cost without changing the underlying computed envelope data.

Automatic sampling:

- Automatic sampling is the default behavior when `自定义计算/显示采样` is off.
- The effective fast/standard/fine settings are adapted from frequency, electrical aperture, active array size, spacing-to-wavelength ratio, and aspect ratio.
- Extreme spacing or narrow-lobe cases automatically receive denser 2D angular sampling, u-v sampling, 3D envelope direction sampling, scan-union direction sampling, and 3D display-grid limits.
- The user-facing goal is one-step operation: set physical parameters, click `计算`, then inspect results without manually hunting for sample counts or step size.
- Custom sampling remains available for debugging, final high-density plots, or controlled comparison against older runs.

GitHub update check:

- Use `帮助 -> 检查更新` to query the latest GitHub Release from `huiw11512-dotcom/BeamCoverage`.
- The check compares `APP_VERSION` with the latest Release tag and opens the GitHub Releases page when a newer version is available.
- The application also performs a lightweight startup check at most once per day. A failed or offline check is ignored so the calculation engine remains fully offline-capable.
- Direct in-place EXE replacement is intentionally not performed by the running application. Publish a release ZIP on GitHub, close old application windows, then replace the local package from the downloaded release.

Change log:

- `CHANGELOG.md` records the product history from MATLAB prototype reproduction through the current desktop version.
- The same file is bundled into the EXE and shown in the `更新说明` tab.
- Release ZIPs also include `CHANGELOG.md` so customer-review packages have visible version history without opening the app.
