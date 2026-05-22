# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A thermal heatmap analysis tool that processes time-series CSV data from thermal cameras to compute statistics within rectangular regions of interest (ROI). The project provides both a CLI interface (`analyze_heatmap_rois.py`) for batch processing and a Streamlit web UI (`app.py`) for interactive analysis.

## Quick Start

**Run Streamlit UI:**
```powershell
.venv\Scripts\python.exe -m streamlit run app.py
```
Accessible at `http://localhost:8501`

**CLI command example (analyze ROIs):**
```powershell
.venv\Scripts\python.exe analyze_heatmap_rois.py analyze \
  --input-dir input --roi-file rois.json --output-dir output --invert-y
```

**CLI command example (select ROIs interactively):**
```powershell
.venv\Scripts\python.exe analyze_heatmap_rois.py select-rois \
  --input-dir input --frame-index 1 --output rois.json --invert-y
```

## Architecture

### Two-tier Design

**analyze_heatmap_rois.py** (1600+ lines)
- Core data loading and processing library
- CLI subcommands: `analyze`, `select-rois`, `render-heatmaps`
- Data model: `ROI` (dataclass), `FrameData` (dataclass with thermal grid + metadata)
- Handles multiple encodings (cp932, shift_jis, utf-8) for CSV input
- Rendering: matplotlib for static images, PIL for cropping, matplotlib.animation for GIF

**app.py** (1600+ lines)
- Streamlit web UI with 5 tabs (データセット設定→ROI設定→分析実行→可視化→保存済み分析)
- Imports core functions from `analyze_heatmap_rois.py`
- Session state management via `st.session_state` and `settings.ini` persistence
- Plotly for interactive heatmap visualization with rectangle selection
- Key state: `frames`, `rois`, `loaded_analysis`, `session_dir`

### CSV Format Expectations

- Rows 0–6: metadata (ignored)
- Row 7: x-coordinate headers (space/comma separated)
- Rows 8+: y rows, each starting with y-value followed by thermal values
- File naming: numeric suffix interpreted as time-series order (e.g., `data_001.csv`, `data_002.csv`)

### Data Flow

1. Load frames: CSV → FrameData (x_coords, y_coords, heatmap numpy array)
2. Define ROIs: interactive selection on Plotly heatmap or manual input
3. Compute statistics: per-ROI mean/std/variance over all frames
4. Render outputs: PNG images, CSV tables, GIF animation
5. Persist: analysis results in `outputs/{dataset_name}/YYYYMMDD_HHmmSS/`

## Key Design Patterns

**Settings Persistence:**
- Controlled by `_SETTINGS_SCHEMA` dict in app.py
- Read/write via `load_settings()` / `save_settings()` at session start/end
- Settings live in `settings.ini` (ConfigParser format)
- Streamlit widget `key=` parameter links widget to session_state automatically

**ROI Management:**
- Max 30 ROIs per analysis
- Each ROI has name (unique), x/y bounds, enabled flag
- JSON serialization for import/export
- Grid division: programmatically generate ROIs in X×Y pattern

**Heatmap Visualization:**
- Plotly for interactive UI (pan, zoom, rectangle selection)
- Matplotlib for static PNG output
- Shared color scale across all frames (vmin/vmax computed globally)
- Support for Y-axis inversion (invert_y flag)
- View crop feature: limit visible axis range to selected rectangle

**Selection Handling:**
- `extract_box_selection()` parses Plotly rectangle event → (x_min, x_max, y_min, y_max)
- Used for both ROI selection (tab②) and visualization crop (tab④)
- Stored in session_state as `_last_selection`, `_viz_last_selection`

## Important Implementation Notes

**Matplotlib Heatmap Rendering:**
- `draw_heatmap(ax, frame, cmap, vmin, vmax, invert_y)`: core image + colorbar
- `render_single_heatmap()`: wraps draw_heatmap + ROI overlays + title
- `render_heatmap_summary()`: 3×3 grid of frames
- `_apply_view_crop()`: applies xlim/ylim to restrict visible range

**Plotly Interactive Heatmap:**
- `build_plotly_heatmap()`: returns figure with modebar buttons restricted (zoom2d/pan2d/lasso2d removed)
- `uirevision="roi_heatmap"` preserves zoom/pan state across reruns
- `on_select="rerun"` + `selection_mode="box"` enables rectangle selection

**Grid Division Modes:**
- "X固定 → Y分割": 1 column, variable rows
- "Y固定 → X分割": variable columns, 1 row
- "X・Y両方分割": variable columns and rows
- Two input methods: pixel size or division count

**Frame Indexing:**
- Internal 0-based frame_index in FrameData
- Display as 1-based in UI (frame_001.png)
- Time axis can be index-based or timestamp-based

## Extending the Code

**Adding new rendering function:**
1. Import matplotlib/PIL as needed
2. Signature: `def render_*(frames, output_path, rois, ...)`
3. Use `compute_color_scale(frames)` for consistent coloring
4. Test with both individual frames and summaries

**Adding new settings:**
1. Add entry to `_SETTINGS_SCHEMA` with (default_str, type)
2. Add to `_DEFAULTS` dict with `_saved.get(key, default_value)`
3. Use `st.text_input(..., key="setting_name")` to link to session_state
4. Settings auto-save at script end via `save_settings(st.session_state)`

**Adding new CLI subcommand:**
1. Add subparser to `build_parser()` in analyze_heatmap_rois.py
2. Implement handler function
3. Wire up in `main()` switch

## Common Debugging

**CSV fails to load:**
- Check encoding; tool tries cp932, shift_jis, utf-8-sig, utf-8 in order
- Verify x-coord row (row 7) and y-data rows (8+) format

**Settings not persisting:**
- Ensure widget has `key=` parameter matching `_SETTINGS_SCHEMA`
- Confirm `save_settings()` call at end of app.py
- Check `settings.ini` exists in repo root

**Heatmap selection not working:**
- Verify Plotly chart has `on_select="rerun"` + `selection_mode="box"`
- Check `extract_box_selection()` returns non-None tuple
- Session state update (`_last_selection`) must precede any use

**GIF generation slow:**
- Use `--frame-step` on CLI or `frame_step` slider in UI
- Or set `skip_gif=True` to skip animation

## Testing

There is no automated test suite. Manual testing is performed via:
1. CLI: `python analyze_heatmap_rois.py <cmd> --help`
2. Streamlit: load sample CSV, define ROIs, run analysis, check outputs
3. File output validation: verify PNG dimensions, CSV headers, JSON structure
