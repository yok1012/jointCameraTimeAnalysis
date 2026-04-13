"""Streamlit UI for jointCameraTimeAnalysis."""
from __future__ import annotations

import configparser
import io
import json
import zipfile
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import streamlit as st

from analyze_heatmap_rois import (
    MAX_ROIS,
    ROI,
    compute_color_scale,
    draw_heatmap,
    draw_roi_overlays,
    extract_roi_region,
    load_all_frames,
    load_rois,
    plot_results,
    render_heatmap_summary,
    render_roi_crops,
    render_single_heatmap,
    save_gif,
    summarize_frames,
    validate_rois,
    write_results_csv,
    FrameData,
)

st.set_page_config(
    page_title="Heatmap ROI Analyzer",
    page_icon="🌡️",
    layout="wide",
)

# ── 設定ファイル永続化 ─────────────────────────────────────────────────────
SETTINGS_FILE = Path("settings.ini")

# settings.ini で管理するキーとデフォルト値・型
_SETTINGS_SCHEMA: dict[str, tuple[str, type]] = {
    "cmap": ("inferno", str),
    "invert_y": ("False", bool),
    "time_axis": ("index", str),
    "output_base": ("outputs", str),
    "crop_x_offset_left": ("0", int),
    "crop_x_offset_right": ("0", int),
}


def load_settings() -> dict[str, object]:
    """settings.ini から設定値を読み込む。ファイルが無ければデフォルトを返す。"""
    cfg = configparser.ConfigParser()
    cfg.read(str(SETTINGS_FILE), encoding="utf-8")
    result: dict[str, object] = {}
    for key, (default, typ) in _SETTINGS_SCHEMA.items():
        raw = cfg.get("app", key, fallback=default)
        if typ is bool:
            result[key] = raw.lower() in ("true", "1", "yes")
        elif typ is int:
            try:
                result[key] = int(raw)
            except ValueError:
                result[key] = int(default)
        else:
            result[key] = raw
    # input_dir_history は JSON リストとして保存
    raw_hist = cfg.get("app", "input_dir_history", fallback="[]")
    try:
        result["input_dir_history"] = json.loads(raw_hist)
    except (json.JSONDecodeError, ValueError):
        result["input_dir_history"] = []
    return result


def save_settings(state: dict[str, object]) -> None:
    """現在の設定値を settings.ini に書き出す。"""
    cfg = configparser.ConfigParser()
    cfg["app"] = {}
    for key in _SETTINGS_SCHEMA:
        cfg["app"][key] = str(state.get(key, _SETTINGS_SCHEMA[key][0]))
    cfg["app"]["input_dir_history"] = json.dumps(
        state.get("input_dir_history", []), ensure_ascii=False
    )
    with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
        cfg.write(f)


# ── 分析結果の永続化 ──────────────────────────────────────────────────────
SAVED_ANALYSES_DIR = Path("saved_analyses")


def save_analysis(
    name: str,
    rows: list[dict],
    rois: list[ROI],
    input_dir: str,
    dataset_name: str,
    session_dir: str | None,
) -> Path:
    """分析結果をJSONファイルとして保存し、ファイルパスを返す。"""
    SAVED_ANALYSES_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = name.replace(" ", "_").replace("/", "_").replace("\\", "_")
    filename = f"{timestamp}_{safe_name}.json"
    data = {
        "name": name,
        "saved_at": datetime.now().isoformat(),
        "input_dir": input_dir,
        "dataset_name": dataset_name,
        "session_dir": str(session_dir) if session_dir else None,
        "rois": [r.to_dict() for r in rois],
        "rows": rows,
    }
    path = SAVED_ANALYSES_DIR / filename
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def list_saved_analyses() -> list[dict]:
    """保存済み分析の一覧をメタデータ付きで返す（新しい順）。"""
    if not SAVED_ANALYSES_DIR.exists():
        return []
    entries: list[dict] = []
    for p in sorted(SAVED_ANALYSES_DIR.glob("*.json"), reverse=True):
        try:
            meta = json.loads(p.read_text(encoding="utf-8"))
            entries.append({
                "path": p,
                "name": meta.get("name", p.stem),
                "saved_at": meta.get("saved_at", ""),
                "input_dir": meta.get("input_dir", ""),
                "dataset_name": meta.get("dataset_name", ""),
                "roi_count": len(meta.get("rois", [])),
                "row_count": len(meta.get("rows", [])),
            })
        except Exception:
            continue
    return entries


def load_analysis(path: Path) -> dict:
    """保存済み分析を読み込む。"""
    return json.loads(path.read_text(encoding="utf-8"))


# ── ROI永続化 ──────────────────────────────────────────────────────────────
ROI_SAVE_FILE = Path("rois_saved.json")


def save_rois_to_file(rois: list[ROI]) -> None:
    """ROI定義をローカルJSONファイルに保存する。"""
    ROI_SAVE_FILE.write_text(
        json.dumps([r.to_dict() for r in rois], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def load_rois_from_file() -> list[ROI]:
    """保存済みROI定義を読み込む。ファイルが無ければ空リストを返す。"""
    if not ROI_SAVE_FILE.exists():
        return []
    try:
        data = json.loads(ROI_SAVE_FILE.read_text(encoding="utf-8"))
        return [
            ROI(
                name=str(item["name"]),
                x_min=int(item["x_min"]),
                x_max=int(item["x_max"]),
                y_min=int(item["y_min"]),
                y_max=int(item["y_max"]),
                enabled=bool(item.get("enabled", True)),
            )
            for item in data
        ]
    except Exception:
        return []


# ── セッションステートの初期化 ──────────────────────────────────────────────
_saved = load_settings()
_DEFAULTS: dict[str, object] = {
    "rois": load_rois_from_file(),
    "frames": [],
    "analysis_rows": [],
    "input_dir": "",
    "cmap": _saved.get("cmap", "inferno"),
    "invert_y": _saved.get("invert_y", False),
    "time_axis": _saved.get("time_axis", "index"),
    "output_base": _saved.get("output_base", "outputs"),
    "dataset_name": "",
    "session_dir": None,
    "frame_index_for_roi": 0,
    "canvas_key_counter": 0,
    "input_dir_history": _saved.get("input_dir_history", []),
    "editing_roi_index": -1,
    "crop_x_offset_left": _saved.get("crop_x_offset_left", 0),
    "crop_x_offset_right": _saved.get("crop_x_offset_right", 0),
    "loaded_analysis": None,
}
for key, value in _DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = value


def resolve_session_dir() -> Path:
    base = Path(st.session_state["output_base"])
    dataset = st.session_state["dataset_name"] or "default"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return base / dataset / timestamp


def get_or_create_session_dir() -> Path:
    if st.session_state["session_dir"] is None:
        session_dir = resolve_session_dir()
        session_dir.mkdir(parents=True, exist_ok=True)
        st.session_state["session_dir"] = session_dir
    return Path(st.session_state["session_dir"])


def fig_to_bytes(fig: plt.Figure) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    return buf.read()


def build_zip(paths: list[Path]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for path in paths:
            if path.exists():
                zf.write(path, path.name)
    buf.seek(0)
    return buf.read()


def heatmap_to_png_bytes(
    frame: FrameData,
    rois: list[ROI],
    cmap: str,
    vmin: float,
    vmax: float,
    invert_y: bool,
) -> bytes:
    fig, ax = plt.subplots(figsize=(8, 6))
    draw_heatmap(ax, frame, cmap, vmin, vmax, invert_y=invert_y)
    if rois:
        draw_roi_overlays(ax, rois)
    ax.set_title(f"Frame {frame.frame_index}: {frame.file_name}")
    fig.tight_layout()
    data = fig_to_bytes(fig)
    plt.close(fig)
    return data


def heatmap_as_display_image(
    frame: FrameData,
    rois: list[ROI],
    cmap: str,
    invert_y: bool,
) -> np.ndarray:
    """Return an RGBA numpy array of the heatmap for canvas background."""
    vmin, vmax = compute_color_scale([frame])
    fig, ax = plt.subplots(figsize=(8, 6))
    draw_heatmap(ax, frame, cmap, vmin, vmax, invert_y=invert_y)
    if rois:
        draw_roi_overlays(ax, rois)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    from PIL import Image
    return Image.open(buf).convert("RGBA")


def generate_grid_rois(
    prefix: str,
    x_min: int, x_max: int, x_divs: int,
    y_min: int, y_max: int, y_divs: int,
    reverse_x: bool = False,
    reverse_y: bool = False,
) -> list[ROI]:
    """Generate ROIs by dividing the given range into a grid of x_divs * y_divs."""
    x_edges = np.linspace(x_min, x_max + 1, x_divs + 1, dtype=int)
    y_edges = np.linspace(y_min, y_max + 1, y_divs + 1, dtype=int)
    x_order = list(reversed(range(x_divs))) if reverse_x else list(range(x_divs))
    y_order = list(reversed(range(y_divs))) if reverse_y else list(range(y_divs))
    rois: list[ROI] = []
    for y_label, yi in enumerate(y_order, 1):
        for x_label, xi in enumerate(x_order, 1):
            name = f"{prefix}_x{x_label}_y{y_label}" if (x_divs > 1 and y_divs > 1) else (
                f"{prefix}_x{x_label}" if x_divs > 1 else f"{prefix}_y{y_label}"
            )
            rois.append(ROI(
                name=name,
                x_min=int(x_edges[xi]),
                x_max=int(x_edges[xi + 1] - 1),
                y_min=int(y_edges[yi]),
                y_max=int(y_edges[yi + 1] - 1),
            ))
    return rois


def build_plotly_heatmap(
    frame: FrameData,
    rois: list[ROI],
    cmap: str,
    invert_y: bool,
    preview_rois: list[ROI] | None = None,
) -> go.Figure:
    """Return a Plotly Figure with the heatmap and existing ROI overlays.

    The figure uses ``dragmode='drawrect'`` so the user can draw rectangles
    directly on the plot.  Coordinates in the shapes correspond 1-to-1 with
    data coordinates, so no pixel→data conversion is needed.
    """
    # Plotly colorscale names differ slightly from matplotlib
    plotly_cmap_map = {
        "inferno": "Inferno", "hot": "Hot", "plasma": "Plasma",
        "viridis": "Viridis", "jet": "Jet", "gray": "Greys",
    }
    plotly_cmap = plotly_cmap_map.get(cmap, "Inferno")

    x0 = int(frame.x_coords.min())
    x1 = int(frame.x_coords.max())
    y0 = int(frame.y_coords.min())
    y1 = int(frame.y_coords.max())

    fig = go.Figure(data=go.Heatmap(
        z=frame.heatmap,
        x=frame.x_coords,
        y=frame.y_coords,
        colorscale=plotly_cmap,
        colorbar=dict(title="temp"),
    ))

    # Draw existing ROI overlays
    COLORS = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
        "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
        "#aec7e8", "#ffbb78", "#98df8a", "#ff9896", "#c5b0d5",
        "#c49c94", "#f7b6d2", "#c7c7c7", "#dbdb8d", "#9edae5",
    ]
    for idx, roi in enumerate(rois):
        color = COLORS[idx % len(COLORS)]
        fig.add_shape(
            type="rect",
            x0=roi.x_min - 0.5, y0=roi.y_min - 0.5,
            x1=roi.x_max + 0.5, y1=roi.y_max + 0.5,
            line=dict(color=color, width=2),
            fillcolor="rgba(0,0,0,0)",
        )
        fig.add_annotation(
            x=roi.x_min, y=roi.y_max + 1,
            text=f"#{idx+1}: {roi.name}",
            showarrow=False,
            font=dict(color=color, size=11),
            bgcolor="rgba(0,0,0,0.4)",
        )

    # プレビュー分割ROI（破線で表示）
    if preview_rois:
        for idx, proi in enumerate(preview_rois):
            fig.add_shape(
                type="rect",
                x0=proi.x_min - 0.5, y0=proi.y_min - 0.5,
                x1=proi.x_max + 0.5, y1=proi.y_max + 0.5,
                line=dict(color="lime", width=2, dash="dash"),
                fillcolor="rgba(0,255,0,0.05)",
            )
            fig.add_annotation(
                x=(proi.x_min + proi.x_max) / 2,
                y=(proi.y_min + proi.y_max) / 2,
                text=proi.name,
                showarrow=False,
                font=dict(color="lime", size=10),
                bgcolor="rgba(0,0,0,0.5)",
            )

    fig.update_layout(
        dragmode="select",
        selectdirection="any",
        xaxis=dict(title="x", range=[x0 - 0.5, x1 + 0.5]),
        yaxis=dict(
            title="y",
            range=[y1 + 0.5, y0 - 0.5] if invert_y else [y0 - 0.5, y1 + 0.5],
            scaleanchor="x",
        ),
        height=600,
        margin=dict(l=60, r=30, t=40, b=60),
    )
    return fig


# ── タブ構成 ────────────────────────────────────────────────────────────────
st.title("🌡️ Heatmap ROI Analyzer")
tabs = st.tabs(["① データセット設定", "② ROI設定", "③ 分析実行", "④ ヒートマップ可視化", "⑤ 保存済み分析"])


# ─────────────────────────────────────────────────────────────────────────────
# TAB 1: データセット設定
# ─────────────────────────────────────────────────────────────────────────────
with tabs[0]:
    st.header("データセット設定")

    col_in, col_out = st.columns(2)
    with col_in:
        st.subheader("入力")

        # 履歴から選択
        history: list[str] = st.session_state["input_dir_history"]
        if history:
            picked = st.selectbox(
                "過去に使用したディレクトリ",
                options=["(直接入力)"] + history,
                index=0,
                key="input_dir_history_picker",
            )
            default_dir = picked if picked != "(直接入力)" else (st.session_state["input_dir"] or "input")
        else:
            default_dir = st.session_state["input_dir"] or "input"

        input_dir_str = st.text_input(
            "入力ディレクトリ（CSVが入っているフォルダ）",
            value=default_dir,
            key="input_dir_widget",
        )
        if st.button("CSVを読み込む"):
            input_dir = Path(input_dir_str)
            if not input_dir.exists():
                st.error(f"ディレクトリが見つかりません: {input_dir}")
            else:
                with st.spinner("CSV読み込み中…"):
                    try:
                        frames = load_all_frames(input_dir)
                        st.session_state["frames"] = frames
                        st.session_state["input_dir"] = input_dir_str
                        st.session_state["session_dir"] = None
                        # 履歴に追加（重複排除、最新を先頭に）
                        hist = [h for h in history if h != input_dir_str]
                        st.session_state["input_dir_history"] = [input_dir_str] + hist[:19]
                        save_settings(st.session_state)
                        st.success(f"{len(frames)} フレームを読み込みました")
                    except Exception as exc:
                        st.error(f"読み込みエラー: {exc}")

    with col_out:
        st.subheader("出力")
        output_base = st.text_input(
            "出力先フォルダ（ベースディレクトリ）",
            value=str(st.session_state["output_base"]),
        )
        dataset_name = st.text_input(
            "データセット名（出力サブフォルダ名）",
            value=str(st.session_state["dataset_name"]),
            placeholder="例: experiment_A",
        )
        st.session_state["output_base"] = output_base
        st.session_state["dataset_name"] = dataset_name
        if output_base and dataset_name:
            preview_path = Path(output_base) / dataset_name / "YYYYMMDD_HHmmSS"
            st.caption(f"出力先プレビュー: `{preview_path}/`")

    st.divider()
    st.subheader("表示設定")
    col_vis1, col_vis2, col_vis3 = st.columns(3)
    with col_vis1:
        cmap = st.selectbox(
            "カラーマップ",
            ["inferno", "hot", "plasma", "viridis", "jet", "gray"],
            index=["inferno", "hot", "plasma", "viridis", "jet", "gray"].index(
                st.session_state["cmap"]
            ),
        )
        st.session_state["cmap"] = cmap
    with col_vis2:
        invert_y = st.toggle("y軸を反転", value=bool(st.session_state["invert_y"]))
        st.session_state["invert_y"] = invert_y
    with col_vis3:
        time_axis = st.radio("時間軸", ["index", "timestamp"], horizontal=True,
                             index=["index", "timestamp"].index(st.session_state["time_axis"]))
        st.session_state["time_axis"] = time_axis
    save_settings(st.session_state)

    if st.session_state["frames"]:
        frames: list[FrameData] = st.session_state["frames"]
        st.divider()
        st.subheader("読み込み済みデータの概要")
        frame_0 = frames[0]
        col_a, col_b, col_c, col_d = st.columns(4)
        col_a.metric("フレーム数", len(frames))
        col_b.metric("x範囲", f"{int(frame_0.x_coords.min())} – {int(frame_0.x_coords.max())}")
        col_c.metric("y範囲", f"{int(frame_0.y_coords.min())} – {int(frame_0.y_coords.max())}")
        col_d.metric("最終フレーム", frames[-1].frame_index)

        st.subheader("自動プレビュー（先頭フレーム）")
        vmin, vmax = compute_color_scale(frames)
        preview_png = heatmap_to_png_bytes(
            frame=frame_0,
            rois=st.session_state["rois"],
            cmap=st.session_state["cmap"],
            vmin=vmin,
            vmax=vmax,
            invert_y=bool(st.session_state["invert_y"]),
        )
        st.image(preview_png, caption=f"Frame {frame_0.frame_index}: {frame_0.file_name}")


# ─────────────────────────────────────────────────────────────────────────────
# TAB 2: ROI設定
# ─────────────────────────────────────────────────────────────────────────────
with tabs[1]:
    st.header("ROI設定")

    if not st.session_state["frames"]:
        st.info("タブ①でCSVを読み込んでください。")
    else:
        frames: list[FrameData] = st.session_state["frames"]
        cmap = st.session_state["cmap"]
        invert_y = st.session_state["invert_y"]

        # 代表フレーム選択
        frame_indices = [f.frame_index for f in frames]
        selected_idx = st.selectbox(
            "代表フレーム（ROI選択に使用）",
            options=frame_indices,
            index=min(st.session_state["frame_index_for_roi"], len(frame_indices) - 1),
        )
        st.session_state["frame_index_for_roi"] = frame_indices.index(selected_idx)
        rep_frame = frames[frame_indices.index(selected_idx)]

        # ── 切り出しオフセット設定 ─────────────────────────────────
        x_lo = int(rep_frame.x_coords.min())
        x_hi = int(rep_frame.x_coords.max())
        y_lo = int(rep_frame.y_coords.min())
        y_hi = int(rep_frame.y_coords.max())

        st.subheader("切り出しオフセット設定")
        st.caption("グリッド分割・手動追加時にX方向を左右から狭めるピクセル数を指定します。ROI座標に直接反映されます。")
        _offset_max = max(0, (x_hi - x_lo) // 2)
        off_col1, off_col2, off_col3 = st.columns([1, 1, 2])
        with off_col1:
            crop_x_offset_left = st.number_input(
                "左オフセット（px）",
                min_value=0, max_value=_offset_max,
                value=min(int(st.session_state["crop_x_offset_left"]), _offset_max),
                step=1, key="crop_x_offset_left_input",
            )
            st.session_state["crop_x_offset_left"] = crop_x_offset_left
        with off_col2:
            crop_x_offset_right = st.number_input(
                "右オフセット（px）",
                min_value=0, max_value=_offset_max,
                value=min(int(st.session_state["crop_x_offset_right"]), _offset_max),
                step=1, key="crop_x_offset_right_input",
            )
            st.session_state["crop_x_offset_right"] = crop_x_offset_right
        with off_col3:
            eff_x_lo = x_lo + crop_x_offset_left
            eff_x_hi = x_hi - crop_x_offset_right
            if eff_x_lo < eff_x_hi:
                st.metric("適用後のX範囲", f"{eff_x_lo} – {eff_x_hi}")
            else:
                st.error("オフセットが大きすぎます（左右の合計がX幅を超えています）")
        save_settings(st.session_state)

        # ── グリッド分割 ROI ──────────────────────────────────────────
        grid_preview_rois: list[ROI] = []

        with st.expander("グリッド分割でROIを一括追加", expanded=False):
            st.markdown(
                "X・Y方向を等間隔に分割してROIを一括生成します。"
                "分割プレビューがヒートマップ上に破線で表示されます。"
            )
            g_col1, g_col2, g_col3 = st.columns(3)
            with g_col1:
                grid_mode = st.radio(
                    "分割モード",
                    ["X固定 → Y分割", "Y固定 → X分割", "グリッド（両方分割）"],
                    key="grid_mode",
                )
            with g_col2:
                grid_x_min = st.number_input("X範囲 min", value=eff_x_lo, min_value=x_lo, max_value=x_hi, key="gx1")
                grid_x_max = st.number_input("X範囲 max", value=eff_x_hi, min_value=x_lo, max_value=x_hi, key="gx2")
            with g_col3:
                grid_y_min = st.number_input("Y範囲 min", value=y_lo, min_value=y_lo, max_value=y_hi, key="gy1")
                grid_y_max = st.number_input("Y範囲 max", value=y_hi, min_value=y_lo, max_value=y_hi, key="gy2")

            remaining = MAX_ROIS - len(st.session_state["rois"])

            input_method = st.radio(
                "分割指定方法",
                ["分割数で指定", "ピクセル数で指定"],
                key="grid_input_method",
                horizontal=True,
            )

            x_range = max(1, grid_x_max - grid_x_min + 1)
            y_range = max(1, grid_y_max - grid_y_min + 1)

            d_col1, d_col2 = st.columns(2)

            if input_method == "ピクセル数で指定":
                # ── ピクセル数指定モード ──
                if grid_mode == "X固定 → Y分割":
                    x_divs = 1
                    y_px = d_col1.number_input("Y方向 1ROIあたりのピクセル数", min_value=1, max_value=y_range, value=max(1, y_range // 2), key="gy_px")
                    y_divs = max(1, y_range // y_px)
                    d_col1.caption(f"→ Y {y_divs} 分割（端数は最後のROIに含まれます）")
                elif grid_mode == "Y固定 → X分割":
                    x_px = d_col1.number_input("X方向 1ROIあたりのピクセル数", min_value=1, max_value=x_range, value=max(1, x_range // 2), key="gx_px")
                    x_divs = max(1, x_range // x_px)
                    y_divs = 1
                    d_col1.caption(f"→ X {x_divs} 分割（端数は最後のROIに含まれます）")
                else:
                    x_px = d_col1.number_input("X方向 1ROIあたりのピクセル数", min_value=1, max_value=x_range, value=max(1, x_range // 2), key="gx_px2")
                    y_px = d_col2.number_input("Y方向 1ROIあたりのピクセル数", min_value=1, max_value=y_range, value=max(1, y_range // 2), key="gy_px2")
                    x_divs = max(1, x_range // x_px)
                    y_divs = max(1, y_range // y_px)
                    d_col1.caption(f"→ X {x_divs} 分割")
                    d_col2.caption(f"→ Y {y_divs} 分割")
            else:
                # ── 分割数指定モード（従来） ──
                if grid_mode == "X固定 → Y分割":
                    x_divs = 1
                    max_y = max(2, min(remaining, 10))
                    y_divs = d_col1.slider("Y分割数", 1, max_y, value=min(2, remaining), key="gy_divs")
                elif grid_mode == "Y固定 → X分割":
                    max_x = max(2, min(remaining, 10))
                    x_divs = d_col1.slider("X分割数", 1, max_x, value=min(2, remaining), key="gx_divs")
                    y_divs = 1
                else:
                    max_x = max(2, min(remaining, 10))
                    x_divs = d_col1.slider("X分割数", 1, max_x, value=min(2, max_x), key="gx_divs2")
                    max_y = max(2, min(remaining // max(x_divs, 1), 10))
                    y_divs = d_col2.slider("Y分割数", 1, max_y, value=min(2, max_y), key="gy_divs2")

            total_rois = x_divs * y_divs
            grid_prefix = st.text_input("ROI名プレフィックス", value="grid", key="grid_prefix")

            rev_col1, rev_col2 = st.columns(2)
            reverse_x = rev_col1.checkbox("X方向を反転（Xmax→Xmin の順に分割）", key="grid_reverse_x")
            reverse_y = rev_col2.checkbox("Y方向を反転（Ymax→Ymin の順に分割）", key="grid_reverse_y")

            if grid_x_min >= grid_x_max or grid_y_min >= grid_y_max:
                st.error("範囲が不正です（min < max にしてください）")
            elif total_rois > remaining:
                st.error(f"生成数 {total_rois} 件が残り枠 {remaining} 件を超えます")
            else:
                grid_preview_rois = generate_grid_rois(
                    grid_prefix, grid_x_min, grid_x_max, x_divs, grid_y_min, grid_y_max, y_divs,
                    reverse_x=reverse_x, reverse_y=reverse_y,
                )
                st.info(f"プレビュー: **{total_rois}** 件のROIを生成（ヒートマップ上に破線で表示中）")
                with st.container(border=True):
                    st.caption("生成されるROI一覧")
                    for proi in grid_preview_rois:
                        st.text(f"  {proi.name}: x={proi.x_min}-{proi.x_max}, y={proi.y_min}-{proi.y_max}")

                if st.button("グリッド分割ROIを一括追加", type="primary", key="grid_add"):
                    existing = st.session_state["rois"]
                    existing_names = {r.name for r in existing}
                    conflicts = [r.name for r in grid_preview_rois if r.name in existing_names]
                    if conflicts:
                        st.error(f"名前が重複しています: {', '.join(conflicts)}")
                    else:
                        st.session_state["rois"] = [*existing, *grid_preview_rois]
                        save_rois_to_file(st.session_state["rois"])
                        st.session_state["canvas_key_counter"] += 1
                        st.rerun()

        # ── Plotly ヒートマップ ──────────────────────────────────────
        st.subheader("ROI範囲指定ヒートマップ")
        st.markdown(
            "**手順**:  \n"
            "① ヒートマップ上でドラッグして範囲を選択  \n"
            "② 下部に表示される座標を確認し、ROI名を入力  \n"
            "③「ROIを追加」ボタンを押す  \n"
            "④ 右側の一覧から各ROIの名前・座標を編集可能  \n"
            f"最大 **{MAX_ROIS}** 件まで登録可能です。"
        )

        plotly_fig = build_plotly_heatmap(
            rep_frame, st.session_state["rois"], cmap, invert_y,
            preview_rois=grid_preview_rois or None,
        )
        plotly_event = st.plotly_chart(
            plotly_fig,
            use_container_width=True,
            key=f"roi_plotly_{st.session_state['canvas_key_counter']}",
            on_select="rerun",
            selection_mode="box",
        )

        # ── ROI追加・一覧 ────────────────────────────────────────────
        col_add, col_roi_list = st.columns([3, 2])

        with col_add:
            st.subheader("描画した矩形からROIを追加")
            rois: list[ROI] = st.session_state["rois"]

            # Plotlyの box selection イベントから座標を取得
            drawn_roi_coords = None

            if plotly_event is not None and plotly_event.get("selection"):
                sel = plotly_event["selection"]
                x_min_data = int(rep_frame.x_coords.min())
                x_max_data = int(rep_frame.x_coords.max())
                y_min_data = int(rep_frame.y_coords.min())
                y_max_data = int(rep_frame.y_coords.max())

                # 方法1: box selection の範囲を直接使用
                if sel.get("box") and len(sel["box"]) > 0:
                    box = sel["box"][0]
                    raw_x = sorted(box.get("x", []))
                    raw_y = sorted(box.get("y", []))
                    if len(raw_x) >= 2 and len(raw_y) >= 2:
                        roi_x_min = max(x_min_data, int(round(raw_x[0])))
                        roi_x_max = min(x_max_data, int(round(raw_x[-1])))
                        roi_y_min = max(y_min_data, int(round(raw_y[0])))
                        roi_y_max = min(y_max_data, int(round(raw_y[-1])))
                        if roi_x_min < roi_x_max and roi_y_min < roi_y_max:
                            drawn_roi_coords = (roi_x_min, roi_x_max, roi_y_min, roi_y_max)

                # 方法2: 選択されたポイントのmin/maxから範囲を算出
                if drawn_roi_coords is None and sel.get("points") and len(sel["points"]) > 0:
                    pts = sel["points"]
                    xs = [p.get("x", 0) for p in pts if "x" in p]
                    ys = [p.get("y", 0) for p in pts if "y" in p]
                    if xs and ys:
                        roi_x_min = max(x_min_data, int(min(xs)))
                        roi_x_max = min(x_max_data, int(max(xs)))
                        roi_y_min = max(y_min_data, int(min(ys)))
                        roi_y_max = min(y_max_data, int(max(ys)))
                        if roi_x_min < roi_x_max and roi_y_min < roi_y_max:
                            drawn_roi_coords = (roi_x_min, roi_x_max, roi_y_min, roi_y_max)

            if drawn_roi_coords:
                roi_x_min, roi_x_max, roi_y_min, roi_y_max = drawn_roi_coords
                st.success(
                    f"選択範囲: **x={roi_x_min}–{roi_x_max}**, **y={roi_y_min}–{roi_y_max}**"
                )
                roi_name = st.text_input(
                    "ROI名",
                    value=f"roi_{len(rois) + 1}",
                    key="new_roi_name",
                )
                if st.button("ROIを追加", disabled=len(rois) >= MAX_ROIS):
                    if any(r.name == roi_name for r in rois):
                        st.error(f"ROI名 '{roi_name}' は既に使われています")
                    else:
                        new_roi = ROI(
                            name=roi_name,
                            x_min=roi_x_min,
                            x_max=roi_x_max,
                            y_min=roi_y_min,
                            y_max=roi_y_max,
                        )
                        st.session_state["rois"] = [*rois, new_roi]
                        save_rois_to_file(st.session_state["rois"])
                        st.session_state["canvas_key_counter"] += 1
                        st.rerun()
            else:
                st.info("ヒートマップ上でドラッグして範囲を選択してください")

            st.divider()
            st.subheader("手動入力")
            with st.expander("座標を直接入力して追加"):
                frame_range = rep_frame
                mc1, mc2 = st.columns(2)
                manual_name = mc1.text_input("ROI名", value=f"roi_{len(rois)+1}", key="manual_name")
                mc1.number_input("x_min", value=int(frame_range.x_coords.min()), key="mx1",
                                  min_value=int(frame_range.x_coords.min()),
                                  max_value=int(frame_range.x_coords.max()))
                mc1.number_input("x_max", value=int(frame_range.x_coords.max()), key="mx2",
                                  min_value=int(frame_range.x_coords.min()),
                                  max_value=int(frame_range.x_coords.max()))
                mc2.number_input("y_min", value=int(frame_range.y_coords.min()), key="my1",
                                  min_value=int(frame_range.y_coords.min()),
                                  max_value=int(frame_range.y_coords.max()))
                mc2.number_input("y_max", value=int(frame_range.y_coords.max()), key="my2",
                                  min_value=int(frame_range.y_coords.min()),
                                  max_value=int(frame_range.y_coords.max()))
                if st.button("手動でROIを追加"):
                    if any(r.name == manual_name for r in rois):
                        st.error(f"ROI名 '{manual_name}' は既に使われています")
                    elif st.session_state["mx1"] >= st.session_state["mx2"]:
                        st.error("x_min は x_max より小さくしてください")
                    elif st.session_state["my1"] >= st.session_state["my2"]:
                        st.error("y_min は y_max より小さくしてください")
                    else:
                        new_roi = ROI(
                            name=manual_name,
                            x_min=st.session_state["mx1"],
                            x_max=st.session_state["mx2"],
                            y_min=st.session_state["my1"],
                            y_max=st.session_state["my2"],
                        )
                        st.session_state["rois"] = [*rois, new_roi]
                        save_rois_to_file(st.session_state["rois"])
                        st.rerun()

        with col_roi_list:
            roi_header_col, roi_reset_col = st.columns([3, 1])
            roi_header_col.subheader(f"現在のROI一覧（{len(st.session_state['rois'])}/{MAX_ROIS}）")
            if st.session_state["rois"] and roi_reset_col.button("全ROIをリセット", type="secondary", key="reset_all_rois"):
                st.session_state["rois"] = []
                save_rois_to_file([])
                st.session_state["canvas_key_counter"] += 1
                st.rerun()
            rois: list[ROI] = st.session_state["rois"]
            if rois:
                x_lo = int(rep_frame.x_coords.min())
                x_hi = int(rep_frame.x_coords.max())
                y_lo = int(rep_frame.y_coords.min())
                y_hi = int(rep_frame.y_coords.max())
                for i, roi in enumerate(rois):
                    status_mark = "" if roi.enabled else " [OFF]"
                    with st.expander(f"#{i + 1}: {roi.name}{status_mark}  (x:{roi.x_min}-{roi.x_max}, y:{roi.y_min}-{roi.y_max})"):
                        # 分析対象チェックボックス
                        new_enabled = st.checkbox(
                            "分析に含める",
                            value=roi.enabled,
                            key=f"enable_roi_{i}",
                        )
                        if new_enabled != roi.enabled:
                            updated = ROI(name=roi.name, x_min=roi.x_min, x_max=roi.x_max,
                                          y_min=roi.y_min, y_max=roi.y_max, enabled=new_enabled)
                            new_rois = list(rois)
                            new_rois[i] = updated
                            st.session_state["rois"] = new_rois
                            save_rois_to_file(new_rois)
                            st.rerun()
                        # 編集フォーム
                        new_name = st.text_input("名前", value=roi.name, key=f"edit_name_{i}")
                        ec1, ec2 = st.columns(2)
                        clamped_xmin = max(x_lo, min(roi.x_min, x_hi))
                        clamped_xmax = max(x_lo, min(roi.x_max, x_hi))
                        clamped_ymin = max(y_lo, min(roi.y_min, y_hi))
                        clamped_ymax = max(y_lo, min(roi.y_max, y_hi))
                        if (clamped_xmin != roi.x_min or clamped_xmax != roi.x_max
                                or clamped_ymin != roi.y_min or clamped_ymax != roi.y_max):
                            st.warning("ROI座標がフレーム範囲外です。値がクランプされています。")
                        new_xmin = ec1.number_input("x_min", value=clamped_xmin, min_value=x_lo, max_value=x_hi, key=f"edit_x1_{i}")
                        new_xmax = ec1.number_input("x_max", value=clamped_xmax, min_value=x_lo, max_value=x_hi, key=f"edit_x2_{i}")
                        new_ymin = ec2.number_input("y_min", value=clamped_ymin, min_value=y_lo, max_value=y_hi, key=f"edit_y1_{i}")
                        new_ymax = ec2.number_input("y_max", value=clamped_ymax, min_value=y_lo, max_value=y_hi, key=f"edit_y2_{i}")
                        btn_c1, btn_c2 = st.columns(2)
                        if btn_c1.button("保存", key=f"save_roi_{i}", type="primary"):
                            # 名前重複チェック (自分自身は除く)
                            if any(r.name == new_name for j, r in enumerate(rois) if j != i):
                                st.error(f"ROI名 '{new_name}' は既に使われています")
                            elif new_xmin >= new_xmax:
                                st.error("x_min < x_max にしてください")
                            elif new_ymin >= new_ymax:
                                st.error("y_min < y_max にしてください")
                            else:
                                updated = ROI(name=new_name, x_min=new_xmin, x_max=new_xmax,
                                              y_min=new_ymin, y_max=new_ymax, enabled=new_enabled)
                                new_rois = list(rois)
                                new_rois[i] = updated
                                st.session_state["rois"] = new_rois
                                save_rois_to_file(new_rois)
                                st.session_state["canvas_key_counter"] += 1
                                st.rerun()
                        if btn_c2.button("削除", key=f"del_roi_{i}"):
                            st.session_state["rois"] = [r for j, r in enumerate(rois) if j != i]
                            save_rois_to_file(st.session_state["rois"])
                            st.session_state["canvas_key_counter"] += 1
                            st.rerun()
            else:
                st.info("ROIがまだ追加されていません（最大10件）")

            st.divider()
            st.subheader("JSON インポート / エクスポート")
            uploaded = st.file_uploader("既存ROI JSONを読み込む", type="json", key="roi_upload")
            if uploaded:
                try:
                    loaded_rois = [
                        ROI(
                            name=str(item["name"]),
                            x_min=int(item["x_min"]),
                            x_max=int(item["x_max"]),
                            y_min=int(item["y_min"]),
                            y_max=int(item["y_max"]),
                            enabled=bool(item.get("enabled", True)),
                        )
                        for item in json.load(uploaded)
                    ]
                    validate_rois(loaded_rois, rep_frame.x_coords, rep_frame.y_coords)
                    st.session_state["rois"] = loaded_rois
                    save_rois_to_file(loaded_rois)
                    st.success(f"{len(loaded_rois)} 件のROIを読み込みました")
                    st.rerun()
                except Exception as exc:
                    st.error(f"読み込みエラー: {exc}")

            export_rois: list[ROI] = st.session_state["rois"]
            if export_rois:
                roi_json = json.dumps(
                    [r.to_dict() for r in export_rois], ensure_ascii=False, indent=2
                )
                st.download_button(
                    "ROI定義をJSONとしてダウンロード",
                    data=roi_json.encode("utf-8"),
                    file_name="rois.json",
                    mime="application/json",
                )


# ─────────────────────────────────────────────────────────────────────────────
# TAB 3: 分析実行
# ─────────────────────────────────────────────────────────────────────────────
with tabs[2]:
    st.header("分析実行")

    frames_ok = bool(st.session_state["frames"])
    rois_ok = bool(st.session_state["rois"])
    active_rois = [r for r in st.session_state["rois"] if r.enabled]
    if not frames_ok:
        st.info("タブ①でCSVを読み込んでください。")
    elif not rois_ok:
        st.info("タブ②でROIを設定してください。")
    elif not active_rois:
        st.warning("有効なROIがありません。タブ②で「分析に含める」にチェックを入れてください。")
    else:
        frames: list[FrameData] = st.session_state["frames"]
        rois: list[ROI] = active_rois
        cmap = st.session_state["cmap"]
        invert_y = st.session_state["invert_y"]
        time_axis = st.session_state["time_axis"]

        disabled_count = len(st.session_state["rois"]) - len(active_rois)
        if disabled_count > 0:
            st.caption(f"({disabled_count} 件のROIが無効化されています)")

        if st.button("▶ 分析を実行", type="primary"):
            with st.spinner("集計中…"):
                try:
                    validate_rois(rois, frames[0].x_coords, frames[0].y_coords)
                    rows = summarize_frames(frames, rois)
                    st.session_state["analysis_rows"] = rows
                    session_dir = get_or_create_session_dir()
                    st.session_state["session_dir"] = session_dir
                    csv_path = session_dir / "roi_timeseries.csv"
                    write_results_csv(rows, csv_path)
                    st.success(f"分析完了。出力先: `{session_dir}`")
                except Exception as exc:
                    st.error(f"分析エラー: {exc}")
                    st.stop()

        rows: list[dict] = st.session_state["analysis_rows"]
        if rows:
            # グラフ表示
            METRICS = ("mean", "std", "variance")
            fig, axes = plt.subplots(len(METRICS), 1, figsize=(12, 10), sharex=True)
            groups: dict = {}
            for row in rows:
                key = (row["roi_name"], row["metric"])
                groups.setdefault(key, []).append(row)
            for series in groups.values():
                series.sort(key=lambda r: int(r["frame_index"]))

            from datetime import datetime as dt
            use_ts = time_axis == "timestamp" and all(r["timestamp"] for r in rows)
            for metric, ax in zip(METRICS, axes):
                for roi in rois:
                    series = groups.get((roi.name, metric), [])
                    if not series:
                        continue
                    if use_ts:
                        xs = [dt.fromisoformat(r["timestamp"]) for r in series]
                    else:
                        xs = [int(r["frame_index"]) for r in series]
                    ys = [float(r["value"]) for r in series]
                    ax.plot(xs, ys, marker="o", label=roi.name)
                ax.set_ylabel(metric)
                ax.legend(loc="best")
                ax.grid(True, alpha=0.3)
            axes[-1].set_xlabel("timestamp" if use_ts else "frame index")
            fig.suptitle("ROI time-series statistics")
            fig.tight_layout()
            st.pyplot(fig)

            # ── 基準ROIとの差分グラフ ──
            st.divider()
            st.subheader("基準ROIとの差分")
            roi_names = [r.name for r in rois]
            base_roi_name = st.selectbox(
                "基準ROI（この ROI との差分を表示）",
                roi_names,
                index=0,
                key="diff_base_roi",
            )

            # 基準ROIの各メトリクスの時系列を辞書化 {metric: {frame_index: value}}
            base_lookup: dict[str, dict[int, float]] = {}
            for metric in METRICS:
                base_series = groups.get((base_roi_name, metric), [])
                base_lookup[metric] = {
                    int(r["frame_index"]): float(r["value"]) for r in base_series
                }

            other_rois = [r for r in rois if r.name != base_roi_name]
            if not other_rois:
                st.info("差分を計算するには2つ以上のROIが必要です。")
            else:
                fig_d, axes_d = plt.subplots(len(METRICS), 1, figsize=(12, 10), sharex=True)
                for metric, ax in zip(METRICS, axes_d):
                    base_by_frame = base_lookup[metric]
                    for roi in other_rois:
                        series = groups.get((roi.name, metric), [])
                        if not series:
                            continue
                        diff_xs = []
                        diff_ys = []
                        for r in series:
                            fi = int(r["frame_index"])
                            if fi in base_by_frame:
                                if use_ts and r.get("timestamp"):
                                    diff_xs.append(dt.fromisoformat(r["timestamp"]))
                                else:
                                    diff_xs.append(fi)
                                diff_ys.append(float(r["value"]) - base_by_frame[fi])
                        if diff_xs:
                            ax.plot(diff_xs, diff_ys, marker="o",
                                    label=f"{roi.name} − {base_roi_name}")
                    ax.set_ylabel(f"Δ {metric}")
                    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
                    ax.legend(loc="best")
                    ax.grid(True, alpha=0.3)
                axes_d[-1].set_xlabel("timestamp" if use_ts else "frame index")
                fig_d.suptitle(f"Difference from {base_roi_name}")
                fig_d.tight_layout()
                st.pyplot(fig_d)

                # 差分グラフも保存
                if st.session_state.get("session_dir"):
                    diff_plot_path = Path(st.session_state["session_dir"]) / "roi_diff.png"
                    fig_d.savefig(diff_plot_path, dpi=150)
                plt.close(fig_d)

            # ROI オーバーレイ表示
            session_dir = Path(st.session_state["session_dir"])
            scale = compute_color_scale(frames)
            rep_frame = frames[0]
            overlay_path = session_dir / f"roi_overlay_frame_{rep_frame.frame_index:03d}.png"
            crop_path = session_dir / f"roi_crops_frame_{rep_frame.frame_index:03d}.png"
            plot_path = session_dir / "roi_timeseries.png"

            with st.spinner("画像を生成中…"):
                render_single_heatmap(rep_frame, overlay_path, rois, cmap, *scale, invert_y=invert_y)
                render_roi_crops(rep_frame, rois, crop_path, cmap, *scale, invert_y=invert_y)
                fig.savefig(plot_path, dpi=150)
            plt.close(fig)

            col_ov, col_cr = st.columns(2)
            with col_ov:
                st.subheader("ROIオーバーレイ")
                st.image(str(overlay_path))
            with col_cr:
                st.subheader("ROI切り出し")
                st.image(str(crop_path))

            # ダウンロード
            st.divider()
            st.subheader("ダウンロード")
            dl1, dl2, dl3 = st.columns(3)
            csv_path = session_dir / "roi_timeseries.csv"
            if csv_path.exists():
                dl1.download_button(
                    "📄 CSV ダウンロード",
                    data=csv_path.read_bytes(),
                    file_name="roi_timeseries.csv",
                    mime="text/csv",
                )
            if plot_path.exists():
                dl2.download_button(
                    "📈 グラフ PNG ダウンロード",
                    data=plot_path.read_bytes(),
                    file_name="roi_timeseries.png",
                    mime="image/png",
                )
            all_paths = [csv_path, plot_path, overlay_path, crop_path]
            dl3.download_button(
                "🗜️ 全ファイル ZIP ダウンロード",
                data=build_zip(all_paths),
                file_name="analysis_results.zip",
                mime="application/zip",
            )

            st.caption(f"保存先: `{session_dir}`")

            # ── 分析結果の保存 ──
            st.divider()
            st.subheader("分析結果をプロジェクトに保存")
            save_col1, save_col2 = st.columns([3, 1])
            default_save_name = st.session_state.get("dataset_name", "") or "analysis"
            analysis_save_name = save_col1.text_input(
                "保存名", value=default_save_name, key="analysis_save_name",
            )
            if save_col2.button("保存", key="save_analysis_btn", type="primary"):
                if not analysis_save_name.strip():
                    st.error("保存名を入力してください")
                else:
                    save_path = save_analysis(
                        name=analysis_save_name.strip(),
                        rows=rows,
                        rois=rois,
                        input_dir=st.session_state.get("input_dir", ""),
                        dataset_name=st.session_state.get("dataset_name", ""),
                        session_dir=str(session_dir) if session_dir else None,
                    )
                    st.success(f"保存しました: `{save_path.name}`")


# ─────────────────────────────────────────────────────────────────────────────
# TAB 4: ヒートマップ可視化
# ─────────────────────────────────────────────────────────────────────────────
with tabs[3]:
    st.header("ヒートマップの時間変化")

    if not st.session_state["frames"]:
        st.info("タブ①でCSVを読み込んでください。")
    else:
        frames: list[FrameData] = st.session_state["frames"]
        rois: list[ROI] = [r for r in st.session_state["rois"] if r.enabled]
        cmap = st.session_state["cmap"]
        invert_y = st.session_state["invert_y"]

        col_s1, col_s2 = st.columns(2)
        with col_s1:
            frame_step = st.number_input(
                "フレームを間引く（1=全フレーム, 2=1フレームおきなど）",
                min_value=1, max_value=max(1, len(frames)),
                value=1, step=1,
            )
        with col_s2:
            skip_gif = st.checkbox("GIFを生成しない（高速）", value=False)

        if st.button("▶ ヒートマップを生成", type="primary"):
            selected = frames[::int(frame_step)]
            vmin, vmax = compute_color_scale(frames)
            session_dir = get_or_create_session_dir()
            st.session_state["session_dir"] = session_dir
            frame_dir = session_dir / "frames"
            frame_dir.mkdir(parents=True, exist_ok=True)

            progress = st.progress(0.0, text="フレームを生成中…")
            rendered_paths: list[Path] = []
            for i, frame in enumerate(selected):
                fp = frame_dir / f"heatmap_{frame.frame_index:03d}.png"
                render_single_heatmap(frame, fp, rois, cmap, vmin, vmax, invert_y=invert_y)
                rendered_paths.append(fp)
                progress.progress((i + 1) / len(selected), text=f"フレーム {frame.frame_index} 生成中…")

            summary_path = session_dir / "heatmap_summary.png"
            render_heatmap_summary(selected, summary_path, rois, cmap, vmin, vmax, invert_y=invert_y)
            progress.progress(1.0, text="完了")

            if not skip_gif and rendered_paths:
                with st.spinner("GIF生成中…"):
                    gif_path = session_dir / "heatmap_animation.gif"
                    save_gif(rendered_paths, gif_path)

            st.success(f"生成完了: `{session_dir}`")
            st.image(str(summary_path), caption="ヒートマップサマリー（先頭9フレーム）")

            # GIF 表示とダウンロード
            gif_path = session_dir / "heatmap_animation.gif"
            if gif_path.exists():
                st.image(str(gif_path), caption="ヒートマップアニメーション")
                st.download_button(
                    "🎞️ GIF ダウンロード",
                    data=gif_path.read_bytes(),
                    file_name="heatmap_animation.gif",
                    mime="image/gif",
                )

            st.divider()
            st.subheader("個別フレームブラウザ")
            if rendered_paths:
                frame_labels = [p.name for p in rendered_paths]
                chosen = st.select_slider("フレームを選択", options=frame_labels)
                chosen_path = frame_dir / chosen
                st.image(str(chosen_path))
                st.download_button(
                    "このフレームをダウンロード",
                    data=chosen_path.read_bytes(),
                    file_name=chosen,
                    mime="image/png",
                )

            st.caption(f"保存先: `{session_dir}`")


# ─────────────────────────────────────────────────────────────────────────────
# TAB 5: 保存済み分析
# ─────────────────────────────────────────────────────────────────────────────
with tabs[4]:
    st.header("保存済み分析")

    saved_list = list_saved_analyses()
    if not saved_list:
        st.info("保存済みの分析結果がありません。タブ③で分析を実行し、保存してください。")
    else:
        st.caption(f"{len(saved_list)} 件の保存済み分析")

        options = [
            f"{e['name']}  ({e['saved_at'][:19]}  ROI:{e['roi_count']}  rows:{e['row_count']})"
            for e in saved_list
        ]
        selected_idx = st.selectbox(
            "読み込む分析を選択",
            range(len(options)),
            format_func=lambda i: options[i],
            key="saved_analysis_selector",
        )
        entry = saved_list[selected_idx]

        btn_col1, btn_col2 = st.columns([1, 1])
        load_clicked = btn_col1.button("読み込む", type="primary", key="load_analysis_btn")
        delete_clicked = btn_col2.button("削除", key="delete_analysis_btn")

        if delete_clicked:
            entry["path"].unlink(missing_ok=True)
            st.success(f"削除しました: {entry['name']}")
            st.rerun()

        if load_clicked:
            data = load_analysis(entry["path"])
            st.session_state["loaded_analysis"] = data
            st.rerun()

        # ── 読み込み済み分析の表示 ──
        loaded = st.session_state.get("loaded_analysis")
        if loaded:
            st.divider()
            st.subheader(f"分析: {loaded['name']}")

            meta_col1, meta_col2, meta_col3 = st.columns(3)
            meta_col1.metric("保存日時", loaded.get("saved_at", "")[:19])
            meta_col2.metric("入力ディレクトリ", loaded.get("input_dir", "-") or "-")
            meta_col3.metric("データセット名", loaded.get("dataset_name", "-") or "-")

            loaded_rois = [
                ROI(
                    name=str(r["name"]),
                    x_min=int(r["x_min"]),
                    x_max=int(r["x_max"]),
                    y_min=int(r["y_min"]),
                    y_max=int(r["y_max"]),
                    enabled=bool(r.get("enabled", True)),
                )
                for r in loaded.get("rois", [])
            ]
            loaded_rows: list[dict] = loaded.get("rows", [])

            with st.expander(f"ROI定義（{len(loaded_rois)} 件）"):
                for i, roi in enumerate(loaded_rois):
                    st.text(f"  #{i+1}: {roi.name}  x={roi.x_min}-{roi.x_max}, y={roi.y_min}-{roi.y_max}")

            if not loaded_rows:
                st.warning("分析データが空です。")
            else:
                METRICS_SAVED = ("mean", "std", "variance")
                fig_s, axes_s = plt.subplots(len(METRICS_SAVED), 1, figsize=(12, 10), sharex=True)
                groups_s: dict = {}
                for row in loaded_rows:
                    key = (row["roi_name"], row["metric"])
                    groups_s.setdefault(key, []).append(row)
                for series in groups_s.values():
                    series.sort(key=lambda r: int(r["frame_index"]))

                use_ts_s = all(r.get("timestamp") for r in loaded_rows)
                for metric, ax in zip(METRICS_SAVED, axes_s):
                    for roi in loaded_rois:
                        series = groups_s.get((roi.name, metric), [])
                        if not series:
                            continue
                        if use_ts_s:
                            from datetime import datetime as dt_s
                            xs = [dt_s.fromisoformat(r["timestamp"]) for r in series]
                        else:
                            xs = [int(r["frame_index"]) for r in series]
                        ys = [float(r["value"]) for r in series]
                        ax.plot(xs, ys, marker="o", label=roi.name)
                    ax.set_ylabel(metric)
                    ax.legend(loc="best")
                    ax.grid(True, alpha=0.3)
                axes_s[-1].set_xlabel("timestamp" if use_ts_s else "frame index")
                fig_s.suptitle(f"ROI time-series statistics — {loaded['name']}")
                fig_s.tight_layout()
                st.pyplot(fig_s)
                plt.close(fig_s)

                saved_session_dir = loaded.get("session_dir")
                if saved_session_dir:
                    sd = Path(saved_session_dir)
                    image_files = sorted(sd.glob("*.png")) if sd.exists() else []
                    if image_files:
                        st.divider()
                        st.subheader("保存済み画像")
                        for img_path in image_files:
                            st.image(str(img_path), caption=img_path.name)

                st.divider()
                csv_lines = []
                if loaded_rows:
                    headers = list(loaded_rows[0].keys())
                    csv_lines.append(",".join(headers))
                    for row in loaded_rows:
                        csv_lines.append(",".join(str(row.get(h, "")) for h in headers))
                csv_text = "\n".join(csv_lines)
                st.download_button(
                    "CSV ダウンロード",
                    data=csv_text.encode("utf-8"),
                    file_name=f"{loaded['name']}_timeseries.csv",
                    mime="text/csv",
                    key="saved_csv_dl",
                )

                if st.button("このROI定義を現在のセッションに読み込む", key="apply_saved_rois"):
                    st.session_state["rois"] = loaded_rois
                    save_rois_to_file(loaded_rois)
                    st.success(f"{len(loaded_rois)} 件のROIをセッションに反映しました")
