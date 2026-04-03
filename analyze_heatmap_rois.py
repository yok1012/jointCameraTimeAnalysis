from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from matplotlib.widgets import RectangleSelector
from PIL import Image


SUPPORTED_ENCODINGS = ("cp932", "shift_jis", "utf-8-sig", "utf-8")
METRICS = ("mean", "std", "variance")
MAX_ROIS = 10


@dataclass(frozen=True)
class ROI:
    name: str
    x_min: int
    x_max: int
    y_min: int
    y_max: int

    def to_dict(self) -> dict[str, int | str]:
        return {
            "name": self.name,
            "x_min": self.x_min,
            "x_max": self.x_max,
            "y_min": self.y_min,
            "y_max": self.y_max,
        }


@dataclass(frozen=True)
class FrameData:
    frame_index: int
    file_name: str
    captured_at: datetime | None
    x_coords: np.ndarray
    y_coords: np.ndarray
    heatmap: np.ndarray


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Analyze thermal heatmap CSV frames with rectangular ROIs."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    analyze_parser = subparsers.add_parser(
        "analyze",
        help="Compute ROI mean/std/variance over time and save plots.",
    )
    analyze_parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("input"),
        help="Directory containing CSV frames.",
    )
    analyze_parser.add_argument(
        "--roi-file",
        type=Path,
        required=True,
        help="JSON file containing up to 10 ROIs.",
    )
    analyze_parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output"),
        help="Directory for plots and CSV output.",
    )
    analyze_parser.add_argument(
        "--time-axis",
        choices=("index", "timestamp"),
        default="index",
        help="X axis for plots.",
    )
    analyze_parser.add_argument(
        "--show",
        action="store_true",
        help="Display the plot window in addition to saving files.",
    )
    analyze_parser.add_argument(
        "--cmap",
        default="inferno",
        help="Matplotlib colormap name used for heatmap outputs.",
    )
    analyze_parser.add_argument(
        "--invert-y",
        action="store_true",
        help="Invert the y axis for heatmap-based outputs.",
    )

    select_parser = subparsers.add_parser(
        "select-rois",
        help="Select ROIs interactively on a representative frame.",
    )
    select_parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("input"),
        help="Directory containing CSV frames.",
    )
    select_parser.add_argument(
        "--frame-index",
        type=int,
        default=1,
        help="Frame index taken from the filename suffix, for example 1 for 10_1.CSV.",
    )
    select_parser.add_argument(
        "--output",
        type=Path,
        default=Path("rois.json"),
        help="Where to save the selected ROIs as JSON.",
    )
    select_parser.add_argument(
        "--cmap",
        default="inferno",
        help="Matplotlib colormap name used for the heatmap preview.",
    )
    select_parser.add_argument(
        "--existing-roi-file",
        type=Path,
        help="Optional JSON file with existing ROIs to display and keep editing.",
    )
    select_parser.add_argument(
        "--preview-output",
        type=Path,
        help="Optional PNG path for saving the selected ROI overlay preview.",
    )
    select_parser.add_argument(
        "--invert-y",
        action="store_true",
        help="Invert the y axis in the interactive heatmap view and saved previews.",
    )

    render_parser = subparsers.add_parser(
        "render-heatmaps",
        help="Render the heatmap time change as frame PNGs and an animated GIF.",
    )
    render_parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("input"),
        help="Directory containing CSV frames.",
    )
    render_parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("heatmap_output"),
        help="Directory for rendered heatmap images.",
    )
    render_parser.add_argument(
        "--roi-file",
        type=Path,
        help="Optional ROI JSON file to overlay on the heatmaps.",
    )
    render_parser.add_argument(
        "--cmap",
        default="inferno",
        help="Matplotlib colormap name used for rendering.",
    )
    render_parser.add_argument(
        "--frame-step",
        type=int,
        default=1,
        help="Render every Nth frame. 1 renders all frames.",
    )
    render_parser.add_argument(
        "--gif-name",
        default="heatmap_animation.gif",
        help="Filename for the animated GIF stored inside output-dir.",
    )
    render_parser.add_argument(
        "--skip-gif",
        action="store_true",
        help="Skip animated GIF generation and only export PNGs.",
    )
    render_parser.add_argument(
        "--invert-y",
        action="store_true",
        help="Invert the y axis for rendered heatmaps and GIF frames.",
    )

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "analyze":
        run_analysis(
            args.input_dir,
            args.roi_file,
            args.output_dir,
            args.time_axis,
            args.show,
            args.cmap,
            args.invert_y,
        )
        return 0

    if args.command == "select-rois":
        run_roi_selector(
            args.input_dir,
            args.frame_index,
            args.output,
            args.cmap,
            args.existing_roi_file,
            args.preview_output,
            args.invert_y,
        )
        return 0

    if args.command == "render-heatmaps":
        run_render_heatmaps(
            args.input_dir,
            args.output_dir,
            args.roi_file,
            args.cmap,
            args.frame_step,
            args.gif_name,
            args.skip_gif,
            args.invert_y,
        )
        return 0

    parser.error("Unknown command")
    return 2


def run_analysis(
    input_dir: Path,
    roi_file: Path,
    output_dir: Path,
    time_axis: str,
    show_plot: bool,
    cmap: str,
    invert_y: bool,
) -> None:
    frames = load_all_frames(input_dir)
    if not frames:
        raise SystemExit(f"No CSV files found in {input_dir}")

    rois = load_rois(roi_file)
    validate_rois(rois, frames[0].x_coords, frames[0].y_coords)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = summarize_frames(frames, rois)
    csv_path = output_dir / "roi_timeseries.csv"
    write_results_csv(rows, csv_path)
    plot_path = output_dir / "roi_timeseries.png"
    plot_results(rows, rois, plot_path, time_axis, show_plot)
    representative = frames[0]
    overlay_path = output_dir / f"roi_overlay_frame_{representative.frame_index:03d}.png"
    crop_path = output_dir / f"roi_crops_frame_{representative.frame_index:03d}.png"
    scale = compute_color_scale(frames)
    render_single_heatmap(representative, overlay_path, rois, cmap, *scale, invert_y=invert_y)
    render_roi_crops(representative, rois, crop_path, cmap, *scale, invert_y=invert_y)

    print(f"Saved analysis CSV: {csv_path}")
    print(f"Saved analysis plot: {plot_path}")
    print(f"Saved ROI overlay image: {overlay_path}")
    print(f"Saved ROI crop image: {crop_path}")


def run_roi_selector(
    input_dir: Path,
    frame_index: int,
    output_path: Path,
    cmap: str,
    existing_roi_file: Path | None,
    preview_output: Path | None,
    invert_y: bool,
) -> None:
    frames = load_all_frames(input_dir)
    if not frames:
        raise SystemExit(f"No CSV files found in {input_dir}")

    frame = next((item for item in frames if item.frame_index == frame_index), None)
    if frame is None:
        raise SystemExit(f"Frame index {frame_index} was not found in {input_dir}")

    existing_rois = load_rois(existing_roi_file) if existing_roi_file else []
    if existing_rois:
        validate_rois(existing_rois, frame.x_coords, frame.y_coords)

    selector = InteractiveROISelector(frame=frame, cmap=cmap, initial_rois=existing_rois, invert_y=invert_y)
    rois = selector.collect()
    if not rois:
        print("No ROIs were selected. Nothing was saved.")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps([roi.to_dict() for roi in rois], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    preview_path = preview_output or output_path.with_suffix(".png")
    crop_path = preview_path.with_name(f"{preview_path.stem}_crops.png")
    scale = compute_color_scale([frame])
    render_single_heatmap(frame, preview_path, rois, cmap, *scale, invert_y=invert_y)
    render_roi_crops(frame, rois, crop_path, cmap, *scale, invert_y=invert_y)
    print(f"Saved {len(rois)} ROI definitions to {output_path}")
    print(f"Saved ROI preview image: {preview_path}")
    print(f"Saved ROI crop image: {crop_path}")


def run_render_heatmaps(
    input_dir: Path,
    output_dir: Path,
    roi_file: Path | None,
    cmap: str,
    frame_step: int,
    gif_name: str,
    skip_gif: bool,
    invert_y: bool,
) -> None:
    if frame_step < 1:
        raise ValueError("--frame-step must be 1 or greater")

    frames = load_all_frames(input_dir)
    if not frames:
        raise SystemExit(f"No CSV files found in {input_dir}")

    rois = load_rois(roi_file) if roi_file else []
    if rois:
        validate_rois(rois, frames[0].x_coords, frames[0].y_coords)

    output_dir.mkdir(parents=True, exist_ok=True)
    frame_dir = output_dir / "frames"
    frame_dir.mkdir(parents=True, exist_ok=True)
    vmin, vmax = compute_color_scale(frames)
    selected_frames = frames[::frame_step]
    rendered_paths: list[Path] = []
    for frame in selected_frames:
        file_path = frame_dir / f"heatmap_{frame.frame_index:03d}.png"
        render_single_heatmap(frame, file_path, rois, cmap, vmin, vmax, invert_y=invert_y)
        rendered_paths.append(file_path)

    summary_path = output_dir / "heatmap_summary.png"
    render_heatmap_summary(selected_frames, summary_path, rois, cmap, vmin, vmax, invert_y=invert_y)
    print(f"Saved {len(rendered_paths)} heatmap frames to {frame_dir}")
    print(f"Saved heatmap summary image: {summary_path}")

    if not skip_gif and rendered_paths:
        gif_path = output_dir / gif_name
        save_gif(rendered_paths, gif_path)
        print(f"Saved heatmap animation: {gif_path}")


def load_all_frames(input_dir: Path) -> list[FrameData]:
    candidates = sorted(
        (path for path in input_dir.iterdir() if path.is_file() and path.suffix.lower() == ".csv"),
        key=lambda path: extract_frame_index(path.name),
    )
    return [load_frame(path) for path in candidates]


def load_frame(path: Path) -> FrameData:
    lines = read_csv_lines(path)
    if len(lines) < 9:
        raise ValueError(f"Unexpected CSV format in {path}: fewer than 9 rows")

    header_row = [cell.strip() for cell in lines[7]]
    x_coords = np.array([int(cell) for cell in header_row[1:] if cell], dtype=int)

    y_coords: list[int] = []
    heatmap_rows: list[list[float]] = []
    for row in lines[8:]:
        if not row:
            continue
        trimmed = [cell.strip() for cell in row]
        if not trimmed[0]:
            continue
        y_coords.append(int(trimmed[0]))
        values = [float(cell) for cell in trimmed[1 : 1 + len(x_coords)]]
        if len(values) != len(x_coords):
            raise ValueError(f"Row width does not match x header in {path}")
        heatmap_rows.append(values)

    captured_at = parse_capture_datetime(lines)
    return FrameData(
        frame_index=extract_frame_index(path.name),
        file_name=path.name,
        captured_at=captured_at,
        x_coords=x_coords,
        y_coords=np.array(y_coords, dtype=int),
        heatmap=np.array(heatmap_rows, dtype=float),
    )


def read_csv_lines(path: Path) -> list[list[str]]:
    last_error: Exception | None = None
    for encoding in SUPPORTED_ENCODINGS:
        try:
            with path.open("r", encoding=encoding, newline="") as handle:
                return list(csv.reader(handle))
        except UnicodeDecodeError as exc:
            last_error = exc
    raise UnicodeDecodeError(
        "csv",
        b"",
        0,
        1,
        f"Could not decode {path} using supported encodings: {SUPPORTED_ENCODINGS}. Last error: {last_error}",
    )


def parse_capture_datetime(lines: list[list[str]]) -> datetime | None:
    date_match = re.search(r"\d{4}/\d{2}/\d{2}", ",".join(lines[1]))
    time_match = re.search(r"\d{2}:\d{2}:\d{2}", ",".join(lines[2]))
    if not date_match or not time_match:
        return None
    return datetime.strptime(
        f"{date_match.group(0)} {time_match.group(0)}",
        "%Y/%m/%d %H:%M:%S",
    )


def extract_frame_index(file_name: str) -> int:
    match = re.search(r"_(\d+)\.csv$", file_name, re.IGNORECASE)
    if not match:
        raise ValueError(f"Could not extract frame index from {file_name}")
    return int(match.group(1))


def load_rois(path: Path) -> list[ROI]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("ROI file must contain a JSON array")
    rois = [
        ROI(
            name=str(item["name"]),
            x_min=int(item["x_min"]),
            x_max=int(item["x_max"]),
            y_min=int(item["y_min"]),
            y_max=int(item["y_max"]),
        )
        for item in payload
    ]
    if not rois:
        raise ValueError("At least one ROI is required")
    if len(rois) > MAX_ROIS:
        raise ValueError(f"At most {MAX_ROIS} ROIs are supported")
    return rois


def validate_rois(rois: list[ROI], x_coords: np.ndarray, y_coords: np.ndarray) -> None:
    names = set()
    min_x, max_x = int(x_coords.min()), int(x_coords.max())
    min_y, max_y = int(y_coords.min()), int(y_coords.max())
    for roi in rois:
        if roi.name in names:
            raise ValueError(f"Duplicate ROI name: {roi.name}")
        names.add(roi.name)
        if roi.x_min > roi.x_max or roi.y_min > roi.y_max:
            raise ValueError(f"ROI bounds are reversed: {roi.name}")
        if roi.x_min < min_x or roi.x_max > max_x or roi.y_min < min_y or roi.y_max > max_y:
            raise ValueError(
                f"ROI {roi.name} is out of bounds. Valid x: {min_x}-{max_x}, valid y: {min_y}-{max_y}"
            )


def summarize_frames(frames: Iterable[FrameData], rois: list[ROI]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for frame in frames:
        for roi in rois:
            region = extract_roi_region(frame, roi)
            rows.extend(build_metric_rows(frame, roi, region))
    return rows


def extract_roi_region(frame: FrameData, roi: ROI) -> np.ndarray:
    x_mask = (frame.x_coords >= roi.x_min) & (frame.x_coords <= roi.x_max)
    y_mask = (frame.y_coords >= roi.y_min) & (frame.y_coords <= roi.y_max)
    if not x_mask.any() or not y_mask.any():
        raise ValueError(f"ROI {roi.name} does not overlap frame coordinates")
    return frame.heatmap[np.ix_(y_mask, x_mask)]


def build_metric_rows(frame: FrameData, roi: ROI, region: np.ndarray) -> list[dict[str, object]]:
    pixel_count = int(region.size)
    metric_values = {
        "mean": float(np.nanmean(region)),
        "std": float(np.nanstd(region)),
        "variance": float(np.nanvar(region)),
    }
    timestamp = frame.captured_at.isoformat() if frame.captured_at else ""
    return [
        {
            "file_name": frame.file_name,
            "frame_index": frame.frame_index,
            "timestamp": timestamp,
            "roi_name": roi.name,
            "metric": metric,
            "value": value,
            "pixel_count": pixel_count,
        }
        for metric, value in metric_values.items()
    ]


def write_results_csv(rows: list[dict[str, object]], path: Path) -> None:
    fieldnames = ["file_name", "frame_index", "timestamp", "roi_name", "metric", "value", "pixel_count"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def compute_color_scale(frames: Iterable[FrameData]) -> tuple[float, float]:
    values = [frame.heatmap for frame in frames]
    stacked = np.concatenate([item.reshape(-1) for item in values])
    return float(np.nanmin(stacked)), float(np.nanmax(stacked))


def render_single_heatmap(
    frame: FrameData,
    output_path: Path,
    rois: list[ROI],
    cmap: str,
    vmin: float,
    vmax: float,
    invert_y: bool = False,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    draw_heatmap(ax, frame, cmap, vmin, vmax, invert_y=invert_y)
    if rois:
        draw_roi_overlays(ax, rois)
    ax.set_title(build_frame_title(frame))
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def render_heatmap_summary(
    frames: list[FrameData],
    output_path: Path,
    rois: list[ROI],
    cmap: str,
    vmin: float,
    vmax: float,
    invert_y: bool = False,
) -> None:
    display_frames = frames[: min(len(frames), 9)]
    columns = min(3, len(display_frames))
    rows = int(np.ceil(len(display_frames) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(4 * columns, 3.5 * rows), squeeze=False)
    flat_axes = axes.flatten()

    for axis, frame in zip(flat_axes, display_frames, strict=False):
        draw_heatmap(axis, frame, cmap, vmin, vmax, invert_y=invert_y)
        if rois:
            draw_roi_overlays(axis, rois)
        axis.set_title(build_frame_title(frame))

    for axis in flat_axes[len(display_frames):]:
        axis.axis("off")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def draw_heatmap(ax, frame: FrameData, cmap: str, vmin: float, vmax: float, invert_y: bool = False) -> None:
    extent = [
        int(frame.x_coords.min()) - 0.5,
        int(frame.x_coords.max()) + 0.5,
        int(frame.y_coords.min()) - 0.5,
        int(frame.y_coords.max()) + 0.5,
    ]
    image = ax.imshow(
        frame.heatmap,
        origin="lower",
        extent=extent,
        aspect="equal",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )
    plt.colorbar(image, ax=ax, label="temperature")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    if invert_y:
        ax.invert_yaxis()


def draw_roi_overlays(ax, rois: list[ROI]) -> None:
    colors = plt.cm.tab10.colors
    for index, roi in enumerate(rois):
        color = colors[index % len(colors)]
        patch = Rectangle(
            (roi.x_min - 0.5, roi.y_min - 0.5),
            roi.x_max - roi.x_min + 1,
            roi.y_max - roi.y_min + 1,
            fill=False,
            linewidth=2,
            edgecolor=color,
        )
        ax.add_patch(patch)
        ax.text(
            roi.x_min,
            roi.y_max + 0.8,
            roi.name,
            color=color,
            fontsize=9,
            weight="bold",
            bbox={"facecolor": "black", "alpha": 0.35, "pad": 2, "edgecolor": "none"},
        )


def render_roi_crops(
    frame: FrameData,
    rois: list[ROI],
    output_path: Path,
    cmap: str,
    vmin: float,
    vmax: float,
    invert_y: bool = False,
) -> None:
    columns = min(2, len(rois))
    rows = int(np.ceil(len(rois) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(5 * columns, 4 * rows), squeeze=False)
    flat_axes = axes.flatten()
    for axis, roi in zip(flat_axes, rois, strict=False):
        region = extract_roi_region(frame, roi)
        image = axis.imshow(region, origin="lower", aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
        axis.set_title(f"{roi.name}: x={roi.x_min}-{roi.x_max}, y={roi.y_min}-{roi.y_max}")
        axis.set_xlabel("local x")
        axis.set_ylabel("local y")
        if invert_y:
            axis.invert_yaxis()
        plt.colorbar(image, ax=axis, label="temperature")

    for axis in flat_axes[len(rois):]:
        axis.axis("off")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def save_gif(frame_paths: list[Path], output_path: Path) -> None:
    images = [Image.open(path) for path in frame_paths]
    if not images:
        return
    try:
        images[0].save(
            output_path,
            save_all=True,
            append_images=images[1:],
            duration=250,
            loop=0,
        )
    finally:
        for image in images:
            image.close()


def build_frame_title(frame: FrameData) -> str:
    timestamp = frame.captured_at.strftime("%Y-%m-%d %H:%M:%S") if frame.captured_at else "timestamp unavailable"
    return f"Frame {frame.frame_index}: {frame.file_name} / {timestamp}"


def plot_results(
    rows: list[dict[str, object]],
    rois: list[ROI],
    output_path: Path,
    time_axis: str,
    show_plot: bool,
) -> None:
    fig, axes = plt.subplots(len(METRICS), 1, figsize=(12, 10), sharex=True)
    axis_by_metric = dict(zip(METRICS, axes, strict=True))
    row_groups = group_rows(rows)

    use_timestamp = time_axis == "timestamp"
    if use_timestamp and not can_use_timestamps(rows):
        print("Timestamps are incomplete, falling back to frame index for plotting.", file=sys.stderr)
        use_timestamp = False

    for metric in METRICS:
        axis = axis_by_metric[metric]
        for roi in rois:
            series = row_groups[(roi.name, metric)]
            x_values, y_values = build_plot_series(series, use_timestamp)
            axis.plot(x_values, y_values, marker="o", label=roi.name)
        axis.set_ylabel(metric)
        axis.grid(True, alpha=0.3)
        axis.legend(loc="best")

    axes[-1].set_xlabel("timestamp" if use_timestamp else "frame index")
    if use_timestamp:
        axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
        fig.autofmt_xdate()

    fig.suptitle("ROI time-series statistics")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    if show_plot:
        plt.show()
    else:
        plt.close(fig)


def group_rows(rows: list[dict[str, object]]) -> dict[tuple[str, str], list[dict[str, object]]]:
    grouped: dict[tuple[str, str], list[dict[str, object]]] = {}
    for row in rows:
        key = (str(row["roi_name"]), str(row["metric"]))
        grouped.setdefault(key, []).append(row)
    for series in grouped.values():
        series.sort(key=lambda item: int(item["frame_index"]))
    return grouped


def can_use_timestamps(rows: list[dict[str, object]]) -> bool:
    return all(bool(row["timestamp"]) for row in rows)


def build_plot_series(series: list[dict[str, object]], use_timestamp: bool) -> tuple[list[object], list[float]]:
    if use_timestamp:
        x_values = [datetime.fromisoformat(str(item["timestamp"])) for item in series]
    else:
        x_values = [int(item["frame_index"]) for item in series]
    y_values = [float(item["value"]) for item in series]
    return x_values, y_values


class InteractiveROISelector:
    def __init__(
        self,
        frame: FrameData,
        cmap: str,
        initial_rois: list[ROI] | None = None,
        invert_y: bool = False,
    ) -> None:
        self.frame = frame
        self.cmap = cmap
        self.rois: list[ROI] = list(initial_rois or [])
        self.current_bounds: tuple[int, int, int, int] | None = None
        self.patches: list[Rectangle] = []
        self.labels = []
        self.status_text = None
        self.invert_y = invert_y

    def collect(self) -> list[ROI]:
        fig, ax = plt.subplots(figsize=(10, 8))
        vmin, vmax = compute_color_scale([self.frame])
        draw_heatmap(ax, self.frame, self.cmap, vmin, vmax, invert_y=self.invert_y)
        ax.set_title(
            "Drag a rectangle, press 'a' to add, 'u' to undo, 's' to save, 'q' to quit"
        )
        self.status_text = ax.text(
            0.01,
            1.02,
            self.build_status_message(),
            transform=ax.transAxes,
            fontsize=9,
            color="white",
            bbox={"facecolor": "black", "alpha": 0.5, "pad": 3, "edgecolor": "none"},
        )
        self.sync_existing_rois(ax)

        selector = RectangleSelector(
            ax,
            self.on_select,
            useblit=True,
            button=[1],
            minspanx=1,
            minspany=1,
            spancoords="data",
            interactive=True,
        )

        def on_key(event) -> None:
            if event.key == "a":
                self.add_current_roi(ax)
                fig.canvas.draw_idle()
            elif event.key == "u":
                self.undo_last(ax)
                fig.canvas.draw_idle()
            elif event.key == "s":
                plt.close(fig)
            elif event.key == "q":
                self.rois.clear()
                plt.close(fig)

        fig.canvas.mpl_connect("key_press_event", on_key)
        plt.show()
        selector.set_active(False)
        return self.rois

    def sync_existing_rois(self, ax) -> None:
        if self.rois:
            for roi in self.rois:
                self.draw_single_roi(ax, roi)
            self.update_status()

    def on_select(self, click_event, release_event) -> None:
        if None in (click_event.xdata, click_event.ydata, release_event.xdata, release_event.ydata):
            print("Selection ignored because it was outside the heatmap axes.")
            return
        x1, x2 = sorted((round(click_event.xdata), round(release_event.xdata)))
        y1, y2 = sorted((round(click_event.ydata), round(release_event.ydata)))
        self.current_bounds = (int(x1), int(x2), int(y1), int(y2))
        print(f"Current selection: x={x1}-{x2}, y={y1}-{y2}")
        self.update_status()

    def add_current_roi(self, ax) -> None:
        if self.current_bounds is None:
            print("Select a rectangle before pressing 'a'.")
            return
        if len(self.rois) >= MAX_ROIS:
            print(f"You already have {MAX_ROIS} ROIs. Save or undo before adding more.")
            return

        x_min, x_max, y_min, y_max = self.current_bounds
        default_name = f"roi_{len(self.rois) + 1}"
        name = input(f"ROI name [{default_name}]: ").strip() or default_name
        if any(roi.name == name for roi in self.rois):
            print(f"ROI name '{name}' already exists.")
            return

        roi = ROI(name=name, x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max)
        validate_rois([*self.rois, roi], self.frame.x_coords, self.frame.y_coords)
        self.rois.append(roi)
        self.draw_single_roi(ax, roi)
        self.update_status()
        print(f"Added ROI '{name}'")

    def undo_last(self, ax) -> None:
        if not self.rois:
            print("There is no ROI to undo.")
            return
        removed = self.rois.pop()
        self.patches.pop().remove()
        self.labels.pop().remove()
        self.update_status()
        print(f"Removed ROI '{removed.name}'")

    def draw_single_roi(self, ax, roi: ROI) -> None:
        color = plt.cm.tab10.colors[len(self.patches) % len(plt.cm.tab10.colors)]
        patch = Rectangle(
            (roi.x_min - 0.5, roi.y_min - 0.5),
            roi.x_max - roi.x_min + 1,
            roi.y_max - roi.y_min + 1,
            fill=False,
            linewidth=2,
            edgecolor=color,
        )
        ax.add_patch(patch)
        label = ax.text(
            roi.x_min,
            roi.y_max + 0.5,
            roi.name,
            color=color,
            fontsize=9,
            weight="bold",
            bbox={"facecolor": "black", "alpha": 0.35, "pad": 2, "edgecolor": "none"},
        )
        self.patches.append(patch)
        self.labels.append(label)

    def build_status_message(self) -> str:
        if self.current_bounds is None:
            selection = "selection: none"
        else:
            x_min, x_max, y_min, y_max = self.current_bounds
            selection = f"selection: x={x_min}-{x_max}, y={y_min}-{y_max}"
        return f"ROIs: {len(self.rois)}/{MAX_ROIS} | {selection}"

    def update_status(self) -> None:
        if self.status_text is not None:
            self.status_text.set_text(self.build_status_message())


if __name__ == "__main__":
    raise SystemExit(main())