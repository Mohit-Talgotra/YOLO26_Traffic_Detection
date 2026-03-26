from __future__ import annotations

from collections import Counter
from statistics import mean, pstdev

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

from config import AppConfig
from counter import CountSummary, DetectionRecord


CHART_COLORS = {
    "car": "#2E8B57",
    "motorcycle": "#1E90FF",
    "bus": "#FF8C00",
    "truck": "#DC143C",
    "left": "#4C78A8",
    "right": "#F58518",
    "low": "#54A24B",
    "medium": "#EECA3B",
    "high": "#E45756",
}


def _round(value: float) -> float:
    return round(value, 2)


def _class_confidences(detections: list[DetectionRecord], class_name: str) -> list[float]:
    return [item.confidence for item in detections if item.class_name == class_name]


def build_image_analytics(
    summary: CountSummary,
    detections: list[DetectionRecord],
    frame_shape: tuple[int, int, int],
    config: AppConfig,
) -> dict[str, object]:
    frame_height, frame_width = frame_shape[:2]
    frame_area = max(frame_height * frame_width, 1)
    overall_confidences = [item.confidence for item in detections]
    bbox_area_ratios = [
        ((item.bbox[2] - item.bbox[0]) * (item.bbox[3] - item.bbox[1])) / frame_area
        for item in detections
    ]

    class_breakdown: dict[str, dict[str, float | int | None]] = {}
    for class_name in config.target_classes:
        class_confidences = _class_confidences(detections, class_name)
        count = summary.counts[class_name]
        class_breakdown[class_name] = {
            "count": count,
            "share_percent": _round((count / summary.total) * 100) if summary.total else 0.0,
            "avg_confidence": _round(mean(class_confidences)) if class_confidences else None,
            "max_confidence": _round(max(class_confidences)) if class_confidences else None,
        }

    region_totals = {
        region_name: sum(region_counts.values())
        for region_name, region_counts in summary.region_counts.items()
    }
    dominant_region = (
        "Balanced"
        if region_totals[config.left_zone_name] == region_totals[config.right_zone_name]
        else max(region_totals, key=region_totals.get)
    )

    return {
        "vehicle_distribution": class_breakdown,
        "region_totals": region_totals,
        "region_share_percent": {
            region_name: _round((region_total / summary.total) * 100) if summary.total else 0.0
            for region_name, region_total in region_totals.items()
        },
        "dominant_region": dominant_region,
        "average_confidence": _round(mean(overall_confidences)) if overall_confidences else None,
        "max_confidence": _round(max(overall_confidences)) if overall_confidences else None,
        "cumulative_bbox_coverage_percent": _round(sum(bbox_area_ratios) * 100) if bbox_area_ratios else 0.0,
        "frame_size": {"width": frame_width, "height": frame_height},
    }


def build_video_analytics(
    frame_reports: list[dict[str, object]],
    fps: float | int,
    config: AppConfig,
) -> dict[str, object]:
    frame_count = len(frame_reports)
    if frame_count == 0:
        return {
            "frames_processed": 0,
            "duration_seconds": 0.0,
            "average_vehicles_per_frame": 0.0,
            "max_vehicles_in_frame": 0,
            "min_vehicles_in_frame": 0,
            "vehicle_count_std_dev": 0.0,
            "density_breakdown": {},
            "class_totals": {class_name: 0 for class_name in config.target_classes},
            "class_average_per_frame": {class_name: 0.0 for class_name in config.target_classes},
            "class_share_percent": {class_name: 0.0 for class_name in config.target_classes},
            "region_totals": {
                config.left_zone_name: 0,
                config.right_zone_name: 0,
            },
            "dominant_region": "Balanced",
            "peak_frames": [],
            "average_detection_confidence": None,
            "note": "Video analytics are based on detections per frame, not unique tracked vehicles.",
        }

    safe_fps = float(fps) if fps and float(fps) > 0 else 25.0
    totals_per_frame = [int(frame["total"]) for frame in frame_reports]
    class_totals = {class_name: 0 for class_name in config.target_classes}
    density_counter: Counter[str] = Counter()
    region_totals = {config.left_zone_name: 0, config.right_zone_name: 0}
    all_confidences: list[float] = []

    for frame in frame_reports:
        density_counter[str(frame["density"])] += 1
        for class_name in config.target_classes:
            class_totals[class_name] += int(frame[class_name])
        regions = frame["regions"]
        for region_name in (config.left_zone_name, config.right_zone_name):
            region_totals[region_name] += sum(int(value) for value in regions[region_name].values())
        for detection in frame["detections"]:
            all_confidences.append(float(detection["confidence"]))

    total_detections = sum(class_totals.values())
    peak_frames = sorted(
        (
            {
                "frame_index": int(frame["frame_index"]),
                "timestamp_seconds": _round(int(frame["frame_index"]) / safe_fps),
                "total": int(frame["total"]),
                "density": str(frame["density"]),
            }
            for frame in frame_reports
        ),
        key=lambda item: item["total"],
        reverse=True,
    )[:3]

    dominant_region = (
        "Balanced"
        if region_totals[config.left_zone_name] == region_totals[config.right_zone_name]
        else max(region_totals, key=region_totals.get)
    )

    return {
        "frames_processed": frame_count,
        "duration_seconds": _round(frame_count / safe_fps),
        "average_vehicles_per_frame": _round(mean(totals_per_frame)),
        "max_vehicles_in_frame": max(totals_per_frame),
        "min_vehicles_in_frame": min(totals_per_frame),
        "vehicle_count_std_dev": _round(pstdev(totals_per_frame)) if frame_count > 1 else 0.0,
        "density_breakdown": {
            density: {
                "frames": count,
                "share_percent": _round((count / frame_count) * 100),
            }
            for density, count in sorted(density_counter.items())
        },
        "class_totals": class_totals,
        "class_average_per_frame": {
            class_name: _round(class_totals[class_name] / frame_count)
            for class_name in config.target_classes
        },
        "class_share_percent": {
            class_name: _round((class_totals[class_name] / total_detections) * 100) if total_detections else 0.0
            for class_name in config.target_classes
        },
        "region_totals": region_totals,
        "region_share_percent": {
            region_name: _round((region_total / total_detections) * 100) if total_detections else 0.0
            for region_name, region_total in region_totals.items()
        },
        "dominant_region": dominant_region,
        "peak_frames": peak_frames,
        "average_detection_confidence": _round(mean(all_confidences)) if all_confidences else None,
        "note": "Video analytics are based on detections per frame, not unique tracked vehicles.",
    }


def print_image_analytics(analytics: dict[str, object]) -> None:
    print("===== Image Analytics =====")
    print(f"Dominant Region: {analytics['dominant_region']}")
    print(f"Average Confidence: {analytics['average_confidence']}")
    print(f"Coverage Estimate: {analytics['cumulative_bbox_coverage_percent']}%")
    for class_name, values in analytics["vehicle_distribution"].items():
        print(
            f"{class_name.title()} Share: {values['share_percent']}% "
            f"(avg conf: {values['avg_confidence']})"
        )


def print_video_analytics(analytics: dict[str, object]) -> None:
    print("===== Video Analytics =====")
    print(f"Duration (s): {analytics['duration_seconds']}")
    print(f"Average Vehicles/Frame: {analytics['average_vehicles_per_frame']}")
    print(f"Min Vehicles In A Frame: {analytics['min_vehicles_in_frame']}")
    print(f"Max Vehicles In A Frame: {analytics['max_vehicles_in_frame']}")
    print(f"Vehicle Count Std Dev: {analytics['vehicle_count_std_dev']}")
    print(f"Dominant Region: {analytics['dominant_region']}")
    print(f"Average Detection Confidence: {analytics['average_detection_confidence']}")
    for density, values in analytics["density_breakdown"].items():
        print(f"{density} Density Frames: {values['frames']} ({values['share_percent']}%)")
    if analytics["peak_frames"]:
        top_frame = analytics["peak_frames"][0]
        print(
            "Peak Frame: "
            f"index {top_frame['frame_index']} at {top_frame['timestamp_seconds']}s "
            f"with {top_frame['total']} vehicles"
        )


def _style_axes(ax) -> None:
    ax.set_facecolor("#ffffff")
    for side in ax.spines.values():
        side.set_visible(False)
    ax.grid(axis="y", color="#d9e2ec", linewidth=0.8, alpha=0.8)
    ax.tick_params(colors="#334e68", labelsize=10)
    ax.title.set_color("#102a43")


def _make_figure(title: str, subtitle: str, rows: int, cols: int, figsize: tuple[float, float]):
    fig = plt.figure(figsize=figsize, facecolor="#f4f7fb")
    fig.text(0.04, 0.965, title, fontsize=24, fontweight="bold", color="#102a43")
    fig.text(0.04, 0.935, subtitle, fontsize=10.5, color="#486581")
    grid = GridSpec(rows, cols, figure=fig, left=0.04, right=0.98, top=0.9, bottom=0.06, hspace=0.38, wspace=0.28)
    return fig, grid


def _figure_to_bgr(fig) -> np.ndarray:
    fig.canvas.draw()
    width, height = fig.canvas.get_width_height()
    rgb = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(height, width, 4)[..., :3]
    plt.close(fig)
    return rgb[:, :, ::-1].copy()


def _add_kpi_card(ax, title: str, value: str, accent: str) -> None:
    ax.set_facecolor("#ffffff")
    ax.set_xticks([])
    ax.set_yticks([])
    for side in ax.spines.values():
        side.set_visible(False)
    ax.axhline(1.0, color=accent, linewidth=6, xmin=0.0, xmax=1.0)
    ax.text(0.06, 0.72, title, transform=ax.transAxes, fontsize=11, color="#486581", fontweight="bold")
    ax.text(0.06, 0.28, value, transform=ax.transAxes, fontsize=22, color="#102a43", fontweight="bold")


def render_image_analytics_dashboard(source_label: str, summary: CountSummary, analytics: dict[str, object], config: AppConfig) -> np.ndarray:
    fig, grid = _make_figure("Traffic Image Analytics", source_label, 3, 12, (18, 11))

    _add_kpi_card(fig.add_subplot(grid[0, 0:3]), "Total Vehicles", str(summary.total), "#2F80ED")
    _add_kpi_card(fig.add_subplot(grid[0, 3:6]), "Density", summary.density, "#27AE60")
    _add_kpi_card(fig.add_subplot(grid[0, 6:9]), "Avg Confidence", str(analytics["average_confidence"]), "#9B51E0")
    _add_kpi_card(fig.add_subplot(grid[0, 9:12]), "Coverage", f"{analytics['cumulative_bbox_coverage_percent']}%", "#F2994A")

    class_names = list(config.target_classes)
    counts = [summary.counts[name] for name in class_names]
    colors = [CHART_COLORS[name] for name in class_names]

    ax_bar = fig.add_subplot(grid[1, 0:5])
    _style_axes(ax_bar)
    bars = ax_bar.bar([name.title() for name in class_names], counts, color=colors, edgecolor="none")
    ax_bar.set_title("Vehicle Count by Class", loc="left", fontsize=14, fontweight="bold")
    for bar, count in zip(bars, counts):
        ax_bar.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2, str(count), ha="center", va="bottom", color="#102a43", fontsize=10, fontweight="bold")

    ax_donut = fig.add_subplot(grid[1, 5:8])
    ax_donut.set_facecolor("#ffffff")
    ax_donut.set_title("Class Share", loc="left", fontsize=14, fontweight="bold", color="#102a43")
    total = max(sum(counts), 1)
    wedges, _ = ax_donut.pie(counts if any(counts) else [1], colors=colors if any(counts) else ["#d9e2ec"], startangle=90, wedgeprops={"width": 0.38, "edgecolor": "white"})
    ax_donut.text(0, 0.05, f"{summary.total}", ha="center", va="center", fontsize=24, fontweight="bold", color="#102a43")
    ax_donut.text(0, -0.18, "vehicles", ha="center", va="center", fontsize=10, color="#486581")
    if any(counts):
        ax_donut.legend(wedges, [f"{name.title()} ({_round(count / total * 100)}%)" for name, count in zip(class_names, counts)], loc="lower center", bbox_to_anchor=(0.5, -0.22), ncol=2, frameon=False, fontsize=9)

    ax_lane = fig.add_subplot(grid[1, 8:12])
    _style_axes(ax_lane)
    region_totals = analytics["region_totals"]
    lane_labels = [config.left_zone_name, config.right_zone_name]
    lane_values = [region_totals[config.left_zone_name], region_totals[config.right_zone_name]]
    lane_colors = [CHART_COLORS["left"], CHART_COLORS["right"]]
    bars = ax_lane.barh(lane_labels, lane_values, color=lane_colors, edgecolor="none")
    ax_lane.set_title("Lane Volume Comparison", loc="left", fontsize=14, fontweight="bold")
    for bar, value in zip(bars, lane_values):
        ax_lane.text(value + 0.2, bar.get_y() + bar.get_height() / 2, str(value), va="center", color="#102a43", fontsize=10, fontweight="bold")

    ax_conf = fig.add_subplot(grid[2, 0:7])
    _style_axes(ax_conf)
    avg_conf = [analytics["vehicle_distribution"][name]["avg_confidence"] or 0.0 for name in class_names]
    max_conf = [analytics["vehicle_distribution"][name]["max_confidence"] or 0.0 for name in class_names]
    x = np.arange(len(class_names))
    width = 0.36
    ax_conf.bar(x - width / 2, avg_conf, width, label="Average", color="#56CCF2")
    ax_conf.bar(x + width / 2, max_conf, width, label="Max", color="#BB6BD9")
    ax_conf.set_xticks(x, [name.title() for name in class_names])
    ax_conf.set_ylim(0, 1.05)
    ax_conf.set_title("Confidence Profile by Class", loc="left", fontsize=14, fontweight="bold")
    ax_conf.legend(frameon=False)

    ax_notes = fig.add_subplot(grid[2, 7:12])
    ax_notes.set_facecolor("#ffffff")
    ax_notes.set_xticks([])
    ax_notes.set_yticks([])
    for side in ax_notes.spines.values():
        side.set_visible(False)
    ax_notes.set_title("Key Insights", loc="left", fontsize=14, fontweight="bold", color="#102a43")
    notes = [
        f"Dominant region: {analytics['dominant_region']}",
        f"Frame size: {analytics['frame_size']['width']} x {analytics['frame_size']['height']}",
        f"Left lane share: {analytics['region_share_percent'][config.left_zone_name]}%",
        f"Right lane share: {analytics['region_share_percent'][config.right_zone_name]}%",
        "Try larger models or lower confidence for distant vehicles.",
    ]
    y = 0.88
    for note in notes:
        ax_notes.text(0.04, y, f"- {note}", transform=ax_notes.transAxes, fontsize=11, color="#334e68")
        y -= 0.16

    return _figure_to_bgr(fig)


def render_video_analytics_dashboard(source_label: str, analytics: dict[str, object], config: AppConfig) -> np.ndarray:
    fig, grid = _make_figure("Traffic Video Analytics", source_label, 3, 12, (18, 11.5))

    _add_kpi_card(fig.add_subplot(grid[0, 0:3]), "Frames", str(analytics["frames_processed"]), "#2F80ED")
    _add_kpi_card(fig.add_subplot(grid[0, 3:6]), "Duration", f"{analytics['duration_seconds']} s", "#27AE60")
    _add_kpi_card(fig.add_subplot(grid[0, 6:9]), "Avg Vehicles/Frame", str(analytics["average_vehicles_per_frame"]), "#9B51E0")
    _add_kpi_card(fig.add_subplot(grid[0, 9:12]), "Avg Confidence", str(analytics["average_detection_confidence"]), "#F2994A")

    class_names = list(config.target_classes)
    class_totals = [analytics["class_totals"][name] for name in class_names]
    colors = [CHART_COLORS[name] for name in class_names]

    ax_class = fig.add_subplot(grid[1, 0:4])
    _style_axes(ax_class)
    bars = ax_class.bar([name.title() for name in class_names], class_totals, color=colors, edgecolor="none")
    ax_class.set_title("Total Detections by Class", loc="left", fontsize=14, fontweight="bold")
    for bar, value in zip(bars, class_totals):
        ax_class.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2, str(value), ha="center", va="bottom", color="#102a43", fontsize=10, fontweight="bold")

    ax_density = fig.add_subplot(grid[1, 4:8])
    _style_axes(ax_density)
    density_order = ["Low", "Medium", "High"]
    density_values = [analytics["density_breakdown"].get(name, {}).get("frames", 0) for name in density_order]
    density_colors = [CHART_COLORS["low"], CHART_COLORS["medium"], CHART_COLORS["high"]]
    bars = ax_density.bar(density_order, density_values, color=density_colors, edgecolor="none")
    ax_density.set_title("Density Breakdown", loc="left", fontsize=14, fontweight="bold")
    for bar, value in zip(bars, density_values):
        ax_density.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2, str(value), ha="center", va="bottom", color="#102a43", fontsize=10, fontweight="bold")

    ax_lane = fig.add_subplot(grid[1, 8:12])
    _style_axes(ax_lane)
    lane_labels = [config.left_zone_name, config.right_zone_name]
    lane_values = [analytics["region_totals"][config.left_zone_name], analytics["region_totals"][config.right_zone_name]]
    bars = ax_lane.barh(lane_labels, lane_values, color=[CHART_COLORS["left"], CHART_COLORS["right"]], edgecolor="none")
    ax_lane.set_title("Lane Volume Comparison", loc="left", fontsize=14, fontweight="bold")
    for bar, value in zip(bars, lane_values):
        ax_lane.text(value + 0.3, bar.get_y() + bar.get_height() / 2, str(value), va="center", color="#102a43", fontsize=10, fontweight="bold")

    ax_peaks = fig.add_subplot(grid[2, 0:6])
    _style_axes(ax_peaks)
    peak_frames = analytics["peak_frames"]
    peak_labels = [f"F{item['frame_index']}\n{item['timestamp_seconds']}s" for item in peak_frames] or ["No peaks"]
    peak_values = [item["total"] for item in peak_frames] or [0]
    bars = ax_peaks.bar(peak_labels, peak_values, color="#5B8FF9", edgecolor="none")
    ax_peaks.set_title("Peak Frames", loc="left", fontsize=14, fontweight="bold")
    for bar, value in zip(bars, peak_values):
        ax_peaks.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2, str(value), ha="center", va="bottom", color="#102a43", fontsize=10, fontweight="bold")

    ax_mix = fig.add_subplot(grid[2, 6:9])
    ax_mix.set_facecolor("#ffffff")
    ax_mix.set_title("Class Share", loc="left", fontsize=14, fontweight="bold", color="#102a43")
    total = max(sum(class_totals), 1)
    wedges, _ = ax_mix.pie(class_totals if any(class_totals) else [1], colors=colors if any(class_totals) else ["#d9e2ec"], startangle=90, wedgeprops={"width": 0.38, "edgecolor": "white"})
    ax_mix.text(0, 0.05, f"{total}", ha="center", va="center", fontsize=22, fontweight="bold", color="#102a43")
    ax_mix.text(0, -0.18, "detections", ha="center", va="center", fontsize=10, color="#486581")
    if any(class_totals):
        ax_mix.legend(wedges, [f"{name.title()} ({analytics['class_share_percent'][name]}%)" for name in class_names], loc="lower center", bbox_to_anchor=(0.5, -0.24), ncol=2, frameon=False, fontsize=9)

    ax_notes = fig.add_subplot(grid[2, 9:12])
    ax_notes.set_facecolor("#ffffff")
    ax_notes.set_xticks([])
    ax_notes.set_yticks([])
    for side in ax_notes.spines.values():
        side.set_visible(False)
    ax_notes.set_title("Key Insights", loc="left", fontsize=14, fontweight="bold", color="#102a43")
    notes = [
        f"Dominant region: {analytics['dominant_region']}",
        f"Min/Max vehicles per frame: {analytics['min_vehicles_in_frame']} / {analytics['max_vehicles_in_frame']}",
        f"Count variability: {analytics['vehicle_count_std_dev']}",
        "Counts are per frame, not unique tracked vehicles.",
    ]
    y = 0.88
    for note in notes:
        ax_notes.text(0.04, y, f"- {note}", transform=ax_notes.transAxes, fontsize=11, color="#334e68")
        y -= 0.16

    return _figure_to_bgr(fig)
