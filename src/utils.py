from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from config import AppConfig
from counter import CountSummary, DetectionRecord


def ensure_output_dir(config: AppConfig) -> Path:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    return config.output_dir


def ensure_project_dirs(config: AppConfig) -> None:
    if not config.project_root.exists():
        raise RuntimeError(f"Project directory does not exist: {config.project_root}")


def timestamp_token() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def resolve_input_path(path_value: str) -> Path:
    resolved = Path(path_value).expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"Input path does not exist: {resolved}")
    return resolved


def detect_source_kind(path_value: str, config: AppConfig) -> str:
    if path_value.lower() in {"0", "webcam"}:
        return "webcam"

    resolved = resolve_input_path(path_value)
    suffix = resolved.suffix.lower()

    if suffix in config.image_extensions:
        return "image"
    if suffix in config.video_extensions:
        return "video"

    raise ValueError(f"Unsupported input type for path: {resolved}")


def build_output_paths(input_label: str, suffix: str, config: AppConfig) -> tuple[Path, Path]:
    ensure_output_dir(config)
    stem = Path(input_label).stem if input_label not in {"0", "webcam"} else "webcam"
    token = timestamp_token()
    media_path = config.output_dir / f"{stem}_{token}{suffix}"
    json_path = config.output_dir / f"{stem}_{token}.json"
    return media_path, json_path


def annotate_frame(
    frame: np.ndarray,
    detections: list[DetectionRecord],
    summary: CountSummary,
    config: AppConfig,
) -> np.ndarray:
    annotated = frame.copy()
    height, width = annotated.shape[:2]
    midpoint = width // 2

    cv2.line(annotated, (midpoint, 0), (midpoint, height), (255, 255, 255), 2)
    cv2.putText(
        annotated,
        config.left_zone_name,
        (10, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        annotated,
        config.right_zone_name,
        (midpoint + 10, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    for detection in detections:
        x1, y1, x2, y2 = detection.bbox
        color = config.class_colors.get(detection.class_name, (255, 255, 255))
        label = f"{detection.class_name} {detection.confidence:.2f}"
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        cv2.rectangle(
            annotated,
            (x1, max(y1 - 26, 0)),
            (x1 + max(150, len(label) * 8), y1),
            color,
            thickness=-1,
        )
        cv2.putText(
            annotated,
            label,
            (x1 + 4, max(y1 - 8, 16)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )

    overlay_lines = [
        f"Total: {summary.total}",
        f"Density: {summary.density}",
        f"Car: {summary.counts['car']}",
        f"Motorcycle: {summary.counts['motorcycle']}",
        f"Bus: {summary.counts['bus']}",
        f"Truck: {summary.counts['truck']}",
    ]

    for index, line in enumerate(overlay_lines):
        y_pos = height - (len(overlay_lines) - index) * 24
        cv2.putText(
            annotated,
            line,
            (10, y_pos),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    return annotated


def print_image_summary(summary: CountSummary) -> None:
    print("===== Vehicle Count Summary =====")
    print(f"Car: {summary.counts['car']}")
    print(f"Motorcycle: {summary.counts['motorcycle']}")
    print(f"Bus: {summary.counts['bus']}")
    print(f"Truck: {summary.counts['truck']}")
    print(f"Total: {summary.total}")
    print(f"Density: {summary.density}")


def print_video_summary(frame_count: int, max_total: int, average_total: float) -> None:
    print("===== Video Count Summary =====")
    print(f"Frames Processed: {frame_count}")
    print(f"Peak Vehicles In A Frame: {max_total}")
    print(f"Average Vehicles Per Frame: {average_total:.2f}")


def save_json_report(payload: dict[str, object], output_path: Path) -> None:
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def save_image(image, output_path):
    output_path = Path(output_path)

    success, encoded = cv2.imencode(".jpg", image)
    if not success:
        raise RuntimeError("Encoding failed")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(str(output_path), "wb") as f:
        f.write(encoded.tobytes())


def create_video_writer(
    output_path: Path, frame_width: int, frame_height: int, fps: float | int
) -> cv2.VideoWriter:
    safe_fps = fps if fps and fps > 0 else 25.0
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, float(safe_fps), (frame_width, frame_height))
    if not writer.isOpened():
        raise RuntimeError(f"Unable to create video writer for: {output_path}")
    return writer
