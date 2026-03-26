from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from analytics import (
    build_image_analytics,
    build_video_analytics,
    print_image_analytics,
    print_video_analytics,
)
from config import CONFIG
from counter import VehicleCounter, detection_to_dict
from detector import YOLOVehicleDetector
from utils import (
    annotate_frame,
    build_output_paths,
    create_video_writer,
    detect_source_kind,
    ensure_project_dirs,
    print_image_summary,
    print_video_summary,
    save_image,
    save_json_report,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Vehicle type classification and counting using Ultralytics YOLOv26."
    )
    parser.add_argument("--mode", choices=("image", "video", "auto"), default="auto")
    parser.add_argument("--path", required=True, help="Path to an image/video file or '0'/'webcam'.")
    parser.add_argument(
        "--conf",
        type=float,
        default=CONFIG.confidence_threshold,
        help=f"Confidence threshold. Default: {CONFIG.confidence_threshold}",
    )
    parser.add_argument(
        "--model",
        default=CONFIG.model_name,
        help=f"Model name or path. Default: {CONFIG.model_name}",
    )
    parser.add_argument(
        "--device",
        default=CONFIG.device,
        help="Inference device: 'auto', 'cpu', 'cuda:0', etc. Default: auto",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Disable live video display for video/webcam mode.",
    )
    return parser.parse_args()


def run_image(path_value: str, detector: YOLOVehicleDetector, counter: VehicleCounter) -> None:
    frame = detector.load_image(path_value)
    detections = detector.predict_frame(frame)
    summary = counter.summarize(detections, frame.shape[1])
    analytics = build_image_analytics(summary, detections, frame.shape, CONFIG)
    annotated = annotate_frame(frame, detections, summary, CONFIG)

    image_output_path, json_output_path = build_output_paths(path_value, ".jpg", CONFIG)
    save_image(annotated, image_output_path)

    report = {
        "source": str(Path(path_value).expanduser().resolve()),
        "mode": "image",
        **summary.to_dict(),
        "analytics": analytics,
        "detections": [detection_to_dict(item) for item in detections],
        "annotated_output": str(image_output_path),
    }
    save_json_report(report, json_output_path)

    print_image_summary(summary)
    print_image_analytics(analytics)
    print(f"Annotated image saved to: {image_output_path}")
    print(f"JSON report saved to: {json_output_path}")


def run_video(
    path_value: str,
    detector: YOLOVehicleDetector,
    counter: VehicleCounter,
    display: bool,
) -> None:
    source = 0 if path_value.lower() in {"0", "webcam"} else str(Path(path_value).expanduser().resolve())
    capture = cv2.VideoCapture(source)

    if not capture.isOpened():
        raise RuntimeError(f"Unable to open video source: {path_value}")

    frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
    frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720
    fps = capture.get(cv2.CAP_PROP_FPS)

    video_output_path, json_output_path = build_output_paths(path_value, ".mp4", CONFIG)
    writer = create_video_writer(video_output_path, frame_width, frame_height, fps)

    frame_reports: list[dict[str, object]] = []
    frame_index = 0
    total_vehicle_sum = 0
    max_total = 0
    display_enabled = display

    try:
        while True:
            success, frame = capture.read()
            if not success:
                break

            detections = detector.predict_frame(frame)
            summary = counter.summarize(detections, frame.shape[1])
            annotated = annotate_frame(frame, detections, summary, CONFIG)
            writer.write(annotated)

            frame_report = {
                "frame_index": frame_index,
                "timestamp_seconds": round(frame_index / (fps if fps and fps > 0 else 25.0), 2),
                **summary.to_dict(),
                "detections": [detection_to_dict(item) for item in detections],
            }
            frame_reports.append(frame_report)
            total_vehicle_sum += summary.total
            max_total = max(max_total, summary.total)

            if display_enabled:
                try:
                    cv2.imshow(CONFIG.display_window_name, annotated)
                    pressed_key = cv2.waitKey(1) & 0xFF
                    if pressed_key == ord("q"):
                        break
                except cv2.error:
                    display_enabled = False

            frame_index += 1
    finally:
        capture.release()
        writer.release()
        cv2.destroyAllWindows()

    average_total = total_vehicle_sum / frame_index if frame_index else 0.0
    analytics = build_video_analytics(frame_reports, fps, CONFIG)
    report = {
        "source": "webcam" if source == 0 else source,
        "mode": "video" if source != 0 else "webcam",
        "frames_processed": frame_index,
        "aggregate": {
            "average_vehicles_per_frame": round(average_total, 2),
            "peak_vehicles_in_a_frame": max_total,
        },
        "analytics": analytics,
        "frames": frame_reports,
        "annotated_output": str(video_output_path),
    }
    save_json_report(report, json_output_path)

    print_video_summary(frame_index, max_total, average_total)
    print_video_analytics(analytics)
    print(f"Annotated video saved to: {video_output_path}")
    print(f"JSON report saved to: {json_output_path}")


def main() -> None:
    args = parse_args()
    CONFIG.confidence_threshold = args.conf  # type: ignore[misc]
    CONFIG.model_name = args.model  # type: ignore[misc]
    CONFIG.device = args.device  # type: ignore[misc]
    ensure_project_dirs(CONFIG)

    selected_mode = args.mode
    if selected_mode == "auto":
        selected_mode = detect_source_kind(args.path, CONFIG)
        if selected_mode == "webcam":
            selected_mode = "video"

    detector = YOLOVehicleDetector(CONFIG)
    counter = VehicleCounter(CONFIG)
    print(f"Using inference device: {detector.device_description()}")

    if selected_mode == "image":
        run_image(args.path, detector, counter)
        return

    if selected_mode == "video":
        run_video(args.path, detector, counter, display=not args.no_display)
        return

    raise ValueError(f"Unsupported mode: {selected_mode}")


if __name__ == "__main__":
    main()
