from __future__ import annotations

from collections import Counter
from statistics import mean, pstdev

from config import AppConfig
from counter import CountSummary, DetectionRecord


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
            'count': count,
            'share_percent': _round((count / summary.total) * 100) if summary.total else 0.0,
            'avg_confidence': _round(mean(class_confidences)) if class_confidences else None,
            'max_confidence': _round(max(class_confidences)) if class_confidences else None,
        }

    region_totals = {
        region_name: sum(region_counts.values())
        for region_name, region_counts in summary.region_counts.items()
    }
    if region_totals[config.left_zone_name] == region_totals[config.right_zone_name]:
        dominant_region = 'Balanced'
    else:
        dominant_region = max(region_totals, key=region_totals.get)

    return {
        'vehicle_distribution': class_breakdown,
        'region_totals': region_totals,
        'region_share_percent': {
            region_name: _round((region_total / summary.total) * 100) if summary.total else 0.0
            for region_name, region_total in region_totals.items()
        },
        'dominant_region': dominant_region,
        'average_confidence': _round(mean(overall_confidences)) if overall_confidences else None,
        'max_confidence': _round(max(overall_confidences)) if overall_confidences else None,
        'cumulative_bbox_coverage_percent': _round(sum(bbox_area_ratios) * 100) if bbox_area_ratios else 0.0,
        'frame_size': {'width': frame_width, 'height': frame_height},
    }


def build_video_analytics(
    frame_reports: list[dict[str, object]],
    fps: float | int,
    config: AppConfig,
) -> dict[str, object]:
    frame_count = len(frame_reports)
    if frame_count == 0:
        return {
            'frames_processed': 0,
            'duration_seconds': 0.0,
            'average_vehicles_per_frame': 0.0,
            'max_vehicles_in_frame': 0,
            'min_vehicles_in_frame': 0,
            'vehicle_count_std_dev': 0.0,
            'density_breakdown': {},
            'class_totals': {class_name: 0 for class_name in config.target_classes},
            'class_average_per_frame': {class_name: 0.0 for class_name in config.target_classes},
            'class_share_percent': {class_name: 0.0 for class_name in config.target_classes},
            'region_totals': {
                config.left_zone_name: 0,
                config.right_zone_name: 0,
            },
            'dominant_region': 'Balanced',
            'peak_frames': [],
            'average_detection_confidence': None,
            'note': 'Video analytics are based on detections per frame, not unique tracked vehicles.',
        }

    safe_fps = float(fps) if fps and float(fps) > 0 else 25.0
    totals_per_frame = [int(frame['total']) for frame in frame_reports]
    class_totals = {class_name: 0 for class_name in config.target_classes}
    density_counter: Counter[str] = Counter()
    region_totals = {config.left_zone_name: 0, config.right_zone_name: 0}
    all_confidences: list[float] = []

    for frame in frame_reports:
        density_counter[str(frame['density'])] += 1
        for class_name in config.target_classes:
            class_totals[class_name] += int(frame[class_name])
        regions = frame['regions']
        for region_name in (config.left_zone_name, config.right_zone_name):
            region_totals[region_name] += sum(int(value) for value in regions[region_name].values())
        for detection in frame['detections']:
            all_confidences.append(float(detection['confidence']))

    total_detections = sum(class_totals.values())
    peak_frames = sorted(
        (
            {
                'frame_index': int(frame['frame_index']),
                'timestamp_seconds': _round(int(frame['frame_index']) / safe_fps),
                'total': int(frame['total']),
                'density': str(frame['density']),
            }
            for frame in frame_reports
        ),
        key=lambda item: item['total'],
        reverse=True,
    )[:3]

    if region_totals[config.left_zone_name] == region_totals[config.right_zone_name]:
        dominant_region = 'Balanced'
    else:
        dominant_region = max(region_totals, key=region_totals.get)

    return {
        'frames_processed': frame_count,
        'duration_seconds': _round(frame_count / safe_fps),
        'average_vehicles_per_frame': _round(mean(totals_per_frame)),
        'max_vehicles_in_frame': max(totals_per_frame),
        'min_vehicles_in_frame': min(totals_per_frame),
        'vehicle_count_std_dev': _round(pstdev(totals_per_frame)) if frame_count > 1 else 0.0,
        'density_breakdown': {
            density: {
                'frames': count,
                'share_percent': _round((count / frame_count) * 100),
            }
            for density, count in sorted(density_counter.items())
        },
        'class_totals': class_totals,
        'class_average_per_frame': {
            class_name: _round(class_totals[class_name] / frame_count)
            for class_name in config.target_classes
        },
        'class_share_percent': {
            class_name: _round((class_totals[class_name] / total_detections) * 100) if total_detections else 0.0
            for class_name in config.target_classes
        },
        'region_totals': region_totals,
        'region_share_percent': {
            region_name: _round((region_total / total_detections) * 100) if total_detections else 0.0
            for region_name, region_total in region_totals.items()
        },
        'dominant_region': dominant_region,
        'peak_frames': peak_frames,
        'average_detection_confidence': _round(mean(all_confidences)) if all_confidences else None,
        'note': 'Video analytics are based on detections per frame, not unique tracked vehicles.',
    }


def print_image_analytics(analytics: dict[str, object]) -> None:
    print('===== Image Analytics =====')
    print(f"Dominant Region: {analytics['dominant_region']}")
    print(f"Average Confidence: {analytics['average_confidence']}")
    print(f"Coverage Estimate: {analytics['cumulative_bbox_coverage_percent']}%")
    for class_name, values in analytics['vehicle_distribution'].items():
        print(
            f"{class_name.title()} Share: {values['share_percent']}% "
            f"(avg conf: {values['avg_confidence']})"
        )


def print_video_analytics(analytics: dict[str, object]) -> None:
    print('===== Video Analytics =====')
    print(f"Duration (s): {analytics['duration_seconds']}")
    print(f"Average Vehicles/Frame: {analytics['average_vehicles_per_frame']}")
    print(f"Min Vehicles In A Frame: {analytics['min_vehicles_in_frame']}")
    print(f"Max Vehicles In A Frame: {analytics['max_vehicles_in_frame']}")
    print(f"Vehicle Count Std Dev: {analytics['vehicle_count_std_dev']}")
    print(f"Dominant Region: {analytics['dominant_region']}")
    print(f"Average Detection Confidence: {analytics['average_detection_confidence']}")
    for density, values in analytics['density_breakdown'].items():
        print(f"{density} Density Frames: {values['frames']} ({values['share_percent']}%)")
    if analytics['peak_frames']:
        top_frame = analytics['peak_frames'][0]
        print(
            'Peak Frame: '
            f"index {top_frame['frame_index']} at {top_frame['timestamp_seconds']}s "
            f"with {top_frame['total']} vehicles"
        )
