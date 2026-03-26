from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

from config import AppConfig


@dataclass(frozen=True)
class DetectionRecord:
    class_name: str
    confidence: float
    bbox: tuple[int, int, int, int]


@dataclass(frozen=True)
class CountSummary:
    counts: dict[str, int]
    region_counts: dict[str, dict[str, int]]
    total: int
    density: str

    def to_dict(self) -> dict[str, object]:
        payload = dict(self.counts)
        payload["total"] = self.total
        payload["density"] = self.density
        payload["regions"] = self.region_counts
        return payload


class VehicleCounter:
    """Counts detections globally and by vertical image region."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config

    def summarize(self, detections: Iterable[DetectionRecord], frame_width: int) -> CountSummary:
        counts = {name: 0 for name in self.config.target_classes}
        region_counts = {
            self.config.left_zone_name: {name: 0 for name in self.config.target_classes},
            self.config.right_zone_name: {name: 0 for name in self.config.target_classes},
        }

        midpoint = frame_width / 2
        total = 0

        for detection in detections:
            if detection.class_name not in counts:
                continue

            counts[detection.class_name] += 1
            total += 1
            x1, _, x2, _ = detection.bbox
            center_x = (x1 + x2) / 2
            zone_name = (
                self.config.left_zone_name if center_x < midpoint else self.config.right_zone_name
            )
            region_counts[zone_name][detection.class_name] += 1

        return CountSummary(
            counts=counts,
            region_counts=region_counts,
            total=total,
            density=self._density_label(total),
        )

    def _density_label(self, total: int) -> str:
        if total <= self.config.low_density_max:
            return "Low"
        if total <= self.config.medium_density_max:
            return "Medium"
        return "High"


def detection_to_dict(detection: DetectionRecord) -> dict[str, object]:
    return asdict(detection)
