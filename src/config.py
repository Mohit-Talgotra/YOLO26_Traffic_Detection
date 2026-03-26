from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class AppConfig:
    """Central application configuration."""

    project_root: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent)
    output_dir: Path = field(init=False)
    model_dir: Path = field(init=False)
    model_name: str = "yolo26n.pt"
    model_release: str = "v8.4.0"
    device: str = "auto"
    confidence_threshold: float = 0.5
    target_classes: tuple[str, ...] = ("car", "motorcycle", "bus", "truck")
    image_extensions: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    video_extensions: tuple[str, ...] = (".mp4", ".avi", ".mov", ".mkv", ".wmv", ".mpeg")
    display_window_name: str = "Vehicle Type Classification and Counting"
    left_zone_name: str = "Left lane"
    right_zone_name: str = "Right lane"
    low_density_max: int = 10
    medium_density_max: int = 25
    class_colors: dict[str, tuple[int, int, int]] = field(
        default_factory=lambda: {
            "car": (80, 180, 60),
            "motorcycle": (30, 144, 255),
            "bus": (255, 140, 0),
            "truck": (220, 20, 60),
        }
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "output_dir", self.project_root / "outputs")
        object.__setattr__(self, "model_dir", self.project_root / "models")


CONFIG = AppConfig()
