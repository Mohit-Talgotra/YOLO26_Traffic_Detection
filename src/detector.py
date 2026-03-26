from __future__ import annotations

import os
import shutil
from pathlib import Path
from urllib.request import urlopen

import cv2
import numpy as np
from ultralytics import YOLO

from config import AppConfig
from counter import DetectionRecord


class YOLOVehicleDetector:
    """Wraps Ultralytics YOLO inference for vehicle-only detection."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.device = self._resolve_device(config.device)
        self.model = YOLO(str(self._resolve_model_path(config.model_name)))
        self.target_name_by_id = self._build_target_map()

        if not self.target_name_by_id:
            raise ValueError(
                "The loaded model does not expose any of the required classes: "
                f"{', '.join(config.target_classes)}"
            )

    def predict_frame(self, frame: np.ndarray) -> list[DetectionRecord]:
        results = self.model.predict(
            source=frame,
            conf=self.config.confidence_threshold,
            device=self.device,
            verbose=False,
        )
        return self._extract_detections(results[0])

    def device_description(self) -> str:
        return self.device

    def load_image(self, image_path: str | Path) -> np.ndarray:
        resolved_path = Path(image_path).expanduser().resolve()
        frame = cv2.imread(str(resolved_path))
        if frame is None:
            raise ValueError(f"Unable to read image: {resolved_path}")
        return frame

    def _resolve_model_path(self, model_name: str) -> Path:
        model_path = Path(model_name).expanduser()
        if model_path.is_absolute() or (model_path.suffix == ".pt" and model_path.parent != Path(".")):
            resolved_path = model_path.resolve()
            if not resolved_path.exists():
                raise FileNotFoundError(f"Model file does not exist: {resolved_path}")
            return resolved_path

        resolved_path = (self.config.model_dir / model_name).resolve()
        if resolved_path.exists():
            return resolved_path

        download_url = (
            "https://github.com/ultralytics/assets/releases/download/"
            f"{self.config.model_release}/{model_name}"
        )
        self._download_model(download_url, resolved_path)
        return resolved_path

    def _download_model(self, url: str, destination: Path) -> None:
        os.makedirs(destination.parent, exist_ok=True)
        temp_path = destination.with_suffix(destination.suffix + ".part")

        try:
            with urlopen(url) as response, open(temp_path, "wb") as output_file:
                shutil.copyfileobj(response, output_file)
            temp_path.replace(destination)
        except Exception as exc:
            temp_path.unlink(missing_ok=True)
            raise RuntimeError(
                "Unable to download the YOLOv26 weights automatically. "
                f"Download {destination.name} manually and place it at: {destination}"
            ) from exc

    def _resolve_device(self, requested_device: str) -> str:
        normalized = requested_device.strip().lower()
        if normalized != "auto":
            return requested_device

        try:
            import torch
        except ImportError:
            return "cpu"

        if torch.cuda.is_available():
            return "cuda:0"
        return "cpu"

    def _build_target_map(self) -> dict[int, str]:
        names = self.model.names
        target_map: dict[int, str] = {}

        if isinstance(names, dict):
            for class_id, class_name in names.items():
                if class_name in self.config.target_classes:
                    target_map[int(class_id)] = class_name
            return target_map

        for class_id, class_name in enumerate(names):
            if class_name in self.config.target_classes:
                target_map[class_id] = class_name
        return target_map

    def _extract_detections(self, result) -> list[DetectionRecord]:
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            return []

        detections: list[DetectionRecord] = []
        xyxy_values = boxes.xyxy.cpu().tolist()
        confidence_values = boxes.conf.cpu().tolist()
        class_values = boxes.cls.cpu().tolist()

        for bbox, confidence, class_id_value in zip(xyxy_values, confidence_values, class_values):
            class_id = int(class_id_value)
            class_name = self.target_name_by_id.get(class_id)
            if class_name is None:
                continue

            x1, y1, x2, y2 = (int(value) for value in bbox)
            detections.append(
                DetectionRecord(
                    class_name=class_name,
                    confidence=float(confidence),
                    bbox=(x1, y1, x2, y2),
                )
            )

        return detections
