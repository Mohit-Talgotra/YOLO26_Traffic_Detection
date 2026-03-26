from __future__ import annotations

from pathlib import Path
from tempfile import NamedTemporaryFile

import cv2
import streamlit as st

from config import CONFIG
from counter import VehicleCounter, detection_to_dict
from detector import YOLOVehicleDetector
from utils import annotate_frame, build_output_paths, save_image, save_json_report


@st.cache_resource
def get_components() -> tuple[YOLOVehicleDetector, VehicleCounter]:
    return YOLOVehicleDetector(CONFIG), VehicleCounter(CONFIG)


def main() -> None:
    st.set_page_config(page_title="Vehicle Type Classification and Counting", layout="wide")
    st.title("Vehicle Type Classification and Counting using YOLOv26")
    st.write("Upload an image to detect and count cars, motorcycles, buses, and trucks.")

    uploaded_file = st.file_uploader("Upload image", type=["jpg", "jpeg", "png", "bmp", "webp"])
    if uploaded_file is None:
        return

    detector, counter = get_components()
    with NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as temp_file:
        temp_file.write(uploaded_file.getbuffer())
        temp_path = Path(temp_file.name)

    frame = detector.load_image(temp_path)
    detections = detector.predict_frame(frame)
    summary = counter.summarize(detections, frame.shape[1])
    annotated = annotate_frame(frame, detections, summary, CONFIG)

    image_output_path, json_output_path = build_output_paths(uploaded_file.name, ".jpg", CONFIG)
    save_image(annotated, image_output_path)

    report = {
        "source": uploaded_file.name,
        "mode": "image",
        **summary.to_dict(),
        "detections": [detection_to_dict(item) for item in detections],
        "annotated_output": str(image_output_path),
    }
    save_json_report(report, json_output_path)

    st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), caption="Annotated result", use_container_width=True)
    st.json(report)
    st.write(f"Annotated image saved to: `{image_output_path}`")
    st.write(f"JSON report saved to: `{json_output_path}`")

    temp_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
