# YOLO26 Traffic Detection

Vehicle type classification and counting for traffic images, videos, and webcam streams using Ultralytics YOLOv26.

## Features

- Detects and counts `car`, `motorcycle`, `bus`, and `truck`
- Supports image, video, and webcam input
- Saves annotated image or video output
- Saves JSON reports for each run
- Classifies traffic density as `Low`, `Medium`, or `High`
- Splits each frame into two vertical regions for left-lane and right-lane counts
- Supports automatic GPU selection with CPU fallback
- Includes an optional Streamlit UI for image uploads
- Includes a Google Colab notebook for direct cloud execution

## Project Structure

```text
YOLO26_Traffic_Detection/
|-- example_inputs/
|   |-- traffic_photo.jpg
|   \-- traffic_video.mp4
|-- models/
|-- outputs/
|-- src/
|   |-- config.py
|   |-- counter.py
|   |-- detector.py
|   |-- main.py
|   |-- streamlit_app.py
|   \-- utils.py
|-- YOLO26_Traffic_Detection_Colab.ipynb
|-- README.md
|-- requirements.txt
```

The application now uses the repository root as its working base, so model weights are stored in `models/` and generated outputs are saved in `outputs/`.

## Requirements

- Python 3.10+
- `ultralytics`
- `opencv-python`
- `numpy`
- `streamlit`
- CUDA-enabled PyTorch if you want GPU inference

Install dependencies from the repository root:

```bash
pip install -r requirements.txt
```

## Running the CLI

Run commands from the repository root.

### Image Inference

```bash
python src/main.py --mode image --path example_inputs/traffic_photo.jpg
```

### Video Inference

```bash
python src/main.py --mode video --path example_inputs/traffic_video.mp4
```

### Webcam Inference

```bash
python src/main.py --mode video --path 0
```

### Auto Mode

```bash
python src/main.py --mode auto --path example_inputs/traffic_photo.jpg
```

## Useful Options

### Use a stronger model

```bash
python src/main.py --mode image --path example_inputs/traffic_photo.jpg --model yolo26m.pt
python src/main.py --mode image --path example_inputs/traffic_photo.jpg --model yolo26l.pt
python src/main.py --mode image --path example_inputs/traffic_photo.jpg --model yolo26x.pt
```

### Select inference device

```bash
python src/main.py --mode image --path example_inputs/traffic_photo.jpg --device auto
python src/main.py --mode image --path example_inputs/traffic_photo.jpg --device cuda:0
python src/main.py --mode image --path example_inputs/traffic_photo.jpg --device cpu
```

### Disable video display window

```bash
python src/main.py --mode video --path example_inputs/traffic_video.mp4 --no-display
```

### Adjust confidence threshold

```bash
python src/main.py --mode image --path example_inputs/traffic_photo.jpg --conf 0.35
```

## Output

Each run produces:

- an annotated image or video in `outputs/`
- a JSON report in `outputs/`
- a console summary

Example console summary:

```text
Using inference device: cuda:0
===== Vehicle Count Summary =====
Car: 10
Motorcycle: 3
Bus: 1
Truck: 2
Total: 16
Density: Medium
```

Example JSON structure for image mode:

```json
{
  "car": 10,
  "motorcycle": 3,
  "bus": 1,
  "truck": 2,
  "total": 16,
  "density": "Medium",
  "regions": {
    "Left lane": {
      "car": 5,
      "motorcycle": 1,
      "bus": 0,
      "truck": 1
    },
    "Right lane": {
      "car": 5,
      "motorcycle": 2,
      "bus": 1,
      "truck": 1
    }
  }
}
```

## Traffic Density Rules

- `0-10`: Low
- `11-25`: Medium
- `26+`: High

## Region-wise Counting

Each frame is divided into two vertical zones:

- `Left lane`
- `Right lane`

Counts are assigned based on the horizontal center of each detected vehicle box.

## GPU Notes

The application supports:

- `auto`: use `cuda:0` if available, otherwise CPU
- explicit GPU selection such as `cuda:0`
- CPU fallback when CUDA is unavailable

At startup, the CLI prints the selected device:

```text
Using inference device: cuda:0
```

If `auto` keeps selecting CPU, your installed PyTorch build likely does not have CUDA enabled.

## Google Colab

Open [YOLO26_Traffic_Detection_Colab.ipynb](./YOLO26_Traffic_Detection_Colab.ipynb) in Google Colab and run the cells in order.

What the notebook does:

- installs dependencies
- clones or opens the project in Colab
- lets you upload an image or video
- runs inference with your chosen model and device
- displays the annotated result inline
- gives you downloadable outputs from the `outputs/` folder

Recommended Colab runtime:

- Runtime -> Change runtime type -> `GPU`

## Streamlit UI

Run the optional web interface from the repository root:

```bash
streamlit run src/streamlit_app.py
```

## Model Weights

By default the app uses `yolo26n.pt`. If the file is not already present in `models/`, the app attempts to download it automatically.

You can also place model files there manually:

```text
models/yolo26n.pt
models/yolo26m.pt
models/yolo26l.pt
models/yolo26x.pt
```

## Notes

- Only the classes `car`, `motorcycle`, `bus`, and `truck` are counted.
- The confidence threshold defaults to `0.5`.
- Video mode reports per-frame counts. It does not perform multi-object tracking across frames.
- Larger models and lower confidence thresholds usually improve recall for distant vehicles.
