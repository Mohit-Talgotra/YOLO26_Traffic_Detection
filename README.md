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

## Project Structure

```text
YOLO26_Traffic_Detection/
|-- example_inputs/
|-- models/
|-- outputs/
|-- src/
|   |-- config.py
|   |-- counter.py
|   |-- detector.py
|   |-- main.py
|   |-- streamlit_app.py
|   |-- utils.py
|   |-- models/
|   \-- outputs/
|-- requirements.txt
```

Note: the current Python code uses paths relative to `src/`, so downloaded model weights and generated outputs are stored in `src/models/` and `src/outputs/`.

## Requirements

Install dependencies from the repository root:

```bash
pip install -r requirements.txt
```

## Running the CLI

Run commands from the repository root.

### Image Inference

```bash
python src/main.py --mode image --path traffic_photo.jpg
```

### Video Inference

```bash
python src/main.py --mode video --path traffic_video.mp4
```

### Webcam Inference

```bash
python src/main.py --mode video --path 0
```

### Auto Mode

```bash
python src/main.py --mode auto --path traffic_photo.jpg
```

## Useful Options

### Use a stronger model

```bash
python src/main.py --mode image --path traffic_photo.jpg --model yolo26m.pt
python src/main.py --mode image --path traffic_photo.jpg --model yolo26l.pt
python src/main.py --mode image --path traffic_photo.jpg --model yolo26x.pt
```

### Select inference device

```bash
python src/main.py --mode image --path traffic_photo.jpg --device auto
python src/main.py --mode image --path traffic_photo.jpg --device cuda:0
python src/main.py --mode image --path traffic_photo.jpg --device cpu
```

### Disable video display window

```bash
python src/main.py --mode video --path traffic_video.mp4 --no-display
```

### Adjust confidence threshold

```bash
python src/main.py --mode image --path traffic_photo.jpg --conf 0.6
```

## Output

Each run produces:

- an annotated image or video
- a JSON report with counts and metadata
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

## Streamlit UI

Run the optional web interface from the repository root:

```bash
streamlit run src/streamlit_app.py
```

The Streamlit app lets you:

- upload an image
- view the annotated result
- inspect the JSON output interactively

## Model Weights

By default the app uses `yolo26n.pt`. If the file is not already present in `src/models/`, the app attempts to download it automatically.

You can also place model files there manually:

```text
src/models/yolo26n.pt
src/models/yolo26m.pt
src/models/yolo26l.pt
src/models/yolo26x.pt
```

## Notes

- Only the classes `car`, `motorcycle`, `bus`, and `truck` are counted.
- The confidence threshold defaults to `0.5`.
- Video mode reports per-frame counts. It does not perform multi-object tracking across frames.
- This project is designed to run on CPU as well, but larger models are much more practical on GPU.
