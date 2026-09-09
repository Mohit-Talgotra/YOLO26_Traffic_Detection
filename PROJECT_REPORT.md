# Project Report: Vehicle Type Classification and Counting using YOLOv26

## 1. Introduction

This project implements an end-to-end traffic analysis pipeline using the Ultralytics YOLOv26 object detector to identify and count four vehicle categories: `car`, `motorcycle`, `bus`, and `truck`. The system accepts both images and videos as input, performs object detection, filters detections by confidence, computes class-wise counts, estimates traffic density, performs region-wise counting using left and right vertical zones, and saves annotated media along with structured JSON and analytics outputs.

The practical goal of the project is not only to detect vehicles, but also to study how YOLO behaves in real traffic scenes where perspective distortion, glare, small distant objects, and occlusion affect performance.

## 2. Problem Description

Traffic monitoring systems require reliable automatic detection and counting of vehicles for congestion analysis, road utilization studies, and intelligent transportation applications. Manual counting is slow, error-prone, and impractical for continuous analysis. A modern object detector such as YOLOv26 offers real-time inference, making it suitable for traffic surveillance tasks.

However, the traffic analysis problem is challenging because:

- vehicles may appear very small when they are far from the camera
- multiple vehicles overlap or partially occlude each other
- road scenes often contain strong sunlight, glare, shadows, and low contrast
- lane geometry is not always front-facing or perfectly aligned with the image frame
- video counting without tracking counts detections per frame rather than unique vehicles over time

Therefore, this project addresses two levels of the problem:

1. vehicle detection and counting
2. interpretation of what the detection results reveal about model behavior in realistic traffic imagery

## 3. Objectives

The project was designed to:

- detect vehicles using YOLOv26
- classify them into car, motorcycle, bus, and truck
- count detections per class
- estimate traffic density as Low, Medium, or High
- compute left-side and right-side traffic distribution
- generate annotated outputs for image and video
- produce machine-readable analytics reports and visual dashboards
- study YOLO performance through result analysis rather than counting alone

## 4. System Overview

The final system consists of these modules:

- `src/config.py`: central configuration for paths, thresholds, classes, and device selection
- `src/detector.py`: model loading, device resolution, inference, and extraction of detections
- `src/counter.py`: class-wise and region-wise counting logic
- `src/utils.py`: path management, saving outputs, and video writer support
- `src/analytics.py`: structured analytics generation and chart-based dashboard visuals
- `src/main.py`: CLI orchestration for image and video modes
- `src/streamlit_app.py`: optional image-based UI

The repository also stores:

- annotated outputs in `outputs/`
- analytics JSON and analytics visuals in `analytics/`
- downloaded model weights in `models/`

## 5. Methodology

### 5.1 Detection Pipeline

The detector uses the Ultralytics YOLO API with pretrained YOLOv26 weights. Only detections above the selected confidence threshold are considered. The classes are filtered dynamically using `model.names`, so the implementation does not rely on hardcoded class indices.

### 5.2 Counting Logic

Each valid detection contributes to:

- class count
- total count
- left or right region count

The left-right assignment is based on the horizontal center of the bounding box. If the center lies left of the image midpoint, the detection is counted in the left region; otherwise it is counted in the right region.

### 5.3 Density Classification

Traffic density is defined as:

- `0-10`: Low
- `11-25`: Medium
- `26+`: High

### 5.4 Analytics

Beyond raw counts, the project computes:

- class distribution percentage
- confidence statistics
- region share percentage
- dominant traffic side
- cumulative coverage estimate for images
- average/minimum/maximum vehicles per frame for video
- peak frames in videos
- density breakdown across the full video

## 6. Experimental Outputs Used for Analysis

This report is based on saved project artifacts generated on March 26, 2026 and March 27, 2026 from the repository outputs.

### Image Analytics Samples

Saved image analytics reports show the following observed results for the same sample traffic image:

- sample run A: `15` cars, `0` motorcycles, `0` buses, `0` trucks, total `15`, density `Medium`
- sample run B: `18` cars, `0` motorcycles, `0` buses, `0` trucks, total `18`, density `Medium`
- average confidence across these image runs: approximately `0.53` to `0.56`
- left-right split examples:
  - run A: left `46.67%`, right `53.33%`
  - run B: left `55.56%`, right `44.44%`

### Video Analytics Sample

The saved video analytics report shows:

- total frames processed: `1800`
- duration: `60.0 seconds`
- average vehicles per frame: `17.06`
- minimum vehicles in a frame: `11`
- maximum vehicles in a frame: `25`
- standard deviation of vehicle count: `2.01`
- density category across all frames: `100% Medium`
- class totals across the full video:
  - cars: `24344`
  - motorcycles: `0`
  - buses: `777`
  - trucks: `5585`
- class share across all detections:
  - cars: `79.28%`
  - buses: `2.53%`
  - trucks: `18.19%`
- lane distribution:
  - left side: `62.0%`
  - right side: `38.0%`
- dominant side: `Left lane`
- peak traffic frames occurred around `14.67s` to `15.07s`, each with `25` detected vehicles
- average detection confidence: `0.59`

## 7. Results and Discussion

### 7.1 Image Results

This is the most revealing part of the project because the sample image contains dense urban traffic, perspective compression, and bright lighting. The detector successfully identifies a significant number of visible cars, but it does not capture every vehicle in the scene.

The best observed image result counted `18` cars, while another repeated run counted `15`. This variation suggests that model selection and inference settings have a noticeable effect on recall. Since the scene contains many small cars near the upper half of the frame, the model performs best on larger, closer, and less occluded vehicles.

A key observation is that the image analytics consistently report `100% car share`. This does not necessarily mean the road truly contains only cars. It indicates that under the present test image and inference setup, the model did not find enough visual evidence to classify any vehicle as motorcycle, bus, or truck. In practical terms, the class distribution in this sample is dominated by car detections because car instances are the most visually obvious and most frequent class in the scene.

The confidence values are moderate rather than extremely high. An average confidence of about `0.53-0.56` means the model is reasonably certain about many detections, but not overwhelmingly so. This is consistent with crowded scenes where vehicles overlap and where the visual appearance of some objects is partially hidden. A maximum confidence around `0.84-0.85` shows that some foreground vehicles are detected strongly, but many background objects remain difficult.

The cumulative bounding-box coverage estimate is roughly one-third of the image area. This is logical for a crowded traffic image because the lower and middle portions of the frame contain larger cars that occupy substantial space. However, the far-distance cars occupy very little area and are more likely to be missed. This shows an important characteristic of YOLO performance: the model is strongly biased toward medium and large objects relative to tiny far-away ones unless higher input resolution, stronger models, or tiling methods are used.

The left-right split is relatively balanced. One image run favors the right side slightly, while another favors the left side. This suggests that the lane volume result is sensitive to which small distant vehicles are detected or missed. In other words, when the scene is dense and counts are near the decision boundary between the two halves, even a small number of additional detections can change which side appears dominant.

### 7.2 Video Results

The video results are more stable and provide stronger evidence of YOLO's behavior over time. Across `1800` frames, the detector maintains an average of `17.06` detections per frame with a relatively low standard deviation of `2.01`. This indicates that the traffic flow in the sample video is fairly consistent and that the detector is producing stable frame-by-frame outputs rather than highly erratic predictions.

The fact that every frame is classified as `Medium` density is significant. It means the observed traffic volume remains within the same operational band throughout the clip. The detector never saw the scene as sparsely populated enough for Low density or crowded enough for High density. This suggests the density thresholds are reasonable for the selected video, but it also reveals that the test clip may not be sufficiently diverse to evaluate the full dynamic range of the system.

Cars dominate the detections at `79.28%` of all frame-level detections, which is expected in a standard road traffic video. Trucks contribute a meaningful `18.19%`, showing that the model can identify larger heavy vehicles reliably across many frames. Buses appear only `2.53%` of the time, which likely reflects both their lower presence in the scene and the fact that buses are less common in the tested road environment. The absence of motorcycle detections can have two explanations: either motorcycles are genuinely absent, or the current setup struggles to detect them because they are small, fast-moving, distant, or visually blended into traffic.

The left side contributes `62%` of detections while the right side contributes `38%`. This reveals a clear asymmetry in traffic occupancy. There are two plausible explanations. First, the road geometry may genuinely place more traffic in the left half. Second, the camera perspective may make vehicles on one side easier to detect due to size, angle, or occlusion differences. Therefore, the left-side dominance should be interpreted as observed visual dominance, not necessarily exact real-world lane count.

The peak frames around `14.67-15.07 seconds` reach `25` vehicles. This is useful because it identifies the busiest temporal segment in the video. The model therefore captures not only average traffic conditions but also short-term bursts of higher occupancy. However, because counting is frame-based rather than track-based, these peaks represent instantaneous detection load rather than unique vehicle arrivals.

The average confidence of `0.59` in the video is slightly better than in the image sample. This can happen because repeated frames give the detector multiple opportunities to see the same vehicle under slightly different positions and sizes, allowing better detections on some frames even if other frames are weaker.

### 7.3 What the Results Reveal About YOLO Performance

The results show that YOLO performs well in the following situations:

- vehicles are relatively large in the frame
- object boundaries are clear
- heavy vehicles such as trucks are visually distinct
- traffic density is moderate and consistent
- repeated video frames allow stable aggregate behavior

The results also show clear limitations:

- distant small cars are often missed
- dense scenes reduce recall because of overlap and occlusion
- class diversity can collapse when only the dominant class is visually easy
- frame-based video counting is not equivalent to unique vehicle counting
- simple left-right splitting is only an approximation of real lane-level traffic

In summary, YOLO is strong at fast visible-object detection, but its recall falls when targets become small, cluttered, or strongly perspective-compressed.

## 8. Inference

From the observed outputs, the following understanding emerges:

1. YOLOv26 is effective for practical traffic monitoring when the goal is approximate density estimation, visible vehicle counting, and class-wise traffic composition.
2. The model performs best on foreground and medium-sized vehicles and is less reliable for very small distant targets.
3. Video analytics are more trustworthy than single-image counts because repeated frames smooth out some frame-level uncertainty.
4. Detection-based counting is useful for traffic pattern analysis, but not sufficient for unique-vehicle flow estimation without tracking.
5. The system already provides meaningful operational insights such as dominant side, density level, class composition, and peak traffic moments, even before adding advanced tracking or custom training.
6. The strongest evidence from the current results is that YOLO is precise enough to identify the main traffic structure, but recall is still the main bottleneck in crowded scenes.

## 9. CPU vs GPU Performance Analysis

An additional practical objective of the project was to understand how hardware choice affects inference usability. During development, earlier runs with `yolo26n.pt` were executed on the CPU using an **Intel i5-12500H**, while the latest full video run was executed on a **T4 GPU with 16 GB VRAM**.

This comparison is important because traffic analytics is not only an accuracy problem but also a throughput problem. A model that is reasonably accurate but too slow becomes difficult to use for long videos, repeated experimentation, or larger model variants.

### 9.1 Test Context

The available runs were not collected as a strict benchmark under fully identical conditions. There are two reasons:

- the CPU runs were mainly earlier development runs, often using `yolo26n.pt`
- the latest successful long video run used GPU acceleration and a more mature pipeline

Therefore, this section should be interpreted as an **observed engineering comparison**, not a laboratory-grade timing benchmark. Even with that limitation, the difference in practical usability between CPU and GPU execution is clear.

### 9.2 CPU Behavior: Intel i5-12500H

The CPU-based setup was sufficient for:

- validating the pipeline
- testing image inference
- debugging output generation
- running lighter models such as `yolo26n.pt`

However, CPU inference showed the expected limitations:

- slower end-to-end execution, especially for video
- reduced practicality when trying stronger models
- lower suitability for repeated experiments with multiple parameter settings
- higher latency for frame-by-frame processing

From a project-development perspective, CPU execution was useful for correctness testing, but it is not the ideal environment for serious traffic-video analytics once model size or video duration increases.

### 9.3 GPU Behavior: T4 16 GB VRAM

The GPU-based setup on the NVIDIA T4 was clearly more suitable for this project. It enabled:

- stable completion of the full `1800`-frame, `60-second` video run
- practical use of larger YOLO variants
- faster iteration when adjusting thresholds and analytics
- much better scalability for real traffic-video processing

The T4 GPU is particularly valuable because the project now includes not only detection but also analytics generation and chart rendering. Even though analytics visualization itself is lightweight compared with detection, the main bottleneck is still repeated model inference over many frames. GPU acceleration directly addresses that bottleneck.

### 9.4 Performance Interpretation

The most important engineering takeaway is not just that the GPU is faster, but **why** that matters for this specific project:

1. On CPU, lightweight models are practical, but stronger models quickly become inconvenient for longer videos.
2. On GPU, stronger models become realistic options, which is crucial because the major accuracy issue observed in this project is missed distant vehicles.
3. Since the project benefits from experimenting with confidence thresholds, larger models, and possibly higher image sizes, GPU acceleration greatly improves the experimentation cycle.
4. For image-only testing, CPU can be acceptable. For video analytics, GPU is strongly preferable.

### 9.5 Hardware Comparison Summary

| Aspect | CPU: Intel i5-12500H | GPU: NVIDIA T4 16 GB VRAM |
|---|---|---|
| Best use case | pipeline validation, small experiments, image inference | full video analytics, larger models, repeated experimentation |
| Practical model size | `yolo26n.pt` is comfortable; larger models become progressively slower | `yolo26m.pt`, `yolo26l.pt`, and even larger variants become practical |
| Video processing usability | workable but slow for long clips | strongly suitable for sustained frame-by-frame inference |
| Real-time potential | limited | much better |
| Scalability | lower | higher |
| Recommended role | baseline execution environment | primary deployment and experimentation environment |

### 9.6 Final Inference on Hardware Choice

For this project, the CPU and GPU serve different roles:

- the **Intel i5-12500H CPU** is adequate for development, debugging, and lightweight inference
- the **T4 GPU** is the correct choice for serious traffic-video analysis and for improving accuracy through stronger YOLO variants

This is especially relevant because the project?s main accuracy challenge is small and distant vehicles. Solving that challenge often requires heavier models, higher inference sizes, or more advanced inference settings, all of which are far more practical on GPU.

## 10. Limitations

The analysis also highlights several current limitations of the project:

- lane logic is based on image halves, not true road-lane segmentation
- video counts are per-frame detections, not unique trajectories
- only a small number of test samples were analyzed in this report
- performance metrics such as precision, recall, and mAP were not measured against labeled ground truth
- the sample image strongly favors the car class, limiting class-diversity evaluation

## 11. Recommendations for Improvement

Based on the results, the most effective next improvements would be:

- use stronger models such as `yolo26m.pt`, `yolo26l.pt`, or `yolo26x.pt`
- reduce confidence threshold slightly to improve recall for small distant vehicles
- increase inference image size
- use tiled inference for crowded traffic scenes
- introduce multi-object tracking for true vehicle flow analytics
- replace left-right splitting with polygon-based lane regions
- fine-tune the model on traffic-specific data if the system is intended for deployment
- evaluate using labeled benchmark images and videos to measure actual precision and recall

## 12. Conclusion

This project successfully demonstrates an end-to-end vehicle detection and traffic analytics system using YOLOv26. It works for both images and videos, produces annotated outputs, generates structured analytics, and supports visual dashboards. More importantly, the results show that the system is not only able to count vehicles, but also capable of revealing how YOLO behaves under practical traffic conditions.

The major insight from the experiments is that YOLO provides stable, useful traffic estimates in moderate-density scenes, especially in video, but its recall decreases in crowded images with small distant vehicles. Therefore, the project is already a strong proof of concept for traffic analytics, while also clearly identifying the next engineering steps required for higher accuracy and more reliable real-world deployment.
