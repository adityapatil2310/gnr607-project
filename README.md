# Satellite Image Processing: NCC Aircraft Detection

This project implements Zero-mean Normalized Cross-Correlation (NCC) template matching entirely from scratch in Python to detect aircraft in satellite imagery. Geometric transformations, sliding-window calculations, and non-maximum suppression are natively implemented without high-level CV libraries.

## Environment & Libraries
- Python 3.10+
- `numpy` 1.24+
- `matplotlib` 3.7+
- `Pillow` 9.5+
- Random Seed: `42`

## Dataset Information
- **Name:** Airbus Aircraft Detection Sample Dataset
- **Source:** Kaggle (https://www.kaggle.com/datasets/airbusgeo/airbus-aircrafts-sample-dataset)
- **License:** CC BY-NC 4.0
- **Citation:** Jeff Faudi, "Airbus Aircrafts Detection Sample Dataset", 2021.
- **Data Used:** 
  - `scene_01.jpg` (Crop `0,0,1500,1500`) - Grayscale float32
  - Template size: 60x60 px, cropped directly from `scene_01.jpg` at object center. Chosen to tightly bound the mean aircraft wingspan in the dataset.

## Parameters & Justifications
| Parameter | Value | Reason |
|---|---|---|
| `OP_THRESH` | 0.55 | Selected from the "knee" of the Precision-Recall curve to balance False Positives and False Negatives. |
| `NMS_RADIUS` | 30 | Half the template size ensures we do not double-count the fuselage/wings of the same aircraft. |
| `TOLERANCE` | 30 px | Ensures detections map strictly to the physical bounds of the GT aircraft center. |
| `CHUNK_SIZE` | 100 | Memory optimization for NCC. Keeps the sliding-window tensors small enough for standard RAM. |

## Running the Pipeline
Run the entire orchestration script with one command:
```bash
python demo.py