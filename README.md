# EviRER-Net

**Evidential Reliability-Aware Robust Multimodal Beam Prediction for mmWave V2I Communications**

## Code Availability

This repository currently provides selected supporting code for **EviRER-Net**, including multimodal data loading and preprocessing utilities.

The complete implementation, including the proposed model architecture, training scripts, and evaluation procedures, will be made publicly available upon acceptance of the associated paper.

## Overview

EviRER-Net is a reliability-aware multimodal framework for millimeter-wave (mmWave) vehicle-to-infrastructure (V2I) beam prediction. It combines RGB images, LiDAR, radar, and GPS while addressing the performance degradation caused by unreliable sensing modalities.

## Experimental Scope

The study uses **Scenarios 31--34 of the DeepSense 6G dataset**. The considered inputs are RGB images, LiDAR point clouds represented in bird's-eye view, radar angle/velocity representations, and GPS positions. The evaluation considers both clean sensing and modality degradation.

## Current Release Scope

The current preview provides the data loader and the offline image, LiDAR, and radar degradation utilities used by the project. The EviRER-Net model, training pipeline, and complete evaluation pipeline are not included in this preview.

## Repository Structure

```text
EviRER-Net-public-preview/
|-- README.md
|-- .gitignore
|-- data2_seq.py
|-- Data_Preprocessing/
|   |-- degrade_image.py
|   |-- degrade_lidar.py
|   `-- degrade_radar.py
`-- THIRD_PARTY_LICENSES/
    `-- DeepSense6G_TII_LICENSE.md
```

- `data2_seq.py`: multimodal sequential data loading, modality-specific preprocessing, degraded-path selection, and beam-target construction.
- `Data_Preprocessing/degrade_image.py`: offline generation of Gaussian-noise, snow, and fog image degradations.
- `Data_Preprocessing/degrade_lidar.py`: offline generation of random point dropout and coordinate jitter for LiDAR point clouds.
- `Data_Preprocessing/degrade_radar.py`: offline generation of feature dropout and Gaussian measurement noise for radar angle/velocity maps.

## Requirements

The preview requires **Python 3.10 or later**.

- Data loader: Pillow, pandas, PyTorch, tqdm, Open3D, SciPy, scikit-learn, and `utm`.
- Image degradation: OpenCV, NumPy, tqdm, and `imagecorruptions`.
- LiDAR and radar degradation: NumPy; tqdm is optional.

## Quick Start

### Image degradation

```bash
python Data_Preprocessing/degrade_image.py \
  --input_dir /path/to/images \
  --output_dir /path/to/degraded_images \
  --degradation snow \
  --severity 1
```

### LiDAR degradation

```bash
python Data_Preprocessing/degrade_lidar.py \
  --input_dir /path/to/lidar_data \
  --output_dir /path/to/degraded_lidar \
  --degradation random_dropout \
  --intensity 0.1 \
  --seed 0
```

### Radar degradation

```bash
python Data_Preprocessing/degrade_radar.py \
  --input_dir /path/to/radar_data \
  --output_dir /path/to/degraded_radar \
  --degradation noise \
  --intensity 0.1 \
  --seed 0
```

## Citation

The formal citation and BibTeX entry will be provided once the paper is accepted or published.

## License and Third-Party Acknowledgments

Portions of `data2_seq.py` are derived from the open-source [DeepSense6G_TII repository](https://github.com/DeepSenseChallengeTeam/DeepSense6G_TII). The original copyright and MIT License notice for those portions are preserved in [`THIRD_PARTY_LICENSES/DeepSense6G_TII_LICENSE.md`](THIRD_PARTY_LICENSES/DeepSense6G_TII_LICENSE.md).
