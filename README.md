from pathlib import Path

readme = r"""# WC Detection using YOLO11

A Google Colab training workflow for detecting **WC (water closet) symbols in architectural floor plans** using Ultralytics YOLO11. The notebook loads a merged YOLO dataset from Google Drive, trains or resumes a model, saves checkpoints back to Drive, and evaluates the best checkpoint on a held-out test split.

## Overview

The notebook workflow:

1. Checks the Colab GPU and Ultralytics environment.
2. Mounts Google Drive and creates folders for dataset files, model weights, and training runs.
3. Loads `dataset_wc.zip` and extracts it into the Colab working directory.
4. Updates `data.yaml` to use the extracted dataset path.
5. Checks that the dataset is single-class and compares image/label counts for each split.
6. Previews labeled training images from the merged data sources.
7. Trains YOLO11s from pretrained `yolo11s.pt`, or resumes from a saved `last.pt` checkpoint.
8. Mirrors checkpoints and training metadata to Google Drive during training.
9. Evaluates `best.pt` on the test split and previews predictions on validation images.

## Model and training configuration

| Setting | Value |
|---|---|
| Task | Object detection |
| Target class | WC |
| Base model | `yolo11s.pt` |
| Image size | `640` |
| Batch size | `16` |
| Epoch ceiling | `100` |
| Training time limit | `3 hours` |
| Early stopping patience | `20` |
| Random seed | `0` |
| Initial learning rate (`lr0`) | `0.01` |
| Final learning-rate factor (`lrf`) | `0.01` |
| Device | CUDA GPU when available; otherwise CPU |

The notebook notes that the dataset contains mixed-resolution sources: approximately 1600-pixel CubiCasa renders and 640-pixel Roboflow tiles. Images are letterboxed to the configured training size. If WC symbols are too small for reliable detection, the notebook suggests trying `imgsz=960`, which will require more compute.

## Dataset format

Upload the merged dataset archive to this Google Drive location:

```text
MyDrive/
└── wc_detect_yolo/
    ├── dataset/
    │   └── dataset_wc.zip
    ├── weights/
    └── runs/
