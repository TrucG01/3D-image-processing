# 3D Image Processing

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Data Flow & Workflow](#data-flow--workflow)
- [Usage](#usage)
- [Video Dataset](#video-dataset)
- [Citation](#citation)
- [Acknowledgments](#acknowledgments)
- [Contact](#contact)

## Overview
This project provides a robust pipeline for processing 3D image sequences, including spatial and temporal blurring, normalization, and parallelized batch operations. It is designed for scientific and research applications where consistent image scaling and efficient processing are required.

## Features
- Spatial and temporal blurring of image sequences
- Normalization based on global maximum pixel value
- Parallel processing for improved performance
- Configurable via YAML file
- Progress bars for processing and saving phases

## Data Flow & Workflow
```mermaid
flowchart TD
    subgraph Setup
        A[config.yaml]
        B[3d_video_capture.py]
    end
    subgraph Processing
        C[Load Images]
        D[Validate Config]
        E[Prepare Tasks]
        F[Parallel Processing]
        G[Spatial & Temporal Blur]
        H[Crop & Subtract]
        I[Normalize]
        J[Save Output Images]
    end
    subgraph Output
        K[OUTPUT_DIR]
    end
    A --> B
    B --> C
    C --> D
    D --> E
    E --> F
    F --> G
    G --> H
    H --> I
    I --> J
    J --> K
```

## Usage
1. Clone the repository.
2. Download the video dataset (see below).
3. Configure `config.yaml` with your input/output directories and processing parameters.
4. Run the main script:
   ```powershell
   python 3d_video_capture.py
   ```
5. Use `ai_ops.py` for repository health checks and meta documentation:
   ```powershell
   python ai_ops.py init
   python ai_ops.py check
   ```

## Prerequisites
- Python 3.9+
- Packages: OpenCV (cv2), NumPy, PyYAML, tqdm
- Windows PowerShell or compatible shell

## Quick Start
1. Create and activate your Python environment.
2. Install dependencies:
   ```powershell
   pip install opencv-python numpy pyyaml tqdm
   ```
3. Configure `config.yaml` (see schema below).
4. Run the pipeline:
   ```powershell
   python 3d_video_capture.py
   ```

## Configuration Schema (example)
```yaml
INPUT_DIR: "Images/Video_000"         # path to input frames
OUTPUT_DIR: "Images/Video_000_processed" # path to write processed frames
FILE_EXTENSION: "png"                 # input file extension
SPATIAL_KERNEL_SIZE: 5                 # e.g., Gaussian blur kernel size
OUTPUT_BIT_DEPTH: 8                    # 8 or 16
NORM_ZERO_POINT: 0                     # normalization offset
NORM_SCALE_FACTOR: 1.0                 # normalization scale
MAX_WORKERS: 4                         # parallel workers
```

## Meta-Kernel (Automation Layer)
Use the meta-kernel to scan, validate config, and check repo health.
- Initialize:
  ```powershell
  python meta_kernel.py init
  ```
- Scan project:
  ```powershell
  python meta_kernel.py scan
  ```
- List tools:
  ```powershell
  python meta_kernel.py list_tools
  ```
- Run tools:
  ```powershell
  python meta_kernel.py run project_structure
  python meta_kernel.py run config_validator
  python meta_kernel.py run checksum_verifier
  python meta_kernel.py run pipeline_profiler
  ```

## Notes
- Ensure dataset usage complies with the license.
- If processed directories (e.g., `Images/Video_XXX_processed`) are missing, verify `OUTPUT_DIR` and permissions.
- For performance, prefer SSD storage and adjust `MAX_WORKERS` to your CPU cores.

## Video Dataset
- Source: [mediatum.ub.tum.de/1596437](https://mediatum.ub.tum.de/1596437)
- License: [Creative Commons Attribution 4.0 International (CC BY 4.0)](http://creativecommons.org/licenses/by/4.0)
- FTP Access: `ftp://m1596437:m1596437@dataserv.ub.tum.de/`

Please ensure you comply with the dataset license when using or redistributing the data.

## Citation
If you use this project or the dataset in your research, please cite:
- The dataset: "Video source: https://mediatum.ub.tum.de/1596437"
- License: "Rights of video dataset: http://creativecommons.org/licenses/by/4.0"

## Acknowledgments
- Technical University of Munich (TUM) for providing the video dataset.
- OpenCV, NumPy, tqdm, PyYAML for their open-source libraries.

## Contact
For questions or contributions, please open an issue or submit a pull request.