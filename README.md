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