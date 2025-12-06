"""
3D Image Sequence Processing Script

This script processes a sequence of images using spatial and temporal blurring, then subtracts the blurred result from the original image.
It normalizes the output based on the global maximum pixel value found across all frames, ensuring consistent scaling. The script uses
parallel processing for speed and displays progress bars for both processing and saving phases.

Requirements:
    - OpenCV (cv2)
    - numpy
    - tqdm
    - PyYAML

Usage:
    python 3d_video_capture.py

Settings:
    - config.yaml: Configuration file for input/output directories and processing parameters
"""

import os
import cv2
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import multiprocessing
from typing import Tuple, List, Optional
import yaml

# ==========================================
#               USER SETTINGS
# ==========================================

# Load config from YAML
CONFIG_PATH = 'config.yaml'

def load_config(config_path: str) -> dict:
    """
    Loads configuration from a YAML file.
    """
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

config = load_config(CONFIG_PATH)

# File I/O
INPUT_DIR = config['INPUT_DIR']
OUTPUT_DIR = config['OUTPUT_DIR']
FILE_EXTENSION = config['FILE_EXTENSION']

# Processing Parameters
# Spatial Kernel: Must be an odd number (3, 5, 7, etc.)
# Larger number = stronger blur, more pixels cropped from edges.
SPATIAL_KERNEL_SIZE = config['SPATIAL_KERNEL_SIZE']

# Normalization Parameters (for 8-bit output)
OUTPUT_BIT_DEPTH = config['OUTPUT_BIT_DEPTH']
NORM_ZERO_POINT = config['NORM_ZERO_POINT']  # Where 0.0 sits (usually middle of 0-255)
NORM_SCALE_FACTOR = config['NORM_SCALE_FACTOR'] # Scaling factor

# Parallel Processing
# Set to None to use all available cores, or an integer (e.g., 4) to limit usage.
MAX_WORKERS = config.get('MAX_WORKERS', None) 

# ==========================================
#           DERIVED PARAMETERS
# ==========================================
# (Do not edit these unless you are changing logic)

# Calculate how much to crop to remove border artifacts.
# e.g., Kernel 3 -> radius 1. Kernel 5 -> radius 2.
CROP_MARGIN = SPATIAL_KERNEL_SIZE // 2 

# Maximum value for clipping (e.g., 255 for 8-bit)
MAX_PIXEL_VAL = (2 ** OUTPUT_BIT_DEPTH) - 1

# ------------------------------------------
#            CONFIG VALIDATION
# ------------------------------------------

def validate_config() -> None:
    problems: List[str] = []

    # Directories
    if not os.path.isdir(INPUT_DIR):
        problems.append(f"INPUT_DIR not found: {INPUT_DIR}")
    if not os.path.isdir(os.path.dirname(OUTPUT_DIR) or '.'):  # parent exists
        problems.append(f"Parent of OUTPUT_DIR not found: {os.path.dirname(OUTPUT_DIR) or '.'}")

    # File extension
    if not FILE_EXTENSION.startswith('.'):  
        problems.append(f"FILE_EXTENSION must start with a dot, got '{FILE_EXTENSION}'")

    # Kernel size
    if not isinstance(SPATIAL_KERNEL_SIZE, int) or SPATIAL_KERNEL_SIZE <= 0:
        problems.append("SPATIAL_KERNEL_SIZE must be a positive integer.")
    elif SPATIAL_KERNEL_SIZE % 2 == 0:
        problems.append("SPATIAL_KERNEL_SIZE must be odd (3,5,7,...).")

    # Bit depth
    if OUTPUT_BIT_DEPTH not in (8, 10, 12, 16):
        problems.append(f"Unsupported OUTPUT_BIT_DEPTH: {OUTPUT_BIT_DEPTH}. Use one of 8,10,12,16.")

    # Normalization
    if not isinstance(NORM_ZERO_POINT, (int, float)):
        problems.append("NORM_ZERO_POINT must be numeric.")
    if not isinstance(NORM_SCALE_FACTOR, (int, float)):
        problems.append("NORM_SCALE_FACTOR must be numeric.")

    if problems:
        print("Configuration errors:")
        for p in problems:
            print(f" - {p}")
        raise ValueError("Invalid configuration. Fix config.yaml and rerun.")

# ==========================================
#           PROCESSING LOGIC
# ==========================================

def process_frame_optimized(args: Tuple[int, str, str, str, str, int, int]) -> Tuple[int, np.ndarray, float]:
    """
    Processes a triplet of frames (Prev, Curr, Next) with spatial and temporal blur, crops, subtracts, and returns result.

    Args:
        args (Tuple):
            idx (int): Output index (0-based)
            file_prev (str): Filename of previous image
            file_curr (str): Filename of current image
            file_next (str): Filename of next image
            in_dir (str): Input directory
            k_size (int): Spatial kernel size
            crop (int): Crop margin

    Returns:
        Tuple[int, np.ndarray, float]:
            idx: Output index
            subtracted: Cropped and subtracted image (float32)
            max_val: Maximum absolute pixel value in subtracted image
    """
    idx, file_prev, file_curr, file_next, in_dir, k_size, crop = args
    
    # Load images as float32 to prevent overflow/rounding errors during math
    # Combining path joins for slight speedup
    path_prev = os.path.join(in_dir, file_prev)
    path_curr = os.path.join(in_dir, file_curr)
    path_next = os.path.join(in_dir, file_next)

    prev_img = cv2.imread(path_prev, cv2.IMREAD_COLOR).astype(np.float32)
    curr_img = cv2.imread(path_curr, cv2.IMREAD_COLOR).astype(np.float32)
    next_img = cv2.imread(path_next, cv2.IMREAD_COLOR).astype(np.float32)

    # 1. Spatial Blur (Box Filter)
    # Replaces manual loop with optimized C++ backend of OpenCV
    blur_prev = cv2.blur(prev_img, (k_size, k_size))
    blur_curr = cv2.blur(curr_img, (k_size, k_size))
    blur_next = cv2.blur(next_img, (k_size, k_size))
    
    # 2. Temporal Blur (Average the 3 frames)
    # This is mathematically equivalent to the 3D convolution in the original script
    blurred_result = (blur_prev + blur_curr + blur_next) / 3.0
    
    # 3. Crop and Subtract
    # We remove the border pixels defined by CROP_MARGIN to avoid edge artifacts
    h, w, _ = curr_img.shape
    
    # Slice format: [start:end]
    # If crop is 1: [1 : h-1]
    cropped_orig = curr_img[crop : h-crop, crop : w-crop]
    cropped_blur = blurred_result[crop : h-crop, crop : w-crop]
    
    subtracted = cropped_orig - cropped_blur
    
    # Track the max value for this specific frame
    max_val = np.max(np.abs(subtracted))
    
    return idx, subtracted, max_val

def main() -> None:
    """
    Main entry point for processing image sequence.
    Loads images, prepares tasks, runs parallel processing, normalizes and saves output images.
    """
    validate_config()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Get sorted list of image files
    image_files = sorted([f for f in os.listdir(INPUT_DIR) if f.endswith(FILE_EXTENSION)])
    
    if len(image_files) < 3:
        print("Error: Need at least 3 images to perform temporal processing.")
        return

    # Pre-allocate list for results
    # We process frames 1 to N-1 (skipping first and last)
    num_tasks = len(image_files) - 2
    subtracted_images = [None] * num_tasks
    global_max_val = 0.0
    
    # Prepare arguments
    tasks = []
    # Loop from index 1 to len-1
    for i in range(1, len(image_files) - 1):
        # We pass the settings into the worker so it doesn't rely on global scope
        tasks.append((
            i - 1,                # Output index (0-based)
            image_files[i-1],     # Prev
            image_files[i],       # Curr
            image_files[i+1],     # Next
            INPUT_DIR,
            SPATIAL_KERNEL_SIZE,
            CROP_MARGIN
        ))

    print(f"Configuration: Kernel={SPATIAL_KERNEL_SIZE}x{SPATIAL_KERNEL_SIZE} | Crop={CROP_MARGIN}px | Cores={MAX_WORKERS or 'All'}")
    print(f"Submitting {len(tasks)} tasks...")

    # Phase 1: Processing
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(process_frame_optimized, task) for task in tasks]
        
        with tqdm(total=len(futures), desc='Processing Frames') as pbar:
            for f in as_completed(futures):
                out_idx, subtracted, max_val = f.result()
                
                subtracted_images[out_idx] = subtracted
                
                if max_val > global_max_val:
                    global_max_val = max_val
                
                pbar.update(1)

    print(f"Global Max Value found: {global_max_val:.4f}")

    # Phase 2: Normalization and Saving
    # (Fast enough to run in main thread, but could be threaded if disk I/O is slow)
    if global_max_val == 0:
        global_max_val = 1.0 # Prevent divide by zero if images are pure black

    for idx, subtracted in tqdm(enumerate(subtracted_images), total=len(subtracted_images), desc='Saving Images'):
        if subtracted is None: continue 

        # Normalization Formula:
        # (Value / Max) scales to [-1.0, 1.0]
        # * Scale + ZeroPoint shifts to [0, 255]
        norm = ((subtracted / global_max_val) * NORM_SCALE_FACTOR + NORM_ZERO_POINT)
        
        # Clip and cast
        norm = norm.clip(0, MAX_PIXEL_VAL).astype(np.uint8)
        
        # Match output filename to input filename
        # tasks[idx][2] retrieves the 'curr' filename from our task list
        original_filename = tasks[idx][2] 
        out_name = os.path.join(OUTPUT_DIR, original_filename)
        
        cv2.imwrite(out_name, norm)

    print(f"Complete. Saved to {OUTPUT_DIR}")

if __name__ == '__main__':
    # Required for Windows Multiprocessing
    multiprocessing.freeze_support()
    main()