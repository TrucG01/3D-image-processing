import os
import cv2
import numpy as np
from tqdm import tqdm
import yaml

def load_config(config_path='config.yaml'):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def make_side_by_side_video(folder1, folder2, output_path, file_ext='.png', fps=30):
    set1 = set(f for f in os.listdir(folder1) if f.endswith(file_ext))
    set2 = set(f for f in os.listdir(folder2) if f.endswith(file_ext))
    common_files = sorted(list(set1 & set2))
    if not common_files:
        raise RuntimeError("No matching files found in both folders.")
    if len(set1) != len(set2):
        print(f"Warning: Only {len(common_files)} matching files found. Skipping unmatched files.")

    # Find first valid image pair for size
    for fname in common_files:
        img1 = cv2.imread(os.path.join(folder1, fname))
        img2 = cv2.imread(os.path.join(folder2, fname))
        if img1 is not None and img2 is not None:
            break
    else:
        raise RuntimeError("No valid image pairs found.")

    # Resize both images to same height
    target_height = min(img1.shape[0], img2.shape[0])
    img1 = cv2.resize(img1, (int(img1.shape[1] * target_height / img1.shape[0]), target_height))
    img2 = cv2.resize(img2, (int(img2.shape[1] * target_height / img2.shape[0]), target_height))
    width = img1.shape[1] + img2.shape[1]
    height = target_height

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    skipped = 0
    for fname in tqdm(common_files, desc="Building Video"):
        img1 = cv2.imread(os.path.join(folder1, fname))
        img2 = cv2.imread(os.path.join(folder2, fname))
        if img1 is None or img2 is None:
            print(f"Warning: Skipping pair {fname} due to missing/corrupt image.")
            skipped += 1
            continue
        img1 = cv2.resize(img1, (int(img1.shape[1] * target_height / img1.shape[0]), target_height))
        img2 = cv2.resize(img2, (int(img2.shape[1] * target_height / img2.shape[0]), target_height))
        combined = np.hstack((img1, img2))
        out.write(combined)

    out.release()
    print(f"Video saved to {output_path}. Skipped {skipped} pairs.")

if __name__ == "__main__":
    import sys
    config = load_config()
    default_folder1 = config['INPUT_DIR']
    default_folder2 = config['OUTPUT_DIR']
    default_ext = config.get('FILE_EXTENSION', '.png')
    default_output = f"side_by_side_{os.path.basename(default_folder1)}_{os.path.basename(default_folder2)}.mp4"

    if len(sys.argv) == 1:
        print(f"Using config.yaml: {default_folder1} + {default_folder2} -> {default_output} (ext: {default_ext})")
        make_side_by_side_video(default_folder1, default_folder2, default_output, file_ext=default_ext)
    elif len(sys.argv) == 4:
        make_side_by_side_video(sys.argv[1], sys.argv[2], sys.argv[3], file_ext=default_ext)
    else:
        print("Usage: python side_by_side_video.py [<folder1> <folder2> <output_video.mp4>]")
