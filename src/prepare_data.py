import os
import json
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
DATASET_DIR = PROJECT_DIR / "data" / "airbus-aircraft-detection"
IMAGES_DIR = DATASET_DIR / "images" / "img"
ANNOTATIONS_DIR = DATASET_DIR / "images" / "ann"

def load_and_crop_data(image_id, crop_box, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Load image and convert to float32 grayscale
    img_path = IMAGES_DIR / image_id
    annotation_path = ANNOTATIONS_DIR / f"{image_id}.json"
    if not img_path.is_file():
        raise FileNotFoundError(f"Dataset image not found: {img_path}")
    if not annotation_path.is_file():
        raise FileNotFoundError(f"Dataset annotation not found: {annotation_path}")

    img = Image.open(img_path).convert('L')
    
    img_array = np.array(img, dtype=np.float32)
    
    # 2. Crop scene
    x0, y0, x1, y1 = crop_box
    scene = img_array[y0:y1, x0:x1]
    
    # 3. Parse per-image polygon annotations
    gt_centers = []
    truncated_count = 0

    with annotation_path.open('r') as f:
        data = json.load(f)
    for obj in data.get('objects', []):
        if obj.get('classTitle') == 'Truncated_airplane':
            truncated_count += 1
            continue

        coords = np.asarray(obj['points']['exterior'], dtype=np.float32)
        cx = (coords[:, 0].min() + coords[:, 0].max()) / 2
        cy = (coords[:, 1].min() + coords[:, 1].max()) / 2

        # Adjust to cropped coordinates
        if x0 <= cx < x1 and y0 <= cy < y1:
            gt_centers.append((cx - x0, cy - y0))
        
    np.save(os.path.join(output_dir, f"{image_id}_scene.npy"), scene)
    np.save(os.path.join(output_dir, f"{image_id}_gt.npy"), np.array(gt_centers))
    
    print(f"[{image_id}] Scene size: {scene.shape}, GT Aircraft: {len(gt_centers)}, Truncated excluded: {truncated_count}")
    return scene, gt_centers

def extract_template(scene, center, size, output_dir, name):
    cx, cy = int(center[0]), int(center[1])
    half = size // 2
    template = scene[cy-half:cy+half, cx-half:cx+half]
    
    np.save(os.path.join(output_dir, f"{name}_template.npy"), template)
    
    plt.figure(figsize=(4, 4))
    plt.imshow(template, cmap='gray')
    plt.title(f"Template {name}")
    plt.savefig(os.path.join(output_dir, f"{name}_template.png"))
    plt.close()
    
    return template

if __name__ == "__main__":
    scene1_id = "c98a544c-4206-4176-b480-35b8bdf5bb14.jpg"
    scene2_id = "dacabde6-c8a7-475b-ab26-81b4e4dd5977.jpg"
    
    # We crop regions of 1500x1500 to keep computation reasonable
    crop1 = (0, 0, 1500, 1500)
    crop2 = (0, 0, 1500, 1500)
    
    output_dir = PROJECT_DIR / "data" / "processed"
    scene1, gt1 = load_and_crop_data(scene1_id, crop1, output_dir)
    load_and_crop_data(scene2_id, crop2, output_dir)
    
    # Crop a 60x60 template from the first aircraft in scene 1
    if len(gt1) > 0:
        extract_template(scene1, gt1[0], 60, output_dir, "main")